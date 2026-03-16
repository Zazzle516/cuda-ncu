#include <iostream>
#include <cuda_runtime.h>

// 官方文档中的 Kernel 1: 存在严重分支分歧的交错寻址版本 
__global__ void reduce0(int *g_idata, int *g_odata) {
    // dynamic shared mem (smemSize) Inside Block
    extern __shared__ int sdata[];

    // 每个线程从 HBM 加载一个元素到 sdata
    // threadIdx.x [0-255]; blockIdx.x [0-(2^17-1)]; blockDim.x=256
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];

    // 对 Block 内部元素的加载进行同步
    __syncthreads();

    // 在共享内存中进行树状归约计算
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // 每次规约一半的元素 log2(256)=8 需要循环 8 次
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将当前 Block 的归约结果写回全局内存
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    // 定义本次 reduce 的计算量 2^25 一维数据
    int N = 1 << 25; 
    size_t bytes = N * sizeof(int);

    int threadsPerBlock = 256;
    // 2^25 / 256 => BlockNum 如果计算量无法整除则还需要一个完整的 Block (2^17)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("blocksPerGrid: %d\n", blocksPerGrid);
    // 向 kernel 传入的 smemSize 单位是 byte 并且共享局限在 Block 内部
    size_t smemSize = threadsPerBlock * sizeof(int);

    // 1. 分配 Host 端内存并初始化 (结果为 N)
    int *h_idata = new int[N];
    for (int i = 0; i < N; i++) {
        h_idata[i] = 1;
    }
    int *h_odata = new int[blocksPerGrid];  // GPU 以 Block 为单位的并行结束后会产生 2^17 个结果

    // 2. 分配 Device 端内存
    int *d_idata, *d_odata;
    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, blocksPerGrid * sizeof(int));

    // 3. 移动输入数据
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    // 4. 设置 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 5. 启动 Kernel 并计时
    cudaEventRecord(start);
    reduce0<<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. 移动输出数据
    cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // 7. 在 CPU 端完成最后一步归约 (对 2^17 个结果求和)
    int final_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        final_sum += h_odata[i];
    }

    // 8. 打印结果和性能指标
    std::cout << "--- Reduction Kernel 1: Interleaved Addressing ---" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;
    std::cout << "Expected Sum: " << N << " | Actual Sum: " << final_sum << std::endl;
    
    if (final_sum == N) {
        std::cout << "Result: SUCCESS!" << std::endl;
    } else {
        std::cout << "Result: FAILED!" << std::endl;
    }

    // 计算带宽：总量 = 输入字节数 + 输出结果字节数
    // 实际归约操作是受限于内存带宽的 (bandwidth-bound) 
    double totalBytes = bytes + (blocksPerGrid * sizeof(int));
    double bandwidth = (totalBytes / 1e9) / (milliseconds / 1000.0);
    
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 9. 清理内存
    cudaFree(d_idata);
    cudaFree(d_odata);
    delete[] h_idata;
    delete[] h_odata;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// nvcc -lineinfo reduction_v1.cu -o reduction_v1.exe