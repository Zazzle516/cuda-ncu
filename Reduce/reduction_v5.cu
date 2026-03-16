#include <iostream>
#include <cuda_runtime.h>

#include <iostream>
#include <cuda_runtime.h>

__inline__ __device__ int warpReduceSum(int val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 现代版 Kernel：结合 Shared Memory 与 Warp Shuffle
__global__ void reduce_modern(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 1. 读取并进行第一次加法，保存在局部寄存器 mySum 中
    int mySum = g_idata[i] + g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    // 2. Block 级别的归约 (在 Shared Memory 中进行，直到步长缩小到 32)
    for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            // 更新寄存器 mySum 和共享内存
            mySum += sdata[tid + s];
            sdata[tid] = mySum; 
        }
        __syncthreads();
    }

    // 3. Warp 级别的归约 
    if (tid < 32) {
        // 【修复核心】：Shuffle 跨不出 32 线程的边界。
        // 在彻底脱离 Shared Memory 之前，必须手动把线程 32-63 算出的那一半结果拉过来！
        mySum += sdata[tid + 32];
        
        // 此时，前 32 个线程的 mySum 才真正集齐了所有数据。
        // 接下来放心交给 Shuffle 指令在寄存器层面完成最后的 16, 8, 4, 2, 1 归约
        mySum = warpReduceSum(mySum);
    }

    // 4. 将 Block 的最终归约结果写入全局内存
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

int main() {
    // 定义本次 reduce 的计算量 2^25 一维数据
    int N = 1 << 25; 
    size_t bytes = N * sizeof(int);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + (threadsPerBlock*2) - 1) / (threadsPerBlock*2);
    printf("blocksPerGrid: %d\n", blocksPerGrid);
    size_t smemSize = threadsPerBlock * sizeof(int);

    // 1. 分配 Host 端内存并初始化
    int *h_idata = new int[N];
    for (int i = 0; i < N; i++) {
        h_idata[i] = 1;
    }
    int *h_odata = new int[blocksPerGrid];

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

    // 5. 启动 Kernel 并计时 (修改为 reduce5)
    cudaEventRecord(start);
    reduce_modern<<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. 移动输出数据
    cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // 7. 在 CPU 端完成最后一步归约
    int final_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        final_sum += h_odata[i];
    }

    // 8. 打印结果和性能指标 (修改打印信息)
    std::cout << "--- Reduction Kernel 5: Unroll the Last Warp ---" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;
    std::cout << "Expected Sum: " << N << " | Actual Sum: " << final_sum << std::endl;

    if (final_sum == N) {
        std::cout << "Result: SUCCESS!" << std::endl;
    } else {
        std::cout << "Result: FAILED!" << std::endl;
    }

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

// nvcc -lineinfo reduction_v5.cu -o reduction_v5.exe