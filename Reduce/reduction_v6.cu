#include <iostream>
#include <cuda_runtime.h>

// 现代设备函数：使用寄存器层面的 Shuffle 完成最后 32 个线程的归约
__inline__ __device__ int warpReduceSum(int val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// 现代版 Kernel V6：完全展开循环 + Warp Shuffle
template <unsigned int blockSize>
__global__ void reduce6_modern(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    
    // 1. 读取并进行第一次加法，保存在局部寄存器 mySum 中
    int mySum = g_idata[i] + g_idata[i + blockSize];
    sdata[tid] = mySum;
    __syncthreads();

    // 2. Block 级别的归约：利用模板参数在编译期完全展开
    // 编译器会自动消除不符合条件的死代码
    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();
    }
    if (blockSize >= 64) {
        // 当执行完这一步，tid 0~31 已经集齐了所有需要的数据
        if (tid < 32) { mySum = mySum + sdata[tid + 32]; } __syncthreads();
    }

    // 3. Warp 级别的归约 
    if (tid < 32) {
        // 由于上面 if (blockSize >= 64) 的展开已经把数据汇聚到了前 32 个线程
        // 我们不需要像 V5 那样再手动拉取 sdata[tid + 32] 了，直接放心交给 Shuffle！
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
    int blocksPerGrid = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
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

    // 5. 启动 Kernel 并计时 (根据 threadsPerBlock 动态派发模板)
    cudaEventRecord(start);
    switch (threadsPerBlock) {
        case 1024:
            reduce6_modern<1024><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata); break;
        case 512:
            reduce6_modern<512><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata); break;
        case 256:
            reduce6_modern<256><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata); break;
        case 128:
            reduce6_modern<128><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata); break;
        case 64:
            reduce6_modern<64><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata); break;
        case 32:
            reduce6_modern<32><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata); break;
        default:
            std::cerr << "Unsupported block size! Please use a power of 2 between 32 and 1024." << std::endl;
            break;
    }
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

    // 8. 打印结果和性能指标
    std::cout << "--- Reduction Kernel 6: Completely Unroll (Modern) ---" << std::endl;
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

// nvcc -lineinfo reduction_v6.cu -o reduction_v6.exe