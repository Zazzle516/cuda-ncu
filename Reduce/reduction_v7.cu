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

// 现代版 Kernel V7: Algorithm Cascading + 完全展开 + Warp Shuffle
template <unsigned int threadPerBlock>
__global__ void reduce7_modern(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (threadPerBlock * 2) + tid;
    // 在之前的版本中  每个线程负责自己的元素读取
    // 现在线程串行执行  数据读取也要串行  刚好跨过 1 个 Grid 的距离
    // gridDim.x = [0, ..., 1024/2048]
    unsigned int gridSize = threadPerBlock * 2 * gridDim.x;

    // 1. 每个线程使用 while 循环处理多个元素  累加到本地寄存器
    int mySum = 0;
    while (i < n) {
        mySum += g_idata[i];
        if (i + threadPerBlock < n) {
            mySum += g_idata[i + threadPerBlock];
        }
        i += gridSize;
    }

    // 将线程自己串行累加的结果放入共享内存  进入并行归约阶段
    sdata[tid] = mySum;
    __syncthreads();

    // 2. Block 级别的归约：利用模板参数在编译期完全展开
    if (threadPerBlock >= 1024) {
        if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads();
    }
    if (threadPerBlock >= 512) {
        if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();
    }
    if (threadPerBlock >= 256) {
        if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
    }
    if (threadPerBlock >= 128) {
        if (tid < 64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();
    }
    if (threadPerBlock >= 64) {
        if (tid < 32) { mySum = mySum + sdata[tid + 32]; } __syncthreads();
    }

    // 3. Warp 级别的归约 
    if (tid < 32) {
        mySum = warpReduceSum(mySum);
    }

    // 4. 将 Block 的最终归约结果写入全局内存
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

int main() {
    unsigned int N = 1 << 25; 
    size_t bytes = N * sizeof(int);
    int threadsPerBlock = 256;

    int maxBlocks = 576;
    int blocksPerGrid = std::min(maxBlocks, (int)((N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2)));
    
    std::cout << "blocksPerGrid: " << blocksPerGrid << " (Capped at " << maxBlocks << ")" << std::endl;
    size_t smemSize = threadsPerBlock * sizeof(int);

    // 1. 分配 Host 端内存并初始化
    int *h_idata = new int[N];
    for (unsigned int i = 0; i < N; i++) {
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

    // 5. 启动 Kernel 并计时  Tip: 传入 N
    cudaEventRecord(start);
    switch (threadsPerBlock) {
        case 1024:
            reduce7_modern<1024><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 512:
            reduce7_modern<512><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 256:
            reduce7_modern<256><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 128:
            reduce7_modern<128><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 64:
            reduce7_modern<64><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 32:
            reduce7_modern<32><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        default:
            std::cerr << "Unsupported block size!" << std::endl;
            break;
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. 移动输出数据
    cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // 7. 在 CPU 端完成最后一步归约  相比于前几个版本对 2^17 个结果  这里计算大幅减少
    int final_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        final_sum += h_odata[i];
    }

    // 8. 打印结果和性能指标
    std::cout << "--- Reduction Kernel 7: Multiple Elements Per Thread ---" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;
    std::cout << "Expected Sum: " << N << " | Actual Sum: " << final_sum << std::endl;

    if (final_sum == (int)N) {
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

// nvcc -lineinfo reduction_v7.cu -o reduction_v7.exe
