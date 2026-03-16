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

// Kernel V9：int4 向量化访存 + 4-Way ILP + Grid-Stride + Warp Shuffle
// n = N = 2^25
template <unsigned int blockSize>
__global__ void reduce8_vectorized(const int* __restrict__ g_idata, int* __restrict__ g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;   // blockDim.x = threadsPerBlock = 256, gridDim.x = maxBlocks = 576

    // 1. 将原数组映射为 int4 视图
    const int4* g_idata4 = reinterpret_cast<const int4*>(g_idata);

    // 重新计算原本 2^25 个 int 元素按照 (int4) 读取的话有多少个
    unsigned int n4 = n / 4; 

    // 定义 4 个独立的累加器  利用 ILP  消除数据依赖
    int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

    // 2. 不再对 tid 进行 Block 范围判断  消除 Warp 分歧
    unsigned int i = global_tid;    // global_tid: [0, 256x576]
    while (i < n4) {
        int4 val = g_idata4[i];     // 

        // 加法并发
        sum0 += val.x;
        sum1 += val.y;
        sum2 += val.z;
        sum3 += val.w;
        
        i += stride;
    }

    // 将 4 个累加器合并成一个本地总和
    int mySum = sum0 + sum1 + sum2 + sum3;

    // 3. Tail Loop)：处理不能被 4 整除的零头数据 (最多 3 个元素)
    unsigned int tail_start = n4 * 4;
    unsigned int tail_idx = tail_start + global_tid;
    while (tail_idx < n) {
        mySum += g_idata[tail_idx];
        tail_idx += stride;
    }

    // 4. 将线程串行累加的结果放入共享内存，进入 Block 并行归约阶段
    sdata[tid] = mySum;
    __syncthreads();

    // 5. Block 级别的归约：利用模板参数在编译期完全展开，消除分支开销
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
        if (tid < 32) { mySum = mySum + sdata[tid + 32]; } __syncthreads();
    }

    // 6. Warp 级别的归约：避免使用 volatile 共享内存，改用现代 Shuffle 指令
    if (tid < 32) {
        mySum = warpReduceSum(mySum);
    }

    // 7. 将 Block 的最终归约结果写入全局内存
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

int main() {
    // 定义本次 reduce 的计算量 2^25 (约 3355 万个元素)
    unsigned int N = 1 << 25; 
    size_t bytes = N * sizeof(int);

    int threadsPerBlock = 256;
    
    // Grid 尺寸上限，对于现代 GPU 1024 到 2048 足以填满所有的 SM (Streaming Multiprocessors)
    int maxBlocks = 576; 
    int blocksPerGrid = std::min(maxBlocks, (int)((N + threadsPerBlock - 1) / threadsPerBlock));
    
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

    // 5. 启动 Kernel 并计时
    cudaEventRecord(start);
    switch (threadsPerBlock) {
        case 1024:
            reduce8_vectorized<1024><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 512:
            reduce8_vectorized<512><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 256:
            reduce8_vectorized<256><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 128:
            reduce8_vectorized<128><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 64:
            reduce8_vectorized<64><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 32:
            reduce8_vectorized<32><<<blocksPerGrid, threadsPerBlock, smemSize>>>(d_idata, d_odata, N); break;
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

    // 7. CPU 端收尾：对每个 Block 产生的 Partial Sum 进行最后的累加
    int final_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        final_sum += h_odata[i];
    }

    // 8. 打印结果和性能指标
    std::cout << "--- Reduction Kernel 9: Vectorized int4 + 4-Way ILP ---" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;
    std::cout << "Expected Sum: " << N << " | Actual Sum: " << final_sum << std::endl;

    if (final_sum == (int)N) {
        std::cout << "Result: SUCCESS!" << std::endl;
    } else {
        std::cout << "Result: FAILED!" << std::endl;
    }

    // 注意：这里的内存读写量只算实际发生的 (读取整个数组 + 写入 partial sums)
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

// nvcc reduction_v8.cu -o reduction_v8.exe

// nvcc -lineinfo reduction_v8.cu -o reduction_v8.exe

// D:\cuda-ncu\Reduce>reduction_v8.exe
// blocksPerGrid: 576 (Capped at 576)
// --- Reduction Kernel 9: Vectorized int4 + 4-Way ILP ---
// Array Size: 33554432 elements
// Expected Sum: 33554432 | Actual Sum: 33554432
// Result: SUCCESS!
// Execution Time: 1.08778 ms
// Effective Bandwidth: 123.389 GB/s