#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

int main() {
    unsigned int N = 1 << 25; 
    size_t bytes = N * sizeof(int);

    // 1. 分配 Host 端内存并初始化
    int *h_idata = new int[N];
    for (unsigned int i = 0; i < N; i++) {
        h_idata[i] = 1;
    }

    // CUB reduce 在显存中完成最后的规约  只需要一个元素空间
    int h_odata = 0; 

    // 2. 分配 Device 端内存
    int *d_idata = nullptr;
    int *d_odata = nullptr;
    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, sizeof(int));

    // 3. 移动输入数据到 GPU
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // 第一步：干跑一次 (传入 d_temp_storage = nullptr)。
    // 这一步不会做任何实际计算，而是让 CUB 根据你的数据类型、大小和当前 GPU 架构，
    // 算一下内部（比如 Decoupled Look-back）需要多少额外的临时显存。
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_idata, d_odata, N);

    // 分配 CUB 请求的临时显存
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 4. 设置 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 5. 正式启动 CUB Kernel 并计时
    // 注意：我们只计时真正发生计算的这一次调用
    cudaEventRecord(start);
    
    // 第二步：传入真正分配好的 d_temp_storage，正式执行归约
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_idata, d_odata, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. 将唯一的最终结果拷回 CPU
    cudaMemcpy(&h_odata, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

    // 7. 打印结果和性能指标
    std::cout << "--- Industrial Standard: CUB DeviceReduce::Sum ---" << std::endl;
    std::cout << "Array Size: " << N << " elements" << std::endl;
    std::cout << "Expected Sum: " << N << " | Actual Sum: " << h_odata << std::endl;

    if (h_odata == (int)N) {
        std::cout << "Result: SUCCESS!" << std::endl;
    } else {
        std::cout << "Result: FAILED!" << std::endl;
    }

    // 计算有效带宽
    double totalBytes = bytes + sizeof(int);
    double bandwidth = (totalBytes / 1e9) / (milliseconds / 1000.0);
    
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 8. 清理内存
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_temp_storage);
    delete[] h_idata;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// nvcc -O3 -lineinfo cub_reduce.cu -o cub_reduce

// D:\cuda-ncu\Reduce>cub_reduce.exe
// --- Industrial Standard: CUB DeviceReduce::Sum ---
// Array Size: 33554432 elements
// Expected Sum: 33554432 | Actual Sum: 33554432
// Result: SUCCESS!
// Execution Time: 0.763328 ms
// Effective Bandwidth: 175.832 GB/s