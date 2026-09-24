/*
1. Atomic_Add函数的作用是对一个整数变量进行原子加操作，确保在多线程环境下对该变量的访问是安全的。它的函数原型如下：
   __device__ int atomicAdd(int* address, int val);
   其中，address是指向要进行加操作的整数变量的指针，val是要加的值。atomicAdd函数会将val加到address所指向的整数变量上，并返回加之前的值。


   1. 场景：多个线程**同时写同一个全局内存地址**，普通`+=`会发生写丢失（数据竞争）
    `atomicAdd(&addr, val)`：硬件保证**读-改-写**原子完成，不会冲突
    代价：慢！**原子操作会序列化，大量原子会严重降速**
    示例：统计数组内大于阈值的元素总和
*/


#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA ERROR: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void auto_add_kernel(const int *A, const int *level, int *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && A[idx] > *level) {
        atomicAdd(result, A[idx]);
    }
}

__global__ void auto_add_float_kernel_seq(const float *A, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, A[idx]);
    }
}

__global__ void auto_add_float_kernel(const float *A, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 用来对比测试性能
        *result += A[idx]; // 这里是错误的，应该使用atomicAdd
    }
}


int main() {
    /* Example: 使用atomicAdd统计数组中大于阈值的元素总和 */
    /*
    int N = 10, level = 5, result = 0;
    int *h_A = new int[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
    }


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int *d_A, *level_device, *result_device;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&level_device, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&result_device, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(level_device, &level, sizeof(int), cudaMemcpyHostToDevice));

    auto_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, level_device, result_device, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&result, result_device, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Result: %d\n", result);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(level_device));
    CUDA_CHECK(cudaFree(result_device));
    delete[] h_A;
    */

    int N = 100000;
    float result = 0;
    float *h_A = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
    }


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_A, *result_device;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&result_device, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result_device, &result, sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    auto_add_float_kernel_seq<<<blocksPerGrid, threadsPerBlock>>>(d_A, result_device, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Seq Time: %f ms\n", milliseconds); 
    CUDA_CHECK(cudaMemcpy(&result, result_device, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Seq Result: %f\n", result);


    //Test 2
    result = 0;
    CUDA_CHECK(cudaMemcpy(result_device, &result, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    auto_add_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, result_device, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time: %f ms\n", milliseconds); 


    CUDA_CHECK(cudaMemcpy(&result, result_device, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Result: %f\n", result);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(result_device));
    delete[] h_A;

    /*result:
        keyvalue@keyvalue-Z390-GAMING-X:cuda-test$ ./day4_practice 
        Seq Time: 0.440576 ms
        Seq Result: 4999916032.000000
        Time: 0.021664 ms
        Result: 201376.000000
    */

    return 0;
}