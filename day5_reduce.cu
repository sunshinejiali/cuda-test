#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
keyvalue@keyvalue-Z390-GAMING-X:cuda-test$ nvcc ./day5_reduce.cu -o day5_reduction
keyvalue@keyvalue-Z390-GAMING-X:cuda-test$ ./day5_reduction 
Atomic     : 3.008 ms  result=1048576.0
Naive reduce: 0.072 ms  result=1048576.0
Opt reduce : 0.067 ms  result=1048576.0
Expected   : 1048576.0
*/


#define CUDA_CHECK_API(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("API ERROR: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_KERNEL_LAUNCH() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("KERNEL LAUNCH ERROR: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define BLOCK_SIZE 256

// ========== 版本1：原子直接累加（baseline） ==========
__global__ void kernel_atomic(const float* A, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(result, A[idx]);
    }
}

// ========== 版本2：朴素规约（有 bank conflict） ==========
// block 内所有线程的数据先加载进 shared memory，然后两两配对累加，
// 不断缩小结果范围，最后 tid=0 保存这个 block 的总和，再用 atomicAdd 把所有 block 的总和汇总到全局 result
__global__ void kernel_reduce_naive(const float* A, float* result, int N) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载到 shared，越界补0
    // 每个线程 tid，把全局数组 A 的一个元素读到共享内存 s_data[tid]
    s_data[tid] = (idx < N) ? A[idx] : 0.0f;
    __syncthreads();

    // 朴素配对：step 从1翻倍，有 bank conflict
    for (int step = 1; step < BLOCK_SIZE; step *= 2) {
        if (tid % (2*step) == 0) {
            s_data[tid] += s_data[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, s_data[0]);
    }
}

// ========== 版本3：交错规约（无 bank conflict，推荐） ==========
__global__ void kernel_reduce_opt(const float* A, float* result, int N) {
    __shared__ float s_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] = (idx < N) ? A[idx] : 0.0f;
    __syncthreads();

    // 交错配对：step 从 BLOCK_SIZE/2 减半，无 bank conflict
    for (int step = BLOCK_SIZE / 2; step > 0; step /= 2) {
        if (tid < step) {
            s_data[tid] += s_data[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, s_data[0]);
    }
}

int main() {
    int N = 1 << 20;  // 1M 元素
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h_A[i] = 1.0f;  // 全1，总和=N

    float *d_A, *d_result;
    CUDA_CHECK_API(cudaMalloc(&d_A, bytes));
    CUDA_CHECK_API(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK_API(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    CUDA_CHECK_API(cudaEventCreate(&start));
    CUDA_CHECK_API(cudaEventCreate(&stop));

    // ---- Test 1: atomic ----
    float h_result = 0.0f;
    CUDA_CHECK_API(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_API(cudaEventRecord(start));
    kernel_atomic<<<blocks, BLOCK_SIZE>>>(d_A, d_result, N);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK_API(cudaDeviceSynchronize());
    CUDA_CHECK_API(cudaEventRecord(stop));
    CUDA_CHECK_API(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK_API(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK_API(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Atomic     : %.3f ms  result=%.1f\n", ms, h_result);

    // ---- Test 2: naive reduce ----
    h_result = 0.0f;
    CUDA_CHECK_API(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_API(cudaEventRecord(start));
    kernel_reduce_naive<<<blocks, BLOCK_SIZE>>>(d_A, d_result, N);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK_API(cudaDeviceSynchronize());
    CUDA_CHECK_API(cudaEventRecord(stop));
    CUDA_CHECK_API(cudaEventSynchronize(stop));
    CUDA_CHECK_API(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK_API(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Naive reduce: %.3f ms  result=%.1f\n", ms, h_result);

    // ---- Test 3: optimized reduce ----
    h_result = 0.0f;
    CUDA_CHECK_API(cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_API(cudaEventRecord(start));
    kernel_reduce_opt<<<blocks, BLOCK_SIZE>>>(d_A, d_result, N);
    CUDA_CHECK_KERNEL_LAUNCH();
    CUDA_CHECK_API(cudaDeviceSynchronize());
    CUDA_CHECK_API(cudaEventRecord(stop));
    CUDA_CHECK_API(cudaEventSynchronize(stop));
    CUDA_CHECK_API(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK_API(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Opt reduce : %.3f ms  result=%.1f\n", ms, h_result);

    printf("Expected   : %.1f\n", (float)N);

    CUDA_CHECK_API(cudaEventDestroy(start));
    CUDA_CHECK_API(cudaEventDestroy(stop));
    CUDA_CHECK_API(cudaFree(d_A));
    CUDA_CHECK_API(cudaFree(d_result));
    free(h_A);
    return 0;
}