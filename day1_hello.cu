/*
| NVIDIA-SMI 595.71.05              Driver Version: 595.71.05      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |                        |                    0 |
| N/A   34C    P8             13W /   70W |       5MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A                     G                                             |
+-----------------------------------------------------------------------------------------+

*/


// #include <stdio.h>

// // GPU上每个线程都会执行这一段代码
// __global__ void hello_gpu() {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     printf("GPU thread %d: Hello CUDA\n", tid);
// }

// int main() {
//     // kernel<<<grid, block>>>();
//     hello_gpu<<<2,4>>>();
//     /*
//     threadIdx.x：线程在 block 内的编号
//     blockIdx.x：block 在 grid 内的编号
//     blockDim.x：一个 block 里面有多少个 thread
//     gridDim.x：grid 里面有多少 block
//     因此全局线程id：blockIdx.x * blockDim.x + threadIdx.x
//     */
//     cudaDeviceSynchronize();
//     return 0;
// }

#include <cuda_runtime.h>
#include <stdio.h>

// 封装错误检查宏，以后所有CUDA调用都加这个
#define CHECK_CUDA(err)\
if(err != cudaSuccess) {\
    printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__);\
    exit(1);\
}

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        printf("GPU thread %d: A[%d] = %f, B[%d] = %f\n", idx, idx, A[idx], idx, B[idx]);
        C[idx] = A[idx] + B[idx];
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // <<<blocksPerGrid, threadsPerBlock>>>是CUDA kernel的启动配置，
    // 表示启动blocksPerGrid个block，
    // 每个block包含threadsPerBlock个线程。
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    // host内存
    float *h_A, *h_B, *h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];
    for(int i=0;i<N;i++){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // device显存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    // host -> device 拷贝
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 调用solve，传入device指针
    solve(d_A, d_B, d_C, N);

    // device -> host 拿回结果
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 校验结果
    bool ok = true;
    for(int i=0;i<N;i++){
        if(h_C[i] != 3.0f) ok = false;
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    // 释放
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}