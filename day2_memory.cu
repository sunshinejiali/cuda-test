// 矩阵的转置
/*
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx_x = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_y = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx_x < rows && idx_y < cols) {
        int cur = idx_x * cols + idx_y;
        output[idx_y * rows + idx_x] = input[cur];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
*/

/*
### 任务 1：朴素矩阵乘法（Global Memory 版本，无 shared mem）

就是你刚才写的矩阵乘法核函数：

- A[M,K], B[K,N], C[M,N]
- 每个线程负责输出 C 矩阵**一个元素**
- 线程索引：`row = blockIdx.y * blockDim.y + threadIdx.y; col = blockIdx.x * blockDim.x + threadIdx.x;`
- 循环累加 `C[row*N + col] += A[row*K + k] * B[k*N + col]`

> 缺点：反复读 Global 显存，性能很差，用来做基准版本。
> 

编译运行，加计时，测出执行耗时。
*/


/*
核心思路：
把 A、B 小块加载进 Block 内

**shared memory**

，利用 shared memory 低延迟，减少全局显存访问次数。
步骤：

1. Block 先把 A tile、B tile 从 global 读到 shared
2. Block 内所有线程从 shared 读取数据做乘累加
3. 同步 `__syncthreads()`：同一个 block 线程屏障，**必须！防止读脏数据**

> `__syncthreads()`：只在 block 内部生效，不要在分支里乱调用。
> 

> 要求：对比朴素版 vs shared 分块版的运行时间，观察加速比。
>
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t error = call; \
    if(error != cudaSuccess) { \
        printf("ERROR Found in CUDA: error(%s)", cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

const int WIDTH = 16;

__global__ void multiple_metrixs(const float* A, const float* B, float* C, int M, int K, int N) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx_x < M && idx_y < N) {
        float sum = 0.0;
        for(int i = 0; i < K; i++) {
            sum += A[idx_x * K + i] * B[i * N + idx_y];
        }
        C[idx_x * N + idx_y] = sum;
        printf("idx_x:%d, idx_y:%d, C[%d]:%f\n", idx_x, idx_y, idx_x * N + idx_y, sum);
    }
}

__global__ void multiple_metrixs_2(const float* A, const float* B, float* C, int M, int K, int N) {

    __shared__ float As[WIDTH][WIDTH];
    __shared__ float Bs[WIDTH][WIDTH];

    int cur_x = threadIdx.x;
    int cur_y = threadIdx.y;

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;

    for(int i = 0; i < (K + WIDTH - 1)/WIDTH; i++) {
        // A[idx_x][i * WIDTH + cur_y]
        if(idx_x < M && (i * WIDTH + cur_y) < K) {
            As[cur_x][cur_y] = A[idx_x * K + i * WIDTH + cur_y];
        } else {
            As[cur_x][cur_y] = 0.0f;
        }
        // B[i * WIDTH + cur_x][idx_y]
        int temp = i * WIDTH + cur_x;
        if(idx_y < N && temp < K) {
            Bs[cur_x][cur_y] = B[temp * N + idx_y];
        } else {
            Bs[cur_x][cur_y] = 0.0f;
        }
        __syncthreads();

        for(int j = 0; j < WIDTH; j++) {
            sum += As[cur_x][j] * Bs[j][cur_y];
        }
        __syncthreads();
    }
    if(idx_x < M && idx_y < N) {
        C[idx_x * N + idx_y] = sum;
    }
}

int main() {
    int M, K, N;
    M = 2;
    K = 3;
    N = 4;

    // 分配主机内存
    float * A, *B, *C;
    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(N * K * sizeof(float));
    C = (float*)malloc(M * N * sizeof(float));

    // 赋值A B
    for(int i = 0; i < M * K; i++) {
        A[i] = static_cast<float>(i + 1); // A = [[1, 2, 3],[4, 5, 6]]
    }
    for(int i = 0; i < K * N; i++) {
        B[i] = static_cast<float>(i + 1); // B = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]]
    }

    // 设备内存分配
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(float)));


    // 拷贝
    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));

    // 调用
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid((threadPerBlock.x + M - 1) / threadPerBlock.x, (threadPerBlock.y + N - 1) / threadPerBlock.y);
    


    //=======================Method 1========================//
    // 增加计时器
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    multiple_metrixs<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // 计算耗时
    float mill_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&mill_time, start, stop));
    printf("Time: %f\n", mill_time);


    // 拷贝
    CHECK_CUDA(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));


    // 输出结果
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf(" %f", C[i * N + j]);
        }
        printf("\n");
    }


    //=======================Method 2========================//

    CHECK_CUDA(cudaEventRecord(start));
    
    multiple_metrixs_2<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // 计算耗时
    mill_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&mill_time, start, stop));
    printf("Time2: %f\n", mill_time);


    // 拷贝
    CHECK_CUDA(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));


    // 输出结果
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            printf(" %f", C[i * N + j]);
        }
        printf("\n");
    }


    //释放设备空间
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));


    free(A);
    free(B);
    free(C);


    return 0;
}