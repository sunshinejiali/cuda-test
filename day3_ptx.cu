#include <stdio.h>

__global__ void divergence_demo(float *out)
{
    int tid = threadIdx.x;
    float v;
    if(tid % 2 == 0) {
        v = sinf((float)tid);
    } else {
        v = cosf((float)tid);
    }
    out[tid] = v;
}

__global__ void no_divergence_demo(float *out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    // warp内分支完全一致，无divergence
    if (threadIdx.x >= 32)
    {
        val = sinf(1.0f);
    }
    else
    {
        val = cosf(2.0f);
    }
    out[tid] = val;
}

int main()
{
    int N = 64;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    divergence_demo<<<1,64>>>(d_out);
    no_divergence_demo<<<1,64>>>(d_out);

    cudaFree(d_out);
    printf("kernel finished\n");
    return 0;
}