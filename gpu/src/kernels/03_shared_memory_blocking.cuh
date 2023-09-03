#include <stdio.h>
#define NN 2048
#define BLOCKSIZE 32

__global__ void gemmKernel(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    printf("x/row:%d y/column:%d\n", x, y);

    // If condition is necessary when M, N aren't multiples of 32 (warp size)
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = tmp; // x = row, y = column
    }
}

void gemmKernelLauncher(float *&A, float *&B, float *&C)
{
    dim3 gridDim(ceil(NN / 32), ceil(NN / 32), 1);
    dim3 blockDim(32 * 32);
    gemmKernel<<<gridDim, blockDim>>>(NN, NN, NN, A, B, C);
    return;
}