#include <iostream>
#include <vector>

#define NN 2048
#define BLOCKSIZE 32

__global__ void gemmKernel(int M, int N, int K, const float *A, const float *B, float *C)
{
    // const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

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
    dim3 gridDim(ceil(NN / BLOCKSIZE), ceil(NN / BLOCKSIZE), 1);
    // Flatten the block dim
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    gemmKernel<<<gridDim, blockDim>>>(NN, NN, NN, A, B, C);
    return;
}