#pragma once

#include <stdio.h>

template <const uint BLOCKSIZE>
__global__ void sharedMemGemmKernel(int M, int N, int K, const float *A, const float *B,
                                    float *C)
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