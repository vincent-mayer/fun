#pragma once

__global__ void naivelyCoalescedGemmKernel(int M, int N, int K, const float *A,
                                           const float *B, float *C)
{
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint x = blockIdx.y * blockDim.y + threadIdx.y;

    // If condition is necessary when M, N aren't multiples of 32 (warp size)
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = tmp; // x = row, y = column
    }
}