#pragma once

__global__ void naiveGemmKernel(int M, int N, int K, const float *A, const float *B,
                                float *C)
{
    // CUDA compute ordered in 3-level hierarchy
    // invocation -> grid -> blocks -> up to 1024 threads
    // Threads share memory of block
    // # of threads = blockDim
    // # of blocks = gridDim
    // blocks contain warps, warps contain 32 threads

    // threads of the same warp are determined by threadIdx.x
    // threadIdx.x influences x, which deterioned the row in the output matrix
    // jumping rows in the output means jumping rows in the input A, while keeping the B
    // column constant
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // If condition is necessary when M, N aren't multiples of 32 (warp size)
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = tmp; // x = row, y = column
    }
}