#pragma once

#include <stdio.h>

template <const uint TILESIZE_X, const uint TILESIZE_Y, const uint RESULTS_PER_THREAD>
__global__ void blocktiling1DGemmKernel(int N, const float *A, const float *B, float *C)
{
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILESIZE_Y][TEILSIZE_X / RESULTS_PER_THREAD];
    __shared__ float Bs[TILESIZE_X / RESULTS_PER_THREAD][TILESIZE_X];

    float threadResults[RESULTS_PER_THREAD] = {0};

    for (int tileIdx = 0; tileIdx < N / TILESIZE; ++tileIdx)
    {
        As[][] = A[];
        Bs[][] = B[];

        __syncthreads();

        // Perform work in the tile leveraging data in shared mem
        for (int i = 0; i < TILESIZE; ++i)
        {
            float tmpB = Bs[][];
            for (int resIdx = 0; resIdx < RESULTS_PER_THREAD; ++resIdx)
            {
                threadResults[resIdx] += As * tmpB;
            }
        }

        __syncthreads();
    }
    for (int resIdx = 0; resIdx < RESULTS_PER_THREAD; ++resIdx)
        C[row * N + col] = threadResults[resIdx];
}