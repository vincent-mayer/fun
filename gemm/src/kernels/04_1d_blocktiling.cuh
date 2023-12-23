#pragma once

#include <stdio.h>

template <const uint TILESIZE, const uint RESULTS_PER_THREAD>
__global__ void blocktiling1DGemmKernel(int N, const float *A, const float *B, float *C)
{
    // Global row and column in output matrixe
    uint grow = blockIdx.y * blockDim.y + threadIdx.y * RESULTS_PER_THREAD;
    uint gcol = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILESIZE][TILESIZE / RESULTS_PER_THREAD];
    __shared__ float Bs[TILESIZE / RESULTS_PER_THREAD][TILESIZE];

    float threadResults[RESULTS_PER_THREAD] = {0.0};
    const uint nTilesCommon = N / (TILESIZE / RESULTS_PER_THREAD);

    for (int tileIdx = 0; tileIdx < nTilesCommon; ++tileIdx)
    {
        for (int i = 0; i < RESULTS_PER_THREAD; ++i)
        {
            As[threadIdx.y + i][threadIdx.x] =
                A[(grow + i) * N + tileIdx * (TILESIZE / RESULTS_PER_THREAD) +
                  threadIdx.x];
        }
        Bs[threadIdx.y][threadIdx.x] =
            B[gcol + tileIdx * N * (TILESIZE / RESULTS_PER_THREAD) + threadIdx.y * N];

        __syncthreads();

        // Perform work in the tile leveraging data in shared mem
        // Loop over the dot product dimension of the tile (common dimension)
        for (int i = 0; i < TILESIZE / RESULTS_PER_THREAD; ++i)
        {
            float tmpB = Bs[i][threadIdx.x];
            // Additionally compute multiple outputs
            for (int resIdx = 0; resIdx < RESULTS_PER_THREAD; ++resIdx)
            {
                threadResults[resIdx] +=
                    As[threadIdx.y * RESULTS_PER_THREAD + resIdx][i] * tmpB;
            }
        }

        __syncthreads();
    }
    for (int resIdx = 0; resIdx < RESULTS_PER_THREAD; ++resIdx)
        C[(grow + resIdx) * N + gcol] = threadResults[resIdx];
}