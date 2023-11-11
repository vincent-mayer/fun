#pragma once

template <const uint TILESIZE>
__global__ void sharedMem2dGemmKernel(const int N, const float *A, const float *B,
                                      float *C)
{
    uint row = blockIdx.y * blockDim.y + threadIdx.y;
    uint col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    float sum = 0.0;

    for (int tileIdx = 0; tileIdx < N / TILESIZE; ++tileIdx)
    {
        As[threadIdx.y][threadIdx.x] = A[row * N + tileIdx * TILESIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[col + tileIdx * N * TILESIZE + threadIdx.y * N];

        __syncthreads();

        // Perform work in the tile leveraging data in shared mem
        for (int i = 0; i < TILESIZE; ++i)
        {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }
    C[row * N + col] = sum;
}