#pragma once

#include <stdio.h>

template <const uint BLOCKSIZE>
__global__ void sharedMemGemmKernel(int M, int N, int K, const float *A, const float *B,
                                    float *C)
{
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // Blockdim is 1D (only x). Translate index between 0-1023 to index in (32, 32) block
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    // Allocate buffer for current block in shared mem
    // Shared mem is shared between all threads of a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // Advance pointers to the starting position
    A += cRow * K * BLOCKSIZE;
    B += cCol * BLOCKSIZE;
    C += cRow * N * BLOCKSIZE + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bIdx = 0; bIdx < K; bIdx += BLOCKSIZE)
    {
        // Each thread loads a single element into shared mem. Since thread rows are
        // consecutive, memory accesses are coalesed.
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // Block level synchronization barrier, makes sure SMEM is fully populated.
        __syncthreads();

        // Advance A and B to next chunk.
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // Perform dot product over row vector of block in A and col vector of block in B
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            tmp +=
                As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];

        // Need to synch again at the end to avoid faster threads fetching the next block
        // into cache before slower threads are done.
        __syncthreads();
    }
    C[threadRow * N + threadCol] = tmp;
}