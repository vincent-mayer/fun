#pragma once

#include <stdio.h>

template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void blocktiling1DGemmKernel(int M, int N, int K, const float *A,
                                        const float *B, float *C)
{
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // Blockdim is 1D (only x). Translate index between 0-1023 to index in (32, 32) block
    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    // Allocate buffer for current block in shared mem
    // Shared mem is shared between all threads of a block
    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];

    // Advance pointers to the starting position
    A += cRow * K * BM;
    B += cCol * BN;
    C += cRow * N * BM + cCol * BN;

    //
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing

    // allocate thread-level cache for results in register file
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (int bIdx = 0; bIdx < K; bIdx += BK)
    {
        // Each thread loads a single element into shared mem. Since thread rows are
        // consecutive, memory accesses are coalesed.
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        // Block level synchronization barrier, makes sure SMEM is fully populated.
        __syncthreads();

        // Advance A and B to next chunk.
        A += BK;
        B += BK * N;

        // Perform dot product over row vector of block in A and col vector of block in B
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            for (int resIdx = 0; resIdx < TM; ++resIdx)
            {
            }
        }
        tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];

        // Need to synch again at the end to avoid faster threads fetching the next block
        // into cache before slower threads are done.
        __syncthreads();
    }
    C[threadRow * N + threadCol] = tmp;
}