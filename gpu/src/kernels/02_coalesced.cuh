#pragma once

template <const uint BLOCKSIZE>
__global__ void coalescedGemmKernel(int M, int N, int K, const float *A, const float *B,
                                    float *C)
{
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // Allocate buffer for current block in shared mem
    // Shared mem is shared between all threads of a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // Advance pointers to the starting position
    A += cRow * K * BLOCKSIZE;
    B += cCol * BLOCKSIZE;
    C += cRow * N * BLOCKSIZE + cCol * BLOCKSIZE;
}