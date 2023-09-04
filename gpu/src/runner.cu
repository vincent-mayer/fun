#include "kernels.cuh"
#include "runner.cuh"

#define N 2048

void runNaive(float *&A, float *&B, float *&C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32, 32);
    naiveGemmKernel<<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runNaivelyCoalesced(float *&A, float *&B, float *&C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32, 32);
    naivelyCoalescedGemmKernel<<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runCoalesced(float *&A, float *&B, float *&C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32 * 32);
    coalescedGemmKernel<32><<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runSharedMem(float *&A, float *&B, float *&C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32 * 32);
    sharedMemGemmKernel<32><<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runKernel(int kernelNum, float *&A, float *&B, float *&C)
{
    switch (kernelNum)
    {
    case 0:
        runNaive(A, B, C);
        break;
    case 1:
        runNaivelyCoalesced(A, B, C);
        break;
    case 2:
        runCoalesced(A, B, C);
        break;
    case 3:
        runSharedMem(A, B, C);
        break;
    default:
        break;
    }
}