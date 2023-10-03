#include "kernels.cuh"
#include "runner.cuh"
#include <stdio.h>

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

void run1DBlocktiling(float *&A, float *&B, float *&C)
{
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(ceil(N / BN), ceil(N / BM));
    dim3 blockDim((BM * BN) / TM);
    blocktiling1DGemmKernel<BM, BN, BK, TM><<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void printDeviceProperties()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",
               (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Clock Rate: %d\n", prop.clockRate);
        printf("  L2 cache size: %d\n", prop.l2CacheSize / 1024);
        printf("  Compute mode: %d\n", prop.computeMode);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warpsize: %d\n", prop.warpSize);
        printf("  SMEM per SM: %.1f\n", prop.sharedMemPerMultiprocessor / 1024.0);
        printf("  SMEM per block: %.1f\n", prop.sharedMemPerBlock / 1024.0);
        printf("  SM count: %d\n", prop.multiProcessorCount);
    }
}

void runKernel(int kernelNum, float *&A, float *&B, float *&C)
{
    printDeviceProperties();

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
    case 4:
        run1DBlocktiling(A, B, C);
    default:
        break;
    }
}