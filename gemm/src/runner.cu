#include "kernels.cuh"
#include "runner.cuh"
#include <stdio.h>

#define N 2048

void runNaive(float *A, float *B, float *C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32, 32);
    naiveGemmKernel<<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runNaivelyCoalesced(float *A, float *B, float *C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32, 32);
    naivelyCoalescedGemmKernel<<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runCoalesced(float *A, float *B, float *C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32 * 32);
    coalescedGemmKernel<32><<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runSharedMem(float *A, float *B, float *C)
{
    dim3 gridDim(ceil(N / 32), ceil(N / 32), 1);
    dim3 blockDim(32 * 32);
    sharedMemGemmKernel<32><<<gridDim, blockDim>>>(N, N, N, A, B, C);
    return;
}

void runSharedMem2d(float *A, float *B, float *C)
{
    const int TILESIZE = 32;
    dim3 gridDim(ceil(N / TILESIZE), ceil(N / TILESIZE));
    dim3 blockDim(TILESIZE, TILESIZE);
    sharedMem2dGemmKernel<32><<<gridDim, blockDim>>>(N, A, B, C);
}

void run1DBlocktiling(float *A, float *B, float *C)
{
    const uint TILESIZE_X = 64;
    const uint TILESIZE_Y = 64;
    const uint RESULTS_PER_THREAD = 8;
    dim3 gridDim(ceil(N / TILESIZE_X), ceil(N / TILESIZE_Y));
    dim3 blockDim(1);
    blocktiling1DGemmKernel<TILESIZE_X, TILESIZE_Y, RESULTS_PER_THREAD>
        <<<gridDim, blockDim>>>(N, A, B, C);
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
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);
        printf("  Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        const auto &tdim = prop.maxThreadsDim;
        printf("  Max threads dim: [%d, %d, %d]\n", tdim[0], tdim[1], tdim[2]);
        const auto &gsize = prop.maxGridSize;
        printf("  Max grid size: [%d, %d, %d]\n", gsize[0], gsize[1], gsize[2]);
    }
}

void runKernel(int kernelNum, float *A, float *B, float *C)
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
    case 4:
        runSharedMem2d(A, B, C);
        break;
    case 5:
        run1DBlocktiling(A, B, C);
        break;
    default:
        break;
    }
}