#include <vector>

#define NN 2048

__global__ void gemmKernel(int M, int N, int K, const float *A, const float *B, float *C)
{
    // CUDA compute ordered in 3-level hierarchy
    // invocation -> grid -> blocks -> up to 1024 threads
    // Threads share memory of block
    // # of threads = blockDim
    // # of blocks = gridDim
    // blocks contain warps, warps contain 32 threads
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // If condition is necessary when M, N aren't multiples of 32 (warp size)
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; i++)
        {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp;
    }
}

void gemmKernelLauncher(std::vector<float> &A, std::vector<float> &B,
                        std::vector<float> &C)
{

    dim3 gridDim(ceil(NN / 32.0), ceil(NN / 32.0), 1);
    dim3 blockDim(32, 32, 1);
    gemmKernel<<<gridDim, blockDim>>>(NN, NN, NN, A.data(), B.data(), C.data());
    return;
}