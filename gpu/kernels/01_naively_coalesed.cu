#define NN 2048

__global__ void gemmKernel(int M, int N, int K, const float *A, const float *B, float *C)
{
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint x = blockIdx.y * blockDim.y + threadIdx.y;
    printf("x/row:%d y/column:%d\n", x, y);

    // If condition is necessary when M, N aren't multiples of 32 (warp size)
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = tmp; // x = row, y = column
    }
}

void gemmKernelLauncher(float *&A, float *&B, float *&C)
{
    dim3 gridDim(ceil(NN / 32.0), ceil(NN / 32.0), 1);
    dim3 blockDim(32, 32, 1);
    gemmKernel<<<gridDim, blockDim>>>(NN, NN, NN, A, B, C);
    return;
}