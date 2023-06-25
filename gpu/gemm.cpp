#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <gemm.h>
#include <iostream>
#include <npy.hpp>
#include <numeric>
#include <thread>
#include <vector>

#define N 2048

constexpr double flop = 2.0 * N * N * N;

using namespace std::chrono;
auto now = high_resolution_clock::now;

int main(int argc, char *argv[])
{
    // Load data to check matmul correctness.
    std::filesystem::path cwd = std::filesystem::current_path();
    std::vector<float> A, B, ExpectedC;
    std::vector<unsigned long> ShapeA{}, ShapeB{}, ShapeExpectedC{};
    bool FortranOrderA, FortranOrderB, FortranOrderExpectedC;
    npy::LoadArrayFromNumpy(cwd / "data/A.npy", ShapeA, FortranOrderA, A);
    npy::LoadArrayFromNumpy(cwd / "data/B.npy", ShapeB, FortranOrderB, B);
    npy::LoadArrayFromNumpy(cwd / "data/C.npy", ShapeExpectedC, FortranOrderExpectedC,
                            ExpectedC);
    // Allocate result of gemm:
    int NumElem = 1;
    for (auto &s : ShapeExpectedC)
        NumElem *= s;
    std::vector<float> C(NumElem, 0.0);

    // Device copies of A, B, C
    float *dA, *dB, *dC;
    int ArrayBytes = NumElem * sizeof(float);
    cudaMalloc((void **)&dA, ArrayBytes);
    cudaMalloc((void **)&dB, ArrayBytes);
    cudaMalloc((void **)&dC, ArrayBytes);

    cudaMemcpy(dA, A.data(), ArrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), ArrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), ArrayBytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < 1; i++)
    {
        std::cout << "before C: " << C[0] << std::endl;

        auto startTime = now();
        gemmKernelLauncher(dA, dB, dC);
        cudaDeviceSynchronize();
        auto duration = duration_cast<microseconds>(now() - startTime);
        auto gflops = (flop / (static_cast<double>(duration.count()) * 1e-6)) * 1e-9;
        cudaMemcpy(C.data(), dC, ArrayBytes, cudaMemcpyDeviceToHost);
        std::cout << "after C: " << C[0] << std::endl;
        std::cout << "expected C: " << ExpectedC[0] << std::endl;

        std::cout << "Duration: " << duration.count() * 1e-6 << " s"
                  << "| GFLOPS: " << gflops << std::endl;
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}