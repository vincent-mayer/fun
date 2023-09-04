#include "runner.cuh"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <npy.hpp>
#include <numeric>
#include <thread>
#include <vector>

#define N 2048

constexpr double flop = 2.0 * N * N * N;

using namespace std::chrono;
auto now = high_resolution_clock::now;

void check_result(std::vector<float> &C, std::vector<float> &ExpectedC)
{
    for (int j = 0; j < C.size(); j++)
        assert(std::abs(C[j] - ExpectedC[j]) < 1e-3);
}

int main(int argc, char *argv[])
{
    // Which kernel to execute:
    int kernelNum;
    std::cout << "Select the kernel by typing a number between 0 and 12: ";
    std::cin >> kernelNum;
    std::cout << "You selected number: " << kernelNum << std::endl;

    if (kernelNum < 0 || kernelNum > 12)
        throw std::invalid_argument("Please choose a number between 0 and 12!");

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
    std::vector<float> C(ExpectedC.size(), 0.0);

    // Device copies of A, B, C
    float *dA, *dB, *dC;
    int ArrayBytes = ExpectedC.size() * sizeof(float);
    cudaMalloc((void **)&dA, ArrayBytes);
    cudaMalloc((void **)&dB, ArrayBytes);
    cudaMalloc((void **)&dC, ArrayBytes);

    cudaMemcpy(dA, A.data(), ArrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), ArrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), ArrayBytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < 30; i++)
    {
        auto startTime = now();
        runKernel(kernelNum, dA, dB, dC);
        cudaMemcpy(C.data(), dC, ArrayBytes, cudaMemcpyDeviceToHost);
        auto duration = duration_cast<microseconds>(now() - startTime);
        auto gflops = (flop / (static_cast<double>(duration.count()) * 1e-6)) * 1e-9;
        std::cout << "Duration: " << duration.count() * 1e-6 << " s"
                  << "| GFLOPS: " << gflops << std::endl;
        if (i == 0)
            check_result(C, ExpectedC);
    }
    std::cout << "Result of matmul is correct." << std::endl;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}