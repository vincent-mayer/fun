#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gemm.h>
#include <io.h>
#include <iostream>
#include <thread>
#include <vector>

#define N 2048

using namespace std::chrono;
auto now = high_resolution_clock::now;

int main(int argc, char *argv[])
{
    std::vector<float> A = readNumpyFloatArray("tmp/A.npy");
    std::vector<float> B = readNumpyFloatArray("tmp/B.npy");
    std::vector<float> expectedC = readNumpyFloatArray("tmp/C.npy");
    std::vector<float> C(expectedC.size(), 0);

    while (true)
    {
        auto startTime = now();

        gemmKernelLauncher(A, B, C);

        auto duration = duration_cast<microseconds>(now() - startTime);
        double flop = 2.0 * N * N * N;
        auto gflops = (flop / (static_cast<double>(duration.count()) * 1e-6)) * 1e-9;
        std::cout << "Duration: " << duration.count() * 1e-6 << " s" << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;
    }

    return 0;
}