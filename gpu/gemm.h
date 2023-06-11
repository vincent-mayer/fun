#ifndef GEMM_CUH_
#define GEMM_CUH_
#include <vector>

void gemmKernelLauncher(std::vector<float> &A, std::vector<float> &B,
                        std::vector<float> &C);
#endif