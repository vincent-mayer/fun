#include <torch/extension.h>

void voxelize_kernel_launcher(const torch::Tensor &points,
                              torch::Tensor &voxels,
                              torch::Tensor &coords,
                              torch::Tensor &points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> xyz_bounds);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("voxelize", &voxelize_kernel_launcher, "Voxelize a point cloud.");
}