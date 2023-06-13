#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void voxelize_kernel(const float *points,
                                int *coords,
                                const int num_points,
                                const int num_features,
                                const float voxel_size_x,
                                const float voxel_size_y,
                                const float voxel_size_z,
                                const float x_min,
                                const float y_min,
                                const float z_min,
                                const float grid_x,
                                const float grid_y,
                                const float grid_z)
{
    // 1D kernel loop; blockDim.x = threads per block, gridDim.x = blocks per grid, blockDim.x * gridDim.x = threads per grid
    // 6 FLOP per point
    for (int index = threadIdx.x + blockIdx.x * blockDim.x;
         index < num_points;
         index += blockDim.x * gridDim.x)
    {
        auto cur_points = points + index * num_features;
        auto cur_coords = coords + index * 3;
        int x = floorf((cur_points[0] - x_min) / voxel_size_x);
        if (x < 0 || x >= grid_x)
        {
            cur_coords[0] = -1;
            continue;
        }
        int y = floorf((cur_points[1] - y_min) / voxel_size_y);
        if (y < 0 || y >= grid_y)
        {
            cur_coords[0] = -1;
            cur_coords[1] = -1;
            continue;
        }
        int z = floorf((cur_points[2] - z_min) / voxel_size_z);
        if (z < 0 || z >= grid_z)
        {
            cur_coords[0] = -1;
            cur_coords[1] = -1;
            cur_coords[2] = -1;
        }
        else
        {
            cur_coords[0] = x;
            cur_coords[1] = y;
            cur_coords[2] = z;
        }
    }
}
/*
    Function: Fills voxel tensor from point cloud.
    Args:
        points              : Input point cloud, (N_points, N_point_features)
        voxels              : Output voxels (N_voxels, N_points_per_voxel, N_point_features)
        coords              : (N_voxels, 3)
        points_per_voxel    : Number of points per voxel (N_voxels, )
        voxel_size          : x-y-z size of voxel in meters (3, )
        xyz_bounds          : xmin-xmax-ymin-ymax-zmin-zmax 3D bounds in which to consider points
*/
void voxelize_kernel_launcher(const torch::Tensor &points,
                              torch::Tensor &voxels,
                              torch::Tensor &coords,
                              torch::Tensor &points_per_voxel,
                              const std::vector<float> voxel_size,
                              const std::vector<float> xyz_bounds)
{
    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const float voxel_size_x = voxel_size[0];
    const float voxel_size_y = voxel_size[1];
    const float voxel_size_z = voxel_size[2];

    const float x_min = xyz_bounds[0];
    const float x_max = xyz_bounds[1];
    const float y_min = xyz_bounds[2];
    const float y_max = xyz_bounds[3];
    const float z_min = xyz_bounds[4];
    const float z_max = xyz_bounds[5];

    const int grid_x = round((x_max - x_min) / voxel_size_x);
    const int grid_y = round((y_max - y_min) / voxel_size_y);
    const int grid_z = round((z_max - z_min) / voxel_size_z);

    // 1. Link point to voxel coordinate
    torch::Tensor temp_coords = torch::zeros({num_points, 3}, points.options().dtype(torch::kInt));
    dim3 blocks(ceilf(num_points / 512));
    dim3 threads(512); // 512 threads per block
    voxelize_kernel<<<blocks, threads>>>(points.contiguous().data_ptr<float>(),
                                         coords.contiguous().data_ptr<int>(),
                                         num_points,
                                         num_features,
                                         voxel_size_x,
                                         voxel_size_y,
                                         voxel_size_z,
                                         x_min,
                                         y_min,
                                         z_min,
                                         grid_x,
                                         grid_y,
                                         grid_z);
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    // 2.
}
