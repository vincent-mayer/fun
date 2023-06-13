#include <chrono>
#include <iostream>
#include <ATen/TensorUtils.h>
#include <torch/torch.h>
// #include <torch/extension.h>
#include <io.h>

constexpr int MAX_VOXELS = 160000;
constexpr int MAX_POINTS = 5;

using namespace std::chrono;
auto now = high_resolution_clock::now;

void dynamic_voxelize_kernel(const torch::TensorAccessor<float, 2> points,
                             torch::TensorAccessor<int, 2> coors,
                             const std::vector<float> voxel_size,
                             const std::vector<float> coors_range,
                             const std::vector<int> grid_size,
                             const int num_points, const int num_features,
                             const int NDim)
{
    const int ndim_minus_1 = NDim - 1;
    bool failed = false;
    // int coor[NDim];
    int *coor = new int[NDim]();
    int c;

    for (int i = 0; i < num_points; ++i)
    {
        failed = false;
        for (int j = 0; j < NDim; ++j)
        {
            c = floor((points[i][j] - coors_range[j]) / voxel_size[j]);
            // necessary to rm points out of range
            if ((c < 0 || c >= grid_size[j]))
            {
                failed = true;
                break;
            }
            coor[j] = c;
        }

        for (int k = 0; k < NDim; ++k)
        {
            if (failed)
                coors[i][k] = -1;
            else
                coors[i][k] = coor[k];
        }
    }

    delete[] coor;
    return;
}

void hard_voxelize_kernel(const torch::TensorAccessor<float, 2> points,
                          torch::TensorAccessor<float, 3> voxels,
                          torch::TensorAccessor<int, 2> coors,
                          torch::TensorAccessor<int, 1> num_points_per_voxel,
                          torch::TensorAccessor<int, 3> coor_to_voxelidx,
                          int &voxel_num, const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const std::vector<int> grid_size,
                          const int max_points, const int max_voxels,
                          const int num_points, const int num_features,
                          const int NDim)
{
    // declare a temp coors
    at::Tensor temp_coors = at::zeros(
        {num_points, NDim}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));

    // First use dynamic voxelization to get coors,
    // then check max points/voxels constraints
    dynamic_voxelize_kernel(points, temp_coors.accessor<int, 2>(),
                            voxel_size, coors_range, grid_size,
                            num_points, num_features, NDim);

    int voxelidx, num;
    auto coor = temp_coors.accessor<int, 2>();

    for (int i = 0; i < num_points; ++i)
    {
        // T_int* coor = temp_coors.data_ptr<int>() + i * NDim;

        if (coor[i][0] == -1)
            continue;

        voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];

        // record voxel
        if (voxelidx == -1)
        {
            voxelidx = voxel_num;
            if (max_voxels != -1 && voxel_num >= max_voxels)
                continue;
            voxel_num += 1;

            coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]] = voxelidx;

            for (int k = 0; k < NDim; ++k)
            {
                coors[voxelidx][k] = coor[i][k];
            }
        }

        // put points into voxel
        num = num_points_per_voxel[voxelidx];
        if (max_points == -1 || num < max_points)
        {
            for (int k = 0; k < num_features; ++k)
            {
                voxels[voxelidx][num][k] = points[i][k];
            }
            num_points_per_voxel[voxelidx] += 1;
        }
    }

    return;
}

int hard_voxelize_cpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3)
{
    // current version tooks about 0.02s_0.03s for one frame on cpu
    // check device

    std::vector<int> grid_size(NDim);
    const int num_points = points.size(0);
    const int num_features = points.size(1);

    for (int i = 0; i < NDim; ++i)
    {
        grid_size[i] =
            round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
    }

    // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
    // printf("cpu coor_to_voxelidx size: [%d, %d, %d]\n", grid_size[2],
    // grid_size[1], grid_size[0]);
    at::Tensor coor_to_voxelidx =
        -at::ones({grid_size[2], grid_size[1], grid_size[0]}, coors.options());

    int voxel_num = 0;
    hard_voxelize_kernel(points.accessor<float, 2>(),
                         voxels.accessor<float, 3>(),
                         coors.accessor<int, 2>(),
                         num_points_per_voxel.accessor<int, 1>(),
                         coor_to_voxelidx.accessor<int, 3>(),
                         voxel_num,
                         voxel_size,
                         coors_range,
                         grid_size,
                         max_points,
                         max_voxels,
                         num_points,
                         num_features,
                         NDim);

    return voxel_num;
}

int main(int argc, char *argv[])
{
    std::vector<float> points_vec = readNumpyFloatArray("data/pcd_17k.npy");
    torch::Tensor points = torch::from_blob(points_vec.data(), (points_vec.size() / 5, 5));
    torch::Tensor voxels = torch::zeros({MAX_VOXELS, MAX_POINTS, 5}).to(torch::kInt32);
    torch::Tensor coors = torch::zeros({MAX_VOXELS, 3}).to(torch::kInt32);
    torch::Tensor num_points_per_voxel = torch::zeros({MAX_VOXELS}).to(torch::kInt32);
    std::vector<float> voxelsize{0.075, 0.075, 0.2};
    std::vector<float> coors_range{-54.0, -54.0, -5.0, 54.0, 54.0, 3.0};
    auto startTime = now();
    auto duration = duration_cast<microseconds>(now() - startTime);
    hard_voxelize_cpu(points, voxels, coors, num_points_per_voxel, voxelsize, coors_range, MAX_POINTS, MAX_VOXELS);
    std::cout << "Duration: " << duration.count() * 1e-6 << " s" << std::endl;
    return 0;
}