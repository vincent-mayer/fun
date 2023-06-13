import pytest
import torch
from voxelize import voxelize
import time

N_POINTS = 252768
N_VOXELS = 160000
MAX_POINTS_PER_VOXEL = 10
VOXEL_SIZE = [0.075, 0.075, 0.2]
XYZ_BOUNDS = [-50.0, 50.0, -50.0, 50.0, -5.0, 5.0]

def test_voxelize_bevfusion():
    points = 5 * torch.randn((N_POINTS, 5), device="cuda", dtype=torch.float32).contiguous()
    voxels = torch.zeros((N_VOXELS, MAX_POINTS_PER_VOXEL, 5), device="cuda", dtype=torch.float32).contiguous()
    coords = torch.zeros((N_VOXELS, 3), device="cuda", dtype=torch.int).contiguous()
    points_per_voxel = torch.zeros((N_VOXELS,), dtype=int, device="cuda")
    voxel_size = VOXEL_SIZE
    xyz_bounds = XYZ_BOUNDS

    start = time.time()
    voxelize(points, voxels, coords, points_per_voxel, voxel_size, xyz_bounds)
    et = time.time() - start
    print(f"Execution time: {et*1e6:.2f} us")
    print(f"coords: max={coords.max().tolist()}, min={coords.min().tolist()}")

if __name__=="__main__":
    test_voxelize_bevfusion()