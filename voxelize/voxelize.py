import torch

POINTS = 252768
MAX_VOXELS = 160000


def voxelize():
    points = torch.randn((1, POINTS, 5), dtype=torch.float32, device="cuda")


if __name__ == "__main__":
    voxelize()
