import numpy as np
import torch
from pprint import pprint

def cubic_interpolate(p: list[float], x: float):
	"""Evaluates a cubic spline at x.
	
	p = [p0, p1, p2, p3]
	f(x) = (-0.5*p0 + 1.5p1 - 1.5p2 + 0.5p3)x^3 + (p0 - 2.5p1 + 2p2 - 0.5p3)x^2 + (-0.5p0 + 0.5p2)x + p1
	"""
	return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])))

def bicubic_interpolate(grid: np.ndarray, x: float, y: float):
	"""Bicubically interpolates a grid point at x and y by first inpolating along the columns (y) and then the row (x)."""
	column_interp = np.zeros(4)
	# Interpolate columns
	column_interp[0] = cubic_interpolate(grid[:, 0], y) # select frist column to interpolate over
	column_interp[1] = cubic_interpolate(grid[:, 1], y)
	column_interp[2] = cubic_interpolate(grid[:, 2], y)
	column_interp[3] = cubic_interpolate(grid[:, 3], y)
	# Interpolate row
	return cubic_interpolate(column_interp, x)

if __name__=="__main__":
	grid = np.array([[1, 3, 3, 4],
					 [7, 2, 3, 4],
					 [1, 6, 3, 6],
					 [2, 5, 7, 2]], dtype=np.float32)
	print("=" * 20 + " input " + "=" * 20)
	pprint(grid)
	results = []
	for x in np.linspace(0, 1, 100):
		for y in np.linspace(0, 1, 100):
			interp = bicubic_interpolate(grid, x, y)
			results.append(interp)
	torch_res = torch.nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)(torch.tensor(grid)[None, ..., None]).transpose(-2, -1).flatten()
	# torch_res = torch.nn.functional.interpolate(torch.tensor(grid)[None, ..., None], scale_factor=2, mode="bicubic").flatten()
	print("=" * 20 + " interp " + "=" * 20)
	for n in range(grid.shape[0] * 2):
		pprint(torch_res[n*8:(n+1)*8])
		
	argmin = torch.abs(torch_res - interp).argmin()
	print("\n", torch.abs(torch_res - interp).min(), argmin) 
	print(interp, torch_res[27])
	
	pprint(sorted(results))