import time
import torch

N = 2048
FLOP = 2 * N * N * N
DEVICE = "mps"
DTYPE = torch.float32

def benchmark_m1():
    a = torch.randn((N, N), dtype=DTYPE, device=DEVICE)
    b = torch.randn((N, N), dtype=DTYPE, device=DEVICE)
    start = time.time()
    c = a @ b
    elapsed_time = (time.time() - start)
    flops = (FLOP / elapsed_time) * 1e-12
    print(f"{(elapsed_time)*1e6:.2f} us; {flops:.2f} TFLOP/S")
    return flops

if __name__=="__main__":
    flops = [benchmark_m1() for _ in range(10)]
    print(f"{min(flops):.2f} TFLOP/S")