import torch
import numpy as np
import time

N = 2048
FLOP = 2 * N * N * N

def benchmark(n: int=5):
    ets = []
    for i in range(n-1):
        A = torch.randn((N, N), dtype=float, device="cuda")
        B = torch.randn((N, N), dtype=float, device="cuda")
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        ets.append(time.perf_counter() - start)
    return ets
    

def main():
    A = torch.randn((N, N), dtype=float, device="cuda")
    B = torch.randn((N, N), dtype=float, device="cuda")
    C = torch.matmul(A, B)
    ets = benchmark()
    gflops = [round((FLOP / t) * 1e-12, 3) for t in ets]
    print(f"{gflops} TFLOP/S  \n{[round(e * 1e3, 1) for e in ets]} ms")
    [np.save(f"tmp/{n}.npy", v.cpu().numpy()) for n, v in {"A": A, "B": B, "C": C}.items()]
    

if __name__=="__main__":
    main()