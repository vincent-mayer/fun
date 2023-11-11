import math
import torch
from torchtyping import TensorType

N = "n"
d = "d"
inf = 1e15

def flash_attention(Q: TensorType[N, d], K: TensorType[N, d], V: TensorType[N, d], M: int) -> TensorType:
    N, d = Q.shape
    Bc = math.ceil(M / (4 * d))
    Br = min(Bc, d)
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)
    
    O = torch.zeros_like(Q)
    l = Q.new_zeros(N)
    m = -inf * Q.new_ones(N)
    
    Qs, Ks, Vs = torch.split(Q, Br), torch.split(K, Bc), torch.split(V, Bc)
    Os, ls, ms = torch.split(O, Br), torch.split(l, Br), torch.split(m, Br)
    
    Os, ls, ms = list(Os), list(ls), list(ms)

    assert len(Qs) == Tr
    assert Qs[0].shape == torch.Size((Br, d))
    assert len(Ks) == Tc
    assert Ks[0].shape == torch.Size((Bc, d))
    
    for j in range(Tc):
        # Load Kj, Vj from HBM to on-chip SRAM
        Kj, Vj = Ks[j], Vs[j]
        for i in range(Tr):
            # Load Qi, Oi, li, mi from HBM to on-chip SRAM
            Qi, Oi, li, mi = Qs[i], Os[i], ls[i], ms[i]
            if li.ndim == 1:
                li, mi = li.unsqueeze(-1), mi.unsqueeze(-1)
            # On-chip compute attention block
            Sij = torch.matmul(Qi, Kj.T)
            
            # On-chip compute intermediates
            mij, _ = torch.max(Sij, dim=1, keepdim=True) # rowmax
            Pij = torch.exp(Sij - mij) # subtract rowmax and exponentiate
            lij = torch.sum(Pij, dim=1, keepdim=True) # rowsum
            
            # On-chip compute
            minew = torch.max(mi, mij) 
            linew = torch.exp(mi - minew) * li + torch.exp(mij - minew) * lij # weighted average
            
            PijVj = torch.matmul(Pij, Vj)
            diaglinewinv = torch.inverse(torch.diag(linew.squeeze()))
            diagli = torch.diag(li.squeeze())
            expOi = torch.exp(mi - minew) * Oi
            diagliexpOi = torch.matmul(diagli, expOi)
            expPijVj = torch.exp(mij - minew) * PijVj
            lastsum = diagliexpOi + expPijVj
            Os[i] = torch.matmul(diaglinewinv, lastsum)
            ls[i] = linew
            ms[i] = minew
    
    return torch.cat(Os)
    

if __name__ == '__main__':
    n, d = 10240, 128
    m = 1 * 1024 * 1024 # 1MB
    
    q = torch.randn((n, d), device="cpu", dtype=torch.float32)
    k = torch.randn((n, d), device="cpu", dtype=torch.float32)
    v = k.clone()
    
    flash_out = flash_attention(q, k, v, m)
    
    torch_out = torch.softmax(torch.matmul(q, k.T), -1)
    torch_out = torch.matmul(torch_out, v)
    
    assert torch.allclose(torch_out, flash_out, atol=1e-3)