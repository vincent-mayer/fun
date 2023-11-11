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
        for i in range(Tr): # Tr scales with sequence-length
            # Load Qi, Oi, li, mi from HBM to on-chip SRAM
            Qi, Oi, li, mi = Qs[i], Os[i], ls[i], ms[i]
            if li.ndim == 1:
                li, mi = li.unsqueeze(-1), mi.unsqueeze(-1)
            # AMEM -> Grid -> vector buffer; 128 cycles
            QKij = torch.matmul(Qi, Kj.T) # 128x128 * 2B = 32kB
            
            # vector buffer -> VPU: requires 33kB
            # Compute: 4 cycles per element (max, sub, exp, sum)
            mij, _ = torch.max(QKij, dim=1, keepdim=True) # 128x128 -> 128x1 rowmax; 128 * 2B = 256B
            Pij = torch.exp(QKij - mij) # 128x128 -> 128x128; overwrite QKij; 128x128 * 2B = 32kB
            lij = torch.sum(Pij, dim=1, keepdim=True) # 128x128 -> 128x1; 128 * 2B = 256B
            minew = torch.max(mi, mij) # 128x1 -> 128x1; 128 * 2B = 256B 
            linew = torch.exp(mi - minew) * li + torch.exp(mij - minew) * lij # 256B
            
            # vector-buffer (Pij)/ AMEM (Vj) -> grid -> vector buffer
            # Compute: 4 cycles per element (mul, mul, add, div)
            PijVj = torch.matmul(Pij, Vj) # 128x128 * 2B = 32kB
            mili = li * torch.exp(mi - minew) # 128x1
            expOi = mili * Oi # 128x128 * 2B = 32kB
            acc = (expOi + torch.exp(mij - minew) * PijVj) / linew
            Os[i] = acc # [128, 128] * 2B * Tr(=SEQLEN / 128)=16 ) = 786kB(!!)
            ls[i] = linew # [128, 1] * 2B * Tr(=SEQLEN / 128)=16 ) = 4096B
            ms[i] = minew # [128, 1] * 2B * Tr(=SEQLEN / 128)=16 ) = 4096B
    
    return torch.cat(Os)
    

if __name__ == '__main__':
    n, d = 2048, 128
    m = 4 * 128 * 128 # 1MB
    
    q = torch.randn((n, d), device="cpu", dtype=torch.float32)
    k = torch.randn((n, d), device="cpu", dtype=torch.float32)
    v = k.clone()
    
    flash_out = flash_attention(q, k, v, m)
    
    torch_out = torch.softmax(torch.matmul(q, k.T), -1)
    torch_out = torch.matmul(torch_out, v)
    
    assert torch.allclose(torch_out, flash_out, atol=1e-4)