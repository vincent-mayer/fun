import torch
from torchtyping import TensorType

S = "s"  # Sequence length
d = "d"  # Head dimension


def _is_masked_out_tile(outer_loop_idx: int, inner_loop_idx: int, tiles_per_inner_iter: int = 2):
    return tiles_per_inner_iter * inner_loop_idx > outer_loop_idx


def _is_diagonal_tile(outer_loop_idx: int, inner_loop_idx: int, tiles_per_inner_iter: int = 2):
    return tiles_per_inner_iter * inner_loop_idx == outer_loop_idx


def _is_subdiagonal_tile(outer_loop_idx: int, inner_loop_idx: int, tiles_per_inner_iter: int = 2):
    return (tiles_per_inner_iter * inner_loop_idx) + 1 == outer_loop_idx


def flash_attention(Q: TensorType[S, d], K: TensorType[S, d], V: TensorType[S, d]) -> TensorType[S, d]:
    """FlashAttention2 on Pyxis following https://arxiv.org/abs/2307.08691.

    We adopted FlashAttention2 to consider a Pyxis scheduling-friendly way of execution and masking.
    Tiles that are completely masked out are not computed. In the inner loop, we consider two
    square QK^T tiles for each AV-tile for scheduling reasons.

    For more details see also https://www.notion.so/recogni/Softmax-Attention-110165feac554f0dbf27774dafd6f037.


    Parameters
    ----------
    Q
        Query tensor of shape [sequence length, head dim]
    K
        Key tensor of shape [sequence length, head dim]
    V
        Value tensor of shape [sequence length, head dim]

    Returns
    -------
        AV result tensor of attention computation with shapoe [sequence length, head dim]
    """
    d = Q.shape[-1]
    Qs, Ks, Vs = torch.split(Q, d), torch.split(K, d), torch.split(V, d)
    AVs = list(torch.split(torch.zeros_like(Q), d))

    # Outer loop over Q, AV tiles
    for i, (Qi, AVi) in enumerate(zip(Qs, AVs)):  # Tr scales with sequence-length; sequence tiles output (AV)
        li, mi = Q.new_zeros(d, 1), float("-inf") * Q.new_ones(d, 1)  # 128x1, 128x1; 512B
        # Inner loop over K/V tiles; Accumulate into one AV tile as output.
        # Load two K and two V tiles for efficient scheduling on hardware.
        for j, (Kj1, Kj2, Vj1, Vj2) in enumerate(zip(Ks[::2], Ks[1::2], Vs[::2], Vs[1::2])):
            if _is_masked_out_tile(i, j):  # Both tiles are fully in the masked out region.
                continue
            # AMEM (Qi, Kj) -> Grid -> vector buffer
            # Compute: 128 x 2 cycles
            Vjs = [Vj1]
            QiKj1 = torch.matmul(Qi, Kj1.T)
            QiKjs = [QiKj1]  # 128x128 -> 128x128; 32kB
            # If the first QK^T sits on the diagonal, mask it; VPU
            if _is_diagonal_tile(i, j):
                QiKjs[0] = QiKjs[0].masked_fill(torch.tril(torch.ones(d, d)) == 0, float("-inf"))
            # Else both tiles are completely in the valid region, so compute second QKT tile.
            else:
                QiKj2 = torch.matmul(Qi, Kj2.T)
                QiKjs.append(QiKj2)  # 128x128 -> 128x128; 32kB
                Vjs.append(Vj2)

            # Apply mask to second QKT tile, if it sits on the diagonal; VPU
            if _is_subdiagonal_tile(i, j):
                QiKjs[1] = QiKjs[1].masked_fill(torch.tril(torch.ones(d, d)) == 0, float("-inf"))

            # vector buffer -> VPU: requires 33kB
            # Compute: 4 cycles per element (max, sub, exp, sum)
            mi_new = mi
            for QiKj in QiKjs:
                mi_new = torch.max(mi_new, torch.max(QiKj, dim=1, keepdim=True)[0])  # 128x1 -> 128x1; 256B

            # Concatenate the two QK^T tiles
            QiKj = torch.cat(QiKjs, -1)

            # SEA: Subtract, exponentiate, accumulate; Per element
            Pij = torch.exp(QiKj - mi_new)  # 128x128 -> 128x128; overwrite QiKj; 32kB
            lij = torch.sum(Pij, dim=1, keepdim=True)  # 128x128 -> 128x1; 256B

            # Compute scaling factor
            alpha = torch.exp(mi - mi_new)  # 128x1 -> 128x1; 256B
            li_new = alpha * li + lij  # 128x1 -> 128x1; 256B

            # vector-buffer (Pij)/ AMEM (Vj) -> grid -> vector buffer
            Vj = torch.cat(Vjs, dim=0)
            PijVj = torch.matmul(Pij, Vj)  # 128x128 -> 128x128; 32kB

            # vector buffer -> VPU: requires 33kB
            # Compute: 2 cycles per element (mul, add)
            AVi = AVi * alpha + PijVj  # 128x128 -> 128x128; 32kB

            # Update statistics (max, denominator)
            li = li_new  # 128x1;
            mi = mi_new  # 128x1;
        # AMEM
        AVs[i] = AVi / li_new  # [128, 128] * 2B
    return torch.cat(AVs)


if __name__ == "__main__":
    s, d = 2048, 128

    q = torch.randn((s, d), device="cpu", dtype=torch.float32)
    k = torch.randn((s, d), device="cpu", dtype=torch.float32)
    v = torch.randn((s, d), device="cpu", dtype=torch.float32)

    flash_out = flash_attention(q, k, v)

    qkt = torch.matmul(q, k.T)
    qkt_masked = qkt.masked_fill(torch.tril(torch.ones(s, s)) == 0, float("-inf"))
    torch_out = torch.matmul(torch.softmax(qkt_masked, -1), v)

    assert torch.allclose(torch_out, flash_out, atol=1e-4)