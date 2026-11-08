"""Check triton_kernels' MXFP4 W4A16 matmul on this box against a hand dequantized fp32 reference.

Mirrors sglang's quark W4A16 path: _swizzle_mxfp4 + matmul with ragged, expert-sorted rows.
    PYTHONPATH=$SGLANG_SRC/python python check_triton_kernels.py
"""

import torch

from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4
from triton_kernels.matmul import FlexCtx, PrecisionConfig, matmul
from triton_kernels.tensor import make_ragged_tensor_metadata

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def dequant(w_packed, scale):
    # [E, N, K/2] uint8 (low nibble = even k) and [E, N, K/32] e8m0 -> [E, N, K] fp32
    lo, hi = w_packed & 0xF, w_packed >> 4
    codes = torch.stack([lo, hi], dim=-1).flatten(-2).long()
    vals = E2M1.to(w_packed.device)[codes]
    return vals * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)


def main(E=4, N=1024, K=2048, tokens=8, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    scale = torch.randint(122, 130, (E, N, K // 32), dtype=torch.uint8, device=dev)
    x = torch.randn(tokens, K, dtype=torch.bfloat16, device=dev)
    ids = torch.randint(0, E, (tokens,), device=dev)

    ref_w = dequant(w, scale)  # before the swizzle, which may consume its inputs

    tw, flex, tscale = _swizzle_mxfp4(w, scale, 4)
    pc = PrecisionConfig(b_mx_scale=tscale, b_microblock_size=32, flex_ctx=FlexCtx(rhs_data=flex))
    combine = torch.argsort(ids, stable=True).to(torch.int32)
    counts = torch.bincount(ids, minlength=E).to(torch.int32)
    ragged = make_ragged_tensor_metadata(counts, tokens)
    out = matmul(x, tw, None, a_ragged_metadata=ragged, gather_indx=combine, precision_config=pc)

    sorted_ids = ids[combine.long()]
    ref = torch.stack(
        [x[t].float() @ ref_w[e].T for t, e in zip(combine.long().tolist(), sorted_ids.tolist())]
    )
    err = (out.float() - ref).abs()
    tol = 2e-2 * ref.abs().max()
    print(f"max abs err {err.max().item():.4g}, ref max {ref.abs().max().item():.4g}, tol {tol.item():.4g}")
    assert err.max() <= tol, "triton_kernels MXFP4 matmul disagrees with the dequantized reference"
    print("PASS")


if __name__ == "__main__":
    main()
