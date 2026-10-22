"""Verified starting point for the rewrite (handoff 03, Phase 1). Not vLLM code yet.

- repack_blocks(): [..., NB, 66|34] uint8 -> (qweight, fp16 scales), same total bytes
- _decode_tile(): per-record scale, per-group index/sign math, per-weight lookup in an
  fp32 [256 x 8] table (the fastest of three variants measured on sm_120)
- _gemm_kernel: tl.dot path (M >= ~16); _gemv_kernel: no tensor cores, split-K (M = 1)

Self-check + mini benchmark: python proto_kernels.py
Checked on RTX 5060 Ti (sm_120), Triton 3.8, 2026-10-22: decode bit-exact vs the GGML
reference, GEMM error == cuBLAS error, M=1 IQ2 2x faster than the draft.
"""

import torch
import triton
import triton.language as tl

from ref_decode import IQ2XXS_GRID

Q8_0, IQ2_XXS = 0, 1
CODEC_ID = {"q8_0": Q8_0, "iq2_xxs": IQ2_XXS}
BLOCK = {"q8_0": 32, "iq2_xxs": 256}
BYTES = {"q8_0": 34, "iq2_xxs": 66}


def repack_blocks(raw: torch.Tensor, codec: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ggml blocks into an aligned payload and fp16 block scales.

    Args:
        raw: uint8 [..., NB, 66] (IQ2_XXS) or [..., NB, 34] (Q8_0).
    Returns:
        qweight: int32 [..., NB * 16] (IQ2_XXS) or int8 [..., NB * 32] (Q8_0).
        scales: fp16 [..., NB].
    """
    scales = raw[..., :2].contiguous().view(torch.float16).squeeze(-1)
    payload = raw[..., 2:].contiguous().flatten(-2)
    return payload.view(torch.int32 if codec == "iq2_xxs" else torch.int8), scales


def iq2_lut(device: torch.device) -> torch.Tensor:
    """fp32 [256 * 8]: row i = ggml iq2xxs_grid[i] magnitudes (8, 25 or 43)."""
    grid = torch.tensor(IQ2XXS_GRID, dtype=torch.int64, device=device)
    return ((grid[:, None] >> (8 * torch.arange(8, device=device))) & 0xFF).float().flatten()


@triton.jit
def _decode_tile(qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
                 CODEC: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Weights [offs_n, k0:k0+BK] as an fp32 [BN, BK] tile. BK is a multiple of 32."""
    R: tl.constexpr = BK // 32
    rr = k0 // 32 + tl.arange(0, R)
    row = offs_n.to(tl.int64)[:, None]
    if CODEC == 1:
        d = tl.load(sc_ptr + row * stride_sn + (rr // 8)[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        idx = tl.load(qw_ptr + row * stride_qn + (2 * rr)[None, :], mask=n_mask[:, None], other=0)
        aux = tl.load(qw_ptr + row * stride_qn + (2 * rr + 1)[None, :],
                      mask=n_mask[:, None], other=0)
        rs = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
        g = tl.arange(0, 4)[None, None, :]
        gidx = (idx[:, :, None] >> (8 * g)) & 255
        s = (aux[:, :, None] >> (7 * g)) & 127
        p = s ^ (s >> 4)
        p = p ^ (p >> 2)
        p = p ^ (p >> 1)
        smask = s | ((p & 1) << 7)
        j = tl.arange(0, 8)[None, None, None, :]
        mag = tl.load(lut_ptr + gidx[:, :, :, None] * 8 + j)
        neg = (smask[:, :, :, None] >> j) & 1
        w = tl.reshape(tl.where(neg != 0, -mag, mag) * rs[:, :, None, None], (BN, BK))
    else:
        d = tl.load(sc_ptr + row * stride_sn + rr[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        q = tl.load(qw_ptr + row * stride_qn + (k0 + tl.arange(0, BK))[None, :],
                    mask=n_mask[:, None], other=0).to(tl.float32)
        w = tl.reshape(tl.reshape(q, (BN, R, 32)) * d[:, :, None], (BN, BK))
    return w


@triton.jit(do_not_specialize=["M"])
def _gemm_kernel(x_ptr, qw_ptr, sc_ptr, lut_ptr, out_ptr, M, N, K, k_per_split,
                 stride_xm, stride_qn, stride_sn,
                 CODEC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out[pid_k] = x[:, k-split] @ W[:, k-split]^T in fp32; out is [SPLIT, M, N]."""
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_n = tl.cdiv(N, BN)
    offs_m = (pid // num_n) * BM + tl.arange(0, BM)
    offs_n = (pid % num_n) * BN + tl.arange(0, BN)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        x = tl.load(x_ptr + offs_m.to(tl.int64)[:, None] * stride_xm
                    + (k0 + tl.arange(0, BK))[None, :], mask=m_mask[:, None], other=0.0)
        w = _decode_tile(qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
                         CODEC, BN, BK)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    out = (pid_k * M + offs_m.to(tl.int64))[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _gemv_kernel(x_ptr, qw_ptr, sc_ptr, lut_ptr, out_ptr, N, K, k_per_split,
                 stride_qn, stride_sn, CODEC: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """M == 1: out[pid_k, n] = x[k-split] . W[n, k-split] in fp32; out is [SPLIT, N]."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BK):
        x = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
        w = _decode_tile(qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
                         CODEC, BN, BK)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + pid_k * N + offs_n, acc, mask=n_mask)


def block_linear(x, qw, sc, lut, codec, BM=16, BN=64, BK=128, split=1, warps=4, stages=2,
                 gemv=False):
    """bf16 [M, K] @ packed [N, K]^T -> bf16 [M, N]. Split-K sums fp32 partials."""
    M, K = x.shape
    N = qw.shape[0]
    assert K % (split * BK) == 0
    part = torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    lut = sc if lut is None else lut
    if gemv:
        assert M == 1
        _gemv_kernel[(triton.cdiv(N, BN), split)](
            x, qw, sc, lut, part, N, K, K // split, qw.stride(0), sc.stride(0),
            CODEC=CODEC_ID[codec], BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    else:
        _gemm_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN), split)](
            x, qw, sc, lut, part, M, N, K, K // split, x.stride(0), qw.stride(0), sc.stride(0),
            CODEC=CODEC_ID[codec], BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return (part[0] if split == 1 else part.sum(0)).to(torch.bfloat16)


if __name__ == "__main__":
    from triton.testing import do_bench

    from ref_decode import ref_decode

    def make_raw(n, nb, codec, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        raw = torch.randint(0, 256, (n, nb, BYTES[codec]), dtype=torch.uint8, device="cuda",
                            generator=gen)
        d = (torch.randn(n, nb, device="cuda", generator=gen) * 2e-3).half()
        d.view(-1)[0::7] = 0.0
        d.view(-1)[1::11] = -d.view(-1)[1::11].abs()
        d.view(-1)[2::13] = 3e-6
        raw[..., :2] = d.unsqueeze(-1).view(torch.uint8)
        return raw

    ok = True
    for codec, n, nb, bk in (("iq2_xxs", 1032, 16, 64), ("iq2_xxs", 1032, 16, 256),
                             ("q8_0", 1032, 33, 32), ("q8_0", 520, 128, 128)):
        raw = make_raw(n, nb, codec, seed=n + nb)
        qw, sc = repack_blocks(raw, codec)
        lut = iq2_lut(raw.device) if codec == "iq2_xxs" else None
        ref = ref_decode(raw, codec)
        K = ref.shape[1]
        eye = torch.eye(K, device="cuda", dtype=torch.bfloat16)
        exact = torch.equal(block_linear(eye, qw, sc, lut, codec, BM=64, BK=bk, stages=1).T,
                            ref.bfloat16())
        worst = 0.0
        for M, gemv in ((1, True), (1, False), (3, False), (16, False), (100, False)):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            want = x.double() @ ref.double().T
            got = block_linear(x, qw, sc, lut, codec, BM=16 if M <= 16 else 64, BK=bk,
                               split=2 if K % (2 * bk) == 0 else 1, gemv=gemv,
                               stages=1 if bk == 256 else 2)  # 99 KB smem on sm_86/89/120
            cub = (x @ ref.bfloat16().T).double()
            worst = max(worst, ((got.double() - want).abs().max()
                                / (cub - want).abs().max()).item())
        ok &= exact and worst <= 1.5
        print(f"{codec} N={n} K={K} BK={bk}: decode bit-exact={exact}, "
              f"max err / cuBLAS err = {worst:.2f}")

    print(f"\nM=1 speed ({torch.cuda.get_device_name()}), best of a small sweep:")
    for codec, n, nb, tag in (("iq2_xxs", 5376, 16, "shared-up"), ("q8_0", 4608, 128, "qkv")):
        raw = make_raw(n, nb, codec, seed=2)
        qw, sc = repack_blocks(raw, codec)
        lut = iq2_lut(raw.device) if codec == "iq2_xxs" else None
        K = nb * BLOCK[codec]
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        best = min(
            (do_bench(lambda: block_linear(x, qw, sc, lut, codec, BN=bn, BK=bk, split=s,
                                           warps=w, gemv=True), rep=40) * 1e3, bn, bk, s, w)
            for bn in (8, 16, 32, 64) for bk in (128, 256) for s in (1, 2, 4, 8)
            for w in (1, 2, 4) if K % (s * bk) == 0)
        wd = ref_decode(raw, codec).bfloat16()
        tc = do_bench(lambda: x @ wd.T, rep=40) * 1e3
        t, bn, bk, s, w = best
        print(f"  {codec:7} {tag:9}: {t:6.1f} us {n * nb * BYTES[codec] / t / 1e3:5.0f} GB/s "
              f"(GEMV BN={bn} BK={bk} split={s} warps={w}) | bf16 cuBLAS {tc:6.1f} us")
    print("ALL OK" if ok else "FAIL")
