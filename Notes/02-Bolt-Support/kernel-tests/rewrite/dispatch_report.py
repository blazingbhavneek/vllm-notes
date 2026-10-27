"""Cross-compile every dispatched config: shared memory per target, sm_90 spills.

Run from the vllm repo root. Any "BAD" line fails Appendix D of the plan.
"""
import itertools, os, re, subprocess, sys, tempfile
import torch, triton
from triton.backends.compiler import GPUTarget
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

BIN = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin")
LIMIT = {80: 163, 86: 99, 89: 99, 90: 227, 100: 227, 120: 99}

def ptxas_spills(ptx, cc):
    arch = {90: "sm_90a", 100: "sm_100a"}.get(cc, f"sm_{cc}")
    exe = os.path.join(BIN, "ptxas-blackwell" if cc >= 100 else "ptxas")
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx)
    out = subprocess.run([exe, "-v", f"-arch={arch}", f.name, "-o", "/dev/null"],
                         capture_output=True, text=True).stderr
    regs = max(map(int, re.findall(r"Used (\d+) registers", out) or [0]))
    spill = sum(map(int, re.findall(r"(\d+) bytes spill stores", out)))
    return regs, spill

def report(name, ck, warps, stages, ccs=(80, 86, 89, 90, 100)):
    row = [name, f"w{warps} s{stages}", f"sm120={ck.metadata.shared/1024:.0f}"]
    worst = ck.metadata.shared / 1024 > LIMIT[120]
    for cc in ccs:
        k = triton.compile(ck.src, target=GPUTarget("cuda", cc, 32),
                           options={"num_warps": warps, "num_stages": stages})
        kb = k.metadata.shared / 1024
        worst |= kb > LIMIT[cc]
        extra = ""
        if cc == 90:
            regs, sp = ptxas_spills(k.asm["ptx"], 90)
            extra = f"(r{regs} spill{sp})"
            worst |= sp > 0
        row.append(f"sm{cc}={kb:.0f}{extra}")
    print(("BAD " if worst else "ok  ") + " ".join(row), flush=True)

dev = "cuda"
N, K, E = 256, 1024, 4
for codec in ("iq2_xxs", "q8_0"):
    c = t._CODEC_ID[codec]
    if codec == "iq2_xxs":
        qw = torch.zeros((N, K // 16), dtype=torch.int32, device=dev); sc = torch.zeros((N, K // 256), dtype=torch.float16, device=dev)
        lut = t.iq2_lut(torch.device(dev))
        qe = torch.zeros((E, N, K // 16), dtype=torch.int32, device=dev); se = torch.zeros((E, N, K // 256), dtype=torch.float16, device=dev)
    else:
        qw = torch.zeros((N, K), dtype=torch.int8, device=dev); sc = torch.zeros((N, K // 32), dtype=torch.float16, device=dev)
        lut = None
        qe = torch.zeros((E, N, K), dtype=torch.int8, device=dev); se = torch.zeros((E, N, K // 32), dtype=torch.float16, device=dev)
    gemv = [(1, 8, 256, 1), (2, 8, 256, 1), (4, 8, 256, 1)] if codec == "iq2_xxs" else [(1, 32, 256, 4)]
    for bm, bn, bk, w in gemv:
        x = torch.zeros((bm, K), dtype=torch.bfloat16, device=dev); o = torch.zeros((1, bm, N), dtype=torch.bfloat16, device=dev)
        ck = t._gemv_kernel.warmup(x, qw, sc, lut, o, bm, N, K, K, qw.stride(0), sc.stride(0), CODEC=c, BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=2, grid=(1,))
        report(f"gemv {codec} BM{bm} BN{bn} BK{bk}", ck, w, 2)
    for bm, bn, bk, w in [(16, 64, 128, 4), (32, 64, 128, 4), (64, 64, 128, 4), (64, 128, 64, 8), (128, 128, 64, 8)]:
        x = torch.zeros((bm, K), dtype=torch.bfloat16, device=dev); o = torch.zeros((1, bm, N), dtype=torch.bfloat16, device=dev)
        ck = t._gemm_kernel.warmup(x, qw, sc, lut, o, bm, N, K, K, qw.stride(0), sc.stride(0), CODEC=c, BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=2, grid=(1,))
        report(f"gemm {codec} BM{bm} BN{bn} BK{bk}", ck, w, 2)
    bn, w = (8, 1) if codec == "iq2_xxs" else (32, 4)
    o = torch.zeros((N, K), dtype=torch.bfloat16, device=dev)
    ck = t._dequant_kernel.warmup(qw, sc, lut, o, N, K, qw.stride(0), sc.stride(0), CODEC=c, BN=bn, BK=256, num_warps=w, num_stages=3, grid=(1,))
    report(f"dequant {codec} BN{bn} BK256", ck, w, 3)
    for bm, bn, bk, w in [(16, 64, 128, 4), (32, 64, 128, 4), (64, 64, 64, 4), (128, 128, 64, 8)]:
        a = torch.zeros((64, K), dtype=torch.bfloat16, device=dev); cc_ = torch.zeros((64, N), dtype=torch.bfloat16, device=dev)
        tw = torch.zeros((64,), dtype=torch.float32, device=dev); ids = torch.zeros((256,), dtype=torch.int32, device=dev)
        eids = torch.zeros((16,), dtype=torch.int32, device=dev); ntp = torch.zeros((1,), dtype=torch.int32, device=dev)
        ck = t._moe_gemm_kernel.warmup(a, qe, se, lut, cc_, tw, ids, eids, ntp, N, K, 256, 64, a.stride(0), qe.stride(0), qe.stride(1), se.stride(0), se.stride(1), cc_.stride(0), top_k=1, MUL_ROUTED_WEIGHT=True, CODEC=c, BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=2, grid=(1,))
        report(f"moe {codec} BM{bm} BN{bn} BK{bk}", ck, w, 2)
