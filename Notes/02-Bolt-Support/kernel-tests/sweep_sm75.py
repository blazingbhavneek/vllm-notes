"""FP16 tl.dot tiles for sm_75 (FMA, no tensor cores): registers, spills, smem."""

import itertools
import os
import re
import subprocess
import tempfile

import torch
import triton
from triton.backends.compiler import GPUTarget

from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

BIN = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin")
dev, N, K, E = "cuda", 256, 1024, 4
f16 = torch.float16


def sm75(ck, warps, stages):
    k = triton.compile(
        ck.src,
        target=GPUTarget("cuda", 75, 32),
        options={"num_warps": warps, "num_stages": stages},
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(k.asm["ptx"])
    out = subprocess.run(
        [os.path.join(BIN, "ptxas"), "-v", "-arch=sm_75", f.name, "-o", "/dev/null"],
        capture_output=True, text=True,
    ).stderr  # fmt: skip
    regs = max(map(int, re.findall(r"Used (\d+) registers", out) or [0]))
    spill = sum(map(int, re.findall(r"(\d+) bytes spill stores", out)))
    return regs, spill, k.metadata.shared // 1024


for codec in ("iq2_xxs", "q8_0"):
    c = t._CODEC_ID[codec]
    if codec == "iq2_xxs":
        qw = torch.zeros((N, K // 16), dtype=torch.int32, device=dev)
        sc = torch.zeros((N, K // 256), dtype=f16, device=dev)
        qe = torch.zeros((E, N, K // 16), dtype=torch.int32, device=dev)
        se = torch.zeros((E, N, K // 256), dtype=f16, device=dev)
        lut = t.iq2_lut(torch.device(dev))
    else:
        qw = torch.zeros((N, K), dtype=torch.int8, device=dev)
        sc = torch.zeros((N, K // 32), dtype=f16, device=dev)
        qe = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
        se = torch.zeros((E, N, K // 32), dtype=f16, device=dev)
        lut = None
    for bm, bn, bk, w in itertools.product((16, 32, 64), (32, 64), (32, 64), (2, 4, 8)):
        x = torch.zeros((bm, K), dtype=f16, device=dev)
        o = torch.zeros((1, bm, N), dtype=f16, device=dev)
        ck = t._gemm_kernel.warmup(
            x, qw, sc, lut, o, bm, N, K, K, qw.stride(0), sc.stride(0), CODEC=c,
            BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=1, grid=(1,),
        )  # fmt: skip
        r, s, kb = sm75(ck, w, 1)
        a = torch.zeros((64, K), dtype=f16, device=dev)
        cc = torch.zeros((64, N), dtype=f16, device=dev)
        ck = t._moe_gemm_kernel.warmup(
            a, qe, se, lut, cc, torch.zeros(64, device=dev),
            torch.zeros(256, dtype=torch.int32, device=dev),
            torch.zeros(16, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev), N, K, 256, 64, a.stride(0),
            qe.stride(0), qe.stride(1), se.stride(0), se.stride(1), cc.stride(0),
            top_k=1, MUL_ROUTED_WEIGHT=True, CODEC=c, BM=bm, BN=bn, BK=bk,
            num_warps=w, num_stages=1, grid=(1,),
        )  # fmt: skip
        mr, ms, mkb = sm75(ck, w, 1)
        print(f"{codec} BM{bm} BN{bn} BK{bk} w{w}: gemm r{r} spill{s} {kb}KB | "
              f"moe r{mr} spill{ms} {mkb}KB", flush=True)
