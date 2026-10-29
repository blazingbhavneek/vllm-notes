#!/usr/bin/env python
"""Print the real MoE / dense block shapes and kernel configs of the Bolt checkpoint."""
import collections

import vllm.model_executor.layers.quantization.modelopt_block_quant_triton as T

seen = collections.Counter()
_orig_moe = T.invoke_block_moe_kernel


def probe_moe(a, qw, sc, lut, c, tw, sti, eid, ntpp, mul, top_k, cfg, codec):
    seen[
        (
            "moe",
            a.size(0),
            qw.size(0),
            qw.size(1),
            a.size(1),
            top_k,
            cfg["BLOCK_SIZE_M"],
            cfg["BN"],
            cfg["BK"],
            cfg["num_warps"],
            cfg["num_stages"],
        )
    ] += 1
    seen[("moe_em", a.size(0), sti.numel())] += 1
    return _orig_moe(a, qw, sc, lut, c, tw, sti, eid, ntpp, mul, top_k, cfg, codec)


T.invoke_block_moe_kernel = probe_moe
import vllm.model_executor.layers.fused_moe.experts.modelopt_block_triton_moe as M  # noqa: E402

M.invoke_block_moe_kernel = probe_moe

_orig_lin = T.launch_linear


def probe_lin(x, qw, sc, lut, codec, cfg):
    seen[("lin", x.size(0), qw.size(0), x.size(1), codec, cfg["BM"], cfg["BN"],
           cfg["BK"], cfg["SPLIT"], cfg["num_warps"])] += 1
    return _orig_lin(x, qw, sc, lut, codec, cfg)


def probe_deq(qw, sc, lut, codec):
    seen[("deq", qw.size(0), sc.size(1), codec)] += 1
    return _orig_deq(qw, sc, lut, codec)


T.launch_linear = probe_lin
_orig_deq = T.dequantize_blocks
T.dequantize_blocks = probe_deq
# `_modelopt_block_linear` resolved these at import time inside the same module.
import vllm.model_executor.layers.quantization.modelopt_block_quant as Q  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(
    model="/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    trust_remote_code=True,
    language_model_only=True,
    max_model_len=1024,
    max_num_seqs=8,
    gpu_memory_utilization=0.27,
    enable_chunked_prefill=True,
    enforce_eager=True,
    seed=0,
    disable_log_stats=True,
)
sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
for b in (1, 8):
    llm.generate(["Q: 1+1? A:"] * b, sp, use_tqdm=False)
print("\n=== MoE launches (M, E, N, K, top_k, BM, BN, BK, warps, stages) ===")
for k, v in sorted(seen.items()):
    if k[0] == "moe":
        print(f"n={v:5d} M={k[1]:4d} E={k[2]:4d} N={k[3]:5d} K={k[4]:5d} topk={k[5]:3d} "
              f"BM={k[6]:3d} BN={k[7]:4d} BK={k[8]:4d} w={k[9]} s={k[10]}")
    elif k[0] == "moe_em":
        print(f"n={v:5d} EM for M={k[1]}: {k[2]}")
    elif k[0] == "lin":
        print(f"n={v:5d} LIN M={k[1]:5d} N={k[2]:6d} K={k[3]:5d} {k[4]:8s} "
              f"BM={k[5]:3d} BN={k[6]:4d} BK={k[7]:4d} SPLIT={k[8]:2d} w={k[9]}")
    else:
        print(f"n={v:5d} DEQ N={k[1]:6d} K={k[2]*256:6d} {k[3]}")
