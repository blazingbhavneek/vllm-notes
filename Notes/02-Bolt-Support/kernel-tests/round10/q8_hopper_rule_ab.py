"""A/B the SM90 single-row Q8_0 rule in `_linear_config` against the generic path.

Run from the vLLM repo root:

    python <notes>/Notes/02-Bolt-Support/kernel-tests/round10/q8_hopper_rule_ab.py \
        --out <dir>/q8_hopper_rule_ab.json

Arms, all through the production `launch_linear` wrapper:
  current : `_linear_config` with the device's real major (the rule applies on
            SM90 to (256|1024|4096, 4096) and (4096, 1024) only).
  generic : the same call with major=8 (GEMV + split-K reduction).
  wide1024 / wideK : the rule's tile forced onto every shape, BK=1024 or BK=K.
The question for the PR: does qkv_proj (4608, 4096), Bolt's biggest Q8_0 dense
layer, want the rule too, and is the rule still a win on this GPU?
Timing: CUDA-graph replay, `do_bench` (L2 flushed before each), best of 7 captures.
"""

import argparse
import json
from pathlib import Path

import torch
import triton

from tests.kernels.quantization.test_modelopt_block_triton import (
    assert_close_to_cublas,
    make_raw,
)
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

# Bolt TP=1 Q8_0 dense layers, plus the rule's (256, 4096) entry.
SHAPES = [
    (256, 4096),
    (1024, 4096),  # fc1_latent
    (4096, 4096),  # o_proj
    (4096, 1024),  # fc2_latent
    (4608, 4096),  # qkv_proj, not in the rule
    (2304, 4096),  # qkv_proj at TP=2
    (131072, 4096),  # lm_head
]


def bench_us(fn):
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return triton.testing.do_bench(graph.replay, return_mode="median") * 1e3


# Captures of one config land on two levels 15-25% apart on an RTX 5060 Ti (a
# ~10 us kernel), so recapture per run, interleave the arms and keep the best.
RUNS = 7


def wide(bk):
    return {"BM": 1, "BN": 4, "BK": bk, "SPLIT": 1, "num_warps": 4, "num_stages": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties()
    sms, smem = props.multi_processor_count, props.shared_memory_per_block_optin
    out = {
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "num_sms": sms,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shapes": [],
    }
    print(f"{props.name} SM{props.major}{props.minor} {sms} SMs")
    for N, K in SHAPES:
        raw = make_raw((N,), K, "q8_0", 0)
        qweight, scales = t.repack_blocks(raw, "q8_0")
        x = torch.randn(1, K, dtype=torch.bfloat16)
        # Exact fp32 weights, as the kernel uses them (bf16 would round them).
        ref = t.dequantize_blocks(qweight, scales, None, "q8_0", torch.float32)
        arms = {
            "current": t._linear_config(1, N, K, "q8_0", sms, smem, props.major),
            "generic": t._linear_config(1, N, K, "q8_0", sms, smem, 8),
            "wide1024": wide(min(K, 1024)),
            "wideK": wide(K),
        }
        row = {"N": N, "K": K}
        fns = {
            arm: lambda cfg=cfg: t.launch_linear(x, qweight, scales, None, "q8_0", cfg)
            for arm, cfg in arms.items()
        }
        for fn in fns.values():
            assert_close_to_cublas(fn(), x, ref)
        runs = {arm: [] for arm in arms}
        for _ in range(RUNS):
            for arm, fn in fns.items():
                runs[arm].append(bench_us(fn))
        for arm, cfg in arms.items():
            row[arm] = {"us": min(runs[arm]), "runs": runs[arm], "config": cfg}
        best = min(arms, key=lambda a: row[a]["us"])
        row["best"] = best
        out["shapes"].append(row)
        print(
            f"{N:6d} {K:5d} "
            + " ".join(f"{a}={row[a]['us']:7.2f}" for a in arms)
            + f"  best={best}",
            flush=True,
        )
        del qweight, scales, ref
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
