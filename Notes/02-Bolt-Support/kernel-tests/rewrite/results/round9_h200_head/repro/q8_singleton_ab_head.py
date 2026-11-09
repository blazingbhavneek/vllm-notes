"""A/B the SM90 single-row Q8_0 dispatch against the generic split-K GEMV.

Run from the vLLM repo root on GPU 1:

    CUDA_VISIBLE_DEVICES=1 /home/seigyo/vllmServer/.venv/bin/python \
        ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/q8_singleton_ab_head.py \
        --out <dir>/q8_singleton_ab.json

Both arms use the production `launch_linear` wrapper. The singleton arm asks
`_linear_config` for its real SM90 choice; the control arm asks for the same
config with `major=8`, which selects the generic Q8_0 GEMV + split-K reduction.
Timing is CUPTI with a CUDA graph, the same method as the repo benchmark.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from tests.kernels.quantization.test_modelopt_block_triton import (
    assert_close_to_cublas,
    make_raw,
)
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

SHAPES = [(256, 4096), (1024, 4096), (4096, 4096), (4096, 1024), (131072, 4096)]


def bench_us(fn):
    for _ in range(25):
        fn()
    torch.accelerator.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties()
    out = {
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "shapes": [],
    }

    print(f"{props.name} SM{props.major}{props.minor} {props.multi_processor_count} SMs")
    print("N K arm us config")
    for N, K in SHAPES:
        raw = make_raw((N,), K, "q8_0", 0)
        qweight, scales = t.repack_blocks(raw, "q8_0")
        x = torch.randn(1, K, dtype=torch.bfloat16)
        ref = t.dequantize_blocks(qweight, scales, None, "q8_0")

        singleton = t._linear_config(
            1, N, K, "q8_0", props.multi_processor_count,
            props.shared_memory_per_block_optin, props.major,
        )  # fmt: skip
        control = t._linear_config(
            1, N, K, "q8_0", props.multi_processor_count,
            props.shared_memory_per_block_optin, 8,
        )  # fmt: skip

        row = {"N": N, "K": K}
        for arm, cfg in (("singleton", singleton), ("splitk_control", control)):
            fn = lambda cfg=cfg: t.launch_linear(x, qweight, scales, None, "q8_0", cfg)
            y = fn()
            assert_close_to_cublas(y, x, ref)
            us = bench_us(fn)
            row[arm] = {"us": us, "config": cfg}
            print(f"{N} {K} {arm} {us:.2f} {cfg}", flush=True)
        row["speedup"] = row["splitk_control"]["us"] / row["singleton"]["us"]
        print(f"{N} {K} speedup {row['speedup']:.2f}x", flush=True)
        out["shapes"].append(row)

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
