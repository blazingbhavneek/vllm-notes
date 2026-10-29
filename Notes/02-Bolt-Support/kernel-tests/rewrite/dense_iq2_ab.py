#!/usr/bin/env python
"""A/B the dense IQ2 GEMV tiles on Bolt's shared-expert shapes at decode M.

`_linear_config` picks BN=8/BK=256/1 warp for IQ2 decode, tuned on an RTX 5060
Ti. This re-checks that pick (and the split-K budget) on the box that serves the
model. Only a candidate set is enumerated here; the dispatch rule stays in
`_linear_config`. Confirm any winner with a serve run -- kernel-level timings
have disagreed with serving on the MoE path.

  CUDA_VISIBLE_DEVICES=0 python dense_iq2_ab.py [M ...]
"""
import statistics
import sys

import torch
import triton
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

SHAPES = [("shared_up", 5376, 4096), ("shared_down", 4096, 5376)]
# (BN, num_warps) pairs that keep ~64 decoded values in flight per lane.
TILES = [(2, 1), (4, 1), (8, 1), (16, 1), (32, 4)]
MAX_SPLIT = 16


def us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


def candidates(M, N, K, num_sms):
    """Every (BN, BK, SPLIT, warps) whose split covers K and fills the GPU."""
    tiles_of = lambda bn: -(-N // bn)  # noqa: E731
    for bn, warps in TILES:
        for bk in (256, 128):
            steps = K // bk
            for split in range(1, min(MAX_SPLIT, steps) + 1):
                if steps % split:
                    continue
                if M <= 4 and tiles_of(bn) * split >= 4 * 32 * num_sms:
                    continue  # far past the decode budget
                yield {"BM": 1, "BN": bn, "BK": bk, "SPLIT": split}, warps


def main() -> None:
    ms = [int(a) for a in sys.argv[1:]] or [1, 2]
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    sms, smem = props.multi_processor_count, props.shared_memory_per_block_optin
    for name, N, K in SHAPES:
        qweight, scales = (
            x.contiguous()
            for x in t.repack_blocks(
                torch.randint(0, 256, (N, K // 256, 66), dtype=torch.uint8), "iq2_xxs"
            )
        )
        lut = t.iq2_lut(dev)
        mb = N * K * 66 / 256 / 1e6
        for M in ms:
            base = t._linear_config(M, N, K, "iq2_xxs", sms, smem)
            x = torch.randn(M, K, dtype=torch.bfloat16)
            out = {}
            print(f"\n=== {name} N={N} K={K} M={M}  {mb:.1f} MB  "
                  f"in-tree {base} ===", flush=True)
            for cfg, warps in candidates(M, N, K, sms):
                cfg["BM"] = triton.next_power_of_2(M)
                try:
                    v = us(
                        lambda c=cfg, w=warps: t.launch_linear(
                            x, qweight, scales, lut, "iq2_xxs", {**c,
                            "num_warps": w, "num_stages": 2}
                        )
                    )
                except Exception as e:  # a tile that cannot be compiled here
                    print(f"  {cfg} w{warps} -> {type(e).__name__}", flush=True)
                    continue
                key = (cfg["BN"], cfg["BK"], cfg["SPLIT"], warps)
                out[key] = v
                print(f"  BN{key[0]:<3} BK{key[1]:<4} SPLIT{key[2]:<3} w{key[3]}  "
                      f"{v:7.1f} us  {mb / v * 1e3:4.0f} GB/s", flush=True)
            best = min(out.items(), key=lambda kv: kv[1])
            in_tree = (base["BN"], base["BK"], base["SPLIT"], base["num_warps"])
            print(f"  best  {best[0]} -> {best[1]:.1f} us; in-tree {in_tree} -> "
                  f"{out.get(in_tree, float('nan')):.1f} us; "
                  f"{out.get(in_tree, float('nan')) / best[1]:.2f}x", flush=True)


main()
