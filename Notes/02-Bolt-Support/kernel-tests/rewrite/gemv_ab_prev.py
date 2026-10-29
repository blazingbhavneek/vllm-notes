#!/usr/bin/env python
"""Old vs new `_gemv_kernel` / `_moe_gemv_kernel`, interleaved in one process.

GPU 0 drifts up to 3x in absolute kernel time between hours on this box, so
comparing a run to a recorded file is worthless. Both module versions are loaded
(`git show HEAD:...` into a file) and timed round-robin on identical tensors.

Run from the vllm repo root:
  git show HEAD:vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py > /tmp/prev_block.py
  CUDA_VISIBLE_DEVICES=0 python ../vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/gemv_ab_prev.py [rounds]
"""
import importlib.util
import statistics
import sys

import torch
import triton

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as new

import vllm.utils.torch_utils as vt  # noqa: E402

# The previous revision re-declares the same custom ops; importing it must not
# register them twice. The kernels are reached through launch_linear/invoke_*.
spec = importlib.util.spec_from_file_location("prev_block", "/tmp/prev_block.py")
prev = importlib.util.module_from_spec(spec)
real, vt.direct_register_custom_op = vt.direct_register_custom_op, lambda *a, **k: None
spec.loader.exec_module(prev)
vt.direct_register_custom_op = real

sys.path.insert(0, ".")
from flashinfer.testing import bench_gpu_time_with_cupti  # noqa: E402

DENSE = [
    ("qkv_proj", "q8_0", 4608, 4096), ("o_proj", "q8_0", 4096, 4096),
    ("fc1_latent", "q8_0", 1024, 4096), ("fc2_latent", "q8_0", 4096, 1024),
    ("shared_up", "iq2_xxs", 5376, 4096), ("shared_down", "iq2_xxs", 4096, 5376),
    ("lm_head", "q8_0", 131072, 4096),
]
MOE = [("moe_240x2048", 240, 2048), ("moe_496x2560", 496, 2560)]
TOP_K, HID = 9, 1024
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def us(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


torch.set_default_device("cuda")
torch.manual_seed(0)
props = torch.cuda.get_device_properties(torch.device("cuda"))
sms, smem = props.multi_processor_count, props.shared_memory_per_block_optin
print(f"{'shape':16s} {'M':>4s} " + " ".join(f"{'prev':>9s} {'new':>9s} {'x':>5s}"
                                             for _ in range(1)) + "  config")

def time_pair(label, M, run_prev, run_new, cfg):
    got = {"prev": [], "new": []}
    for _ in range(ROUNDS):
        for name, fn in (("prev", run_prev), ("new", run_new)):
            got[name].append(us(fn))
    p, n = statistics.median(got["prev"]), statistics.median(got["new"])
    print(f"{label:16s} {M:4d}  {p:9.1f} {n:9.1f} {p / n:4.2f}x  {cfg}")


for name, codec, N, K in DENSE:
    for M in (1, 2, 4):
        raw = torch.randint(0, 256, (N, K // (256 if codec == "iq2_xxs" else 32),
                                    66 if codec == "iq2_xxs" else 34), dtype=torch.uint8)
        qw, sc = new.repack_blocks(raw, codec)
        lut = new.iq2_lut(torch.device("cuda")) if codec == "iq2_xxs" else None
        x = torch.randn(M, K, dtype=torch.bfloat16)
        cfg = new._linear_config(M, N, K, codec, sms, smem)
        pcfg = prev._linear_config(M, N, K, codec, sms, smem)
        time_pair(f"{name} {codec[:2]}", M,
                  lambda: prev.launch_linear(x, qw, sc, lut, codec, pcfg),
                  lambda: new.launch_linear(x, qw, sc, lut, codec, cfg), str(cfg))

lut = new.iq2_lut(torch.device("cuda"))
for name, E, inter in MOE:
    w1, s1 = (x.contiguous() for x in new.repack_blocks(
        torch.randint(0, 256, (E, inter, HID // 256, 66), dtype=torch.uint8), "iq2_xxs"))
    gen = torch.Generator(device="cuda").manual_seed(0)
    for M in (1, 8):
        ids = torch.rand(M, E, generator=gen, device="cuda").topk(TOP_K).indices.to(torch.int32)
        wts = torch.rand(M, TOP_K, device="cuda").softmax(-1)
        a = torch.randn(M * TOP_K, HID, dtype=torch.bfloat16)
        c = torch.empty((M, TOP_K, inter), dtype=torch.bfloat16)
        routing = moe_align_block_size(ids, 1, E, None)
        cfg = new.moe_gemv_config(M, E, TOP_K, "iq2_xxs")
        time_pair(f"{name} gemv", M,
                  lambda: prev.invoke_block_moe_gemv_kernel(
                      a, w1, s1, lut, c, None, *routing, False, TOP_K, cfg, "iq2_xxs"),
                  lambda: new.invoke_block_moe_gemv_kernel(
                      a, w1, s1, lut, c, None, *routing, False, TOP_K, cfg, "iq2_xxs"),
                  str(cfg))
