"""Q8_0 dense: can vLLM's existing mixed-precision machinery (Marlin) replace the
PR's Triton Q8_0 path? Checks selection, accuracy, memory and speed on this GPU.

Run: python q8_marlin_check.py [select|accuracy|speed|all]
"""

import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS, choose_mp_linear_kernel
from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
    MarlinLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
    repack_blocks,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

U8 = scalar_types.uint8b128
# Bolt Q8_0 dense rows, [N, K] (from the todo's layer table / config.json).
SHAPES = {
    "qkv_proj": (4608, 4096),
    "o_proj": (4096, 4096),
    "fc1_latent": (1024, 4096),
    "fc2_latent": (4096, 1024),
    "lm_head": (131072, 4096),
}


def make_raw(n, k, seed):
    """Same generator as tests/kernels/quantization/test_modelopt_block_triton.py."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randint(0, 256, (n, k // 32, 34), dtype=torch.uint8, device="cuda",
                        generator=gen)
    d = (torch.randn((n, k // 32), device="cuda", generator=gen) * 2e-3).half()
    raw[..., :2] = d.unsqueeze(-1).view(torch.uint8)
    return raw


def cfg(n, k, act, kp=None, np_=None):
    return MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(kp or k, np_ or n),
        weight_type=U8,
        act_type=act,
        group_size=32,
        zero_points=False,
    )


def marlin_layer(q, scales, act, scale_dtype=None):
    """q int8 [N, K], scales fp16 [N, K/32] -> Marlin layer via the MP-kernel path
    (same param classes as compressed_tensors_wNa16)."""
    n, k = q.shape
    layer = torch.nn.Module()
    noop = lambda *a, **kw: None  # noqa: E731
    layer.register_parameter("weight_packed", PackedvLLMParameter(
        data=pack_quantized_values_into_int32(q.int() + 128, U8, packed_dim=1).contiguous(),
        input_dim=1, output_dim=0, packed_dim=1, packed_factor=4,
        weight_loader=noop))
    layer.register_parameter("weight_scale", GroupQuantScaleParameter(
        data=scales.to(scale_dtype or act).clone(), output_dim=0, input_dim=1,
        weight_loader=noop))
    kernel = MarlinLinearKernel(cfg(n, k, act), "weight_packed", "weight_scale", None)
    kernel.process_weights_after_loading(layer)
    return lambda x: kernel.apply_weights(layer, x), layer


def triton_op(qw, sc):
    return lambda x: torch.ops.vllm.modelopt_block_linear(x, qw, sc, None, "q8_0")


def select():
    print(f"GPU {torch.cuda.get_device_name()} cc {current_platform.get_device_capability()}")
    with set_current_vllm_config(VllmConfig()):
        for act in (torch.bfloat16, torch.float16):
            for name, (n, k) in SHAPES.items():
                c = cfg(n, k, act)
                print(f"{act} {name:11s} -> {choose_mp_linear_kernel(c).__name__}")
        c = cfg(4096, 4096, torch.bfloat16)
        print("\nper-kernel (o_proj, bf16):")
        for kern in _POSSIBLE_KERNELS[current_platform._enum]:
            ok, why = kern.can_implement(c)
            print(f"  {kern.__name__:28s} min_cc={kern.get_min_capability():3d} "
                  f"can_implement={ok} {why or ''}")
        print("\nMarlin with TP=2 partitions:")
        for name, (n, k) in SHAPES.items():
            for kind, kp, np_ in (("row", k // 2, n), ("col", k, n // 2)):
                ok, why = MarlinLinearKernel.can_implement(
                    cfg(n, k, torch.bfloat16, kp, np_))
                print(f"  {name:11s} {kind}: {ok} {why or ''}")


def errs(out, want):
    d = out.double() - want
    return (d.abs().max() / want.abs().max()).item(), (d.norm() / want.norm()).item()


def accuracy():
    print("max|err|/max|y|   rel-Frobenius   (vs exact fp64 product of the Q8_0 weights)")
    for name in ("o_proj", "fc2_latent"):
        n, k = SHAPES[name]
        qw, sc = repack_blocks(make_raw(n, k, seed=1), "q8_0")
        w = qw.double() * sc.double().repeat_interleave(32, dim=1)
        for act in (torch.bfloat16, torch.float16):
            marlin, _ = marlin_layer(qw, sc, act)
            tri = triton_op(qw, sc)
            for m in (1, 16, 64, 256):
                x = torch.randn(m, k, device="cuda", dtype=act)
                want = x.double() @ w.T
                # Reference with the scale rounded to the activation dtype first,
                # to isolate Marlin's scale-rounding error.
                w_sr = qw.double() * sc.to(act).double().repeat_interleave(32, dim=1)
                row = {
                    "triton(PR)": errs(tri(x), want),
                    "marlin": errs(marlin(x), want),
                    "cuBLAS bf16-W": errs(x @ w.to(act).T, want),
                    "exact,scale->act": errs((x.double() @ w_sr.T), want),
                }
                print(f"{name:10s} {str(act):15s} M={m:<4d} " + "  ".join(
                    f"{kname} {a:.2e}/{b:.2e}" for kname, (a, b) in row.items()))
    # Safety: what if FP16 checkpoint scales reach Marlin unconverted under BF16?
    n, k = SHAPES["o_proj"]
    qw, sc = repack_blocks(make_raw(n, k, seed=2), "q8_0")
    w = qw.double() * sc.double().repeat_interleave(32, dim=1)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    try:
        f, _ = marlin_layer(qw, sc, torch.bfloat16, scale_dtype=torch.float16)
        print("fp16 scales + bf16 act (unconverted):", errs(f(x), x.double() @ w.T))
    except Exception as e:  # noqa: BLE001
        print("fp16 scales + bf16 act (unconverted): raises", type(e).__name__,
              str(e)[:200])


def speed():
    ms = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    act = torch.bfloat16
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    print(f"us per call, CUDA graph, weights rotated over >= 4x L2 "
          f"({l2 / 2**20:.0f} MiB) so every call reads DRAM. "
          f"cuBLAS = bf16 weights (2x memory).")
    for name, (n, k) in SHAPES.items():
        q_bytes = n * k + n * (k // 32) * 2
        reps = max(1, -(-4 * l2 // q_bytes))
        tri, mar, cub, layers = [], [], [], []
        for r in range(reps):
            qw, sc = repack_blocks(make_raw(n, k, seed=3 + r), "q8_0")
            f, layer = marlin_layer(qw, sc, act)
            mar.append(f)
            layers.append(layer)
            tri.append(triton_op(qw, sc))
            cub.append((qw.float() * sc.float().repeat_interleave(32, dim=1)).to(act))
        m_bytes = sum(p.numel() * p.element_size() for p in layers[0].parameters())
        print(f"\n{name} N={n} K={k} x{reps} copies: PR {q_bytes/2**20:.1f} MiB, "
              f"Marlin {m_bytes/2**20:.1f} MiB, bf16 {n*k*2/2**20:.1f} MiB per copy")
        print(f"{'M':>5s} {'triton(PR)':>11s} {'marlin':>9s} {'cuBLAS':>9s} "
              f"{'PR/marlin':>10s} {'PR GB/s':>8s} {'marlin GB/s':>11s}")
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=act)

            def run(fs):
                for f in fs:
                    f(x)

            t_tri = do_bench_cudagraph(lambda: run(tri), rep=300) * 1e3 / reps
            t_mar = do_bench_cudagraph(lambda: run(mar), rep=300) * 1e3 / reps
            t_cub = do_bench_cudagraph(
                lambda: [F.linear(x, w) for w in cub], rep=300) * 1e3 / reps
            print(f"{m:5d} {t_tri:11.1f} {t_mar:9.1f} {t_cub:9.1f} "
                  f"{t_tri/t_mar:9.2f}x {q_bytes/t_tri/1e3:8.0f} "
                  f"{m_bytes/t_mar/1e3:11.0f}")
        del tri, mar, cub, layers
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.manual_seed(0)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29517", backend="nccl")
        initialize_model_parallel(1, 1)
        for step in (select, accuracy, speed):
            if what in ("all", step.__name__):
                print(f"\n===== {step.__name__} =====")
                step()
