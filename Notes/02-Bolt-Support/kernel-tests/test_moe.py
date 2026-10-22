"""moe_forward vs fp64 reference. vLLM ops replaced by pure-torch stand-ins."""

import os
import sys
import types

import torch

from harness import hopper, make_packed, oracle_decode


def fake_align(topk_ids, block_size, num_experts, expert_map=None,
               pad_sorted_ids=False, ignore_invalid_experts=False):
    flat = topk_ids.flatten().long()
    numel = flat.numel()
    maxp = numel + num_experts * (block_size - 1)
    sorted_ids = torch.full((maxp,), numel, dtype=torch.int32, device=flat.device)
    # poison unused blocks: kernel must exit on num_post before reading them
    expert_ids = torch.full(((maxp + block_size - 1) // block_size,), 1 << 28,
                            dtype=torch.int32, device=flat.device)
    pos = 0
    for e in range(num_experts):
        idx = (flat == e).nonzero().flatten()
        if idx.numel() == 0:
            continue
        padded = -(-idx.numel() // block_size) * block_size
        sorted_ids[pos:pos + idx.numel()] = idx.int()
        expert_ids[pos // block_size:(pos + padded) // block_size] = e
        pos += padded
    return sorted_ids, expert_ids, torch.tensor([pos], dtype=torch.int32,
                                                device=flat.device)


def fake_moe_sum(inp, out):
    out.copy_(inp.float().sum(1).to(out.dtype))


for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe"):
    sys.modules.setdefault(name, types.ModuleType(name))
m = types.ModuleType("vllm._custom_ops")
m.moe_sum = fake_moe_sum
sys.modules["vllm._custom_ops"] = m
m = types.ModuleType("vllm.model_executor.layers.fused_moe.moe_align_block_size")
m.moe_align_block_size = fake_align
sys.modules[m.__name__] = m

if os.environ.get("FORCE_STAGES"):
    st = int(os.environ["FORCE_STAGES"])
    for k in ("_moe_pass1_iq2_kernel", "_moe_pass2_iq2_kernel"):
        kern = getattr(hopper, k)

        class _Wrap:
            def __init__(self, kern):
                self.kern = kern

            def __getitem__(self, grid):
                def launch(*a, **kw):
                    kw["num_stages"] = st
                    return self.kern[grid](*a, **kw)
                return launch

        setattr(hopper, k, _Wrap(kern))


def reference(x, w13, w2, tw, ti, round_mid):
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for t in range(x.shape[0]):
        for j in range(ti.shape[1]):
            e = ti[t, j]
            h = torch.relu(x[t].double() @ w13[e].double().T) ** 2
            if round_mid:  # mirror kernel: c1 stored as bf16
                h = h.bfloat16().double()
            out[t] += (h @ w2[e].double().T) * tw[t, j].double()
    return out


fails = []
torch.manual_seed(0)
for E, K, I, topk in [(64, 1024, 1536, 9), (128, 1024, 2560, 9), (8, 512, 256, 2)]:
    w13p = make_packed((E, I, K // 256), "iq2_xxs", seed=E, scale=1e-3)
    w2p = make_packed((E, K, I // 256), "iq2_xxs", seed=E + 1, scale=1e-3)
    w13 = oracle_decode(w13p, "iq2_xxs").bfloat16()
    w2 = oracle_decode(w2p, "iq2_xxs").bfloat16()
    for M in (1, 3, 16, 33, 200):
        x = (torch.randn(M, K, device="cuda") * 0.5).bfloat16()
        logits = torch.randn(M, E, device="cuda")
        tw, ti = torch.topk(torch.softmax(logits, -1), topk)
        ti = ti.int()
        try:
            got = hopper.moe_forward(x, w13p, w2p, tw, ti).double()
        except Exception as ex:  # noqa: BLE001
            print(f"FAIL E={E} I={I} M={M}: {type(ex).__name__}: {str(ex)[:100]}")
            fails.append((E, I, M))
            continue
        ref = reference(x, w13, w2, tw, ti, round_mid=True)
        scale = ref.abs().max().item()
        rel = (got - ref).abs().max().item() / scale
        ok = torch.isfinite(got).all().item() and rel < 1e-2
        print(f"{'PASS' if ok else 'FAIL'} E={E} K={K} I={I} topk={topk} M={M} "
              f"rel={rel:.2e}")
        if not ok:
            fails.append((E, I, M))
print("FAILURES:", fails or "none")
