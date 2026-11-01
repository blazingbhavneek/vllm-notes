"""Full routed-MoE A/B for packed-register decode, direct routing on both arms."""

import sys, statistics

sys.path.insert(0, ".")
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from packed_register_ab import packed_dot4
from tests.kernels.quantization.test_modelopt_block_triton import (
    make_moe,
    make_routing,
    assert_moe_close_to_bf16,
)
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import (
    modelopt_block_triton_moe as moe,
)
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t
from vllm.v1.worker.workspace import init_workspace_manager


@triton.jit
def packed_moe_kernel(
    xp,
    qp,
    sp,
    gp,
    op,
    ids,
    weights,
    N,
    K,
    SX,
    SQE,
    SQN,
    SSE,
    SSN,
    SC,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    RELU2: tl.constexpr = False,
):
    slot = tl.program_id(0)
    e = tl.load(ids + slot).to(tl.int64)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    xp += slot // TOPK * SX
    qp += e * SQE
    sp += e * SSE
    group = tl.arange(0, BK // 4)
    acc = tl.zeros((BN, BK // 4), tl.float32)
    for k0 in range(0, K, BK):
        k4 = k0 // 4 + group
        rr = k4 // 8
        idx = tl.load(
            qp + ns.to(tl.int64)[:, None] * SQN + 2 * rr[None, :],
            ns[:, None] < N,
            other=0,
        )
        aux = tl.load(
            qp + ns.to(tl.int64)[:, None] * SQN + 2 * rr[None, :] + 1,
            ns[:, None] < N,
            other=0,
        )
        d = tl.load(
            sp + ns.to(tl.int64)[:, None] * SSN + (rr // 8)[None, :],
            ns[:, None] < N,
            other=0.0,
        ).to(tl.float32)
        rs = d * (0.5 + ((aux >> 28) & 15).to(tl.float32)) * 0.25
        g = (k4 % 8) // 2
        ix = (idx >> (8 * g[None, :])) & 255
        sg = (aux >> (7 * g[None, :])) & 127
        parity = sg ^ (sg >> 4)
        parity ^= parity >> 2
        parity ^= parity >> 1
        sg |= (parity & 1) << 7
        sg = (sg >> (4 * (k4 % 2)[None, :])) & 15
        gw = tl.load(gp + ix * 2 + (k4 % 2)[None, :])
        x0 = tl.load(xp.to(tl.pointer_type(tl.int32)) + 2 * k4)
        x1 = tl.load(xp.to(tl.pointer_type(tl.int32)) + 2 * k4 + 1)
        acc += packed_dot4(gw, sg, rs, x0[None, :], x1[None, :])
    val = tl.sum(acc, 1)
    if MUL:
        val *= tl.load(weights + slot).to(tl.float32)
    if RELU2:
        val = tl.maximum(val.to(op.dtype.element_ty).to(tl.float32), 0.0)
        val = val * val
    tl.store(op + slot * SC + ns, val, ns < N)


def us(fn):
    for _ in range(10):
        fn()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1000


torch.set_default_device("cuda")
torch.manual_seed(0)
init_workspace_manager(torch.device("cuda"))
grid = torch.tensor(t._IQ2XXS_GRID, dtype=torch.int64).view(torch.int32)
base_launch = moe.invoke_block_moe_gemv_kernel
base_config = moe.moe_gemv_config
selected = (8, 256)


def packed_launch(
    a,
    q,
    s,
    lut,
    c,
    tw,
    sorted_ids,
    expert_ids,
    ntp,
    mul,
    top_k,
    config,
    codec,
    apply_relu2=False,
):
    bn, bk = selected
    assert sorted_ids is None
    packed_moe_kernel[(expert_ids.numel(), triton.cdiv(q.shape[1], bn))](
        a,
        q,
        s,
        grid,
        c,
        expert_ids,
        tw,
        q.shape[1],
        a.shape[1],
        a.stride(0),
        q.stride(0),
        q.stride(1),
        s.stride(0),
        s.stride(1),
        c.stride(-2),
        TOPK=top_k,
        MUL=mul,
        BN=bn,
        BK=bk,
        RELU2=apply_relu2,
        num_warps=1,
        num_stages=2,
    )


for E, I in [(512, 2048)]:
    run, w1, w2 = make_moe(
        "iq2_xxs", MoEActivation.RELU2_NO_MUL, E=E, K=1024, inter=I, top_k=9
    )
    for M in (1, 8):
        x = torch.randn(M, 1024, dtype=torch.bfloat16)
        weights, ids = make_routing(M, E, 9)
        moe.invoke_block_moe_gemv_kernel = base_launch
        reference = run(x, weights, ids).clone()
        base = us(lambda: run(x, weights, ids))
        results = []
        moe.invoke_block_moe_gemv_kernel = packed_launch
        for bn in (1, 2, 4, 8, 16):
            for bk in (128, 256, 512, 1024):
                selected = (bn, bk)
                got = run(x, weights, ids)
                torch.testing.assert_close(
                    got, reference, rtol=0, atol=reference.abs().max().item() / 256
                )
                v = us(lambda: run(x, weights, ids))
                results.append((v, selected))
        for v, cfg in sorted(results)[:5]:
            print(
                E,
                I,
                M,
                "base",
                round(base, 2),
                "packed",
                round(v, 2),
                "speedup",
                round(base / v, 3),
                cfg,
                flush=True,
            )
moe.invoke_block_moe_gemv_kernel = base_launch
