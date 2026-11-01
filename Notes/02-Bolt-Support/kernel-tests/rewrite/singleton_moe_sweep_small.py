"""Adaptive singleton-expert GEMV inside the grouped GEMM grid."""

import sys, statistics

sys.path.insert(0, ".")
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_moe, make_routing
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
    _decode_tile,
    _CODEC_ID,
)
from vllm.model_executor.layers.fused_moe.experts import (
    modelopt_block_triton_moe as moe,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.v1.worker.workspace import init_workspace_manager


@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def singleton_kernel(
    a_ptr,
    qw_ptr,
    sc_ptr,
    lut_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_qe,
    stride_qn,
    stride_se,
    stride_sn,
    stride_cm,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    CODEC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """`fused_moe_kernel` with B decoded from packed blocks of expert e."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    pid_m = pid // num_n
    if pid_m * BM >= tl.load(num_tokens_post_padded_ptr):
        return
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BM + tl.arange(0, BM))
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = (pid % num_n) * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    a_rows = a_ptr + (offs_token // top_k)[:, None] * stride_am
    qw_ptr += e * stride_qe
    sc_ptr += e * stride_se
    if BM <= 64:
        first = tl.load(sorted_token_ids_ptr + pid_m * BM).to(tl.int64)
        second = tl.load(sorted_token_ids_ptr + pid_m * BM + 1)
        if second >= num_valid_tokens:
            row = a_ptr + (first // top_k) * stride_am
            single = tl.zeros((BN,), dtype=tl.float32)
            for k0 in range(0, K, BK):
                w = (
                    _decode_tile(
                        qw_ptr,
                        sc_ptr,
                        lut_ptr,
                        offs_n,
                        n_mask,
                        k0,
                        stride_qn,
                        stride_sn,
                        CODEC,
                        True,
                        BN,
                        BK,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                x = tl.load(row + k0 + tl.arange(0, BK)).to(tl.float32)
                single += tl.sum(w * x[None, :], axis=1)
            if MUL_ROUTED_WEIGHT:
                single *= tl.load(topk_weights_ptr + first).to(tl.float32)
            tl.store(
                c_ptr + first * stride_cm + offs_n,
                single.to(c_ptr.dtype.element_ty),
                n_mask,
            )
            return
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(
            a_rows + (k0 + tl.arange(0, BK))[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        w = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
            CODEC, False, BN, BK,
        )  # fmt: skip
        acc += tl.dot(a, tl.trans(w.to(tl.bfloat16)))
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc *= weight.to(tl.float32)[:, None]
    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


def invoke(a, q, s, lut, c, tw, ids, eids, ntp, mul, top_k, cfg, codec):
    bm = cfg["BLOCK_SIZE_M"]
    em = ids.size(0)
    if a.size(0) < bm:
        em = min(em, a.size(0) * top_k * bm)
    n = q.size(1)
    singleton_kernel[(triton.cdiv(em, bm) * triton.cdiv(n, cfg["BN"]),)](
        a,
        q,
        s,
        lut,
        c,
        tw,
        ids,
        eids,
        ntp,
        n,
        a.size(1),
        em,
        a.size(0) * top_k,
        a.stride(0),
        q.stride(0),
        q.stride(1),
        s.stride(0),
        s.stride(1),
        c.stride(-2),
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul,
        CODEC=_CODEC_ID[codec],
        BM=bm,
        BN=cfg["BN"],
        BK=cfg["BK"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def us(fn):
    for _ in range(10):
        fn()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1000


def main():
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda"))
    run, w1, w2 = make_moe(
        "iq2_xxs", MoEActivation.RELU2_NO_MUL, E=512, K=1024, inter=2048, top_k=9
    )
    del w1, w2
    base = moe.invoke_block_moe_kernel
    base_cfg = moe.moe_config
    configs = [(64,32,256,4),(64,16,256,4),(64,16,128,4),
               (64,32,128,4),(64,16,256,2)]
    for M in (16,32):
        for pattern in ('diverse','identical'):
            x=torch.randn(M,1024,dtype=torch.bfloat16);weights,ids=make_routing(M,512,9)
            if pattern=='identical':ids=ids[:1].expand(M,-1).contiguous()
            moe.invoke_block_moe_kernel=base
            # Disable the new in-tree fast path for the baseline arm.
            moe.moe_config=lambda *args:{**base_cfg(*args), 'SINGLETON_GEMV':False}
            ref=run(x,weights,ids).clone()
            old_us=us(lambda:run(x,weights,ids))
            for bm,bn,bk,w in configs:
                moe.invoke_block_moe_kernel=invoke
                cfg={'BLOCK_SIZE_M':bm,'BN':bn,'BK':bk,'num_warps':w,'num_stages':1}
                moe.moe_config=lambda *args:cfg
                new=run(x,weights,ids).clone()
                gap=((new-ref).abs().max()/ref.abs().max()).item()
                assert gap<1/128,(M,pattern,cfg,gap)
                value=statistics.median(us(lambda:run(x,weights,ids)) for _ in range(2))
                print(M,pattern,cfg,'old_us',round(old_us,2),'new_us',round(value,2),
                      'speedup',round(old_us/value,3),'scaled_gap',gap,flush=True)
    moe.moe_config=base_cfg
    moe.invoke_block_moe_kernel=base
if __name__=='__main__':main()
