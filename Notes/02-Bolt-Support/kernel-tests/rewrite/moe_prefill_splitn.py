import sys,statistics
sys.path.insert(0,'.')
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_moe,make_routing
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import modelopt_block_triton_moe as moe
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import _CODEC_ID
from vllm.triton_utils import triton,tl
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import _decode_tile
@triton.jit(do_not_specialize=["EM", "num_valid_tokens"])
def splitn_kernel(
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
    offs_n = (pid % num_n) * BN + tl.arange(0, BN // 2)
    n_mask = offs_n < N
    offs_n2 = offs_n + BN // 2
    n_mask2 = offs_n2 < N
    a_rows = a_ptr + (offs_token // top_k)[:, None] * stride_am
    qw_ptr += e * stride_qe
    sc_ptr += e * stride_se
    acc0 = tl.zeros((BM, BN // 2), dtype=tl.float32)
    acc1 = tl.zeros((BM, BN // 2), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(
            a_rows + (k0 + tl.arange(0, BK))[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        w = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n, n_mask, k0, stride_qn, stride_sn,
            CODEC, False, BN // 2, BK,
        )  # fmt: skip
        acc0 += tl.dot(a, tl.trans(w.to(tl.bfloat16)))
        w2 = _decode_tile(
            qw_ptr, sc_ptr, lut_ptr, offs_n2, n_mask2, k0, stride_qn, stride_sn,
            CODEC, False, BN // 2, BK,
        )
        acc1 += tl.dot(a, tl.trans(w2.to(tl.bfloat16)))
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc0 *= weight.to(tl.float32)[:, None]
        acc1 *= weight.to(tl.float32)[:, None]
    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :],
        acc0.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )



    tl.store(
        c_ptr + offs_token[:, None] * stride_cm + offs_n2[None, :],
        acc1.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask2[None, :],
    )
def invoke(a, q, s, lut, c, tw, ids, eids, ntp, mul, top_k, cfg, codec):
    bm = cfg["BLOCK_SIZE_M"]
    em = ids.size(0)
    if a.size(0) < bm:
        em = min(em, a.size(0) * top_k * bm)
    n = q.size(1)
    splitn_kernel[(triton.cdiv(em, bm) * triton.cdiv(n, cfg["BN"]),)](
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



torch.set_default_device('cuda');torch.manual_seed(0)
init_workspace_manager(torch.device('cuda'))
run,w1,w2=make_moe('iq2_xxs',MoEActivation.RELU2_NO_MUL,E=512,K=1024,inter=2048,top_k=9)
del w1,w2
x=torch.randn(4096,1024,dtype=torch.bfloat16);weights,ids=make_routing(4096,512,9)
base=moe.invoke_block_moe_kernel
config=moe.moe_config
moe.moe_config=lambda *args:{**config(*args),'num_warps':4}
ref=None
for name,launch in [('original',base),('split_n',invoke)]:
    moe.invoke_block_moe_kernel=launch
    out=run(x,weights,ids).clone()
    if ref is None:ref=out
    gap=((out-ref).abs().max()/ref.abs().max()).item()
    assert gap<1/256,gap
    fn=lambda:run(x,weights,ids)
    for _ in range(10):fn()
    times=[statistics.median(bench_gpu_time_with_cupti(fn,use_cuda_graph=True))*1000 for _ in range(3)]
    print(name,'median_us',round(statistics.median(times),2),'scaled_gap',gap,flush=True)
moe.invoke_block_moe_kernel=base
moe.moe_config=config
