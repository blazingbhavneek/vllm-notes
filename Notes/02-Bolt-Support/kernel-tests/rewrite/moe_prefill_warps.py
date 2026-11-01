"""A/B large-prefill warp counts; original IQ2 BM128 spills with four warps."""
import sys,statistics
sys.path.insert(0,'.')
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_moe,make_routing
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import modelopt_block_triton_moe as moe
from vllm.v1.worker.workspace import init_workspace_manager

torch.set_default_device('cuda');torch.manual_seed(0)
init_workspace_manager(torch.device('cuda'))
run,w1,w2=make_moe('iq2_xxs',MoEActivation.RELU2_NO_MUL,E=512,K=1024,inter=2048,top_k=9)
del w1,w2
M=4096
x=torch.randn(M,1024,dtype=torch.bfloat16);weights,ids=make_routing(M,512,9)
base=moe.moe_config
ref=None
for warps in (4,8,16):
    moe.moe_config=lambda *args:{**base(*args),'num_warps':warps}
    out=run(x,weights,ids).clone()
    if ref is None:ref=out
    gap=((out-ref).abs().max()/ref.abs().max()).item()
    assert gap<1/128,gap
    fn=lambda:run(x,weights,ids)
    for _ in range(10):fn()
    times=[statistics.median(bench_gpu_time_with_cupti(fn,use_cuda_graph=True))*1000 for _ in range(3)]
    print('M',M,'warps',warps,'median_us',round(statistics.median(times),2),'scaled_gap',gap,flush=True)
moe.moe_config=base
