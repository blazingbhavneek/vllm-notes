"""Experimental Hopper multi-CTA GEMV: no production dispatch changes."""
import json,sys,statistics,time
sys.path.insert(0,'.')
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_raw,assert_close_to_cublas,ref_decode
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t
from vllm.triton_utils import triton

torch.set_default_device('cuda');torch.manual_seed(0)
props=torch.cuda.get_device_properties(0)
for codec,N,K in [('iq2_xxs',5376,4096),('q8_0',1024,4096)]:
 raw=make_raw((N,),K,codec,0);qw,s=t.repack_blocks(raw,codec);lut=t.iq2_lut(torch.device('cuda')) if codec=='iq2_xxs' else None
 x=torch.randn(1,K,dtype=torch.bfloat16);cfg=t._linear_config(1,N,K,codec,props.multi_processor_count,props.shared_memory_per_block_optin)
 baseline=lambda:t.launch_linear(x,qw,s,lut,codec,cfg)
 ref=baseline();print('BASE',codec,float(statistics.median(bench_gpu_time_with_cupti(baseline,use_cuda_graph=True))*1000),flush=True)
 for ctas in (1,2,4,8):
  for bn,warps in ((8,2),(8,4),(4,4)):
   out=torch.empty((1,N),dtype=torch.bfloat16)
   def run():
    return t._gemv_kernel[(triton.cdiv(N,bn),1)](x,qw,s,lut,out,1,N,K,K,qw.stride(0),s.stride(0),CODEC=t._CODEC_ID[codec],BM=1,BN=bn,BK=K,num_warps=warps,num_stages=1,num_ctas=ctas)
   try:
    ck=run();torch.cuda.synchronize()
    assert_close_to_cublas(out,x,ref_decode(raw,codec))
    error=float((out-ref).abs().max()/ref.abs().max())
    value=statistics.median(bench_gpu_time_with_cupti(run,use_cuda_graph=True))*1000
    print('CLUSTER',codec,'ctas',ctas,'bn',bn,'warps',warps,'us',value,'regs',ck.n_regs,'spills',ck.n_spills,'smem',ck.metadata.shared,'error',error,flush=True)
   except Exception as exc:
    print('UNAVAILABLE',codec,ctas,bn,warps,type(exc).__name__,str(exc)[-350:],flush=True)
