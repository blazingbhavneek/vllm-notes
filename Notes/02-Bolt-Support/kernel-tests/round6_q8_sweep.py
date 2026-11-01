"""Q8 single-row reduction launch ablation; production wrapper timing."""
import sys,statistics,json
sys.path.insert(0,'.')
import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from tests.kernels.quantization.test_modelopt_block_triton import make_raw,assert_close_to_cublas,ref_decode
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t

torch.set_default_device('cuda');torch.manual_seed(0)
p=torch.cuda.get_device_properties(0);results=[]
for N,K in [(256,4096),(4096,4096),(1024,4096),(4096,1024),(131072,4096)]:
 raw=make_raw((N,),K,'q8_0',0);qw,s=t.repack_blocks(raw,'q8_0');x=torch.randn(1,K,dtype=torch.bfloat16)
 ref=ref_decode(raw,'q8_0');base=t._linear_config(1,N,K,'q8_0',p.multi_processor_count,p.shared_memory_per_block_optin)
 configs=[base]+[dict(BM=1,BN=bn,BK=bk,SPLIT=split,num_warps=w,num_stages=1) for bn,w in [(4,4),(8,4),(16,4),(32,4),(8,8)] for bk in sorted(set([K,1024,512])) if bk<=K for split in [1]]
 rows=[]
 for cfg in configs:
  try:
   fn=lambda:t.launch_linear(x,qw,s,None,'q8_0',cfg)
   y=fn();assert_close_to_cublas(y,x,ref)
   us=statistics.median(bench_gpu_time_with_cupti(fn,use_cuda_graph=True))*1000
   rows.append({'N':N,'K':K,'config':cfg,'us':us});print(json.dumps(rows[-1]),flush=True)
  except Exception as e:print('FAILED',N,K,cfg,str(e)[-200:],flush=True)
 results.extend(rows)
 print('BEST',N,K,min(rows,key=lambda r:r['us']),flush=True)
open('/tmp/bolt-round6-q8-sweep.json','w').write(json.dumps(results,indent=2))
