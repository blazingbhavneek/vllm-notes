"""CUDA driver resource/occupancy queries of actual SM90 Triton cubins.
These are theoretical residency limits, not achieved occupancy counters.
"""
import ctypes as c,json,pathlib,collections,re
cuda=c.CDLL('libcuda.so.1')
def call(name,*args):
 r=getattr(cuda,name)(*args)
 if r:raise RuntimeError((name,r))
call('cuInit',0);dev=c.c_int();call('cuDeviceGet',c.byref(dev),0)
ctx=c.c_void_p();call('cuDevicePrimaryCtxRetain',c.byref(ctx),dev);call('cuCtxSetCurrent',ctx)
rows=[];seen=set()
for name in ['_gemv_kernel','_moe_gemv_kernel','_moe_gemm_kernel','_splitk_reduce_kernel']:
 count=0
 for p in pathlib.Path('/home/seigyo/.triton/cache').glob('*/'+name+'.json'):
  d=json.loads(p.read_text())
  if d.get('arch')!='sm90' or d.get('num_ctas',1)!=1:continue
  cub=p.with_suffix('.cubin')
  if not cub.exists():continue
  mod=c.c_void_p();fn=c.c_void_p()
  try:
   call('cuModuleLoad',c.byref(mod),str(cub).encode());call('cuModuleGetFunction',c.byref(fn),mod,name.encode())
   attrs={}
   for k,i in [('max_threads',0),('static_shared',1),('local_bytes',3),('registers',4)]:
    v=c.c_int();call('cuFuncGetAttribute',c.byref(v),i,fn);attrs[k]=v.value
   shared=d['shared'];threads=d['num_warps']*32
   key=(name,threads,shared,attrs['registers'],attrs['local_bytes'])
   if key in seen:continue
   seen.add(key)
   if shared:call('cuFuncSetAttribute',fn,8,shared)
   blocks=c.c_int();call('cuOccupancyMaxActiveBlocksPerMultiprocessor',c.byref(blocks),fn,threads,c.c_size_t(shared))
   row=dict(kernel=name,cubin=str(cub),threads=threads,dynamic_shared=shared,active_blocks_per_sm=blocks.value,theoretical_warp_occupancy=blocks.value*threads/2048,**attrs)
   rows.append(row);print(json.dumps(row),flush=True);count+=1
   if count>=32:break
  finally:
   if mod.value:call('cuModuleUnload',mod)
pathlib.Path('/home/seigyo/vllmServer/vllm-notes/Notes/02-Bolt-Support/kernel-tests/rewrite/results/round6_serving_profile/static_occupancy.json').write_text(json.dumps(rows,indent=2))
call('cuDevicePrimaryCtxRelease',dev)
