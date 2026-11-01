"""Does Triton emit tensor-core MMA for tl.dot on sm_75 (fp16)?"""
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def k(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    r = tl.arange(0, BM)[:, None] * BK + tl.arange(0, BK)[None, :]
    s = tl.arange(0, BN)[:, None] * BK + tl.arange(0, BK)[None, :]
    c = tl.dot(tl.load(a_ptr + r), tl.trans(tl.load(b_ptr + s)))
    tl.store(c_ptr + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], c)


a = torch.zeros(64, 64, dtype=torch.float16, device="cuda")
c = torch.zeros(64, 64, dtype=torch.float32, device="cuda")
ck = k.warmup(a, a, c, BM=64, BN=64, BK=64, grid=(1,))
for cc in (75, 80):
    ptx = triton.compile(ck.src, target=GPUTarget("cuda", cc, 32),
                         options={"num_warps": 4}).asm["ptx"].splitlines()
    mma = [l.strip() for l in ptx if "mma.sync" in l]
    print(f"sm{cc}: {len(mma)} mma.sync ({mma[:1]}), "
          f"{sum('fma.rn.f32' in l for l in ptx)} fma.rn.f32, "
          f"maxnreg={[l.strip() for l in ptx if 'maxnreg' in l]}")
