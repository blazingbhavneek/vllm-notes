# llama.cpp at HEAD 6d5ac3c8d (2026-10-29) - IQ2_XXS reference lane

All paths relative to `research/clones/llama.cpp/`. Repo is a read-only reference; nothing was modified.

## Verdict

llama.cpp's M=1 IQ2_XXS decode on NVIDIA is the **same structural shape as our Triton GEMV** - per-thread
global-memory codebook gather + fp32 scalar accumulate per 32-value block - and it is **not** presented or
provable as being within 2x of the memory bandwidth floor at M=1; the repo's own tuning knobs (the
"low bit types load few weight bytes per thread, ... adds memory parallelism" comment at
`ggml/src/ggml-cuda/mmvq.cu:581`, the `iq_slow_*` type lists that disable optimizations for IQ2_*/IQ3_* at
`ggml/src/ggml-cuda/mmvq.cu:1150-1174`) are implicit admissions that i-quant GEMV sits well off the
bandwidth floor on NVIDIA. The two things that *are* genuinely better than our current decode are
(a) the **fully-integer dot**: the 8 magnitudes are used as packed int8 bytes accumulated with 8x
`__dp4a`, the per-32 scale is applied *after* the dot as an integer multiply, and one fp32 FMA per 32
values closes the accumulator (`ggml/src/ggml-cuda/vecdotq.cuh:1111-1141`); and (b) **batch reuse via
device-side int8 repack**: at batch >8 IQ2_XXS is decoded once per tile into int8 Q8_0-layout tiles in
shared memory and consumed by dp4a or int8 tensor cores (`mmq-load-tiles.cuh:1136-1218`), which is
confirmed proof-of-work that an int8/IMMA codebook path is exact-for-weights - but it only pays when a
tile's decoded rows are reused across columns (J>=9), which our M=1 top-9 MoE does not have per expert.
Copyable value: the dp4a integer-decode bundle (with `unpack_ksigns` popcount-parity signs, 4 ALU ops, no
sign-table load) is worth prototyping against our fp32 FMA decode; nothing else in llama.cpp breaks the
dependent load->gather->unpack chain differently than we do, and NVIDIA MMVQ deliberately keeps the
codebook in plain global memory (no smem staging, no __constant__) - consistent with our measurement that
the gather is free and the decode ALU is the wall.

## Answers to the lane brief

### 1. Decode-size (M=1) IQ2_XXS on CUDA/HIP

- **Kernel**: `mul_mat_vec_q` (MMVQ). Dispatch: `ggml_cuda_should_use_mmvq` returns true for
  `ne11 <= MMVQ_MAX_BATCH_SIZE = 8` (`ggml/src/ggml-cuda/mmvq.cuh:3`, `ggml/src/ggml-cuda/mmvq.cu:319-325`);
  `ggml_cuda_mul_mat` prefers MMVQ over MMQ below that (`ggml/src/ggml-cuda/ggml-cuda.cu:2061-2069`).
  IQ2_XXS dispatch at `mmvq.cu:1396-1398`; kernel at `mmvq.cu:620-820`.
- **Thread mapping**: `VDR_IQ2_XXS_Q8_1_MMVQ = 2` (`vecdotq.cuh:1108`), `QI2_XXS = 16`, `QR2_XXS = 4`
  (`ggml-common.h:142-143`, `common.cuh:1097-1102`). On generic NVIDIA (incl. H200/SM90) with
  `ncols_dst=1`: `nwarps = 4` (`mmvq.cu:455-467`), `rows_per_cuda_block = 1` (two-row blocks are granted
  only to Q1_0/Q2_K/Q3_K/IQ1_S/IQ1_M, `mmvq.cu:582-605`).
- **Per thread per vec_dot call** (`vec_dot_iq2_xxs_q8_1`, `vecdotq.cuh:1111-1141`): one 32-value sub-block
  = one 4-byte load of 4 codebook-row indices + one 4-byte `aux32` (4x 7-bit sign groups + parity-folded
  8th signs + 4-bit block scale) + **one 8-byte gather** `((const uint2*)iq2xxs_grid)[aux8[k]]` per codebook
  row (`vecdotq.cuh:1123`) + 8 `ggml_cuda_dp4a` (real `__dp4a` on CC>=61, `common.cuh:741-743`) against the
  8 int32 words of one Q8_1 activation block. So **32 weight values decoded per thread per call**, ILP = 8
  independent-ish dp4a + 4 table gathers per call - there is no "8 vs 16 blocks per accumulator" unroll;
  the fp32 accumulator `tmp[j][i]` receives **one fp32 add per 32 values** (`mmvq.cu:737, 780-786`).
- **Codebook storage**: `iq2xxs_grid` is `static const __device__ uint64_t[256]` = 2 KB in **plain global
  memory** (`ggml-common.h:493-494` CUDA/HIP macro, table at `ggml-common.h:560`), bytes are magnitudes
  {8,25,43} = 0x08/0x19/0x2b. No `__constant__`, no shared-memory staging on the NVIDIA fast path; the
  divergent 8 B row gathers hit L1/L2. (Metal is the one that uses `constant` memory,
  `ggml-common.h:486-487`; Vulkan/Metal GEMVs stage the table to shared/threadgroup memory.)
- Activations are pre-quantized to **Q8_1** (int8 + fp32 scale/32) on device before the GEMV
  (`mmvq.cu:1583`, quantize step upstream of `ggml_cuda_op_mul_mat_vec_q`).

### 2. Accumulation dtype for IQ2_XXS

- **M=1 MMVQ: int8 dp4a**, not fp32 FMA, not HFMA2, not tensor cores.
  `sumi = ggml_cuda_dp4a(grid0, u0, sumi)` x8 (`vecdotq.cuh:1125-1136`), then integer
  `ls = aux32 >> 27 | 1; sumi = sumi * ls / 8;` and a single fp32 `d * sumi` per 32 values
  (`vecdotq.cuh:1137-1140`). **Yes - magnitudes are used as integers with the per-32 scale applied after
  the dot product**, and this is the shipping M=1 path, not an experiment.
- **Batch >8 MMQ: int8 tensor cores (IMMA) or dp4a.** `ggml_cuda_should_use_mmq` explicitly lists
  IQ2_XXS (`mmq.cu:404-411`); per-tile loader converts to shared-memory int8 (`mmq.cuh:576-745`,
  `ggml_cuda_mmq_load_tiles_iq2_xxs`), vec_dot is `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a`
  (`mmq-vec-dot.cuh:110`) or `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` (`mmq-vec-dot.cuh:143`, selected at
  `mmq.cuh:828-832`); float `x_df[i*sram_stride + kqsx] = d * ls / 8` scale is applied after the integer
  tile dot (`mmq-load-tiles.cuh:1195-1216`).
- **Confirmed negative resolution**: the historical "MMQ excludes IQ2_*" is *gone*. It was lifted by
  `69c487f4e` "CUDA: MMQ code deduplication + iquant support (#8495)" (2024-07-20) - before that commit
  `ggml_cuda_should_use_mmq`'s switch had no IQ1/IQ2/IQ3 cases and they fell through to cuBLAS + fp32
  dequantize. At HEAD there is no comment excluding IQ types from integer MMA; IQ2_XXS runs *through*
  shared-memory int8 conversion into s8 MMA tiles. The only remaining exclusions are shape/SMEM gates
  (`mmq.cu:424-430`, 48 KiB smem floor).
- fp32/half dequant of IQ2_XXS survives only on the cuBLAS fallback path
  (`dequantize.cuh:270-290`, scalar `grid[j] * sign * d` per value).

### 3. Device-side repack/convert for the fast paths

- **CUDA MMQ**: `ggml_cuda_mmq_load_tiles_iq2_xxs` (`mmq-load-tiles.cuh:1136-1218`) converts
  IQ2_XXS -> **signed int8 magnitude bytes in 4-byte lanes** (via `__vcmpne4`/`__vsub4` xor-negate) written
  into shared memory in the MMQ Q8_0 tile layout (dp4a or MMA fragment layout), plus one **fp32 scale per
  32 values** (`d * ls / 8`). Cost: +128 bytes smem per row per 128-K tile (int8 x1 = same bytes as one
  Q8 row per 256 values + scale array); zero extra global bytes, ephemeral per kernel. What it eliminates:
  the sign/codebook decode is paid **once per loaded tile and reused across the whole J-column tile**, and
  the inner dot becomes pure dp4a/IMMA. No `to_k_q`, no `convert_unalign`, no persistent
  repacked-weight/"prepare" buffer exists for CUDA i-quants at HEAD (grep: none).
- **Metal**: not a layout repack but per-block staging of the LUT + `ksigns_iq2xs` into threadgroup memory
  (`kernels/mul_mv.metal:1913-1924`); tensor-core path dequantizes IQ2_XXS to **fp16 fragments**
  (`kernels/mul_mv_mma.metal:616`, `kernels/dequantize.h:510`).
- **Vulkan**: `init_iq_shmem` copies the 256-entry grid into shared memory per workgroup + barrier
  (`vulkan-shaders/types.glsl:1081-1090`, called from `dequant_iq2_xxs.comp`); no persistent repack.
- **CPU**: new `tiled` GEMM host-unpacks IQ2_XXS into int8 tile codes stored as **(value + 128)** with
  precomputed int32 `(2*ls+1)` scales (`ggml/src/ggml-cpu/tiled/tiled.cpp:283-313`) and uses an
  "act-bias" unsigned MAC since Wmax <= 62 (`tiled-kernel.cpp:544`) - sign handling deleted entirely from
  the inner loop.

### 4. CPU AVX2/AVX512/NEON contrast

- **AVX2** (`ggml/src/ggml-cpu/arch/x86/quants.c:2724-2778`): codebook row materialized by **4 scalar byte
  reads per 32 values** (`_mm256_set_epi64x(iq2xxs_grid[aux8[3]], ..., iq2xxs_grid[aux8[0]])`); no shuffle,
  no expanded table. Signs are applied to the **activations**, not the codebook: a 128-row precomputed
  `keven_signs_q2xs` +/-1 int8 table is looked up per 7-bit sign group and fed to `_mm256_sign_epi8(q8, s2)`
  (lines 2750-2757). Integer chain: `_mm256_maddubs_epi16(grid, q8_signed)` -> `_mm256_madd_epi16(dot,
  2*ls+1)` (per-32 scale folded into the pair-wise multiply) -> int32 accum -> **one fp32 FMADD per 256
  block**, final `0.125f` folded. Nothing is pre-multiplied per table row; the only precomputes are the
  128-entry even-parity sign table and the 0.125 factor.
- **No AVX512-specific IQ2_XXS** kernel exists at HEAD (only `__AVX2__`/`__AVX__`/generic branches in that
  file); no negative/signed 512-entry table for iq2 on x86 (the signed-512 idea is an iq2_xs/AVX512 thing
  that this tree does not have for xxS).
- **NEON** (`arch/arm/quants.c:3634-3687`): `vld1_s8(iq2xxs_grid + aux8[k])` 16-byte table loads (two rows
  at once), `vmulq_s8` with the signs64 +/-1 row, `vdotq_s32` (SDOT) vs q8; per-32 scale applied in fp32
  `(0.5f + (aux32 >> 28))` per sub-block, `0.25f` at the end.
- **Generic** (`quants.c:906-943`): scalar `grid[j] * q8[j] * (signs & kmask ? -1 : 1)` with `ksigns_iq2xs`
  byte table.
- **Honest Triton equivalents**: scalar table reads -> `tl.load` of `iq2xxs_grid[idx_vec]` (we already do
  this; their AVX2 is *less* fancy than our gather). `maddubs + madd(2ls+1)` -> int8/int16 `tl.dot` or
  int32 lane ops. The GPU-shaped trick worth stealing is CUDA's, not x86's: see Techniques.

### 5. Other backends - structural differences from CUDA

- **Metal** (`kernels/mul_mv.metal:1873-1975`): fp32 scalar FMA per value, `sum += yl[8l+j] * grid[j] *
  (signs & kmask ? -1 : 1)`; grid + 128-byte sign table staged to threadgroup memory per threadgroup;
  per-thread = one 32-value sub-block; `simd_sum` warp reduction; supports K-split blocks
  (`FC_mul_mv_split`). LUT lives in Metal `constant` memory when not staged (`ggml-common.h:486`). No
  integer dot at all. Batch/tensor-core path is fp16-dequant MMA (`mul_mv_mma.metal:616`,
  `mul_mm.metal:913`).
- **Vulkan** (`vulkan-shaders/dequant_iq2_xxs.comp`): each thread handles exactly one 32-value sub-block
  (4 idx, 4x7 signs, 4-bit scale), recomputes the parity sign with `bitCount` (same parity trick as CUDA's
  `unpack_ksigns`), output is f32/f16 dequant fused into the GEMV; grid is smem-staged with a workgroup
  barrier. `vulkan-shaders-gen.cpp:254-258` documents the LUT load granularity ("8" bytes for iq2_xxs).
  Pipelines use forced 16-lane subgroups (`ggml-vulkan.cpp:2950`). No integer accumulation, no
  persistent-repack/precompute buffers found (grep "repack" in ggml-vulkan.cpp: none).
- **SYCL** (`ggml/src/ggml-sycl/vecdotq.hpp:1533-1560`): the *worst* case - scalar `sumi += q8[j]*grid[j]*
  (signs & kmask ? -1 : 1)` with a `ksigns_iq2xs` lookup, no dp4a; VDR=1 (one 32-value sub-block per
  thread). Has a fused-dequant-to-int8 XMX GEMM gate for iq2_xxs for batch<=64
  (`fused-gemm.hpp:24, 54`) - Intel's answer mirrors CUDA MMQ's "convert to int8 near the cores".
- **WebGPU** (`wgsl-shaders/mul_mat_vec_acc.tmpl:1077-1132`): byte-extraction fp32 GEMV, per-value
  `select(gw_hi, gw_lo, ...)`, no LUT staging.
- **No comment or PR reference anywhere in the tree claims Metal/Vulkan/SYCL is faster than CUDA for
  i-quants.** The only speed-ordering documentation in the tree is NVIDIA/AMD-internal: per-arch, per-type
  MMVQ<->MMQ crossover tables (`mmvq.cu:147-296`, tuned per PR #26079/#28912/#28285, with an explicit
  "k-quants cost more to decode ... so MMQ wins sooner" comment at `mmvq.cu:325-326`) and the
  `iq_slow_turing`/`iq_slow_other` lists (`mmvq.cu:1150-1174`) where small_k was found to *regress* IQ2_*/
  IQ3_* GEMV on NVIDIA hardware.

### 6. History (ggml-cuda + ggml-cpu, i-quant GPU speed)

- `dd5ae0640` 2024-01-08 #4773 "SOTA 2-bit quants" - IQ2_XXS lands (with byte-table signs).
- `8e14e3ddb` 2024-01-?? #5187 "Faster AVX2 dot product for IQ2_XS" - sign-into-activations via
  `_mm_sign_epi8` established on CPU.
- `cb5fad4c6` 2024-07-01 #8215 "CUDA: refactor and optimize IQ MMVQ" - dp4a unified via
  `ggml_cuda_dp4a`, VDR re-tuned (IQ2_XXS MMVQ VDR -> 2), 2-byte/4-byte block-load helpers added.
- `69c487f4e` 2024-07-20 #8495 "CUDA: MMQ code deduplication + **iquant support**" - IQ1/IQ2/IQ3/IQ4
  admitted to MMQ by converting to **q8_0-layout tiles in shared memory** (`load_tiles_iq2_xxs`,
  `MMQ_DP4A_TXS_Q8_0` / `MMQ_MMA_TILE_X_K_Q8_0`). This is where the integer tensor-core exclusion for
  i-quants was removed.
- `27b93cbd1` 2026-02-15 #19624 "optimize iq2xxs/iq2xs/iq3xxs dequantization" - three keeper micro-opts:
  (1) one 8-byte grid-row load instead of two 4-byte, (2) `unpack_ksigns`: derive the 8th sign from
  `popc(v)&1` and broadcast `s * 0x01010101` instead of loading `ksigns_iq2xs`/`ksigns64` (kills one
  dependent table load per 4 bytes), (3) scale rewritten `sum*(2s+1)/8` - commit message: "saves 3
  registers for mul_mat_vec_q (152 -> 149) according to nsight".
- `2b5621094` 2026-08-20 #26079 - per-HW/per-type MMVQ<->MMQ decode crossover switch points (i-quants keep
  MMVQ to batch 8 on NVIDIA P100/Volta/Ada/Blackwell; Turing-Ampere MoE drops to 7 for IQ2_S etc.).
- `f46bc30cb` 2026-09-23 #27962 "HIP: optimize IQ2/IQ3 (`__vsub4` `__vcmpne4`) using SWAR" - AMD ALU-only
  sign decode.
- `2744e48d1` 2026-10-17 #30517 "CUDA: faster MMVQ for low-bit types" - two weight rows per block for
  Q1_0/Q2_K/Q3_K/IQ1_* to "add memory parallelism" (IQ2_XXS not promoted on generic NVIDIA; IQ2_S/IQ3_S
  promoted only on GB10), L2 prefetch distance tuning (DGX Spark only).
- CPU side kept evolving separately: `f470bc36b` #13892 arch split; RVV vec-dots (#18784/#18859/#20633);
  `d834d44e6` 2026-09-26 #27851 new `tiled` mul_mat with the iq2_xxs `(value+128)` act-bias unpack
  (`tiled.cpp:283-313`), `3dbcb7fca` #30296 accumulator reduction.
- What stuck on GPU: dp4a integer dots for every i-quant MMVQ, the parity-popcount sign unpack,
  shared-memory int8 conversion for batched MMQ, per-arch crossovers. What didn't: small_k on i-quants
  (explicitly disabled as "slow"), fp32/half codebook accumulation (gone from all fast paths),
  constant-memory or smem codebooks on NVIDIA (never adopted).

### 7. Verdict for us

No - nothing in llama.cpp at HEAD gets M=1 IQ2_XXS within ~2x of the memory floor on NVIDIA, and its own
code explains why: at M=1 the kernel is a warp-reduction GEMV where every thread re-does the
gathers+dp4a decode for its 32-value slice with **8 weight bytes + 32 activation bytes per thread per
iteration and one fp32 add per 32 values** (`mmvq.cu:743-786`, `vecdotq.cuh:1111-1141`), with
acknowledged poor memory parallelism at low bits (`mmvq.cu:581-583`) and acknowledged slow decode on some
archs (`mmvq.cu:1150-1174`). Their escape routes require column reuse (MMQ smem int8 conversion at J>=9)
or bigger output tiles - neither exists at M=1 per expert. That is a citable, structural agreement with
our measurement: **inside the codebook format, at M=1, everyone is decode-latency-bound, and only buying
bytes (our plane rep) or int-decoding with scale-after-dot (their dp4a bundle) changes the constant.**

## Techniques found

- **Integer (dp4a) codebook dot with post-dot scale** - the shipping CUDA M=1 path: decode 4 grid rows to
  4x4 int8 bytes with xor-negate, 8x `__dp4a` vs Q8_1 activation bytes, `sumi*ls/8` integer, one fp32 FMA
  per 32 values. `vecdotq.cuh:1111-1141`; scale-after-dot also at `mmq-load-tiles.cuh:1195`. Cost:
  activations must be int8+scale (Q8_1, per-32 fp32 d) - weights stay exact, activations are requantized
  (llama.cpp accepts this at decode time on every platform). **Maps to Triton as**: one extra captured
  kernel quantizes x to int8+scales per 32; in `_decode_tile`, keep grid rows as int8, apply sign with
  `^`/`-` (or int8 `tl.where`), accumulate via `tl.inline_asm_elementwise("dp4a.b.s32.s32.s32 $0,$1,$2,$3")`
  over int32-packed lanes (8 dp4a per 32 values), then `acc += (d_x*d_y) * (sumi * ls)` once. Replaces our
  fp32 unpack+FMA chain (~64 ops/32vals) with ~8 dp4a + ~20 int ops.
- **Popcount-parity sign broadcast (`unpack_ksigns`)** - derive all 8 sign masks for a codebook row from
  the 7-bit aux byte with `p = popc(v)&1; s = (v ^ (p<<7)) * 0x01010101;` then `__vcmpne4(s & 0x08040201)`
  style selectors; **no 128-entry sign table load, no per-value shift/mask chain** (`vecdotq.cuh:97-104`;
  commit `27b93cbd1` cites -3 regs). **Maps to Triton as**: after loading the aux 4-byte word as uint32,
  compute parity by xor-fold (`v ^ (v>>1) ^ (v>>2) ...`) & 1, build mask word with `* 0x01010101`, and do
  sign via `grid ^ mask` + `- mask` on packed int8 lanes / per-lane `& 0x01020408` compares. One load
  feeds all 8 sign decisions of the row.
- **Device-side IQ2_XXS -> int8 shared-memory tile conversion + IMMA for batched GEMM** -
  `ggml_cuda_mmq_load_tiles_iq2_xxs` (`mmq-load-tiles.cuh:1136-1218`) + `q8_0_q8_1_mma`
  (`mmq.cuh:828-832`): exact int8 weight codes, float per-32 scales kept separately, decode paid once per
  tile across the J tile. Only pays at ncols>=9 (our dense layers at M>8; not our M=1 MoE). **Maps to
  Triton as**: for the batched `_gemm_kernel`, materialize `wq8 [BN, BK] int8` + `wd [BN, BK/32] fp32` in
  the K-loop prologue from the packed format, then `tl.dot(wq8, xq8, out_dtype=tl.int32)` per 128-K and
  scale after; byte cost = 0 extra global bytes, smem only.
- **One 8-byte gather per codebook row from plain `__device__` global table (2 KB, no constant/smem) on
  NVIDIA** - `vecdotq.cuh:1123`, `ggml-common.h:493/560`; corroborates our "gather is free" probe: they
  never needed `__constant__` or smem staging on CUDA even after 2.5 years of tuning (Metal/Vulkan stage
  it, NVIDIA deliberately doesn't). **Maps to Triton as**: nothing to change; confirms our current
  `tl.load(iq2_lut + idx)` is the right structure, and that a `tl.reshape`/broadcast variant is a dead end
  (their only layout change was row-load width 2x4B -> 1x8B in #19624).
- **Host/producer-side int8 tile unpack with bias-128 codes (CPU `tiled` GEMM)** -
  codes stored as (value+128), scales pre-multiplied as int32, sign branch deleted from the inner loop via
  activation-bias (`tiled.cpp:283-313`, `tiled-kernel.cpp:544-549`). **Maps to Triton as**: if we ever
  build a persistent repack (the +70% bytes option), the *right* repack is biased int8 per value
  (value+128, 1 byte/value = 4.86x bytes - note cost) with (2*ls+1) pre-folded scales so the kernel is
  pure u8*s8 madd; anything less loses to the LUT path.
- **MoE-adjacent plumbing worth noting (not decode)**: llama.cpp fuses gate+up+GLU and the shared expert
  into the *same* MMVQ/MMQ launch (`mmvq.cu:652-719`, `ggml-cuda.cu:1775-1860`) and has a dedicated
  multi-token MUL_MAT_ID MMVQ kernel (`mmvq.cu:1200-1210`, `mul_mat_vec_q_moe_launch`) - i.e. their answer
  to a Nemotron-shaped MoE decode is fewer launches + shared activation loads, not a faster iq2 decode.

## Dead ends confirmed

- **Constant-memory / shared-memory codebook on NVIDIA for the M=1 GEMV**: never adopted in 2.5 years of
  tuning; the table is still `static const __device__` global at HEAD (`ggml-common.h:493`), and the only
  NVIDIA-side LUT change was the 8-byte load width (`27b93cbd1`). Matches our probe "gather is free".
- **fp32 (or half2) accumulation of decoded codebook values on GPU**: exists only in the cuBLAS-fallback
  dequantize kernel (`dequantize.cuh:270-290`) and in Metal/SYCL/WebGPU GEMVs - the NVIDIA fast paths
  moved fully to int8 dp4a (#8215) precisely because the fp decode was the bottleneck. If fp32-FMA decode
  were fine, CUDA wouldn't have abandoned it.
- **MMQ/int8-tensor-core at tiny M**: real but gated to `ne11 > 8` (`mmvq.cuh:3`, `mmq.cu:382-430`);
  tile heights and the smem-conversion cost do not amortize at M=1 - do not attempt an IMMA path for the
  M=1 MoE GEMV.
- **`small_k` (rows-per-block = nwarps) for IQ2_*/IQ3_* on NVIDIA**: explicitly disabled as a regression
  (`iq_slow_turing`, `iq_slow_other` lists, `mmvq.cu:1150-1174`) - a same-shaped "more rows per CTA" knob
  for our Triton GEMV is documented-negative on NVIDIA hardware for these types.
- **Per-value scale folding into the codebook**: every backend (CUDA dp4a, AVX2 madd, NEON sdot, Vulkan,
  MMQ) keeps the per-32 scale *outside* the dot and applies it once per 32 values - independently arriving
  at the same conclusion as our measurement that folding the scale into the inner product is slower.

## Sources

- Repo: llama.cpp clone at `research/clones/llama.cpp`, HEAD `6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3`
  (2026-10-29).
- CUDA: `ggml/src/ggml-cuda/mmvq.cu:3, 56, 147-296, 319-326, 455-467, 581-605, 620-820, 1135-1215, 1396`;
  `ggml/src/ggml-cuda/vecdotq.cuh:97-104, 1108-1141`; `ggml/src/ggml-cuda/mmq.cu:48-70, 382-430`;
  `ggml/src/ggml-cuda/mmq.cuh:89, 409, 576-745, 828-837`;
  `ggml/src/ggml-cuda/mmq-load-tiles.cuh:1136-1218`; `ggml/src/ggml-cuda/mmq-vec-dot.cuh:110, 143`;
  `ggml/src/ggml-cuda/common.cuh:712-748, 1096-1102`; `ggml/src/ggml-cuda/dequantize.cuh:270-290`;
  `ggml/src/ggml-cuda/ggml-cuda.cu:1775-1860, 2008-2076`; `ggml/src/ggml-common.h:142-143, 470-505, 560`.
- CPU: `ggml/src/ggml-cpu/arch/x86/quants.c:2724-2838`; `ggml/src/ggml-cpu/arch/arm/quants.c:3634-3692`;
  `ggml/src/ggml-cpu/quants.c:906-943`; `ggml/src/ggml-cpu/tiled/tiled.cpp:283-313`;
  `ggml/src/ggml-cpu/tiled/tiled-kernel.cpp:536-560`.
- Metal: `ggml/src/ggml-metal/kernels/mul_mv.metal:1873-1975`; `kernels/dequantize.h:510`;
  `kernels/mul_mv_mma.metal:616`; `kernels/mul_mm.metal:913`.
- Vulkan: `ggml/src/ggml-vulkan/vulkan-shaders/dequant_iq2_xxs.comp`;
  `vulkan-shaders/types.glsl:1081-1090`; `vulkan-shaders/vulkan-shaders-gen.cpp:249-258`;
  `ggml-vulkan.cpp:2950, 3104`.
- SYCL: `ggml/src/ggml-sycl/vecdotq.hpp:1516-1560`; `ggml/src/ggml-sycl/fused-gemm.hpp:9-60`.
- WebGPU: `ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl:1077-1132`.
- Commits: `dd5ae0640` (#4773, 2024-01-08), `8e14e3ddb` (#5187), `cb5fad4c6` (#8215, 2024-07-01),
  `69c487f4e` (#8495, 2024-07-20), `27b93cbd1` (#19624, 2026-02-15), `2b5621094` (#26079, 2026-08-20),
  `f46bc30cb` (#27962, 2026-09-23), `2744e48d1` (#30517, 2026-10-17), `d834d44e6` (#27851, 2026-09-26).
  In-code PR ref: `mmvq.cu:147-148` cites discussion
  `github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627` for MoE MMVQ/MMQ crossovers.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Report written to task-3.md with file:line evidence for all 7 lane questions, verdict, top transferable techniques, and confirmed dead ends; residual risks listed."
    }
  ],
  "changedFiles": [
    "/home/seigyo/.pi/agent/sessions/--home-seigyo-vllmServer--/subagent-artifacts/outputs/b414d1d3-6cf4-4e07-b3ed-37a37f2fc21d/task-3.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git log/show -S/-i greps on ggml-cuda and ggml-cpu, scoped grep/read of vecdotq.cuh, mmvq.cu, mmq*.cuh, CPU arch quants, Metal/Vulkan/SYCL/WebGPU shaders",
      "result": "passed",
      "summary": "Located M=1 dp4a decode, MMQ int8 smem conversion, table storage, CPU contrast, cross-backend structure, and 9 dated commits"
    }
  ],
  "validationOutput": [
    "HEAD confirmed 6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3 2026-10-29",
    "vecdotq.cuh:1111-1141 dp4a + scale-after-dot confirmed in source; mmq.cu:404-411 IQ2_XXS in MMQ confirmed; git show 69c487f4e confirms pre-2024 exclusion lifted"
  ],
  "residualRisks": [
    "No in-repo benchmark numbers for IQ2_XXS decode throughput on NVIDIA; the 'not within 2x of floor' verdict is structural (code shape + tuning comments) plus our own prior measurements, not a llama.cpp number",
    "dp4a technique transfer assumes we accept Q8_1 int8 activation requantization, which changes activation numerics (weights stay bit-exact); Triton int8 tl.dot at M=1 and dp4a inline-asm lowering were not prototyped",
    "This tree is dated 2026-10-29 and diverges from upstream as of my knowledge (MMQ iquant MMA paths, tiled CPU GEMM); external PR discussion links were not fetched",
    "Did not verify whether ggml_cuda_kernel launch of the Q8_1 activation quantize adds a separate captured kernel cost in their decode (it does per code layout, but was not timed)"
  ],
  "noStagedFiles": true,
  "diffSummary": "One report artifact written outside the repos; no repository files modified",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "Verdict: llama.cpp at HEAD does NOT get M=1 IQ2_XXS within ~2x of the memory floor on NVIDIA; at M=1 it runs the same gather+decode GEMV shape we do (4 warps/row, 32 values/thread, global 2KB LUT), and its own tuning (memory-parallelism comment mmvq.cu:581, iq_slow_* regression lists mmvq.cu:1150-1174) documents why. Top 3 transferable: (1) fully-integer decode - xor-negate grid magnitudes into packed int8 and accumulate with 8x dp4a vs Q8_1 activations, per-32 scale applied after the dot, one fp32 FMA per 32 values (vecdotq.cuh:1111-1141; Triton via tl.inline_asm_elementwise dp4a); (2) popcount-parity sign broadcast replacing the 128-entry sign table with 4 int ops per 4-byte aux word (unpack_ksigns, vecdotq.cuh:97-104, commit 27b93cbd1 -3 regs); (3) batch-only IQ2_XXS->int8 shared-memory tile conversion + IMMA with post-dot float scales (mmq-load-tiles.cuh:1136-1218, mmq.cuh:828-832) - the confirmed-proof pattern if we ever batch M>8 dense layers, dead at M=1 per expert. Strongest confirmed dead end: int8 tensor cores / MMQ-style repack for M=1 (gated to ncols>8, tile height cannot amortize), plus NVIDIA deliberately never adopted constant/smem codebooks for the GEMV, matching our 'gather is free' probe."
}
```
