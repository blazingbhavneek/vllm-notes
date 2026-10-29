# TensorFold / EXL3 codebook lane

Repo: `clones/TensorFold`, HEAD `7c068895`. Read-only pass; nothing in the checkout was
modified, no builds, no GPU work.

## Verdict

There is something to copy, and it is not the codebook. TensorFold ships **two** IQ2_XXS-
class decode kernels for **the same model we ship** (Nemotron-3.5-Bolt-74B-A9B-24GB, experts
in GGML IQ2_XXS, projections in Q8_0 — `CHANGELOG.md:604`, commit `3915bcea`), and the one
that matters (`zig/kernels/cuda/ggml.cu`) kills the `unpack → sign → reshape` tail in two
moves that cost **zero weight bytes**: (1) a load-time **byte permutation** of the 66-byte
blocks into 16-column tiles, proven to be a pure permutation by its own unit test
(`zig/src/cuda/ggml.zig:204-224`, "a byte permutation, the same size" in `experts.zig:24`),
so each lane's 16-byte load lands *already in MMA A-fragment position*; and (2) decode
straight into **packed bf16x2** with `prmt` + a bias subtract + one integer multiply for the
signs, ~2.5 SASS ops/value and one `LDS.64` per 8 values, with no sign table at all
(`ggml.cu:30-38`). The per-32 scale is then applied **once per 32-value group on the fp32
accumulator**, not per value and not folded into the weights (`ggml.cu:145-166`), which is
why the inner loop is multiply-free without paying our measured 2-3x penalty. The
EXL3 kernels prove the same shape of answer from the other side: at 2 bpw they run at
**221 GB/s of a 237 GB/s streaming ceiling** and are within 5% from 1 to 16 rows
(`docs/recipes/exl3.md:141-150`), i.e. near the memory floor, with 154-168 registers a
thread and ~8-25% occupancy (`docs/receipts/glm-kv-pages-0b094b74/gate5/limits-nvcc13.3.log:680-691`)
— they buy bandwidth with **per-thread prefetch depth, not with warps**. What does *not*
transfer is EXL3's headline trick: EXL3 has no codebook table at all, its "codebook" is a
closed-form arithmetic hash of the 16-bit state, so there is nothing to look up; our 256×8
grid of {8,25,43} is real data and we are stuck with the gather (which we already measured
as free). Expect the packed-bf16 decode + pre-permuted layout to be worth a large slice of
the 12.5 us gap between our 18.8 us GEMV and the 6.3 us floor, at the cost of ~2 KB shared
per CTA and an accumulation-order change (last-ulp, not an accuracy change).

## Answers to the lane's seven questions

**1. What is resident for decode, and is anything pre-expanded at load time?**
Two different answers, and both pre-expand *nothing* into global memory.

- *EXL3* keeps no codebook in memory at all. The codebook is a closed-form function of a
  16-bit stream state (`exl3_decode.cuh:86-108`, `exl3_candidate.cuh:23-36`): `mul1` =
  `h = 1024 + byte_sum(s * 0x83DCD12D)` then `value = h * fp16(0x1EEE) + fp16(0xC931)`, one
  rounding; `3inst`/`mcg` = `x = (s * A + B) mod 2^32; x = (x & 0x8FFF8FFF) ^ 0x3B603B60;
  value = fp16(lo16) + fp16(hi16)`. So codebook bytes = 0, bits/value = the trellis width
  only (2..8, half-steps allowed), resolved **per value at matmul time, always**. At load
  time they permute the *read order* only: "`load` reads one group with safetensors and
  copies the trellis, **never changing it**, into the read order the kernel wants
  (`layout="strips"`)" — `docs/recipes/exl3.md:95-96`. Cost: 0 extra bytes.
- *IQ2_XXS* (`ggml.cu`) keeps the ggml grid, `__device__ const uint64_t tf_iq2xxs_grid[256]`
  = 2 KB of magnitudes, each byte `128 + m`, `m ∈ {8,25,43}`
  (`ggml_iq2xxs_tables.cuh:17`), copied once per CTA into **2 KB of shared** as `uint2`
  (`ggml.cu:111-124`). Signs get **no table**: ggml's `tf_ksigns_iq2xs[128]` is shipped but
  unread, because signs are arithmetic (`ggml.cu:29-30, 111`). Resident weight bytes are
  the checkpoint's 66 bytes/256 values, permuted: a tile is 16 columns × 66 bytes = 1056
  bytes (`experts.zig:24`, `ggml.zig:26-29`). The repack is asserted to be a permutation:
  `test "the repack is a permutation of each matrix's bytes"` (`ggml.zig:204-224`).
- The one place they *do* materialize fp16 weights is **prefill**, not decode: "W_q is
  decoded once a chunk into fp16, a fixed-tile fp16 GEMM with fp32 accumulation multiplies
  it" (`docs/recipes/exl3.md:120-126`). Decode never expands. That is their implicit
  judgement on our "+70% bytes" question: they buy the fp16 expansion only when 16+ rows
  amortize it, and they measure that amortization ("within 5% from 1 row to 16, because one
  program's weights are decoded once and reused by all its rows", `exl3.md:141`).

**2. M=1 decode kernel structure.**
Weights are the **A** operand and rows are the **B** operand, deliberately inverted
(`ggml.cu:1`: "weights as A and rows as B"), so the mma's `n`=8 slots carry up to 8
*rows/members* and M=1 simply wastes 7 of them — harmless because the kernel is bandwidth-
and issue-bound, not compute-bound. One warp owns a **16-output-column × K** tile; at M=1 a
lane resolves **one grid entry = 8 magnitudes** per column-half per 32-value group, into
exactly the 4 `.b32` registers of the m16n8k16 A fragment (`ggml.cu:54-72`, `a[2][4]`),
then issues 2 mmas. The dependent chain is broken *geometrically, not speculatively*: the
load-time byte permutation puts the bytes a lane needs in one contiguous 16-byte run
(`uptr`, `ggml.cu:45-47`: `tile + (q>>1)*1056 + (q&1)*512 + (lane&3)*128 + (lane>>2)*16`,
comment "a warp's load is 512 distinct bytes"), so `load → 16B → 4 fragment registers`
needs **no reshape at all**, only 4 `__shfl_sync` per group to transpose between the 4
lanes that share a column (`ggml.cu:262-267`). Lookups come from **shared memory, staged
once per CTA — explicitly not constant memory**: "IQ2_XXS's grid in shared memory (**indices
diverge across a warp**); signs and the bf16 conversion are arithmetic" (`ggml.cu:111`).
That is the same reason our own broadcast probe was *slower* (25.8 us vs 18.8 us).
Latency is hidden by per-thread in-flight loads, not occupancy: `U = 2` "loads a warp has
in flight, twice over" (`ggml.cu:186`), `fetch(q + U*KW)` issued **before** the current
group is consumed (`ggml.cu:229-241`), and the weights are dispatched before the table copy
so the LDS does not wait on the LDG (`ggml.cu:218-219`: "the weights are on their way while
the tables load"). Loads are `ld.global.nc.L1::no_allocate.v4.u32` (`ggml.cu:13-24`) —
streaming weights must not evict L1. EXL3's dense linear does the identical thing with
`PF = K2 <= 12`: `load_step` pulls a whole 8-tile k step (up to `8*lane_words` = 24-32
registers) into `nxt` while the current one decodes (`exl3_decode.cuh:85-99`,
`exl3_linear.cu:124,155,169`).

**3. Accumulation dtype and scale order.**
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for both GGML formats
(`experts.cuh:44-51`) and `...f32.f16.f16.f32` for EXL3 (`exl3_decode.cuh:130-136`). No
HFMA2 anywhere, no integer accumulation on any codebook path — the only int8 mma in the
tree is `w4a8_prefill.cu:32`, a prefill kernel. Their gate 4 actively *forbids* non-fp32
accumulation: the harness replays each row's sums in fp16 and bf16 on the CPU and requires
each to miss by **≥100×** the exact-sum bound, with a planted "dot summed in half" control
(`docs/kernel-contract.md:44-50`, visible failing in
`docs/receipts/glm-fp4local-80517fc3/single/kernel-contract.log`). Scale order is the
transferable part: the accumulator is **zeroed per 32-value group**, the two mmas fill it,
and the group scale enters as 8 fp32 FMAs on the 16-column result
(`ggml.cu:145-166`): `p = 0; mma(p, a[s], b); … acc[nt][j] = fmaf(p[nt][j], sc[s], acc[nt][j])`.
So the inner loop **is** multiply-free — the decoded values are exact bf16 integers
(8/25/43 exactly representable) and `d * (0.5 + grid/16) * 0.25` is applied once per 32
values per output column instead of once per value. Note this scales the *accumulator*,
which is why it is free; folding the scale into the *operand* is what cost us 2-3x.

**4. Tensor-core path and the table-decode instruction sequence.**
Both kernels land decoded values directly in the `m16n8k16` layout — no shared staging, no
`ldmatrix` on the weight side (`ldmatrix` appears only in `x3gm2.cuh:5-8` for activations).
EXL3 gets this for free by defining the tile's *storage* permutation as the fragment
permutation: "Value p lands in its tile at row `2*(l%4) + (j&1) + 8*((j>>1)&1)`, column
`l//4 + 8*(j>>2)` … lane l holds values 8l..8l+7, **exactly its B fragments of the two
mma.m16n8k16 of the tile**" (`docs/recipes/exl3.md:64-67`, and `exl3_decode.cuh:126-127`).
The decode sequences, as written:

- IQ2 grid byte pair → signed bf16 pair (`ggml.cu:30-38, 54-72`), 5 ops per 2 values:
  `PRMT g, 0x43, 0x4140` (two grid bytes → two bf16 with exponent 0x43 = 128+m) →
  `HADD2 -0x43004300` (bias out, exact) → `IMAD sb, 0x40008000` (scatter sign bit 0/1 to
  bit 15/31) → `LOP3 & 0x80008000` → `LOP3 ^ ` into the pair. Per 8 values: 1 `LDS.64`
  gather + 1 `POPC` (the parity fold, `ggml.cu:61`) + 2 shifts + ~20 of the above ≈ 2.9
  ops/value. The nibble sibling is `nib2`: `(w >> sh) & 0x000F000F | 0x43004300` then
  `HADD2 -128` (`experts.cuh:36-41`) — 3 integer ops for 2 values, no table.
- EXL3 `3inst`/`mcg` pair (`exl3_decode.cuh:96-107`): `IMAD` → `LOP3` (`(x & 0x8FFF8FFF) ^
  0x3B603B60` is a single LOP3) → `PRMT 0x5410` and `PRMT 0x7632` to interleave the two
  states' halves → `HADD2`. `mul1` pair: `IMAD` → `DP4A(x, 0x01010101, 0x6400)` (byte sum
  as an fp16 bit pattern, `exl3_decode.cuh:89-90`) → `PRMT 0x5410` → `HFMA2` with
  `half2(0x1EEE), half2(0xC931)`. "two values cost ~5 instructions and the per-value decode
  cost does not grow with the bit width" (`docs/recipes/exl3-universal-experts.md:39-43`).
- Bit extraction to the 16-bit states uses `__funnelshift_l/r` on 2-3 loaded words — one
  variable shift lines up the lane's window and every subsequent state is a **constant**
  shift (`exl3_decode.cuh:64-81`, `exl3_candidate.cuh:44-53`); the grouped-expert version
  does it as one 64-bit merge per run and proves per compile time that "a group of GV
  values never spans more than two 32-bit words", with `GV` degrading 8→4→2 as width grows
  (`experts_grouped.cuh:44-53, 86-110`, `exl3-universal-experts.md:33-38`).

**5. What the receipts actually assert.**
`docs/receipts/…/limits.log` and `limits-nvcc*.log` are gate-5 output: per kernel symbol,
per compiled arch, `N registers, S static shared bytes, B launch-bounds threads`, checked
against `regs*32` rounding and the per-block/SM register and smem ceilings
(`zig/src/core/contract/limits.zig:30-64, 140-153`). They are the closest thing in the tree
to "what the decode kernel is limited by". The numbers (sm_80 / sm_120):

| kernel | regs | static smem | bounds |
|---|---|---|---|
| `tf_exl3_linear_2_2_4` (2 bpw, mul1, 4 warps) | 168 / 154 | 16 B | 128 thr |
| `tf_exl3_linear_4_2_4` | 168 / 154 | 16 B | 128 thr |
| `tf_exl3_linear_8_0_4` (4 bpw `3inst`) | **254** | 16 B | 128 thr |
| `tf_exl3_linear_8_0_8` | **255** | 16 B | 256 thr |
| `tf_exl3_linear_*_*_8` (8 warps) | 176-241 | 16 B | 256 thr |
| `tf_exl3_grouped_{0,1,2}` (experts) | 127-128 | **32832 B** | – |
| `tf_exl3_tc_*` (candidate tensor-core) | 128 | 8196 B | 256 |
| `tf_exl3_prompt_*` | 75-80 | 32768 B | 256 |
| `tf_exl3_rot_in` | 19-22 | 0 | 128 |
| `tf_exl3_codebook`, `tf_exl3_unpack`, `tf_exl3_rotate_*` | 6-28 | 0 | 32 |

(`docs/receipts/glm-kv-pages-0b094b74/gate5/limits-nvcc13.3.log:674-760`,
`docs/receipts/glm-fp4local-80517fc3/single/limits.log`.) Read: the codebook decode kernel
is **register-limited to ~1 block/SM** — 168 regs × 128 thr = 21504 regs/block of 65536, so
3 blocks/SM = 384 threads = 25% of a 1536-thread SM, and the 4-bit `3inst` case is pinned at
the 255-register ceiling = 1 block/SM = 8% occupancy — and it still hits 93% of the DRAM
ceiling. That is the strongest available evidence that a codebook GEMV at low occupancy is
fine **provided each thread keeps enough loads in flight**, which is exactly what `U=2` and
`PF` buy. Shared is used for the 2 KB codebook + fixed-order reductions, never for staging
weights at decode. `kernel-contract.log` asserts the *correctness* gates (non-finite
classes, five-run byte identity, shape envelope, real-fp32 sums) with planted controls that
must fail; it constrains **accumulation order and determinism**, not registers. Two gaps to
know: `tf_ggml_iq2_*` is **not registered** in the limits gate anywhere in this tree, so
there are **no register numbers for their IQ2 GEMV**, and the fatbins are built for
sm_80/86/89/120/121 only — **sm_90 (our H200) is not compiled at all**, so every number
above is for a neighbouring arch.

**6. The MoE angle.**
Experts are one contiguous array of *packed tiles* `[E][n/16][k/256][1056]` bytes
(`zig/src/cuda/experts.cuh` consumer: `experts.zig:24`), so the per-expert gather is a
scalar base-pointer add **per thread block**, not a per-element gather:
`wt = W + (it.e * NT16 + tile) * (KG/G) * BYTES` (`ggml.cu:206`). That is the "free gather"
layout choice: **batch by expert in the block index, batch by member inside the mma.** The
plan hands each CTA `(expert, first member, count)` items plus a `members` list grouped by
expert (`zig/src/cuda/grouped.zig:14-21`, `ggml.cu:127-135`), and up to 8 members of one
expert occupy the mma's 8 `n` slots (`epilogue_rows`, `ggml.cu:171-186`). Nothing
accumulates with atomics: K-split partials and warp sums go through shared **in a fixed
warp order**, so "a row's bits never depend on the window it arrived in"
(`exl3-universal-experts.md:47-49`, `experts_grouped.cuh:259-281`), and the split is chosen
from the **shape alone** — `plan(k, n)` — never from the data, which is what makes CUDA-graph
replay and chunk-invariance provable (`exl3.md:110-116`). Their split-K decision rule is
directly actionable for us: `loads = k/32/(4 for iq2)`, `tiles = loads < 16 ? 4 : 1`
(`ggml.zig:55-56, 168-171`) — for our `up_proj [2048,1024]` experts, `loads = 8 < 16`, so
they run **one warp per 16-column tile, 4 tiles a CTA, no split-K and no reduce kernel at
all**, getting parallelism from tiles; and when K *is* long, the split lives **inside the
CTA through shared memory** (`ggml.cu:280-291`, `exl3_linear.cu:126,217-247`), never as a
separate global-split + reduce launch. Their EXL3 grouped-expert kernel reaches
**204-210 GB/s at 1 row** vs ExLlamaV3's `exl3_moe_mixedk` 49.9-86.7 GB/s (2.4-3.6x), and
**is within 2% of `exl3_moe_coop` on uniform layers** — i.e. the win was removing the
mixed-K dispatch penalty, not beating a well-shaped uniform kernel
(`exl3-universal-experts.md:17-27`).

**7. Verdict for our 5.7 MB / 18.8 us GEMV.**
The single idea is **decode into packed bf16x2 in registers, in a layout that the load-time
byte permutation has already fixed** — i.e. move the `unpack-8-values / apply-sign / reshape
-into-dot-layout` tail out of the kernel and into the one-time repack, then do the unpack as
`prmt`+bias-subtract+`mul`-sign on int32 lanes. It costs **no weight bytes** (permutation),
**2 KB of shared per CTA** for the grid, and ~2 bf16 sign bits of nothing: magnitudes 8/25/43
are exact bf16 integers and our activations are already bf16, so the products are exact and
the fp32 accumulation is inside the HMMA — the only behavioural change is summation order
(last-ulp), not added rounding of the weights (it is *strictly better* than dequantising to
bf16, which rounds `d*mag`). It does not need a tensor core to pay off: even if we keep an
fp32 `tl.dot`, emitting the magnitudes as one int32 holding **two** bf16 values and
bitcasting halves the op count of the unpack and deletes the reshape entirely, because the
permutation makes "value p of the block" land in the register lane the dot wants.
Register cost, on the nearest measured analogue, is the price to watch: 154-168 regs/thread
and 16 B static smem is what a 4-warp codebook-MMA decode costs, and the 4-bit `3inst`
variant hits 254-255, so budget ~150-170 and expect ~25% occupancy — which their
93%-of-ceiling number says is survivable *if* the loop keeps ≥2 16-byte loads per thread in
flight. Second-order but cheap: adopt their split-K rule (no split-K for our 1024-K experts;
4 tiles a CTA) and `cache_modifier=".cg"` to stop weight streaming polluting L1.
**What does not transfer, plainly:** EXL3's answer to "break the lookup chain" is "have no
lookup" — the codebook is an algebraic hash `s → fp16` (`exl3_decode.cuh:86-108`), so there
is no table to stage, no gather, and no `__constant__` serialization problem. Their code
proves the limit of that argument for us: they still keep a **2 KB-per-CTA shared table for
anything that is genuinely tabular** (`ggml.cu:111-124`, with "indices diverge across a
warp" as the stated reason) and they never pre-expand a table into the weights. Our format's
256×8 grid is data, not a curve; the lookup stays, the reshape does not have to.

## Techniques found

- **Load-time byte permutation into MMA-shaped tiles (zero bytes).** Reorders each 66-byte
  IQ2_XXS block into 16-column × 66-byte tiles so lane `(gq,t)`'s 16-byte load is exactly
  the 8 magnitudes + 7 sign bits + scale nibble it needs, in fragment order.
  `ggml.cu:41-52, 433-456`; `ggml.zig:58-95`; permutation proof `ggml.zig:204-224`; "a byte
  permutation, the same size" `CHANGELOG.md` commit `3915bcea`. Cost: 0 bytes, 2 KB shared,
  one pass at load. **Maps to Triton as**: write a `repack_blocks` variant that emits the
  same permutation, then in the kernel `tl.load` a `uint32[4]` per lane and *skip the
  reshape/permute entirely*; verify with a host `packHost`-style bijection test.
- **Packed bf16x2 codebook decode: `prmt` + bias-subtract + multiply-scattered signs.**
  `ggml.cu:30-38` (`sign_mask`, `grid_pair`), nibble sibling `experts.cuh:36-41`. ~2.5 ops
  per value, one `LDS.64` per 8 values, **no sign table**. **Maps to Triton as**: gather the
  grid row as two int32, then
  `v = tl.inline_asm_elementwise("prmt.b32 $0,$1,0x43,0x4140;", …)` (or, without inline asm,
  `(w >> 8*k) & 0x00ff00ff | 0x43004300`), bitcast to `bfloat16`, `- 128.0`, and XOR the
  sign word `((sb * 0x40008000) & 0x80008000)` bitcast to bf16 — all on int32 lanes so two
  values ride one register.
- **Per-32-group scale applied to the fp32 accumulator, not to the operand.**
  `ggml.cu:145-166` (`p` zeroed per group, 8 `fmaf`s out). This is the multiply-free inner
  loop that does *not* cost the 2-3x our operand-side fold cost. **Maps to Triton as**: use
  `BK = 32` for the `tl.dot` (or 4 dots inside a 128-wide k-block) and
  `acc += tl.dot(a32, b32) * scale[:, None]`.
- **Shared-memory codebook staged once per CTA, never constant memory, never broadcast.**
  `ggml.cu:111-124` + the "indices diverge across a warp" comment. Confirms our own
  measurement that the gather is free and the broadcast is 37% slower. **Maps to Triton as**:
  nothing to change — keep `iq2_lut` as a gather; just stop it being a 256×8 *fp32* table by
  keeping it as packed int32 pairs so the LDS is 64-bit and the result is pre-packed.
- **Per-thread prefetch depth as the bandwidth lever at low occupancy.** `U = 2` "loads a
  warp has in flight, twice over" + `fetch(q + U*KW)` before consuming (`ggml.cu:186,
  229-241`), weights fetched before the table copy (`ggml.cu:218-219`); EXL3's
  `PF = K2 <= 12` next-k-step prefetch (`exl3_linear.cu:124,155,169`), worth **40% at 2 bpw
  (156 → 221 GB/s)** and 10-15% at 3-5 bpw (`exl3.md:143-147`). **Maps to Triton as**:
  `num_stages ≥ 2` with a manual double-buffer of the *packed* weight registers, or simply
  two independent k sub-ranges per program so the compiler software-pipelines the loads.
- **`ld.global.nc.L1::no_allocate` for weight streaming.** `ggml.cu:13-24`. **Maps to
  Triton as** `tl.load(..., cache_modifier=".cg", eviction_policy="evict_first")`.
- **Invert the operands at M=1: weights = A, rows = B, so `n`=8 batches members.**
  `ggml.cu:1, 145-166, 171-186`. Lets one mma serve up to 8 same-expert members and wastes
  the rest at M=1 without hurting a bandwidth-bound kernel. **Maps to Triton as**: for MoE,
  put the ≤8 members of an expert on the `M` axis of `tl.dot` and the 16 output columns on
  `N`, weights as the other operand — the tile shape our `_moe_gemv_kernel` does not use.
- **Split-K inside the CTA via shared, fixed warp order, no atomics, or none at all.**
  `ggml.cu:280-291`; rule `loads < 16 → 4 tiles a CTA, no split` `ggml.zig:55-56, 168-171`;
  "Nothing accumulates with atomics… a row's bits do not depend on the window it arrived in"
  `exl3-universal-experts.md:47-49`. CUDA-graph-safe by construction and it removes our
  `_splitk_reduce_kernel` from the M=1 path. **Maps to Triton as**: for the 1024-K experts,
  grid over `N/16 × 4 tiles` with `SPLIT_K = 1`; for the 4096-K dense, split K only across
  warps inside one program and reduce with `tl.sum` over a `[WARPS, BN]` tile.

## Dead ends confirmed

- **Constant-memory or broadcast codebook reads.** They explicitly use shared because
  "indices diverge across a warp" (`ggml.cu:111`) — divergent constant loads serialize. Our
  own broadcast probe (25.8 us vs 18.8 us) agrees.
- **`cp.async` / shared-memory staging of weight tiles for decode.** "An earlier version of
  this kernel streamed the tiles through shared memory with `cp.async`; that was **three to
  five times slower** than these per-lane cached loads on GB10, whose L1 serves a warp's
  256-byte search from one line" (`docs/recipes/exl3.md:149-151`).
- **Prefetching at ≥6 bpw.** "Above 6 bits a step is 224 or 256 B of words a warp and
  prefetching it measured **1-2% slower** than loading per tile, so those widths load per
  tile" (`exl3.md:146-148`). Prefetch depth is a low-bit lever only.
- **Pre-expanding quantized weights to fp16/bf16 for decode.** Done for prefill only
  (`exl3.md:120-126`); decode keeps the packed form and relies on row reuse ("within 5% from
  1 row to 16", `exl3.md:141`). Consistent with our own 1.70x-bytes-for-1.70x-speed plane
  result: no bytes-for-time win at M=1.
- **fp16/bf16 accumulation, in any form.** Forbidden by their own gate 4, which requires
  fp16 and bf16 replays to miss by ≥100× the exact-sum bound and plants a "dot summed in
  half" control that must fail (`docs/kernel-contract.md:44-50`; run output
  `docs/receipts/glm-fp4local-80517fc3/single/kernel-contract.log`). Half accumulation is
  not a legal shortcut even where it would be faster.
- **Shape-only split planning is not free.** "rows 8 and 16 of the 3-bit shapes, where the
  split a shape alone picks is **3-19% off** the best cell of the (K split, warps) grid"
  (`exl3.md:148-149`) — the price they pay for graph/chunk invariance; expect the same if we
  adopt their fixed-plan rule.
- **Beating a well-shaped uniform-expert kernel.** "this work does not beat `exl3_moe_coop`
  on uniform layers, it removes the mixed-K penalty" (`exl3-universal-experts.md:29-31`). If
  our 512-expert layer is already uniform (it is), the grouped-dispatch rewrite buys nothing
  by itself.

## Sources

- Repo `clones/TensorFold` @ `7c068895`; IQ2/Q8 kernels from `3915bcea` + `b2f6d854` (PR
  #897 lane), EXL3 CUDA from #42 (Python, `v0.6.6` tag) and #496/#681/#740/#741/#806 (native).
- `zig/kernels/cuda/ggml.cu` — IQ2_XXS + Q8_0 on mma.sync: `13-24` streaming loads,
  `29-38` sign/magnitude arithmetic, `41-72` IQ2 decode into A fragments, `111-124` shared
  codebook, `145-166` group scale on the accumulator, `186-241` prefetch loop, `262-267`
  lane transpose, `280-291` in-CTA split-K, `433-456` the repack.
- `zig/kernels/cuda/ggml_iq2xxs_tables.cuh:5,17` — the shipped-but-unused ksigns table and
  the 2 KB grid; `zig/kernels/cuda/experts.cuh:11-51` — `K128`, `nib2`, `mma` (bf16→f32).
- `zig/src/cuda/ggml.zig:26-29, 55-56, 58-95, 168-171, 204-224`; `zig/src/cuda/experts.zig:24,112-118`;
  `zig/src/cuda/grouped.zig:14-21`.
- `zig/kernels/cuda/exl3_decode.cuh:26-136`; `zig/kernels/cuda/exl3_linear.cu:85-99,124-180,217-247`;
  `zig/kernels/cuda/exl3_block/experts_grouped.cuh:16-110,130-178,259-281`;
  `zig/kernels/cuda/exl3_candidate.cuh:23-70`.
- `docs/recipes/exl3.md:64-67,95-96,110-116,120-126,139-151,158-166`;
  `docs/recipes/exl3-universal-experts.md:17-49`; `docs/recipes/cuda.md:1-60`.
- `docs/kernel-contract.md:20-50`; `zig/src/core/contract/limits.zig:30-64,140-153`;
  `docs/receipts/glm-kv-pages-0b094b74/gate5/limits-nvcc13.3.log:674-760`;
  `docs/receipts/glm-fp4local-80517fc3/single/{limits,kernel-contract}.log`.
- `CHANGELOG.md:604` (Bolt on CUDA, 153-155 tok/s on an RTX PRO 6000 sm_120); commit
  `3915bcea` message ("lm_head's Q8_0 GEMV reads at **1.65-1.68 TB/s on an RTX 5090**",
  ≈93% of that card's 1.79 TB/s); `THIRD_PARTY_NOTICES.md:211-214`.

## Gaps / residual risk

- **No IQ2 GB/s number exists anywhere in this checkout**, and `tf_ggml_iq2_*` is unregistered
  in the limits gate — the IQ2 structure is credible (near-floor on EXL3 and on Q8_0 lm_head,
  same model, same formats) but its measured decode speed is unevidenced here.
- All register numbers are sm_80/86/89/120/121; **sm_90 is not compiled**, so the
  168-reg / 25%-occupancy claim must be re-measured with `--resource-usage` or Triton's
  `n_regs`/`n_spills` before we trust it on H200.
- Everything here is CUDA C++ with hand-written `mma.sync`; the Triton formulations above are
  my translation, not something this repo demonstrates.
- The accumulator-side scale changes summation order; bit-exactness of the IQ2 *values* is
  preserved, but our kernel's output bytes will move in the last ulps versus the current
  fp32-FMA path — check against `ref_decode`-based tolerances, not byte equality.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Report written to the authoritative path with the context file's Verdict/Techniques/Dead ends/Sources format, answering all 7 lane questions with file:line citations (ggml.cu:29-72,111-124,145-166,186-241,433-456; exl3_decode.cuh:64-136; experts_grouped.cuh:16-110; ggml.zig:58-95,204-224; limits-nvcc13.3.log:674-760), plus a Gaps section listing residual risks."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read/grep/find/ls + read-only git log (git log --oneline, git log -1 --format=%B) over the TensorFold checkout",
      "result": "passed",
      "summary": "Located and read both IQ2_XXS and EXL3 decode kernels, the recipes, the kernel contract and the limits receipts; no writes to the checkout"
    }
  ],
  "validationOutput": [
    "git log --oneline -3 => 7c068895; IQ2 kernels from commits 3915bcea and b2f6d854",
    "limits-nvcc13.3.log: tf_exl3_linear_2_2_4 = 168 regs sm_80 / 154 sm_120, 16 B static smem; tf_exl3_linear_8_0_4 = 254 regs; tf_exl3_grouped_* = 127-128 regs, 32832 B static smem; archs compiled are sm_80/86/89/120/121 only (no sm_90)",
    "exl3.md numbers: 221-226 GB/s of a 237 GB/s ceiling, rows 1 vs 16 within 5%; cp.async staging 3-5x slower; prefetch worth 40% at 2 bpw, 1-2% slower above 6 bpw",
    "exl3-universal-experts.md: 204-210 GB/s at 1 row vs ExLlamaV3 mixedk 49.9-86.7 GB/s; within 2% of coop on uniform layers",
    "commit 3915bcea: repack is 'a byte permutation, the same size'; Q8_0 lm_head GEMV 1.65-1.68 TB/s on RTX 5090"
  ],
  "residualRisks": [
    "No IQ2 GEMV benchmark or register record exists in this checkout (tf_ggml_iq2_* is not registered in the limits gate); the near-floor evidence is for EXL3 and Q8_0 on the same model/format family, not for the IQ2 kernel itself.",
    "All register/occupancy numbers are for sm_80/86/89/120/121; sm_90 (H200) is never compiled in this tree, so the 154-168 register budget and ~25% occupancy must be re-measured before relying on it.",
    "The Triton formulations (prmt-style packed bf16 decode via inline_asm_elementwise or int32 shift/mask/or + bitcast; BK=32 dot with accumulator-side scale) are my translation, not demonstrated by this repo.",
    "Applying the per-32 scale to the fp32 accumulator changes summation order: IQ2 value bit-exactness is preserved but GEMV output bytes will differ in the last ulps from the current fp32-FMA path.",
    "The task named a second write target (research/08-tensorfold.md); the run-specific authoritative output path overrode it, so the parent must copy this file there if the numbered report is still wanted."
  ],
  "noStagedFiles": true,
  "diffSummary": "No repository changes; single report artifact written outside the repos to the run's authoritative output path.",
  "reviewFindings": [
    "no blockers: read-only discipline held - no writes, no builds, no git write commands inside clones/TensorFold"
  ],
  "manualNotes": "Biggest finding is that this repo already implements our exact checkpoint (Nemotron-3.5-Bolt IQ2_XXS + Q8_0) on mma.sync in zig/kernels/cuda/ggml.cu; the transferable core is (a) load-time byte permutation into MMA-shaped tiles at zero byte cost, (b) packed bf16x2 decode with prmt + 128-bias subtract + multiply-scattered sign bits and no sign table, and (c) the per-32 scale applied to the fp32 accumulator per 32-value group rather than to an operand. Strongest confirmed dead end: EXL3 removes the lookup by making the codebook a closed-form hash, which our tabular 256x8 grid cannot imitate - their own code falls back to a 2 KB-per-CTA shared table (never constant memory) for genuinely tabular data."
}
```
