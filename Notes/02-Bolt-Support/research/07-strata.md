# Strata engine lane — how a self-written i-quant GPU kernel set runs IQ2_XXS at decode

Repo: `clones/Strata`, tag `v0.2.7.1`, HEAD `46c152a1310f77cd2f0ce2c7cd213962df14fe83`.
Read-only inspection; nothing in the checkout was created, edited or deleted, and only
`git log/show/diff` reads were run. No build, no GPU run.

Strata has **three independent IQ2_XXS decode implementations**, which is why it is a useful
second opinion:

| path | file | role | accumulator |
|---|---|---|---|
| shipped CUDA engine | `src/kernels/cuda/iq_kernels.cu` | llama.cpp `vecdotq` transcribed + their own "decode-once" `Split<TY>` traits | `__dp4a` → int32 |
| DeepSeek-V4.1 engine (Project Maya port) | `src/dsv41/ds4_iq.cu` | their own token/window/GEMV kernels | `__dp4a` → int32 |
| prefill / low-row | `src/prefill/moe_ws_gu_iq2.cu`, `src/prefill/moe_fused_iq.cu` | weight-stationary + int8 tensor cores | `mad.wide.s16` / `mma.sync m16n8k32 s8` → int32 |

All three accumulate i-quant dots **in the integer domain**. None of them does per-value fp32 FMA.

## Verdict

Yes, there is one thing to copy, and it is cheap: Strata never multiplies an IQ2 magnitude as a float.
Every path decodes a 32-value sub-block into **eight packed int32 registers of four signed int8
magnitudes** and does the dot with eight `__dp4a` (or sixteen `mad.wide.s16`, or one `mma.sync
m16n8k32.s8`), applying `d·(2s+1)/8` once per 32 values — so the per-value instruction count of the
dot is a quarter of a float FMA chain, and the per-value float-op count is zero. The sign, which is
the expensive dependent step in our chain, is **precomputed into a table and folded into a PRMT
selector**: `sel = tabA[idx_byte] | tabB[sign_field]` then `__byte_perm(0x002B1908, 0x00D5E7F8, sel)`
yields four signed int8 magnitudes in one instruction (`src/prefill/moe_ws_gu_iq2.cu:133-165`) — and
that works only because IQ2_XXS magnitudes are exactly `{8,25,43}`, our format's exact grid. Cost:
**zero extra weight bytes**, ~1.5 KB of shared/LUT, and the accuracy price is on the *activation*
side (int8 per 32 with an fp32 scale), not the weight side. Their own measured evidence for the
"bigger, more-folded lookup table" direction is `-15 %` on sm_120 and `-30 %` on sm_86 for one
layer (`git 71abb6c1`), and their own measured evidence against the *other* table direction is
`2.12x` for a divergent `__constant__` codebook (`src/kernels/cuda/s_gemv.cu:20-31`). Worth a day.

## Techniques found

### 1. PRMT-composed sign-folded codebook selector — one instruction per 4 values, no extra bytes
- **What.** Two tiny shared tables generated from ggml's grid at kernel start:
  `tabA[256]` = each grid row as **eight 2-bit magnitude codes** in one u32 (code 0/1/2 for 8/25/43),
  `tabB[128]` = each 7-bit sign field (parity folded in) as **eight nibbles carrying bit 2**.
  `sel = tabA[idx] | tabB[sign]` is then literally a PRMT selector over the constant pair
  `PX = 0x002B1908` = bytes `{+8,+25,+43,0}` and `PY = 0x00D5E7F8` = bytes `{-8,-25,-43,0}`:
  nibble value `code | 4*sign` picks the right signed byte. Cost per 8 values: 2 LDS + 1 IOR +
  2 PRMT (+ 4 PRMT if int16 pairs are wanted). No `popc`, no `vcmpne4`, no `vsub4`, no parity chain.
- **Where.** `src/prefill/moe_ws_gu_iq2.cu:35` (PX/PY), `:133-141` (`grid_codes`),
  `:144-150` (`sign_spread`), `:163-165` (`sel` + `__byte_perm`), `:13-15` (the header note:
  *"every iq2xxs/iq2xs/iq2s_grid byte is 0x08, 0x19 or 0x2B ... one PRMT per 4 values, checked
  against MMQ's expressions for every entry and sign byte"*).
- **Cost.** 256+128 u32 = 1.5 KB shared. Zero weight bytes. Bit-exact by exhaustive check upstream.
- **Maps to Triton as.** `tl.inline_asm_elementwise("prmt.b32 $0,$1,$2,$3;", ...)` with the two
  constants and a packed nibble selector; the two tables are `tl.load` of a 384-entry int32 LUT we
  build next to the existing `iq2_lut` (which already stores the 2-bit codes at offset `256*8`,
  `modelopt_block_quant_triton.py:103-111` — so `tabA` already exists in our LUT, `tabB` is new).

### 2. Integer-domain dot with the scale applied once per 32 values, and a *stated* exactness bound
- **What.** Per 32 values: eight `__dp4a` into one int32, then
  `v = __fmul_rn(s.d, (float)(s.s0*lo + s.s1*hi))` — the `(2s+1)` magnitude-scale multiply happens in
  **integers**, `d` (the fp16 block scale × 1/8) is one fp32 multiply, and the whole sub-block enters
  the fp32 accumulator with one FMA: `ag[i] = fmaf(dx, sub_dot<T>(sg, xa, xb), ag[i])`.
- **Where.** `src/dsv41/ds4_iq.cu:357-381` (`sub_dot`), `:182-183`
  (`s.s0 = s.s1 = 2*(a1>>28)+1; s.d = h2f(b)*0.125f`), `:660-662` (the loop),
  `src/dsv41/ds4_iq.cuh:9-13` (*"a 32-value sub-block's sum is an exact integer dot (dp4a) times its
  scales in F32 — no integer division as llama.cpp's vecdotq takes (its sums truncate), so these are
  the double reference's sums to F32 rounding"*).
  The exactness bound that makes reordering legal: `src/prefill/moe_ws.cuh:13-14`
  *"A 32-value block's integer sum is exact (|sum| < 2^22 for every format here), so the order of its
  MACs is free"*, and `src/prefill/moe_fused_iq.cu:6-11` `|sum q a| <= 32*127*127 < 2^22`.
- **Cost.** Requires int8 activations per 32 values with an fp32 scale (their `q8_quant`:
  `sc = max|x|/127`, round-half-even, `tests/dsv41/iq_expert_test.cpp:52-62`). Accuracy bars they
  record for that: token `gate_up` **1e-5**, token `down` **1e-6**, F16-tile prompt path
  **1e-2 / 3e-3**, IQ1_S **1.5e-2** (`tests/dsv41/iq_expert_test.cpp:470,489,556,585`).
- **Maps to Triton as.** Decode to an int8 weight tile `wq [BK,BN]` with signs folded (technique 1),
  quantize `x` to int8 per 32 into `xq [BK]`, broadcast to the 16-row `tl.dot` M dim, and
  `acc_i32 += tl.dot(xq8, wq8)` (int8 in / int32 out → IMMA on SM90). The `< 2^22` bound is the
  licence: **any split-K / atomic order is bit-safe**, which is exactly the CUDA-graph objection we
  could not answer when we rejected fusing the split-K reduce into the GEMV.

### 3. The 32-value quant block is one thread's private unit
- **What.** Token kernels: one warp per output row, 32 lanes per row, and lane `l` owns sub-block
  `ib0 = l & 7` of block `blk0 = l >> 3` for the whole loop, so a lane always keeps the *same*
  sub-block position and never shuffles inside a quant block. A warp step therefore covers
  32 sub-blocks = 1024 contiguous values, and the row reduction is one `lanes_sum<RW>` butterfly at
  the end.
- **Where.** `src/dsv41/ds4_iq.cu:599-606` (`kGuRW = 1, kDnRW = 2`), `:646-648`
  (`ib0 = lane & 7, blk0 = lane >> 3; for (blk = blk0; blk < nblk; blk += kStep)`), `:600-605`.
- **Cost.** Nothing; it is a lane→data assignment.
- **Maps to Triton as.** Arrange the blocked layout so each thread's K slice is exactly one aligned
  32-value group (`BK` split so `BK/num_threads == 32`, i.e. BK 1024 at 32 threads/warp × 4 warps),
  which lets `aux32`/`idx32` be loaded as one 32-bit word per thread and kills the intra-group
  broadcast/reduction entirely.

### 4. Funnel-shift staging to keep the 66-byte-block loads aligned
- **What.** IQ2_XXS rows are 660/740/820 B, so a 256-value block starts `2*(kc&1)` bytes into a
  4-byte word. Instead of per-lane branches or 16-bit loads, the whole CTA issues **aligned** 32-bit
  loads and recovers the word with `__funnelshift_r(lo, hi, 16)`; `d` is handled the mirror way
  (odd block's `d` = high half of the word two bytes before the block). Every load is issued before
  its stores, and nothing outside a row is read.
- **Where.** `src/prefill/moe_ws_gu_iq2.cu:16-22` (the note) and `:238-262` (the code:
  `if (kc & 1) { v[i] = __ldg(...+2...) } else { lo/hi + __funnelshift_r }`).
  Their own diagnosis of the alternative is in `docs/EXPERTDEC.md:107-110`: *"Not done here: the
  weight side (the 8-9 two-byte loads per call and row of `Split<TG>::load`; a weight tile staged in
  shared memory with 16-byte loads would be the next step)."*
- **Cost.** Two aligned 32-bit loads + 1 shift per word instead of one unaligned word; zero bytes.
  Our row stride is `4096/256*66 = 1056 B`, so the same odd/even alternation applies to us.
- **Maps to Triton as.** Load `qw` as `int32` at `(row*1056 + 8*blk)//4` with an even/odd pair and
  `(a >> 16) | (b << 16)`; Triton folds this to SHF. Alternatively repack blocks so each row's
  `qs`/`aux` regions are separate aligned arrays (see technique 6).

### 5. Fold-more-of-the-decode-into-one-lookup, gated by codebook size
- **What.** A `Fast<T>` table that pre-folds the delta sign *and* the code into one lookup:
  IQ1_M's 12-bit `(neg, 11-bit index) → 8 int8 codes`, 2048×16 B = **32 KB of shared**. They turn it
  on per-type and *only* for the kernels that are decode-bound, and they made persistent grids
  conditional on the codebook being worth keeping resident.
- **Where.** `src/dsv41/ds4_iq.cu:505-509` — *"The token kernels are bound by instructions per
  weight, not by memory, so on CUDA a type may decode through a larger shared-memory table that
  folds more of the work into one lookup. The arithmetic is unchanged ... so the results are the same
  bits."* `:510-513` (`Fast<kIQ1_M> { WORDS = 8192 }`, `kIq1mCodes[4096]`),
  `:519-521` (*"the larger codebook's shared memory costs them occupancy: they keep `decode<T>`"*),
  `:523-530` (`kPersist = kTabWords<T>*4 >= 8192`), `:1123-1149` (persistent grid rationale),
  `src/dsv41/ds4_iq1m_codes.inc` (99 KB generated table).
- **Measured.** `git 71abb6c1`: *"IQ1_M decode through a 4096-entry code table (delta sign folded
  in): bit-identical, IQ1_M layer **-15 % sm_120 / -30 % sm_86**."*
- **Cost.** 32 KB shared for IQ1_M; occupancy loss on many-row kernels. For IQ2_XXS they did **not**
  build the folded table (`Fast<kIQ2_XXS>` default `WORDS = 0`) — the product form would be
  256×128×8 B = 256 KB, but technique 1's two-table OR gets the same effect for 1.5 KB.

### 6. Sign-mask table instead of a parity chain — shipped for IQ3_XXS/IQ3_S/IQ1_M, *not* for IQ2_XXS
- **What.** Staged shared masks `ml/mh`, one per 7-bit sign field (parity folded in), so the sign is
  `(grid_word ^ mask) + (mask & 0x01010101)` — **two ALU ops per 4 values**, replacing
  `popc` + `xor<<7` + `*0x01010101` + two `__vcmpne4` + two `__vsub4`.
- **Where.** `src/kernels/cuda/iq_kernels.cu:1773-1789` (`stage_iq3xxs_masks`), `:1051-1056` (used in
  `Split<18>::load<true>`), `:1804-1860` (`stage_iq_grid`, 512 u32 = 2 KB IQ2_XXS grid staged once per
  block, `kStageIqGrid` at `:911` includes IQ2_XXS). `Split<16>` (`:913-948`) stages the *grid* but
  still runs the `unpack_ksigns`+`vcmpne4`+`vsub4` chain — **the sign-mask table was never extended
  to IQ2_XXS in this repo.** That is a free win sitting on the table.
- **Maps to Triton as.** A 128-entry int32 LUT for `ml` and one for `mh`, both built at load time
  next to `iq2_lut`; `w = (g ^ ml) + (ml & 0x01010101)`. Exactly equivalent to the parity chain, so
  bit-exactness is provable, not hoped for.

### 7. Integer exactness kills the I2F instruction (magic-number seeding)
- **What.** Seed the integer accumulator with `0x4B400000` = the bits of `1.5·2^23`; because
  `|sum| < 2^22`, `__int_as_float(sum)` **is** `1.5e7 + sum`, so `float(sum)` is one `FADD` and no
  quarter-rate `I2F`. Same trick used as the MMA C operand (`mma` starts from `C = 0x4B400000`, so
  `as_float(D) - 1.5·2^23` is the dot).
- **Where.** `src/prefill/moe_ws.cuh:26-28`, `src/prefill/moe_ws_gu_iq2.cu:97,181,189`,
  `src/prefill/moe_fused_iq.cu:6-11,60`.
- **Maps to Triton as.** `acc.to(tl.float32, bitcast=True) - 12582912.0` on an int32 accumulator.

### 8. Int8 MMA expert path: decode to int8 in shared, `ldmatrix`, `mma.sync m16n8k32.s8`
- **What.** Per 64 values of K a thread decodes one 32-value sub-block of one weight row **one stage
  ahead into registers**, then writes 32 int8 + the scales to double-buffered shared memory
  (`WLD = 80 B`/row = 64 int8 + 16, chosen conflict-free); activations arrive by `cp.async` four
  stages deep; fragments by `ldmatrix`; products `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
  Codebooks in shared, up to 8 KB (IQ2_S).
- **Where.** `src/prefill/moe_fused_iq.cu:1-21` (whole arithmetic+load-stage spec), `:66,69,107`
  (`WLD`, `ASTAGES`, `grid_bytes`), `:165` (the MMA), `:195` (`ldmatrix`), `:243-251,417-455`
  (load stage / `convert`), `:578-583` (shared layout).
- **Measured.** `docs/DETAILS.md:1075-1076`: one layer runs at **1.92x / 1.37x / 1.30x** the MMQ path
  at 2,048 / 4,864 / 8,192 tokens. `docs/DETAILS.md:1217-1218`: **IQ2_XXS and IQ2_XS keep MMQ** —
  the fused int8 path was not adopted for IQ2_XXS gate/up, and on a 48-SM card the fused kernels were
  **4.7 % slower**.
- **Relevance.** This is the existence proof for an int8-in/int32-out tensor-core i-quant dot in this
  format family, and the negative selection of IQ2_XXS is a caution flag on it.

### 9. No padded-block waste at decode, by construction — and a row-count split for the prefill tiles
- **What.** Decode: `gate_up_kernel` is a **warp-per-output-row** GEMV, grid
  `(n_ff/(8·RW), groups)`, `groups` = the active experts, and the number of entries per group is a
  **template parameter dispatched at runtime** (`ne<=1 → NE=1`, `<=4 → 4`, `<=8 → 8`, else 16). The
  M=1 case is a pure GEMV; there is no 16/32/64-row tile to waste on it. An expert slot with a null
  blob pointer makes the whole block exit before it loads anything (`group()` returns false,
  `if (!loaded)` guards the codebook copy).
- **Where.** `src/dsv41/ds4_iq.cu:582-597` (`group()`, null blob → skip),
  `:610-671` (`gate_up_kernel`), `:1168-1181` (`ne` → NE dispatch), `:608-609` (grid).
  Prefill: experts with **≤ `kLightRows`=16 rows** go to the token-shaped "light" kernels ("the
  weights read once at the memory's speed, **where a 64-row GEMM tile did 64 rows' work for them**"),
  and the tiles get `skip_le = 16` so the two never overlap: `:939-947`, `:949-1031`, `:1234-1247`.
  The weight-stationary kernels say the same thing in one line: *"an expert's rows are its own count,
  not the group's largest"* (`include/strata/prefill/moe_ws.hpp:6-8`).
- **Their answer to our question 4.** Yes, one launch per product for all active experts; no
  alignment padding on rows; the padded-tile waste is removed by a **row-count threshold that routes
  small experts to the GEMV-shaped kernel**, plus a grouped path (`grp_start/grp_ent`) so several
  entries sharing one expert reuse one decode.

## Dead ends confirmed

1. **`__constant__` codebook with a divergent index: 2.12x, measured, and they wrote the reasoning
   down.** `src/kernels/cuda/s_gemv.cu:20-31`: *"IT WAS IN `__constant__` FOR THE WRONG REASON, AND
   IT COST 2.12x ... constant memory is fast when the access is UNIFORM ... a NON-LINEAR codebook
   means every lane reads a DIFFERENT INDEX, which is the pattern it handles worst."* Probe on
   `attn_qkv` (IQ4_XS, 10240×2560), memory traffic and instruction count held identical:
   **`IQ4NL 0.2032 ms / Affine 0.0960 ms, ratio 2.12x`**. Fix: 16 bytes staged into shared, once per
   block (`decode_tbl`, `:41-46`, `:126-128`). Consistent with our "gather is FREE" probe.
2. **Staging the activations in shared memory for a GEMV: worse in every paired comparison.**
   `src/kernels/cuda/s2_gemv_fast.cu:10-13`: *"staging it once per block measured WORSE in every
   single paired comparison (59-76 G w/s staged against 64-87 G w/s unstaged, across four sweep
   passes and three thread counts). The `__syncthreads` and the copy cost more than the L1/L2 hits
   they replace — the same 5 KB is already resident and already broadcast."* (Do **not** copy the
   `expdec` activation-staging idea into a single-activation GEMV; it only paid where one expert
   serves several rows — see `docs/EXPERTDEC.md:14-18`.)
3. **A constant-memory code *table* to replace shifts/masks/I2F: measured NEUTRAL.**
   `src/kernels/cuda/s2_gemv_fast.cu:15-17` — *"Measured NEUTRAL: the two orderings swap between
   passes with no consistent winner."* The whole file is titled *"THIS FILE IS NAMED `fast` AND IS
   NOT FASTER. DO NOT ADOPT IT AS THE FAST PATH."*
4. **Persistent grid + resident codebook did *not* help IQ2_XXS.** Persistence is enabled only for
   codebooks of 8 KiB+, explicitly because the small-codebook types regressed with few active
   experts: `src/dsv41/ds4_iq.cu:523-530`, `git a70b0069` *"persistent work-item grids only for
   codebooks of 8 KiB+ (IQ1_M/IQ1_S/IQ2_S), lazy codebook load, small-codebook types keep the release
   grids and kernels (no regression with few active experts)"*. IQ2_XXS's table is 2 KB — below the
   bar. Do not expect a resident-codebook win for IQ2_XXS.
5. **Two prompt-side integer attempts, both slower, not shipped.** `docs/release-notes-0.2.2.md:37`:
   *"Two attempts to speed up the prompt (a faster dp4a in llama.cpp's MMQ kernels, which this card
   lacks, and a **register-tiled MMQ dot**) were slower and are not included."*
6. **The bigger folded table costs occupancy when the kernel is dot-bound, not decode-bound.**
   `src/dsv41/ds4_iq.cu:519-521`; and `light_kernel` uses it only for `NE <= 4`
   (`:1022 constexpr bool kF = NE <= 4`), while `light_kernel_rows` (5-16 rows, *"compute bound,
   registers tuned"*) does not (`:1034-1041`).
7. **The int8 fused path was rejected for IQ2_XXS specifically.** `docs/DETAILS.md:1217-1218`:
   fused int8 covers IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS gate/up; *"**IQ2_XXS and IQ2_XS keep MMQ**"*, plus
   4.7 % slower on a 48-SM card. So the int8-MMA route is proven for the *family*, not for our type.
8. **Their own honesty about the noise floor**, worth quoting when we compare against any of the
   numbers below: `src/kernels/cuda/s2_gemv_fast.cu:19-23` — *"run-to-run variance on this machine is
   now ~20 % within a single configuration ... The noise floor now EXCEEDS the effect sizes being
   chased."* (Same lesson as our shared-H200 rule.)

## Answers to the lane's six questions

**1. Decode-size matvec against IQ2_XXS: which kernel, how the table is accessed, how many weights a
thread decodes.** Two shapes of answer, same structure.

- Shipped engine (`src/kernels/cuda/iq_kernels.cu:913-948`, `Split<16>`):
  `mmvq_multi_kernel` / `native_gu_multi_kernel` / `expdec_gu_kernel`. The 2 KB `iq2xxs_grid`
  (256 × uint2) is **staged into `__shared__` once per block** by `stage_iq_grid<16>` (`:1804-1813`;
  `kStageIqGrid<16> = true` at `:911`, unconditional on CUDA), i.e. `LDS.64` per grid entry, **not**
  `__constant__` and not `__ldg` (the `__ldg` form is only used for the Q2_0/trellis loaders,
  `:638`). The non-staged fallback reads it as global `const uint2*` (`vec_dot_iq2_xxs_q8_1`,
  `:117`). **One thread decodes 32 weights** into `struct W { int g[8]; int ls; float dw; }` =
  8 registers of four signed int8 magnitudes + one 4-bit scale + one fp32 `d`.
- DeepSeek engine (`src/dsv41/ds4_iq.cu:610-671`, `mv_kernel` `:909-936`): `__shared__ tab`, loaded
  once per block (lazily, `:628`), **one thread = one 32-value sub-block per step** (4 grid lookups
  → 8 codes each → 32 int8 in `Sub::q[8]`, `:173-184`).

**2. What accumulates.** Integer. `__dp4a` into int32, and **yes — the magnitude scale is multiplied
in the integer domain**: `float v = __fmul_rn(s.d, (float)(s.s0*lo + s.s1*hi))` with
`s.s0 = s.s1 = 2*(aux>>27)+1` (`ds4_iq.cu:367,182`). The shipped engine keeps llama.cpp's
`sumi = sumi * ls / 8` (integer multiply **and integer divide**) for bit-compatibility with the
MMQ oracle (`iq_kernels.cu:944,954`) — and `ds4_iq.cuh:11-13` records that this truncates and is the
one thing their rewrite fixes. No fp8 anywhere on these paths. fp16 `HFMA2` appears only in the
Q2_0 weight-stationary down kernel, where the half2 sums are exact integers ≤ 2048
(`moe_ws_down.cu:32-34`), and in the trellis WMMA path where the activation is split `x = hi + lo`
in two fp16 for ~2⁻²² (`ds4_tqw.cuh:4-6`). Activation scheme = **int8 per 32 values with an fp32
scale** (`sc = max|x|/127`, round-half-even); recorded accuracy caveat = the tolerances in
technique 2 (1e-5 decode / 1e-6 down / 1e-2–3e-3 prompt tiles), plus the explicit statement that
weights are exact and only the activation is quantized.

**3. `mmq_tab` / `fused_iq` / `tqw`.**
- **`mmq_tab` is not a table-optimized MMA.** It is llama.cpp's MMQ outer scheduler with a
  **per-expert pointer table** (`WeightTab` of 16 `const char*` args) so the prompt MMQ reads experts
  in their cache slots instead of a gathered copy. Layout: nothing changes; a zeroed guard of
  `kSlotGuard` bytes after each matrix replaces the tail memset (over-read: 0 / 72 / 136 B by type,
  `tests/cuda/prefill_mmq_tab_test.cu:395-404`). Dependent step removed: the gather copy. Bitwise
  identical to the gathered path; costs **36-38 MiB of VRAM per GPU** (so it is off by default,
  `docs/DETAILS.md:1012`) and measures **-18 to -24 % "read in" time** on 4×RTX 3090
  (`docs/DETAILS.md:1423`). Files: `src/prefill/moe_mmq_tab.cuh:1-25`, `moe_mmq_tab.hpp`.
- **`fused_iq` is the fused int8 prefill kernel** — see technique 8. Layout produced: per decoded
  weight row per stage, `WLD = 80` bytes = **64 int8 magnitudes (signs folded) + 16 B of scales,
  padded to a conflict-free stride**, double-buffered in shared; activations `cp.async` 4 stages deep
  as q8_1-ish 80 B per 64 values; consumed by `ldmatrix` + `mma.sync m16n8k32.s8`. Dependent step
  removed: the per-tile dequantize-to-float that MMQ does once per tile instead of once per pass.
- **`tqw` is a genuine wider-word repack**, for the trellis types: layout `words[tile][P][K][32 lanes]`
  so **a lane's 16 steps of every 128-weight block are one contiguous, fully coalesced 128-bit
  window**, which turns the trellis's cross-lane bit surgery into `__shfl_xor(...,16)` + two windows.
  No extra bytes (reorder only). Codebook = an 8 KB `__shared__` `half2` table indexed by a hash of
  the state, or *no table at all*: `step_words` CODE 1 computes the codebook output with
  two multiply-add-mask-xor ops (`ds4_tqw.cuh:27-37`). Files: `src/dsv41/ds4_tqw.cuh:1-10,42-50,101-124`,
  `tests/dsv41/tqw_hip_test.cpp:5-6,105-131` (bench: 12 gate/up matrices of 2304×5120, GB/s of weight
  bytes). **The transferable form of our measured "plane rep" win is this: reorder, don't add bytes.**

**4. `iq_expert` at decode.** One launch per product (`gate_up`, `down`) covering all active experts
via `grid.y = groups`; a warp per weight row; the *entries* (tokens routed to that expert) are a
template parameter `NE ∈ {1,4,8,16}` chosen at runtime, so M=1 costs exactly one dot per sub-block.
No row alignment or padding anywhere in the decode path; empty plan entries exit before the codebook
copy. Padding is only an issue in the prefill GEMM tiles, and their answer is the ≤16-row "light"
split (`ds4_iq.cu:939-947`, `kLightRows = 16`, `SetArgs::skip_le`) plus, in the prefill engine,
*"an expert's rows are its own count, not the group's largest"* (`moe_ws.hpp:6-8`). What they say
about the waste: *"the weights read once at the memory's speed, where a 64-row GEMM tile did 64 rows'
work for them"* (`ds4_iq.cu:942-943`). This is the same conclusion we reached with our decode GEMV.

**5. Numbers found, with their source line. None of these were measured by me.**

| number | what | source |
|---|---|---|
| `IQ4NL 0.2032 ms / Affine 0.0960 ms = 2.12x` | divergent `__constant__` codebook vs shared, IQ4_XS GEMV 10240×2560 | `src/kernels/cuda/s_gemv.cu:26-28` |
| `66 G weights/s = 0.55 weights/cycle/SM`, `~13 instructions per weight`, `~7.2 instr/cycle/SM` vs a 4/cycle issue ceiling → **instruction issue is the limit** | decode GEMV of 2-bit weights, naive 1-thread-per-row | `src/kernels/cuda/s2_gemv_quads.cu:3-6` |
| `10.5 → 86.2 G weights/s` | amortising loads over 4 elements (2 → 0.5 memory ops/element) | `src/kernels/cuda/s2_gemv_fast.cu:7` |
| `59-76 staged vs 64-87 unstaged G w/s`, 4 passes × 3 thread counts | activation staging in shared: **worse** | `src/kernels/cuda/s2_gemv_fast.cu:11-13` |
| `±20 %` run-to-run variance; "the noise floor now exceeds the effect sizes" | their measurement honesty note | `src/kernels/cuda/s2_gemv_fast.cu:19-22` |
| IQ1_M layer **−15 % sm_120 / −30 % sm_86** | folded 4096-entry code table, bit-identical | `git 71abb6c1` |
| IQ2_XXS gate/up, 20 experts, **T=1: 42 us (0.98x with EXPDEC)**, T=4: 97 us (0.84x), RTX 2080 Ti | their best IQ2_XXS decode table (10 experts/call ⇒ ≈4.2 us/expert ≈ **200 GB/s**, on a 448 GB/s card ⇒ **~2.2x off DRAM floor**) | `docs/EXPERTDEC.md:80-88` |
| IQ3_S gate/up 72 → 0.73x; IQ3_XXS 61 → 0.79x; IQ2_XS/S 60-62 → 0.75x (T=1) | activation staged in shared, same bits | `docs/EXPERTDEC.md:81-88` |
| per-expert / per-entry `5.20 / 1.26` → `3.73 / 0.64` us | EXPDEC, RTX 2080 Ti, IQ3_S | `docs/EXPERTDEC.md:93-94` |
| gate/up kernel 1.60→1.42 ms (T=1, **−11 %**), slope per row 0.86→0.60 ms; end-to-end **+1.1 %** decode | in-engine nsys, 131k decode | `docs/EXPERTDEC.md:97-105` |
| `ws / per-group MMQ = 0.26-0.33` at 2.5 rows/expert, `0.46-0.52` at 10, `0.67-0.76` at 35, `0.75-0.82` at 160 | weight-stationary IQ2_XXS gate/up, GP100 | `src/prefill/moe_ws_gu_iq2.cu:9-11` |
| `0.24-0.29` at 2.5 rows/expert … `0.61-0.74` at 160 | weight-stationary down (Q2_0/IQ4_NL), GP100 | `src/prefill/moe_ws_down.cu:7` |
| `1.92x / 1.37x / 1.30x` the MMQ path at 2048/4864/8192 tokens | fused int8 layer, Q2_0/Q2_0 | `docs/DETAILS.md:1075-1076` |
| `4.7 % slower on 20K prompts` (48-SM card); IQ2_XXS/IQ2_XS keep MMQ | fused int8 **not** adopted for IQ2_XXS | `docs/DETAILS.md:1217-1218` |
| `STRATA_MMQ_DIRECT`: read-in 331→251 ms, 369→290, 773→634, 838→675; costs 36-38 MiB/GPU | pointer-table MMQ | `docs/DETAILS.md:1423,1012` |
| P100 trellis decode **+30 %** (38.5/45.6/42.7 tok/s) from 4 KB codebook in shared + 8-lane rows + staged activations | the codebook-in-shared pattern, measured end to end | `docs/release-notes-0.2.2.md:37` |
| decode `168.8 → 192.0 tok/s (+13.8 %)` from "sm_120 set, wider matvec table, fused prompt experts"; `Q5_K head at 5 tokens 899 → 491 us` | per-arch rows/variant tables for the interleaved matvec | `docs/DETAILS.md:1409`, `docs/MMVQ_IL_TABLE.md:8-9` |
| IQ4_XS 2560×10240 at 2 cols `112 → 65 us`; Q6_K 12288×2560 at 4 cols `189 → 127 us` | interleaved matvec table, P40 | `docs/OLDER_GPUS.md:364` |
| dense dequant `dq_rows_kernel`: coalesced-through-shared store, R9700 8K prompt 14474→13840 ms (566→592 tok/s); `0.5 G weights/s` for the naive store form (a warp's store touched 32 cache lines) | write-side, but the same coalescing lesson | `ds4_iq.cu:1287-1289`, `git 8abe826f, 03056168` |
| CPU-lane AVX2 rows vs ggml-cpu: single token **x1.03 (IQ2_XXS) / x1.13 (IQ2_S)**, 4 rows **x2.4 / x2.3** | ggml parity A/B | `git 49c157fa` (body) |
| accuracy: worst `4.6e-08` token gate/up, `4.6e-08` down, `8.9e-03` F16 tiles, CPU lane `3.5e-08` | i-quant expert parity test | `git 49c157fa` (body) |
| over-read past a matrix that the guard must cover: **0 / 72 / 136 B** for IQ3_S / IQ4_NL / Q8_0 on sm_86 | `overread_bytes` | `tests/cuda/prefill_mmq_tab_test.cu:395-404` |

**6. Verdict on the one thing to prototype against our 18.8 us.** Prototype the
**sign-folded two-table PRMT decode feeding an integer dot**, on our existing tiling and the existing
66-byte blocks. Concretely, on the shared_up 5376×4096 tile: build `tabA[256]` (2-bit magnitude codes
— already in our LUT at offset `256*8`) and `tabB[128]` (parity-folded sign → nibble bit 2), emit
`prmt.b32 0x002B1908, 0x00D5E7F8, (tabA[idx]|tabB[aux])` per 4 values via
`tl.inline_asm_elementwise`, quantize the activation row to int8 per 32, and replace `tl.sum(w*x)`
with an int8 `tl.dot` into an int32 accumulator, then `acc.to(f32, bitcast=True) - 12582912.0` and one
`FFMA` with `d*(2s+1)/8`. Expectation: our "decode without the dot" probe (17.7 us) already says the
dot tree is only 1.1 us of the 18.8, so **the decode side (parity chain + unpack + layout) is the part
this removes**, and Strata's own numbers for exactly that lever are −15 % to −30 % on a layer
(`71abb6c1`) with **zero extra weight bytes**. Two caveats to price before starting: (a) int8
activations are a real accuracy change on an IQ2 checkpoint — Strata bars it at 1e-5 relative for
decode, which may or may not fit ours, and their weights-only-exact formulation is the fallback (keep
the PRMT decode, drop to fp32 math after it: still removes the parity chain and the unpack, and is
bit-exact); (b) `moe_fused_iq` is the int8-tensor-core precedent and it was *not* adopted for
IQ2_XXS (`docs/DETAILS.md:1217`), so treat the MMA form as the risky half and the PRMT decode as the
safe half. Also copy technique 4 (funnel-shift aligned loads) if the 1056-byte row stride shows up as
16-bit loads in the SASS.

## Sources

- Repo `Strata`, tag **v0.2.7.1**, HEAD `46c152a1310f77cd2f0ce2c7cd213962df14fe83`;
  i-quant GPU code ported from Project Maya v1.1.17 (`bsbdatascience/maya-ds4`, MIT) per
  `third_party/project-maya/NOTICE.md`; block layouts and grids from ggml `ggml-common.h`;
  MMQ mirrored from llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`
  (`src/prefill/moe_mmq_tab.cuh:9-11`).
- Decode kernels / codebook handling: `src/kernels/cuda/iq_kernels.cu` (`:108-131`, `:896-960`,
  `:1037-1085`, `:1773-1789`, `:1804-1881`); `src/dsv41/ds4_iq.cu` (`:33-63`, `:70-91`, `:98-108`,
  `:157-184`, `:352-381`, `:505-568`, `:599-671`, `:903-1041`, `:1104-1181`, `:1284-1400`);
  `src/dsv41/ds4_iq.cuh` (`:1-14`, `:33-64`, `:108-126`).
- PRMT decode: `src/prefill/moe_ws_gu_iq2.cu` (`:1-22`, `:34-35`, `:125-280`);
  `src/prefill/moe_ws.cuh` (`:11-28`); `src/prefill/moe_ws_down.cu` (`:1-34`);
  `include/strata/prefill/moe_ws.hpp` (`:1-40`).
- Int8 tensor-core i-quant: `src/prefill/moe_fused_iq.cu` (`:1-21`, `:60-115`, `:165-236`, `:243-320`,
  `:417-470`, `:578-583`); `include/strata/prefill/moe_fused_iq.hpp`;
  `include/strata/kernels/dp4a.hpp` (`:1-40`).
- Pointer-table MMQ: `src/prefill/moe_mmq_tab.cuh` (`:1-25`), `src/prefill/moe_mmq.cu`,
  `include/strata/prefill/moe_mmq.hpp`, `include/strata/prefill/moe_mmq_tab.hpp`,
  `tests/cuda/prefill_mmq_tab_test.cu`, `src/prefill/mmq_resident_sort.hpp`.
- Wider-word repack: `src/dsv41/ds4_tqw.cuh`, `src/dsv41/ds4_tq.cu` (`:1-30`),
  `tests/dsv41/tqw_hip_test.cpp`, `tools/dsv41_lab/wmma_emu.hpp` (`repack_tqw`, referenced).
- Measured dead ends: `src/kernels/cuda/s_gemv.cu:20-46`, `src/kernels/cuda/s2_gemv_fast.cu:1-26`,
  `src/kernels/cuda/s2_gemv_quads.cu:1-16`, `docs/release-notes-0.2.2.md:37`,
  `git a70b0069 / f5fc88c5 / 71abb6c1 / 0553ba6c / 1cace5c8`.
- Numbers: `docs/EXPERTDEC.md` (whole file), `docs/DETAILS.md:1012,1075-1077,1217-1218,1409,1423`,
  `docs/MMVQ_IL_TABLE.md:1-30`, `docs/OLDER_GPUS.md:256,364`, `docs/release-notes-0.2.2.md:37`,
  `docs/release-notes-0.2.4.md:14-16,29`.
- Tests worth reading: `tests/dsv41/iq_expert_test.cpp` (tolerances + `--bench`/`--bench-big`
  harnesses), `tests/dsv41/gg_dot_bench.cpp` (CPU-lane A/B + double-reference pass criterion),
  `tests/cuda/prefill_fused_iq_test.cpp` (int8-vs-MMQ error rule: ≤1.5× MMQ's RMS, ≤2× its worst row),
  `tests/cuda/prefill_ws_parity_test.cpp`, `tests/hip/intrinsics.cpp` (`__dp4a`/`__vsub4`/`__vcmpne4`
  bit-parity across vendors — the reason the same source builds for NVIDIA and AMD bit-identically).
- No PR/issue URLs were cited by the code I read beyond the internal references
  `#136` (fused int8), `#3752` (weight-stationary), `#242`, `#778`, `#3453`, `#3779`, `#3935`.
