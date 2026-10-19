# 18 · Design for Hopper support

**What you'll learn:** the whole plan, phase by phase: what each phase changes, why it's designed that way, what it costs, and what "done" means. Note 20 turns this into PR-sized tasks for the implementing agent.

---

## 0. Principles

1. **Reuse what vLLM has.** Humming, Marlin, DeepGEMM and CUTLASS already run every number format here on Hopper. Don't port b12x.
2. **Smallest correct step first.** Load → correct → small → fast. Each phase is useful alone.
3. **Never materialize big copies.** FP4 weights stay FP4 in VRAM. No FP8 or BF16 expert copies.
4. **Math changes need evals.** Phases A to C don't change any number the kernel sees (bit-identical to an uncompressed checkpoint). Only Phase D changes the math.

## Phase 0: baseline (no code)

**Goal:** know what "correct" and "fast" look like before changing anything.

1. Run `nvidia/Qwen3.8-Flash-Next-NVFP4` (vanilla) on **1 × H200**, and `nvidia/GLM-5.3-Flash-NVFP4` on **2 × H200**, with upstream vLLM.
2. Run each twice: default backends (Humming), then `--moe-backend marlin --linear-backend marlin`.
3. Record:
   - which backends the log picked;
   - GSM8K (5-shot);
   - reference logits for KLD (note 21);
   - `vllm bench serve` throughput at 1 and 32 users;
   - peak memory.
4. Any crash or wrong answer here is a **Hopper bug in existing code**. File it (with the duplicate check), because FP4-Local will hit the same bug.

**Done when:** both vanilla checkpoints run with at least one backend, and the numbers are saved.

## Phase A: load FP4-Local (decode CSF at load time)

**Goal:** `vllm serve nvidia/Qwen3.8-Flash-Next-FP4-Local` and `nvidia/GLM-5.3-Flash-FP4-Local` load and run on Hopper (and on every other GPU) with upstream vLLM.

### A1 · Find every shard the index lists

- **Where:** `DefaultModelLoader._prepare_weights` (`default_loader.py:147`).
- **What:** after the root glob, always add the files named in `model.safetensors.index.json`'s `weight_map` (deduplicated), not only when the glob is empty. Download them too, when the model comes from the Hub.
- **Why:** Qwen keeps all shards in `tensors/`; GLM keeps `native-QB.safetensors` in `tensors/` *next to* top-level shards. PR #60646 fixes only the first case.

### A2 · Slab CSF decode

- PR #60646 (`vllm/model_executor/model_loader/csf_scales.py`): wrap the weight iterator; when both `.nvfp4_csf_fixed` and `.nvfp4_csf_exceptions` of a scale have arrived, yield one decoded `<name>` (E4M3) tensor. Downstream code sees an ordinary checkpoint.
- **Help it land; don't duplicate it.** Review it, test it on Hopper, and build A3 on top of it, with the author's agreement.

### A3 · `csf_32x2_v1` decode

- Same file. Collect the five `.csf32_*` streams per scale name. `rows = len(bases)`, `cols = 2 · len(codes) / rows`.
- Vectorize (torch on the CPU): precompute the block order once per shape as an index tensor, scatter the nibbles in one operation, add bases, then apply exceptions page by page (or fully vectorized: expand each page's prefixes to a per-record block index with `repeat_interleave`).
- Reference: `scripts/csf32_decode.py` (simple loops, verified). The vectorized version must match it.

### A4 · Validate `weight_scale_encoding`

- In `ModelOptMixedPrecisionConfig.__init__` (`modelopt.py:1572`): accept a missing value, `"csf"` and `"csf_32x2_v1"`; raise a clear error for anything else ("unknown NVFP4 scale encoding X; upgrade vLLM").

### Costs and limits

- CPU decode at load: Qwen has 73,728 scale matrices (7 GiB decoded), GLM 36,288 (17.7 GiB). Vectorized decode runs at GB/s, so it adds seconds to load time, not minutes. Measure it.
- VRAM holds **raw** scales: +3.3 GiB (Qwen), +8.1 GiB (GLM) compared with the community stack. Phase C wins that back.

**Done when:** both models load on H200 and on any Blackwell GPU, the decoded scales are bit-identical to the reference decoders, and GSM8K is within noise of Phase 0's vanilla numbers.

## Phase B: make every layer correct on Hopper

**Goal:** correct output for every layer type in note 16's tables, with both Humming and Marlin.

The test matrix:

| Axis | Values |
|---|---|
| Model | Qwen FP4-Local (TP 1), GLM FP4-Local (TP 2, and EP 2) |
| Backend | default (Humming), `--moe-backend marlin --linear-backend marlin` |
| Features | MTP speculative decoding on / off; image input on / off; prefix caching; CUDA graphs on / `--enforce-eager` |
| Checks | GSM8K; KLD of Humming vs Marlin (≈ 0); KLD vs Phase 0 vanilla (small) |

Likely problem spots (note 16, section 4): small-N MXFP8 layers (`in_proj_a/b`, N = 48), TP-split expert widths, the open Marlin MoE bugs, fused shared experts, and the BF16 tensors from `native-QB`.

Each bug found = one small PR with a unit test that fails before the fix.

**Done when:** every cell of the matrix passes, or has a filed issue with a workaround.

## Phase C: keep CSF compressed in VRAM

**Goal:** VRAM holds the CSF streams, not raw scales. Saves about 3.0 GiB (Qwen) and 7.3 GiB (GLM) after counting scratch.

### C1 · The layout problem

After loading, Humming and Marlin **rewrite** the scales into their own layouts (note 15, step 6). Marlin, for example, permutes them and re-encodes E4M3 as S0E5M3. So a runtime decoder must produce *the kernel's* layout. The plan:

1. Start with **Marlin**, whose scale transform is simple:
   - a fixed **permutation** that depends only on the matrix shape. Precompute one small index map per shape (gate/up and down), shared by all experts and layers: about 0.8 MB for Qwen `w13`.
   - a **per-byte** E4M3 → S0E5M3 transform, done with a 256-entry lookup table. Fold any per-layer rescale factor into the table at load.
2. Then **Humming**: read `humming/transform.py` to get its scale layout; it may need its own index map. ❓ Open question for jinzhen-lin.

### C2 · Decode only what's needed

Decoding *all* of a layer's scales on every step is cheap for prefill but **far too slow for decode**:

- Qwen, per layer: read about 80 MB of CSF, write 150 MB → about 48 µs at 4.8 TB/s → about 2.3 ms per forward pass over 48 layers.
- But a batch-1 decode step reads only about 25 MB of expert weights per layer (10 routed experts). Decoding all scales would cost 10 × more memory traffic than the weights themselves.

So follow b12x (note 14):

| Routed (token, expert) pairs in the step | Decode |
|---|---|
| few (decode) | only the routed experts, skipping duplicates |
| many | mark the active experts, then decode those |
| ≥ 16 × number of experts (long prefill) | all experts |

### C3 · Scratch and wiring

- **One scratch buffer per scale shape**, sized for one layer's raw scales (Qwen ≈ 150 MB, GLM ≈ 0.42 GiB), shared by all layers of that shape. Double-buffer it (×2) if you overlap the next layer's decode with the current layer's compute.
- At load: keep the CSF streams as layer buffers; point the kernel's `w13_scale` / `w2_scale` at the shared scratch.
- At forward: before the MoE kernel, call `decode_scales(layer_csf, topk_ids, scratch)`. Same CUDA stream, so ordering is automatic.
- Register the decoder as a **custom op**, so `torch.compile` treats it as opaque and CUDA graphs capture it. `topk_ids` already has a static shape per graph size.
- Expert slots that weren't decoded hold stale data. That's fine, because the kernel only reads the routed experts. Add a debug mode that fills the scratch with NaN first, to prove it.

### C4 · Tests

- Bit-exactness: for random `topk_ids`, the decoded slots equal Phase A's load-time scales after the kernel's transform.
- End to end: same GSM8K / KLD as Phase A, since it's bit-identical math.
- Memory: `torch.cuda.max_memory_allocated` drops by the expected amount.
- Speed: decode-step latency is within a few percent of Phase A.

**Done when:** an opt-in flag (for example `--model-loader-extra-config '{"keep_csf_scales": true}'`, name to be agreed in review) gives bit-identical output, the memory saving, and no meaningful slowdown.

## Phase D: W4A8 prefill (FP8 math), optional

**Goal:** faster long prefills on Hopper by using FP8 tensor cores for the expert GEMMs, with weights still FP4 in VRAM. Math explained in note 17.

### D0 · Benchmark gate (do this first)

Humming already runs **MXFP4 × FP8** on sm_90 (`(kMxfp4Static, kFp8Dynamic128Sym)` in `fused_humming_moe.py`, "Runs via WGMMA software dequant, so it works on Hopper"). Its speed is a good upper bound for NVFP4 × FP8.

1. Make synthetic MXFP4 expert weights of Qwen's and GLM's shapes.
2. Time Humming MoE W4A16 vs W4A8 at 80, 160, 320 and 455 tokens per expert.
3. Estimate the end-to-end prefill gain (the expert GEMM share of prefill time × the speedup).

**Stop if** the end-to-end gain is under about 15 %. Write the numbers up; that's a useful result too.

### D1 · Kernel: "fused E4M3 residual scale" in Humming

- Add a weight transform beside `use_fused_e8m0_scale`: store per-row exponent `E[n]` (int8) and the E4M3 group scales divided by `2^E[n]`.
- In the unpack: FP4 → FP16, × residual (FP16), → E4M3 (round to nearest even).
- In the epilogue: × `2^E[n]` (or fold it into the per-128 promotion).
- Allow this mode past the group-size rule (`humming/schema/humming.py:581`, `kernel/humming.py:396-415`).
- **Talk to jinzhen-lin first** (Humming issue or Slack): the design, the API name, whether they'd accept it.

### D2 · vLLM wiring

- An opt-in setting for NVFP4 MoE on sm_90: W4A16 for small batches, W4A8 when tokens per expert exceed a threshold. Both configs must share the same packed weights in VRAM (no second copy). ❓ Check whether Humming's sm_90 packed layout can serve both 16-bit and 8-bit activations.
- Off by default until evals pass.

### D3 · Accuracy validation

- KLD of W4A8 vs W4A16 on both models (note 21), and GSM8K. Compare with the "acceptable" range used for FP8 KV-cache or W8A8 changes in vLLM.
- Run a quality report like b12x's `nvfp4_mx_residual_quality_report` on the real checkpoints: share of rounded weights and worst-case errors per layer.

**Done when:** a measured prefill speedup, an eval table, and a maintainer-approved flag.

## Non-goals (and why)

| Not doing | Because |
|---|---|
| Port b12x to Hopper | every kernel would need a rewrite; vLLM's Hopper kernels already cover these formats |
| W4A4 emulation | no FP4 tensor cores; emulation is slower than W4A16 |
| Store FP8 copies of the expert weights | doubles expert memory |
| Convert NVFP4 → MXFP4 at load | changes the trained model (note 17) |
| MXFP8 W8A8 for attention on Hopper | separate optimization; not needed for correctness |
| GGUF work | separate track (G1–G4 in `issue-tracker.md`) |

## Order and dependencies

```
Phase 0 ──► A1 ──► A2 (#60646) ──► A3 ──► B ──► C (Marlin first, then Humming)
             └──► A4                       └──► D0 ──(gain?)──► D1 ──► D2 ──► D3
```

---

**Next:** [19 · VRAM budget on H200](19-VRAM-Budget-On-H200.md)
