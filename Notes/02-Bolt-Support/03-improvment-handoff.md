# 03 · Handoff: make the non-SM12x ModelOpt IQ2_XXS / Q8_0 path production-ready

Written 2026-10-22, for the implementer working on the work box (H200 + RTX 4000 Ada).

> **Status 2026-10-27:** Phases 1–2 are done and Phase 3 is started.
> - The code is on the fork branch `feature/modelopt-block-triton`: the draft rebased, then the rewrite on top.
> - Start from that branch, not the `2064cc2efb` steps in §4.2.
> - Read `04-progress-log.md` first. It has the current hashes, what changed from this plan and why, the 5060 Ti numbers, one open conflict with §0.2.2, and the ordered H200 to-do list.

- **Start point:** commit `2064cc2efb` on branch `feature/wip-nvidia-bolt-low-sm-support` in the fork `blazingbhavneek/vllm`. This doc calls that commit **"the draft"**.
- **Upstream issue:** vllm-project/vllm#62609 (Bolt can't run on A100/H100/H200 because these formats require b12x on SM12x). The user has claimed it.
- **Goal:** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` serves fast and correctly on H200 (and any NVIDIA GPU from sm_80 up) with vLLM defaults. Weights stay packed. The result is a PR vLLM maintainers can merge.
- **Companion files in this folder:**
  - `review.md`: what's wrong with the draft. Findings have IDs: B1–B4, K1–K5, S1–S4. This doc refers to those IDs.
  - `kernel-tests/proto_kernels.py`: **verified starting code** for the new kernels: the repack, the decode helper, a `tl.dot` GEMM and a split-K GEMV. Checked bit-exact on sm_120.
  - `kernel-tests/ref_decode.py`: a pure-torch GGML reference decoder plus the IQ2_XXS grid table. Checked bit-exact against gguf-py (llama.cpp).
  - `kernel-tests/e2e_tiny_bolt.py`: a tiny Nemotron-H model run end to end through `LLM()` in eager, compiled and CUDA-graph modes.
  - `kernel-tests/harness.py`, `test_correctness.py`, `test_moe.py`, `bench.py`: run the **draft's** kernels standalone, for baseline numbers.

---

## 0. Read this first

### 0.1 How to use this document

1. Do the phases in order: §4 → §9. Each phase ends with a **Gate**, a list of commands and pass criteria. Do not start the next phase until the gate passes.
2. When something fails, look it up in **§10 Troubleshooting** before improvising.
3. Keep a log in `Notes/02-Bolt-Support/04-progress-log.md`, one entry per experiment, in this shape (the format of vLLM's `kernel-microbenchmark` skill):
   ```
   ### <date> <short title>
   Question: ...      Change: ...      Correctness: <test + result>
   Result: <numbers, with the exact command>      Observation: ...      Next: ...
   ```
   Every number that ends up in the PR must be in this log, with its command.
4. Code in this doc is a skeleton. Names, signatures and the important lines are exact; fill in the obvious parts. Do not redesign it. If something here is wrong, write down why in the log, make the smallest fix, and keep going.

### 0.2 Hard rules

Breaking any of these gets the PR rejected or breaks users.

1. **No change to b12x behavior on SM120/SM121.** With `auto` backends and b12x available, SM12x must take exactly the same path as today. The existing tests (Gate 2) prove this.
2. **Weights stay packed.** Never create a BF16/FP16 copy of a weight matrix: not at load time, not per forward, not "only for prefill", not for the embedding table. Rearranging the packed bytes once at load time is allowed and required (§3.3), as long as the total size stays the same.
3. **No CPU sync and no CPU→GPU copy in forward.** That means no:
   - `.item()`, `.tolist()`, `.cpu()`
   - `torch.unique`, `nonzero`, boolean-mask indexing
   - `if <tensor>:`
   - `torch.tensor(python_list, device=...)`
   - device tensors created lazily in forward

   Every lookup table is created in `process_weights_after_loading`.
4. **No new env vars**, and no module-level mutable state: no `_CACHE = None` globals flipped at runtime, no `threading.Lock`.
5. **No `try: import triton`.** Use `from vllm.triton_utils import tl, triton`.
6. **No Python branching on token count in code that `torch.compile` traces.** Put it inside a custom op (§3.4). vLLM traces once and drops all guards, so such a branch freezes at whatever token count was traced (review B2, confirmed in real vLLM).
7. **Every Triton config must fit 99 KB of shared memory**, unless it is only chosen when the device reports more (`torch.cuda.get_device_properties(dev).shared_memory_per_block_optin`).
8. **Unsupported combinations fail in `__init__` with a clear `ValueError`**, never at the first forward.
9. **Work-box etiquette:**
   - A protected server runs on **port 8000** with served name **`gemma-4-31B`**. Never stop it, never bind port 8000, never reuse that name.
   - The GPU is **shared with other tenants**. Check free memory before every launch (§4.1).
   - Benchmark A and B back to back, and note the contention.
10. **Follow `vllm/AGENTS.md`.**
    - Python only through `uv` and the repo `.venv`, never system `pip`.
    - Run `pre-commit`.
    - The PR description must disclose AI assistance.
    - Read each vLLM skill before the phase that needs it:

    | Skill | Read before |
    |---|---|
    | `.agents/skills/triton-kernel-writing/SKILL.md` | Phases 1 and 3 |
    | `.agents/skills/kernel-microbenchmark/SKILL.md` | Phase 3 |
    | `.agents/skills/model-microbenchmark/SKILL.md` | Phase 3, optional |
    | `.agents/skills/pr-checklist/SKILL.md` | Phase 5 |
    | `.agents/skills/debug-ima/SKILL.md` | Any illegal memory access |

### 0.3 Reading list (≈30 min, in this order)

1. `review.md` (this folder): the findings table.
2. `vllm/model_executor/layers/quantization/modelopt_block_quant.py` at the draft: the two quant methods you'll change.
3. `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`: `TritonExperts` and **`TritonWNA16Experts` (line ~637)**. Your MoE class is a near-copy of `TritonWNA16Experts`.
4. `vllm/model_executor/layers/quantization/utils/marlin_utils.py` around line 470–500: how `marlin_gemm` is registered as a custom op with a fake impl. Copy that pattern.
5. `kernel-tests/proto_kernels.py`: the kernels you start from.

---

## 1. Background

### 1.1 Bolt's quantized layers

Checkpoint: 22.43 GiB on disk, ModelOpt `MIXED_PRECISION` (producer modelopt 0.37.0). These counts come from its `config.json` on Hugging Face (fetched 2026-10-22).

| Format | Bolt tensors | Who runs it |
|---|---|---|
| **IQ2_XXS** (66 B per 256 weights, 2.06 bit) | routed experts `experts.N.up_proj/down_proj` (27,648 tensors in 40 MoE layers); `shared_experts.up_proj/down_proj` (40 + 40) | **this work** |
| **Q8_0** (34 B per 32 weights, 8.5 bit) | attention `q/k/v/o_proj` (8 layers); `fc1_latent_proj`, `fc2_latent_proj` (40 + 40); `lm_head` | **this work** |
| W4A16_NVFP4 | mamba `in_proj`, `out_proj` (40 + 40) | existing FP4 Marlin path, out of scope |
| BF16 | embeddings, vision tower (RADIO), norms | unchanged |

Text model (`llm_config`, architecture `NemotronHPuzzleForCausalLM` inside `NemotronH_Omni_Reasoning_V3`; vLLM maps it to `nano_nemotron_vl` + `nemotron_h`):

- hidden 4096
- 88 layers: 40 mamba + 40 MoE + 8 attention
- vocab 131072
- attention: 32 heads, 2 KV heads, head dim 128
- MoE: latent (experts work in a **1024**-dim latent space), ReLU² (non-gated), sigmoid top-**9** routing, 1 shared expert (intermediate 5376)
- per-layer expert count E ∈ {240, 304, 368, 432, 496}, intermediate I ∈ {1536, 2048, 2304, 2560}; the full list is in Appendix C

### 1.2 Block formats, byte level

**Q8_0**, 34 bytes per 32 weights:
- bytes 0–1: fp16 `d`
- bytes 2–33: int8 `q[0..31]`
- `w[i] = d * q[i]`

**IQ2_XXS**, 66 bytes per 256 weights:
- bytes 0–1: fp16 `d`
- then 8 **records** of 8 bytes; record `r` covers weights `32r … 32r+31`:
  - bytes 0–3: four grid indexes `idx[g]` (uint8), one per group of 8 weights, g = 0..3
  - bytes 4–7: little-endian uint32 `aux`:
    - bits `7g … 7g+6`: 7-bit sign index `s[g]`
    - bits 28–31: 4-bit scale `ls`
- record scale `rs = d * (0.5 + ls) * 0.25`
- 8-bit sign mask `m[g] = s[g] | (parity(s[g]) << 7)`. This is ggml's `ksigns_iq2xs`, computed instead of stored.
- weight `w[32r + 8g + j] = rs * grid[idx[g]][j] * (−1 if (m[g] >> j) & 1 else +1)`, for j = 0..7
- `grid` = ggml's `iq2xxs_grid`: 256 rows × 8 bytes, values ∈ {8, 25, 43} (Appendix A)

IQ2_XS (74-byte blocks) exists in the format but Bolt doesn't use it. **Leave IQ2_XS b12x-only.**

### 1.3 How vLLM runs these today

`vllm/model_executor/layers/quantization/modelopt_block_quant.py` has two methods:

- `ModelOptBlockQuantLinearMethod`: dense layers, `lm_head`, Q8_0 embeddings. Uses the b12x dense API.
- `ModelOptBlockQuantMoEMethod`: routed experts. Uses `B12xExperts` inside `mk.FusedMoEKernel`.

Upstream, both raise "require b12x on SM12x" everywhere else. That's #62609.

### 1.4 The draft, and why it isn't mergeable

The draft adds `modelopt_block_quant_sm90.py` (250 lines) and `modelopt_block_quant_hopper.py` (900 lines), gated on SM90 only. Its math is right: bit-exact against gguf-py. But:

| ID | Problem |
|---|---|
| B1 | It imports the IQ2 grid from b12x's private `_lib`. This works today only because `flashinfer-python==0.7.2`, a core vLLM CUDA dependency, bundles a top-level `b12x`. It's still a private experimental module. Embed the table. |
| B2 | `torch.compile` freezes its token-count dispatch (`if M == 1`, split-K choice) at trace time. |
| B3 | Configs need 104–160 KB of shared memory, so they crash on 99 KB GPUs (sm_86/89/120). |
| B4 | The IQ2 table is created lazily inside forward, so CUDA-graph capture crashes when IQ2 sits on a traced layer. |
| K1 | Decode does per-weight work that belongs to the 8- or 32-weight group, so it's ALU-bound. |
| K2 | Register spills on sm_90a. |
| K3 | The 66/34-byte layout only allows byte loads. |
| K4 | MoE tile fixed at BM=16. |
| K5 | Embedding decodes the whole table every step (1 GiB scratch at Bolt size). |
| S1 | int32 offsets. |
| S2 | Unsupported combos fail late. |
| S3 | `--moe-backend` / `--linear-backend` silently ignored. |
| S4 | Env var, globals, dead code. |

### 1.5 Which Bolt layers run inside `torch.compile`

- `nemotron_h.py:228–252` passes `fc1_latent_proj`/`fc2_latent_proj` (as `routed_input_transform` / `routed_output_transform`) and `shared_experts` into the FusedMoE layer. So they execute **inside the MoE custom op, which runs eagerly**.
- `lm_head` runs in `compute_logits`, outside the compiled graph.
- So in Bolt only the **8 attention `qkv_proj`/`o_proj` (Q8_0)** layers are traced. That's why the draft "worked" for Bolt despite B2/B4.
- Other checkpoints will put IQ2 on traced layers. The tiny e2e model does exactly that on purpose.

---

## 2. Definition of done

### 2.1 Checklist (all must be true before opening the PR)

**Functional**
- [ ] Bolt serves on H200 with defaults (torch.compile + CUDA graphs, no `--enforce-eager`, no backend flags).
- [ ] The startup log shows `Using Triton kernels for ModelOpt IQ2_XXS …` and `… Q8_0 …` (§3.2).
- [ ] Same code passes the kernel tests and the tiny e2e on the RTX 4000 Ada (sm_89, 99 KB shared memory).
- [ ] With b12x available, SM12x with `auto` still uses b12x (existing tests pass). `--linear-backend triton --moe-backend triton` forces the new path there.
- [ ] Unsupported requests fail in `__init__` with clear messages:
  - IQ2_XS off SM12x
  - `--linear-backend marlin` (or any other non-matching backend)
  - `--moe-backend b12x` on H200

**Memory**
- [ ] "Model loading took X GiB" is within +0.1 GiB of the draft's number. The draft keeps raw blocks, so it's the right reference.
- [ ] No weight-sized buffers in forward. Embedding scratch is O(tokens × hidden), not O(vocab × hidden).

**Numerics**
- [ ] Kernel tests: decode bit-exact against the reference; GEMM error ≤ 1.5× the error of BF16 cuBLAS on the same dequantized weights.
- [ ] Tiny e2e: quantized eager-vs-graphs agreement is at least as good as the BF16 control's (§8.3).
- [ ] GSM8K (1319 questions, 5-shot) within **1.5 points** of the draft baseline measured the same day (§8.1).

**Performance** (H200, A/B same day): see §2.2.

**Robustness**
- [ ] `--jit-monitor-mode error` survives a varied-length load test: no Triton compiles after startup (§8.4).
- [ ] `VLLM_GPU_SYNC_CHECK=error` survives the same test (§8.4).
- [ ] The kernels used on sm_90 have `n_spills == 0`, and every config fits shared memory on sm_80/86/89/90/100/120 (Appendix D).

**PR**
- [ ] The draft's two files and its env var are gone. New non-test code is ≤ ~700 lines; the draft was 1,388 added.
- [ ] Docs updated, `pre-commit` clean, `pr-checklist` done, description complete (§9).

### 2.2 Performance targets (H200)

Two baselines, both measured by you the same day: **bf16 cuBLAS** = `x @ W.T` on the dequantized weight; **draft** = commit `2064cc2efb`.

| Case | Merge bar (must) | Goal |
|---|---|---|
| Dense, M = 1, every Bolt shape (Appendix C) | faster than bf16 cuBLAS **and** ≥ 1.5× faster than the draft | IQ2 ≤ 0.4× cuBLAS time; Q8_0 ≤ 0.6× |
| Dense Q8_0 `lm_head`, M = 1 (570 MB packed) | ≥ 55% of copy bandwidth (§7.1) | ≥ 75% |
| Dense, M = 2…64 | faster than bf16 cuBLAS for every Bolt shape | — |
| Dense, M ≥ 2048 | ≤ 2.0× bf16 cuBLAS time | ≤ 1.3× |
| MoE layer, M ≤ 8 tokens, largest Bolt layer (E=432, I=2560, top-9) | ≥ 1.5× faster than the draft's `moe_forward` | ≥ 3× |
| MoE layer, M = 4096 | ≤ 2.5× the time of vLLM's BF16 `TritonExperts` on dequantized weights | ≤ 1.5× |
| End to end, `vllm bench serve`, concurrency 1 / 8 / 32 / 128 | never slower than the draft; ≥ 1.3× at concurrency 1 | ≥ 2× |
| Startup | no runtime JIT after "Graph capturing finished"; no more than +60 s over the draft | — |

The 1.3× end-to-end bar is lower than the kernel bars because NVFP4 mamba projections, attention and the SSM are unchanged and take a large share of each step.

**What the home prototype showed** (RTX 5060 Ti, 448 GB/s, M = 1):

| Kernel | Time | vs bf16 cuBLAS | vs draft | Bandwidth |
|---|---|---|---|---|
| Q8_0 `lm_head` GEMV | 1.40 ms | 2.6 ms | 3× faster (draft: 4.3 ms) | 91% of peak |
| Q8_0 `qkv` GEMV | 57 µs | 108 µs | — | 78% of peak |
| IQ2_XXS shared-expert up GEMV | 48 µs | 123 µs | 2× faster (draft: 96 µs) | 26% of peak |

IQ2 is limited by decode work, not memory. On H200 the IQ2 fraction of bandwidth will be lower still. That's why the IQ2 bars are relative, and why §7.4 has an IQ2-specific ladder.

---

## 3. Target design

### 3.1 Files

| File | Action | Size budget |
|---|---|---|
| `vllm/model_executor/layers/quantization/modelopt_block_quant_triton.py` | **new**. Contains: the embedded IQ2 grid, `repack_blocks`, `iq2_lut`, `_decode_tile`, `_gemm_kernel`, `_gemv_kernel`, `_moe_gemm_kernel`, `_moe_gemv_kernel` (Phase 3, if needed), `_q8_embedding_kernel`, config helpers, and two custom ops (`modelopt_block_linear`, `modelopt_block_embedding`) | ≤ 500 lines |
| `vllm/model_executor/layers/fused_moe/experts/modelopt_block_triton_moe.py` | **new**: `ModelOptBlockTritonExperts(TritonExperts)` | ≤ 100 lines |
| `vllm/model_executor/layers/quantization/modelopt_block_quant.py` | **modify**: a `_use_triton` helper, one flag per method, a Triton branch in `process_weights_after_loading` / `apply` / `embedding`. **Remove** every draft import and the `sm90_fallback` code | ≤ +70 net vs upstream |
| `vllm/model_executor/layers/quantization/modelopt_block_quant_sm90.py` | **delete** | |
| `vllm/model_executor/layers/quantization/modelopt_block_quant_hopper.py` | **delete** | |
| `tests/kernels/quantization/test_modelopt_block_dequant.py` | **delete**, replaced by `test_modelopt_block_triton.py` | |
| `tests/kernels/quantization/test_modelopt_block_triton.py` | **new**: kernel tests (§5.4) | |
| `tests/quantization/test_modelopt.py` | **extend**: selection tests plus the tiny e2e (§6.5) | |
| `benchmarks/kernels/benchmark_block_dequant_gemm.py` | **delete**, replaced by the next row | |
| `benchmarks/kernels/benchmark_modelopt_block_triton.py` | **new** (§7.1) | |
| `docs/features/quantization/b12x.md` | **modify**: add a "Other GPUs (Triton)" section (§9.2) | |

CI notes:
- `tests/kernels/quantization` runs on **L4 (sm_89, 99 KB)** in CI (`.buildkite/test_areas/kernels.yaml:440`), so CI will catch shared-memory regressions.
- `tests/quantization` runs on **H200 MIG 35 GB** (`.buildkite/test_areas/quantization.yaml`).

### 3.2 Backend selection

One helper in `modelopt_block_quant.py`, used by both methods:

```python
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_TRITON_BLOCK_CODECS = ("iq2_xxs", "q8_0")


def _use_triton(codec: str, backend: str, b12x_supported: bool, kind: str) -> bool:
    """Pick b12x (False) or the Triton kernels (True); raise if neither applies."""
    if backend in ("auto", "b12x") and b12x_supported:
        return False
    if (
        backend in ("auto", "triton")
        and codec in _TRITON_BLOCK_CODECS
        and current_platform.is_cuda()
        and current_platform.has_device_capability(80)
    ):
        logger.info_once(
            "Using Triton kernels for ModelOpt %s %s weights.", codec.upper(), kind
        )
        return True
    raise ValueError(
        f"ModelOpt {codec.upper()} {kind} weights run on b12x (SM120/SM121) or, for "
        f"IQ2_XXS and Q8_0 on NVIDIA SM80+, on Triton; the requested {kind} backend "
        f"{backend!r} is not available here."
    )
```

What the helper returns in each case:

| Situation | `auto` | `b12x` | `triton` | anything else |
|---|---|---|---|---|
| SM12x, b12x available | b12x (unchanged) | b12x | Triton | ValueError |
| SM12x, b12x missing | Triton | ValueError | Triton | ValueError |
| sm_80 … sm_100 | Triton | ValueError | Triton | ValueError |
| IQ2_XS, not SM12x | ValueError | ValueError | ValueError | ValueError |

Call sites (replace the draft's `sm90_fallback` lines):

```python
# ModelOptBlockQuantMoEMethod.__init__
self.use_triton = _use_triton(
    self.codec, moe_config.moe_backend, B12xExperts._supports_current_device(), "MoE"
)

# ModelOptBlockQuantLinearMethod.__init__
api = get_b12x_blockscaled()
b12x_supported = (
    api is not None and hasattr(api, "BlockQuantLinearWeight") and api.is_supported()
)
self.use_triton = _use_triton(
    self.codec, _get_linear_backend(quantization=self.codec), b12x_supported, "linear"
)
```

Keep every other existing check in the MoE `__init__`: BF16 activations, SiLU/ReLU² only, no bias, standard SiLU params, and **TP-only via `B12xExperts._supports_parallel_config`**. The Triton experts class has the same parallel limits (§3.7).

### 3.3 Weight layout after loading

The loaders stay exactly as they are: they write raw ggml blocks, `[N, NB, 66|34]` uint8. In the Triton branch of `process_weights_after_loading`, split each raw tensor **once** into:

| Tensor | IQ2_XXS | Q8_0 |
|---|---|---|
| `weight` (replaced) | int32 `[N, NB*16]`: the 8 records per block, as 16 words; record `rr` = words `2rr` (indexes) and `2rr+1` (aux) | int8 `[N, NB*32]` |
| `weight_scale` (new) | fp16 `[N, NB]` | fp16 `[N, NB]` |
| MoE | `w13_weight` `[E, N, NB*16]`, `w13_weight_scale` `[E, N, NB]`; same for `w2` | int8 `[E, N, NB*32]` + scales |

Total bytes are identical (64 + 2 = 66; 32 + 2 = 34). This fixes K3: payload rows are 16-byte aligned, so Triton can vectorize. It also makes MoE weights 3-D, so `FusedMoEExpertsModular.moe_problem_size` works unchanged.

Use `repack_blocks()` from `proto_kernels.py` as is.

Peak extra memory is one tensor's worth while repacking. Free the raw tensor right away with `replace_parameter`.

```python
# dense, Triton branch of process_weights_after_loading
qweight, scales = repack_blocks(layer.weight.data, self.codec)
replace_parameter(layer, "weight", qweight)
layer.register_parameter("weight_scale", torch.nn.Parameter(scales, requires_grad=False))
layer.register_buffer(
    "iq2_lut",
    iq2_lut(qweight.device) if self.codec == "iq2_xxs" else None,
    persistent=False,
)
```

`iq2_lut(device)` builds the fp32 `[256*8]` table (8 KB) from the embedded grid. Decorate it with `functools.cache`; it's only ever called here, eagerly, so every layer on a device shares one tensor. **Never call it from forward.** That fixes B1 and B4.

The dummy loader (`--load-format dummy`) fills packed weights with random bytes, so ~3% of fp16 scales are Inf/NaN. That's fine for speed benchmarks (NaN doesn't slow the GPU). For correctness tests, overwrite `*weight_scale` after load (`e2e_tiny_bolt.py: sanitize_scales`).

### 3.4 What is traced and what runs in custom ops

| Call | Traced by Dynamo? | Rule |
|---|---|---|
| `ModelOptBlockQuantLinearMethod.apply` (attention, other dense) | **yes** | Body is one call: `torch.ops.vllm.modelopt_block_linear(...)`. All token-count dispatch happens inside the op. |
| `ModelOptBlockQuantLinearMethod.embedding` | yes | One call: `torch.ops.vllm.modelopt_block_embedding(...)` |
| MoE `apply` → `FusedMoEKernel.apply` → experts | no (MoE runner custom op) | Python heuristics on M are fine |
| `lm_head` via `compute_logits` | no | still goes through the same custom op (harmless) |

Register both ops like `marlin_gemm` (`marlin_utils.py:500`):

```python
from vllm.utils.torch_utils import direct_register_custom_op


def _modelopt_block_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lut: torch.Tensor | None,
    codec: str,
) -> torch.Tensor:
    """bf16 [M, K] @ packed [N, K]^T -> bf16 [M, N]; picks the kernel from M."""
    ...  # §3.5 dispatch table


def _modelopt_block_linear_fake(x, qweight, scales, lut, codec):
    return x.new_empty((x.shape[0], qweight.shape[0]))


direct_register_custom_op(
    "modelopt_block_linear", _modelopt_block_linear, fake_impl=_modelopt_block_linear_fake
)
```

In `apply`:

```python
if self.use_triton:
    out = torch.ops.vllm.modelopt_block_linear(
        x.reshape(-1, x.shape[-1]), layer.weight, layer.weight_scale, layer.iq2_lut,
        self.codec,
    )
    if bias is not None:
        out.add_(bias)
    return out.view(*x.shape[:-1], -1)
```

The MoE `apply` needs **no** Triton branch. It already calls `self.moe_kernel.apply(...)`, and the kernel object is either b12x's or yours. Delete the draft's `if self.sm90_fallback:` lines.

### 3.5 Kernels

Start from `kernel-tests/proto_kernels.py`; it's verified. Its `_decode_tile` does:

- **Per record** `[BN, R]`: block scale `d`, `rs = d*(0.5+ls)*0.25`.
- **Per group** `[BN, R, 4]`: grid index, 7-bit sign index, parity → 8-bit sign mask.
- **Per weight** `[BN, R, 4, 8]`: `mag = lut[gidx*8 + j]` (fp32 lookup), sign select, `* rs`, reshape to `[BN, BK]`.

On sm_120 this was the fastest of three variants:

| IQ2 M=1 variant | Time | Bandwidth |
|---|---|---|
| decode-free floor | 28 µs | — |
| group words + shifts | 80 µs | — |
| **per-group math + fp32 lookup** | **48 µs** | 118 GB/s |

Dense dispatch inside `_modelopt_block_linear` is the **initial** table. Phase 3 tunes it from measurements, and you write the tuned table back here.

| M | Kernel | BM | BN | BK | warps | split-K |
|---|---|---|---|---|---|---|
| 1 | `_gemv_kernel` | – | 32 (Q8) / 8 (IQ2) | 256 | 4 (Q8) / 1 (IQ2) | pick `S` so that `cdiv(N,BN)·S ≥ 2·num_SMs`; `S` divides `K/BK`; S ≤ 16 |
| 2–16 | `_gemm_kernel` | 16 | 64 | 128 | 4 | same rule |
| 17–64 | `_gemm_kernel` | 32 / 64 | 64 | 128 | 4 | same rule |
| > 64 | `_gemm_kernel` | 128 if smem ≥ 160 KB else 64 | 128 | 64 | 8 | 1 |

Kernel rules (from vLLM's `triton-kernel-writing` skill, applied here):

- **`@triton.jit(do_not_specialize=["M"])`** on every kernel that takes M. MoE kernels: also `"EM"` and `"num_valid_tokens"`. Otherwise Triton compiles a new variant at runtime for M == 1 and M % 16 == 0.
- **int64 offsets** for anything multiplied by a row index: `offs_m.to(tl.int64)`, `offs_n.to(tl.int64)`. `lm_head` M × N passes 2³¹ at M = 16384 (S1).
- **Grid:** put the big product (M-tiles × N-tiles) in `grid[0]` and split-K in `grid[1]`. `grid[1]` and `grid[2]` must stay ≤ 65,535.
- **Split-K must be deterministic:** fp32 partials `[S, M, N]`, then `partials.sum(0).to(bf16)` (two small torch ops inside the custom op). No `tl.atomic_add`.
- **BK** is a multiple of 32 and must divide K:
  - IQ2: K is always a multiple of 256, so 32/64/128/256 all divide.
  - Q8_0: choose the largest of 256/128/64/32 that divides `K // S`.
- **Accumulate in fp32.** Convert the decoded tile to bf16 right before `tl.dot`. That rounding matches what the b12x reference path and cuBLAS see.
- `stages` = the largest of 4/3/2/1 that fits. Check it with Appendix D; don't trust estimates.

Q8_0 embedding (`_q8_embedding_kernel`):
- Grid `(T, cdiv(H, BH))`, BH = 1024.
- One program loads its row id, then decodes `BH` columns of that row: int8 × fp16 scale → bf16.
- Wrap it in the `modelopt_block_embedding` custom op (fake returns `qweight.new_empty((T, H), dtype=torch.bfloat16)`).
- This fixes K5: no full-table decode.

```python
@triton.jit
def _q8_embedding_kernel(ids_ptr, qw_ptr, sc_ptr, out_ptr, H, stride_qn, stride_sn,
                         stride_om, BH: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    mask = offs < H
    row = tl.load(ids_ptr + t).to(tl.int64)
    q = tl.load(qw_ptr + row * stride_qn + offs, mask=mask).to(tl.float32)
    d = tl.load(sc_ptr + row * stride_sn + offs // 32, mask=mask).to(tl.float32)
    tl.store(out_ptr + t.to(tl.int64) * stride_om + offs, (q * d).to(tl.bfloat16), mask=mask)
```

### 3.6 MoE kernel

`_moe_gemm_kernel` is vLLM's `fused_moe_kernel` structure (see `fused_moe.py`, `fused_moe_kernel_gptq_awq`) with the B tile produced by `_decode_tile`:

- **Inputs:** `a_ptr` (`hidden_states` `[M, K]` for GEMM1, `cache2` `[M*topk, N]` for GEMM2), `qw_ptr` + `stride_qe` (expert stride), `sc_ptr` + `stride_se`, `lut_ptr`, `c_ptr`, `topk_weights_ptr`, `sorted_token_ids_ptr`, `expert_ids_ptr`, `num_tokens_post_padded_ptr`, `N`, `K`, `EM`, `num_valid_tokens`, strides; constexprs `top_k`, `MUL_ROUTED_WEIGHT`, `CODEC`, `BM`, `BN`, `BK`.
- **Body:**
  - `pid → (pid_m, pid_n)`.
  - Return early if `pid_m * BM >= num_tokens_post_padded`.
  - `offs_token = load(sorted_token_ids[pid_m*BM + arange(BM)])`; `token_mask = offs_token < num_valid_tokens`.
  - `e = load(expert_ids[pid_m])`.
  - A rows = `offs_token // top_k`.
  - B = `_decode_tile(qw_ptr + e*stride_qe, sc_ptr + e*stride_se, ...)`.
  - `acc += tl.dot(a, tl.trans(b.to(bf16)))`.
  - If `MUL_ROUTED_WEIGHT`: multiply by `topk_weights[offs_token]` in fp32.
  - Store bf16 to `c[offs_token, offs_n]`.
- **Initial config** (Phase 3 tunes it):

```python
def _moe_config(M: int, E: int, top_k: int, smem: int) -> dict:
    per_expert = M * top_k / E
    bm = 16 if per_expert <= 16 else 32 if per_expert <= 32 else 64
    if per_expert > 96 and smem >= 160 * 1024:
        bm = 128
    return {"BLOCK_SIZE_M": bm, "BN": 64 if bm <= 32 else 128, "BK": 128,
            "num_warps": 4 if bm <= 32 else 8}
```

BM tracks tokens per expert, which fixes K4.

### 3.7 MoE experts class

New file `vllm/model_executor/layers/fused_moe/experts/modelopt_block_triton_moe.py`. **Copy `TritonWNA16Experts` (triton_moe.py:637–820)** and change only:

1. `__init__(self, moe_config, quant_config, codec: str, lut: torch.Tensor | None)`: store `self.codec`, `self.lut`, then `super().__init__(moe_config, quant_config)`.
2. `_supports_current_device`: `current_platform.is_cuda() and current_platform.has_device_capability(80)`.
3. `_supports_activation`: `MoEActivation.SILU, MoEActivation.RELU2_NO_MUL`.
4. `_supports_parallel_config`: copy `B12xExperts._supports_parallel_config` (TP only: no EP, no all2all, no EPLB).
5. `apply`: same body as `TritonWNA16Experts.apply` with these changes:
   - `config = _moe_config(num_tokens, E, top_k_num, smem)` instead of `try_get_optimal_moe_config` (which can't handle these dtypes).
   - Both GEMMs call your `invoke_block_moe_kernel(...)`, passing `self.w1_scale` / `self.w2_scale` (from the quant config, see below), `self.lut`, `self.codec`.
   - Keep `self.activation(...)` between the GEMMs (ReLU² and SiLU-and-mul come free), and keep `self.moe_sum(intermediate_cache3, output)`.
   - Keep `moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E, expert_map)`.
6. `moe_sum`: inherited. LoRA is unsupported: raise `NotImplementedError` if a LoRA path is requested, like `B12xExperts.moe_sum`.

In `ModelOptBlockQuantMoEMethod`:

```python
def get_fused_moe_quant_config(self, layer):
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(dtype=self.codec, scale=getattr(layer, "w13_weight_scale", None)),
        _w2=FusedMoEQuantDesc(dtype=self.codec, scale=getattr(layer, "w2_weight_scale", None)),
    )

def process_weights_after_loading(self, layer):
    if self.use_triton:
        for name in ("w13", "w2"):
            qweight, scales = repack_blocks(getattr(layer, f"{name}_weight").data, self.codec)
            replace_parameter(layer, f"{name}_weight", qweight)
            layer.register_parameter(
                f"{name}_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
            )
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    prepare_finalize = maybe_make_prepare_finalize(...)  # unchanged
    if self.use_triton:
        lut = iq2_lut(layer.w13_weight.device) if self.codec == "iq2_xxs" else None
        experts = ModelOptBlockTritonExperts(self.moe, self.moe_quant_config, self.codec, lut)
    else:
        experts = B12xExperts(self.moe, self.moe_quant_config)
    self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)
    if not self.use_triton:
        experts.process_weights_after_loading(layer)
```

Shared experts keep working: the runner passes `shared_experts` through `moe_kernel.apply` exactly as for b12x.

---

## 4. Phase 0: setup and baseline (no code changes)

### 4.1 Check the GPU every time before launching anything

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
ss -ltnp | grep -E ':(8000|8080|8012)\b'   # 8000 is protected; use 8080, or 8012 if busy
```

Bolt needs ≈ 23 GiB of weights plus 4–8 GiB for graphs, activations and the mamba/KV cache.

- **If less than 32 GiB is free:** wait.
- **If utilization is above 30% from other tenants:** run correctness work only. Put benchmarks off until it's quieter, or run the A and B arms back to back and say so in the log.

### 4.2 Repository

```bash
cd ~/vllm                                      # the fork checkout at work (adjust path)
git fetch origin
git checkout feature/wip-nvidia-bolt-low-sm-support   # tip must be 2064cc2efb
git log --oneline -1                           # expect: 2064cc2efb inital support done...
git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null || true
git fetch upstream
git rebase upstream/main                       # one commit; resolve conflicts if any (§10)
git worktree add ../vllm-draft 2064cc2efb      # frozen draft, for the kernel A/B scripts
```

Duplicate-work check (AGENTS.md, mandatory). Paste the outputs into the log:

```bash
gh issue view 62609 --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "62609 in:body"
gh pr list --repo vllm-project/vllm --state open --search "IQ2_XXS"
gh pr list --repo vllm-project/vllm --state open --search "modelopt block triton"
gh pr list --repo vllm-project/vllm --state open --search "b12x SM90 OR Hopper OR A100"
```

**If an open PR already does this:** stop and tell the user. Don't continue.

### 4.3 Environment

```bash
cd ~/vllm
uv venv --python 3.12 .venv && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
uv pip install -r requirements/lint.txt && pre-commit install
uv pip install gguf                              # test oracle only, not a vLLM dependency
python -c "import torch, triton; print(torch.__version__, triton.__version__)"
python -c "import b12x, importlib.metadata as m; print(m.packages_distributions()['b12x'])"
# expect ['flashinfer-python']: b12x arrives bundled with flashinfer, that's normal (B1)
```

If the work box already has a working venv (`/home/seigyo/vllmServer/.venv`), don't modify it. Make this new one next to the checkout.

### 4.4 Baseline: serve the draft

The draft is on the branch after the rebase.

```bash
MODEL=/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
mkdir -p ~/bolt-runs/baseline
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --port 8080 --served-model-name bolt-test \
  --trust-remote-code --max-num-seqs 256 --gpu-memory-utilization 0.27 \
  --reasoning-parser nemotron_v3 2>&1 | tee ~/bolt-runs/baseline/serve.log
```

- `--max-num-seqs 256` is needed at utilization 0.27: the mamba cache has ~313 blocks there, and the default would want 1024.
- **If startup fails on memory:** see §10.

Record from the log:
- `Model loading took X GiB`
- `Graph capturing finished in N secs, took Y GiB`
- total startup time

### 4.5 Baseline: accuracy (run while the server is up)

```bash
python tests/evals/gsm8k/gsm8k_eval.py --port 8080 --model bolt-test \
  --num-questions 1319 --num-shots 5 --max-tokens 512 \
  --save-results ~/bolt-runs/baseline/gsm8k_5shot.json
python tests/evals/gsm8k/gsm8k_eval.py --port 8080 --use-chat-completions \
  --max-tokens 1024 --chat-template-kwargs '{"enable_thinking": false}' \
  --num-questions 1319 --save-results ~/bolt-runs/baseline/gsm8k_chat.json
```

- The first command (completions, 5-shot) is **the A/B number**.
- The second (chat, thinking off) is a sanity check that chat formatting works.
- Also read the model card in `$MODEL/README.md` for any GSM8K number NVIDIA publishes, and note it.

### 4.6 Baseline: speed (same server)

```bash
for C in 1 8 32 128; do
  N=$(( C < 4 ? 32 : C * 8 ))
  vllm bench serve --base-url http://127.0.0.1:8080 --model $MODEL \
    --served-model-name bolt-test --dataset-name random --random-input-len 1024 \
    --random-output-len 256 --ignore-eos --num-prompts $N --max-concurrency $C \
    --save-result --result-dir ~/bolt-runs/baseline --result-filename serve_c$C.json
done
```

Record output tok/s, median TPOT and median TTFT per concurrency. Then stop the server (Ctrl-C in its terminal, and only that one).

### 4.7 Baseline: draft kernels on H200

```bash
cd /path/to/vllmServer/Notes/02-Bolt-Support/kernel-tests
VLLM_SRC=~/vllm-draft ~/vllm/.venv/bin/python test_correctness.py   # draft correct on H200?
VLLM_SRC=~/vllm-draft ~/vllm/.venv/bin/python test_moe.py
VLLM_SRC=~/vllm-draft ~/vllm/.venv/bin/python bench.py | tee ~/bolt-runs/baseline/draft_kernels.txt
```

On H200 leave `FORCE_STAGES` unset, so the draft's real configs run.

Also run `proto_kernels.py`. It gives first H200 numbers for the new kernels and proves they work on Hopper:

```bash
~/vllm/.venv/bin/python proto_kernels.py | tee ~/bolt-runs/baseline/proto_h200.txt
```

### 4.8 Baseline: tiny e2e with the draft

```bash
cd .../kernel-tests
for m in eager compiled graphs; do ~/vllm/.venv/bin/python e2e_tiny_bolt.py $m; done
~/vllm/.venv/bin/python e2e_tiny_bolt.py compare
for m in eager compiled graphs; do ~/vllm/.venv/bin/python e2e_tiny_bolt.py $m --bf16; done
~/vllm/.venv/bin/python e2e_tiny_bolt.py compare --bf16
```

Expected with the draft on H200: **`graphs` crashes at capture** with "Cannot copy between CPU and CUDA tensors during CUDA graph capture" (B4: IQ2 is on attention in this tiny model). Record it. Your rewrite must make it pass.

### Gate 0

- [ ] Duplicate check clean.
- [ ] `baseline/` holds serve.log, gsm8k_5shot.json, gsm8k_chat.json, serve_c{1,8,32,128}.json, draft_kernels.txt, proto_h200.txt.
- [ ] The tiny-e2e result is recorded.
- [ ] `proto_kernels.py` printed `ALL OK` on H200.

---

## 5. Phase 1: new kernel module and kernel tests (no vLLM wiring yet)

### 5.1 Create `modelopt_block_quant_triton.py`

1. Header: SPDX lines like every vLLM file; a one-line module docstring; `from vllm.triton_utils import tl, triton`.
2. `_IQ2XXS_GRID = (...)`: the 256 ints from Appendix A, copied exactly.
3. Copy from `proto_kernels.py`: `repack_blocks`, `iq2_lut` (add `@functools.cache`), `_decode_tile`, `_gemm_kernel`, `_gemv_kernel`.
4. Write `_moe_gemm_kernel` (§3.6) and `_q8_embedding_kernel` (§3.5).
5. Write the launch helpers: `_linear_config(M, N, K, codec, num_sms, smem)` → dict, and `_moe_config(...)`. Read `num_sms` and `smem` once per call from `torch.cuda.get_device_properties(x.device)`; Python-side cost only, and it's inside a custom op.
6. Write `_modelopt_block_linear` and `_modelopt_block_embedding`, plus their fakes, and register both (§3.4).
7. Write `invoke_block_moe_kernel(...)`, called by the experts class.

Style:
- 88-character lines.
- Google-style docstrings on the public helpers and custom ops: shapes and dtypes in, shapes and dtypes out.
- No comments that restate code.
- Match the surrounding vLLM style.

### 5.2 Things that go wrong in Triton here

| Symptom | Fix |
|---|---|
| `OutOfResources: shared memory, Required: 133120, Hardware limit: 101376` | Reduce stages first, then BK, then BM. On 99 KB GPUs, BM=64 + BK=256 + 2 stages = 130 KB (hit at home); BM=64 + BK=256 + 1 stage fits. Make `_linear_config` read `smem` and drop stages until it fits; check with Appendix D. |
| `tl.dot` error about shapes < 16 | BM, BN and BK must each be ≥ 16 for `tl.dot`. M = 1 goes to `_gemv_kernel`, which has no `tl.dot`. |
| Results wrong only when N isn't a multiple of BN | A mask is missing on a load or store. Every load of weights/scales needs `mask=n_mask[:, None]`, every store needs `m_mask & n_mask`. |
| Results wrong for Q8_0 with odd K/32 (e.g. K = 1056) | BK doesn't divide K. Choose BK from the divisors (§3.5). |
| Wrong values from the closure in a benchmark loop | Python late binding (it happened at home). Bind loop vars as default args: `def run(part=part, BN=BN, ...)`. |

### 5.3 Do not optimize yet

Phase 1 ends when it's correct everywhere. Speed is Phase 3.

### 5.4 Kernel tests: `tests/kernels/quantization/test_modelopt_block_triton.py`

Read `tests/AGENTS.md` first. Keep tests minimal, one behavior each, no mocks.

```python
pytestmark = pytest.mark.skipif(
    not current_platform.has_device_capability(80), reason="needs NVIDIA SM80+"
)
```

Put `ref_decode()` from `kernel-tests/ref_decode.py` at the top of the test file (torch only, ~25 lines). Import the grid from the module under test. The checksum test below guards the table itself.

| Test | What it asserts | Why |
|---|---|---|
| `test_iq2_grid_matches_ggml` | `sha256(np.array(_IQ2XXS_GRID, '<u8').tobytes()) == "05826b5d…dd54"`; and if `gguf` imports (`pytest.importorskip`), equals `gguf.quants.IQ2_XXS.grid` | a corrupted table silently breaks every IQ2 model |
| `test_decode_bit_exact[codec]` | `modelopt_block_linear(eye(K), …).T == ref_decode(raw).bfloat16()` exactly; raw has random bytes plus special scales (0, negative, 3e-6 subnormal, 60000) | independent oracle; tests decode through the public op |
| `test_linear_matches_reference[codec, M]` | M ∈ {1, 2, 3, 4, 8, 15, 16, 17, 33, 64, 65, 128, 255, 256, 1000, 2048}; N = 1032 (not a multiple of any BN); K = 4096 (IQ2) / 1056 (Q8_0). Assert `max|out − ref| ≤ 1.5 · max|cuBLAS − ref|`, where ref = `x.double() @ ref_decode(raw).double().T` and cuBLAS = `x @ ref_decode(raw).bfloat16().T` | covers every dispatch branch and both masks |
| `test_linear_offset_view` | x is a slice `big[3:3+M]` of a larger tensor | pointer-alignment specialization (skill rule) |
| `test_linear_int64_offsets` | Q8_0, M = 16400, N = 131072, K = 4096; check the last 8 output rows against the reference. Skip if `torch.cuda.mem_get_info()[0] < 12 << 30` | M·N > 2³¹ (S1) |
| `test_linear_compile_does_not_freeze_dispatch` | `torch.compile(f, dynamic=True, options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe})`; call it at M=256, then M=1. Profile both the compiled and eager calls at M=1 and assert the same kernel names. Also assert correctness. | B2 regression guard |
| `test_linear_cuda_graph_replay` | capture at M ∈ {1, 4, 17}; replay with new x twice; correct, and `torch.cuda.memory_allocated()` unchanged | graph safety |
| `test_moe_matches_reference[codec, activation, M]` | E=16, K=1024, I=512, top-k ∈ {2, 9}, M ∈ {1, 7, 64, 333}, activation ∈ {RELU2_NO_MUL, SILU}; experts via `mk.FusedMoEKernel(MoEPrepareAndFinalizeNoEP(), ModelOptBlockTritonExperts(...))`; reference = loop over tokens and experts with `ref_decode` weights in fp64, relative max error ≤ 1e-2 | MoE path incl. SiLU |
| `test_moe_cuda_graph_new_routing` | capture once, replay with new x **and** new `topk_ids`/`topk_weights` (copied in place); correct and memory-stable | the routing changes every step in serving |
| `test_q8_embedding_exact` | ids with repeats, out of order, int32 and int64; output == `ref_decode(raw[ids]).bfloat16()` exactly | K5 fix |

`MoEPrepareAndFinalizeNoEP` lives in `vllm/model_executor/layers/fused_moe/prepare_finalize/` (grep for the class). Reuse fixtures from `tests/kernels/moe/` where they fit.

### Gate 1 (H200 and RTX 4000 Ada)

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_modelopt_block_triton.py -v
```

- [ ] All pass on H200.
- [ ] All pass on the RTX 4000 Ada. That's the sm_89 / 99 KB check, and CI runs this file on L4 (also sm_89).
- [ ] Appendix D report: every config used fits on sm_80/86/89/90/100/120.

---

## 6. Phase 2: wire into vLLM, delete the draft, repo tests

### 6.1 Edit `modelopt_block_quant.py`

1. Delete the import block `from .modelopt_block_quant_sm90 import (...)`.
2. Add `_use_triton` (§3.2) and the imports it needs.
3. MoE method:
   - Replace the draft's `sm90_fallback` lines in `__init__` with the `self.use_triton = _use_triton(...)` call.
   - Restore the upstream checks that the draft guarded with `not self.sm90_fallback`; the helper now covers the backend check.
   - Replace `process_weights_after_loading` and `get_fused_moe_quant_config` with §3.7's versions.
   - `apply`: delete the `if self.sm90_fallback:` branch.
4. Linear method:
   - `__init__`: the helper call (§3.2).
   - `process_weights_after_loading`: start with `if self.use_triton: <§3.3 block>; return`. Embedding and non-embedding both repack. The b12x code below stays unchanged.
   - `apply`: §3.4 block before the b12x code.
   - `embedding`: `if self.use_triton: return torch.ops.vllm.modelopt_block_embedding(input_.reshape(-1), layer.weight, layer.weight_scale).view(*input_.shape, -1)`.
5. `get_b12x_warmup_unit` and `b12x_warmup_provider` stay b12x-only. The Triton path sets neither.

### 6.2 Delete the draft files

```bash
git rm vllm/model_executor/layers/quantization/modelopt_block_quant_sm90.py \
       vllm/model_executor/layers/quantization/modelopt_block_quant_hopper.py \
       tests/kernels/quantization/test_modelopt_block_dequant.py \
       benchmarks/kernels/benchmark_block_dequant_gemm.py
grep -rn "sm90_fallback\|VLLM_SM90_BLOCK_KERNEL\|prewarm_device_checks\|modelopt_block_quant_hopper\|modelopt_block_quant_sm90" vllm tests benchmarks docs
# expect no output
```

### 6.3 Prove there's no hidden b12x dependency

```bash
.venv/bin/python - <<'EOF'
import sys
sys.modules["b12x"] = None          # any `import b12x` now raises ImportError
import torch
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t
raw = torch.randint(0, 256, (256, 4, 66), dtype=torch.uint8, device="cuda")
raw[..., :2] = (torch.randn(256, 4, device="cuda") * 1e-3).half().unsqueeze(-1).view(torch.uint8)
qw, sc = t.repack_blocks(raw, "iq2_xxs")
y = torch.ops.vllm.modelopt_block_linear(
    torch.randn(5, 1024, device="cuda", dtype=torch.bfloat16), qw, sc, t.iq2_lut(raw.device), "iq2_xxs")
print("ok", y.shape, torch.isfinite(y).all().item())
EOF
```

### 6.4 Run the existing tests (must still pass, no edits to them)

```bash
.venv/bin/python -m pytest tests/quantization/test_modelopt.py -k "block or iq2 or mixed_precision" -v
.venv/bin/python -m pytest tests/model_executor/kernels/test_b12x_linear.py tests/model_executor/warmup/test_b12x_warmup.py -v
```

On H200 the b12x tests may skip; that's fine. At home on sm_120 they ran (62 + 31 passed on the draft). Ask the user to rerun them at home on the final branch. They monkeypatch `B12xExperts._supports_current_device` to True, so they cover the b12x branch on any GPU.

**If one fails:** you changed b12x behavior. Fix your code, not the test.

### 6.5 New repo tests in `tests/quantization/test_modelopt.py`

1. **Selection tests.** Copy the style of `test_modelopt_mixed_precision_dispatches_iq2_xs_experts` (line ~1059).
   - Monkeypatch `B12xExperts._supports_current_device` to `False`.
   - Build the method with `moe_backend` set to `"auto"`, `"triton"`, `"b12x"` and `"marlin"`.
   - Assert `use_triton is True` for the first two, and `pytest.raises(ValueError, match="ModelOpt")` for the others.
   - Assert `iq2_xs` raises with `"auto"`.
   - Do the same for the linear method: build under `set_current_vllm_config(VllmConfig(kernel_config=KernelConfig(linear_backend=...)))`. Look at how nearby tests build configs.
   - Skip if capability < 80.
2. **Tiny e2e** `test_modelopt_block_triton_tiny_nemotron_h`. Port `kernel-tests/e2e_tiny_bolt.py`:
   - Write the config JSON into `tmp_path`.
   - `LLM(load_format="dummy", enforce_eager=True)` vs `LLM(load_format="dummy")` (compiled + CUDA graphs, small capture sizes).
   - Sanitize scales with `llm.apply_model(...)`. Set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` via `monkeypatch.setenv` if multiprocessing needs it.
   - Greedy, `logprobs=5`.
   - Compare with `tests/models/utils.py:check_logprobs_close`.
   - Free the first LLM before building the second (`del llm; gc.collect(); torch.cuda.empty_cache()`).
   - **Known quirk:** even the BF16 control shows prompt 0 diverging from step 0 between eager and compiled. That's vLLM-side (seen at home, 2026-10-22). Use prompts 1+ for the assertion, or assert `check_logprobs_close`, which tolerates it if the tokens stay in each other's top-5. Never assert exact token equality.

### Gate 2

```bash
.venv/bin/python -m pytest tests/kernels/quantization/test_modelopt_block_triton.py tests/quantization/test_modelopt.py -v
cd .../kernel-tests && for m in eager compiled graphs; do ~/vllm/.venv/bin/python e2e_tiny_bolt.py $m; done \
  && ~/vllm/.venv/bin/python e2e_tiny_bolt.py compare
```

- [ ] All tests pass on H200.
- [ ] Tiny e2e passes in all three modes. **`graphs` must now capture** (B4 fixed), with a non-b12x path on all 9 layers.
- [ ] Bolt serves (§4.4 command). The log shows the two `Using Triton kernels …` lines and no `sm90`. "Model loading took" is within +0.1 GiB of baseline.
- [ ] Quick GSM8K, 250 questions: `--num-questions 250`, within ±3 points of the baseline's first 250. Full GSM8K comes in Phase 4.

---

## 7. Phase 3: performance

Read `.agents/skills/kernel-microbenchmark/SKILL.md` first.

### 7.1 The benchmark script `benchmarks/kernels/benchmark_modelopt_block_triton.py`

- **Timing:** `flashinfer.testing.bench_gpu_time_with_cupti` (CUDA graph, cold L2), median. Follow `.agents/skills/kernel-microbenchmark/benchmarks/cupti_microbenchmark.py`. Warm up 25 iterations first.
- **Shapes:** Appendix C. Dense M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}. MoE M ∈ {1, 2, 4, 8, 16, 64, 256, 1024, 4096}.
- **Columns:** shape, M, our µs, bf16 cuBLAS µs, ratio, GB/s, TFLOP/s, chosen config.
  - GB/s = `(N·K·bytes_per_weight + 2·M·K + 2·M·N) / time`, where bytes per weight is 66/256 or 34/32. These are the theoretical bytes; don't count re-reads.
  - TFLOP/s = `2·M·N·K / time`.
- **MoE baseline:** vLLM's BF16 `fused_experts` (`TritonExperts`) on dequantized weights for the same layer.
  - Use random top-9 routing with a fixed seed.
  - The skill warns that dummy gates route every token to the same experts. Random routing avoids that.
- **Copy-bandwidth reference:** time `dst.copy_(src)` on two 1 GiB uint8 tensors. BW = 2 GiB / time. Print it in the header. Expect about 3.8–4.3 TB/s on an idle H200.
- `--sweep` flag: for each shape and M, sweep BN ∈ {8, 16, 32, 64, 128}, BK ∈ {64, 128, 256}, split ∈ {1, 2, 4, 8, 16}, warps ∈ {1, 2, 4, 8}, stages ∈ {1, 2, 3, 4}. Skip configs that don't fit (catch `OutOfResources`) and print the best.
- Correctness check before timing: every config, against `ref_decode`. Explicit tolerance as in §5.4.

### 7.2 Loop

1. Run the benchmark with the current dispatch tables.
2. Compare against §2.2.
3. For each failing row, run `--sweep` on that row.
4. Copy the winning configs into `_linear_config` / `_moe_config` as **small, readable rules**: buckets on M, N and K, never a giant per-shape dict.
5. Re-run the whole table: a rule that helps one shape can hurt another.
6. Log every round.

### 7.3 Dense ladder: do these in order and stop when the bars pass

| If you see | Do |
|---|---|
| M=1 Q8_0 below 55% of copy BW | (a) Split-K so programs ≥ 2× SMs. (b) BN 32 → 64 → 128. (c) Check the PTX for vector loads: `grep -c "ld.global.v4" ` on `kernel.asm["ptx"]` (Appendix D). If zero, the payload pointer or stride isn't 16-byte divisible: pass `qweight.stride(0)` as an int and make sure `NB*32` is a multiple of 16. |
| M=1 IQ2 slower than cuBLAS | (a) BN=8, warps=1, BK=256, split 4–8; that was best at home. (b) Make sure the lookup table is fp32 (`iq2_lut`), not int words. (c) See the IQ2 ladder in §7.4. |
| M = 2–16 slower than cuBLAS | (a) Try `_gemv_kernel` generalized to MB rows (constexpr MB ∈ {2, 4, 8}, acc `[MB, BN]`) against `_gemm_kernel` with BM=16. (b) Increase split-K. (c) **Swap AB**: compute `tl.dot(w_tile, tl.trans(x_tile))` → `[BN, BM]`, so the weight dimension is the large MMA dimension. Store transposed. On Hopper this lets wgmma use the 64-row dimension for weights instead of padding the tokens. |
| M ≥ 2048 slower than 2× cuBLAS | (a) Check wgmma: `"wgmma" in kernel.asm["ptx"]`. If not, BM/BN aren't multiples of 64 or warps < 4. (b) Raise stages to 3–4; H200 has 227 KB. (c) BK 64 vs 128. (d) Check spills (Appendix D): if `n_spills > 0`, drop BK or BN, or raise warps to 8. (e) GROUP_M swizzle (`fused_moe_kernel` has it) for L2 reuse of x. |
| `n_spills > 0` on sm_90 | The decoded tile `[BN, R, 4, 8]` is too big for the warp count. Halve BK, or BN, or double warps. Re-check speed. |

### 7.4 IQ2 ladder (IQ2 is limited by decode work, not memory)

Measure each step and keep it only if it helps on **both** H200 and the 4000 Ada.

1. **Fewer conversions:** keep `rs` in fp32 and multiply once per weight, as the prototype does. Check that the compiled PTX has no redundant cvt instructions.
2. **bf16 lookup table plus sign by XOR:**
   - Table `bf16[256*8]`.
   - Load the magnitude bits: `bits = tl.load(lut_bf16_bits + idx)` as int16.
   - `bits ^= (neg << 15)`, then bitcast to bf16, then multiply by `rs` in fp32.
   - This replaces the `tl.where` select.
3. **Load both record words at once:** an `[BN, R, 2]` load, then `tl.split`, giving one 8-byte vector load per record.
4. **MoE decode path (M ≤ 8):** add `_moe_gemv_kernel`:
   - grid `(cdiv(N, BN), M*top_k, split)`
   - each program serves one (token, expert) pair: `e = topk_ids[pair]`, A row = `pair // top_k` (GEMM1) or `pair` (GEMM2)
   - no `moe_align_block_size`, no BM padding
   - Use it when `M * top_k <= 64`. It's eager code inside the MoE op, so a Python `if` is fine.
   - This will likely be the biggest single MoE decode win, because the dense GEMV beat the `tl.dot` kernel by 2–3× at M=1 at home.
5. **Only if 1–4 still miss the bar, and only with the user's approval:**
   - Quantize activations to int8 per 32-block (like llama.cpp's Q8_1) and use integer dot products on the grid bytes.
   - That changes numerics against b12x's a16 path, so it needs its own GSM8K A/B.
   - It must be a separate follow-up PR, never this one.

### Gate 3

- [ ] Every row of §2.2 meets its merge bar on H200. Paste the benchmark table into the log.
- [ ] The tuned tables are written back as rules in the module.
- [ ] Gate 1 tests are still green on H200 and the 4000 Ada.

---

## 8. Phase 4: full validation

### 8.1 Accuracy A/B

Serve the new branch with exactly the §4.4 command, then run the §4.5 GSM8K commands into `~/bolt-runs/new/`.

| Δ vs baseline (5-shot completions, 1319 questions) | Action |
|---|---|
| ≤ 1.5 points | pass |
| 1.5 – 3 | Re-run both arms once (shared-GPU noise can't change greedy output, but restarts can change batch composition). If it persists, diff per-question results: do the failures cluster? Re-run the new branch with `--enforce-eager`. If eager is fine, suspect compile or graphs (§10). |
| > 3 | Bug. Bisect: run Gate 1 tests on real Bolt shapes, then §8.2 with real weights. |

### 8.2 Layer parity on real Bolt weights

Write `Notes/.../kernel-tests/bolt_layer_parity.py` (not a repo test). Using `safetensors.safe_open` on the model files (the index is `model.safetensors.index.json`), for one tensor of each kind:

- `layers.1.mixer.experts.0.up_proj.weight` (IQ2)
- `shared_experts.up_proj.weight` (IQ2)
- `layers.7.mixer.q_proj.weight` (Q8_0; check `layers_block_type` for an attention index)
- `fc1_latent_proj.weight` (Q8_0)
- `lm_head.weight` (Q8_0)

Then:
1. `repack_blocks` the raw tensor.
2. Check decode bit-exactness with the identity trick on a 256-column slice.
3. Compare `modelopt_block_linear` with x ~ N(0,1) (M ∈ {1, 64, 2048}) against `ref_decode` in fp64, using the §5.4 tolerance rule.

Also check that every scale is finite: `torch.isfinite(scales).all()`.

### 8.3 Model-level consistency (tiny model, both GPUs)

`e2e_tiny_bolt.py` all modes, plus `compare` and `compare --bf16`.

Pass: the quantized run's divergence pattern is no worse than the BF16 control's. Every prompt that matches in the control must match or be `close=True` in the quantized run.

### 8.4 Robustness on Bolt

```bash
# server with the JIT monitor in error mode
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL ...same flags as §4.4... \
  --jit-monitor-mode error --jit-monitor-verbose 2>&1 | tee ~/bolt-runs/new/serve_jit.log
# varied lengths to hit every M bucket and odd alignments
for L in 1 7 33 129 777 3001; do
  vllm bench serve --base-url http://127.0.0.1:8080 --model $MODEL --served-model-name bolt-test \
    --dataset-name random --random-input-len $L --random-output-len 64 --num-prompts 64 \
    --max-concurrency 37
done
```

- Pass: the server never errors and the log has no JIT-monitor complaints after "Graph capturing finished".
- Then restart with `VLLM_GPU_SYNC_CHECK=error` in the environment (no JIT flags) and run the same loop. Pass: no sync errors.

### 8.5 Speed A/B

Re-run §4.6 into `~/bolt-runs/new/`, back to back with a fresh baseline run if the GPU load changed.

Optionally, per-kernel breakdown:

```bash
vllm bench forward --model $MODEL --load-format auto --trust-remote-code \
  --gpu-memory-utilization 0.27 --max-num-seqs 256 \
  --case decode=1*1@2048 --case decode=32*1@2048 --case prefill=2048 --profile-dir ~/bolt-runs/new/fwd
python -m vllm.benchmarks.forward.profile ~/bolt-runs/baseline/fwd/decode.rank0.kernels.json ~/bolt-runs/new/fwd/decode.rank0.kernels.json
```

- Run the same `vllm bench forward` command on the draft first, with the output in `baseline/fwd`.
- If `bench forward` doesn't support this hybrid VL model (it uses Model Runner V2), skip it and say so in the log.

### 8.6 Other GPUs

- **RTX 4000 Ada:** Gate 1 tests, tiny e2e (all modes), and the benchmark at the Appendix C dense shapes. Bolt itself doesn't fit (22.4 GiB > 20 GB).
- **sm_80/86** (no hardware): the Appendix D cross-compile report must show every config fits. Say in the PR that A100/A10 were checked by cross-compiling only.
- **ROCm:** not supported. The selection helper requires `is_cuda()`. Say so in the docs.

### Gate 4

- [ ] §8.1 pass.
- [ ] §8.2 bit-exact everywhere.
- [ ] §8.3 pass on both GPUs.
- [ ] §8.4 clean.
- [ ] §8.5 meets the §2.2 end-to-end bar.
- [ ] §8.6 done.

---

## 9. Phase 5: docs, cleanup, PR

### 9.1 Diff hygiene

```bash
git diff --stat upstream/main
pre-commit run --files $(git diff --name-only upstream/main)
```

- New non-test code should be ≤ ~700 lines.
- No unrelated formatting changes.
- No comments that restate code.
- Inline any helper that has exactly one trivial call site (pr-checklist 3.3).

### 9.2 Docs: `docs/features/quantization/b12x.md`

Add a short section, e.g. `## Other NVIDIA GPUs (Triton kernels)`:

- IQ2_XXS and Q8_0 ModelOpt block weights (linear, MoE, Q8_0 embedding) run through Triton kernels on NVIDIA SM80+ when b12x isn't available.
- IQ2_XS still requires b12x.
- Force the path with `--linear-backend triton --moe-backend triton`. This works on SM12x too.
- Tensor parallelism only (no expert parallelism), same as b12x.
- Weights stay packed; memory use matches the checkpoint.
- ROCm isn't supported.

Also fix the sentence at the top that says these formats are SM120/SM121 only.

### 9.3 Commit

```
[Quantization] Triton path for ModelOpt IQ2_XXS/Q8_0 block weights on SM80+

<2-4 lines: what, why (#62609), how selected>

Co-authored-by: <your agent name/model>
Signed-off-by: <user's name> <user's email>
```

- Use `git commit -s`: the DCO check needs `Signed-off-by` from the human submitter.
- Squash to one commit or a small, clean series.
- Ask the user before pushing.

### 9.4 PR

Run the `pr-checklist` skill and fix what it finds. Then fill `.github/PULL_REQUEST_TEMPLATE.md`:

- **Summary:** 1–2 lines. `Fixes #62609`.
- **Why it isn't duplicate work:** the §4.2 search results. #59604 is the b12x path (merged); #59486 was superseded; #59093 (Q8_0 via Humming) was closed.
- **Claims:**
  - Bolt serves on H200.
  - IQ2_XXS/Q8_0 work on SM80+.
  - SM12x is unchanged.
  - Weights stay packed.
- **Validation:**
  - test commands and results
  - GSM8K table (baseline vs new, 5-shot completions plus chat)
  - `vllm bench serve` table per concurrency
  - kernel benchmark table (Appendix C shapes, M sweep)
  - JIT-monitor and sync-check results
  - which GPUs (H200, RTX 4000 Ada) and what was only cross-compiled (sm_80/86)
- **Limitations:** TP only; IQ2_XS b12x-only; ROCm unsupported; IQ2 is limited by decode work (link the int8-activation follow-up idea).
- **AI assistance:** state it plainly, and that the human submitter reviewed every line and ran the tests.

Before opening, show the user the PR text and the diff summary, and let them post it. Then update the comment on #62609 with the PR link.

---

## 10. Troubleshooting (if X, do Y)

| Symptom | Likely cause | Do |
|---|---|---|
| `ValueError: ... require b12x on SM12x` on H200 | old upstream check still there, or `_use_triton` not called | §6.1 step 3/4; grep for the string |
| `ValueError: ModelOpt ... backend 'auto' is not available` on H200 | `has_device_capability(80)` false, or codec is `iq2_xs` | print `current_platform.get_device_capability()` and the codec |
| `ModuleNotFoundError: b12x...` anywhere in the Triton path | a stray b12x import | §6.3 script; grep the new module for `b12x` (must be none) |
| `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture` | a tensor created in forward (lookup table, `torch.tensor(...)`, `.to(device)`) | move it to `process_weights_after_loading` (§3.3) |
| `torch._dynamo.exc.Unsupported` / `ObservedAttributeError` during compile, pointing at your module | Dynamo traced into Python helpers | the traced `apply` must contain only the custom-op call. Check `torch.ops.vllm.modelopt_block_linear` is registered at import (the module is imported at the top of `modelopt_block_quant.py`) |
| Fake impl shape errors during compile | the fake returns the wrong shape or dtype | fake must return `x.new_empty((x.shape[0], qweight.shape[0]))`, bf16 |
| Eager correct, compiled wrong | dispatch frozen (an M branch outside the op) or wrong fake | `test_linear_compile_does_not_freeze_dispatch`; move logic into the op |
| `OutOfResources: shared memory` | config too big for this GPU | §5.2; make the config read `shared_memory_per_block_optin` |
| `illegal memory access` | a mask or offset bug, or int32 overflow | `CUDA_LAUNCH_BLOCKING=1`; run the failing test under `compute-sanitizer --tool memcheck`; follow `.agents/skills/debug-ima`; check every load/store mask and int64 cast |
| NaN/Inf output with real weights | scale bytes misread (wrong byte order, wrong offset after repack) | `repack_blocks` must take bytes 0–1 as fp16; `torch.isfinite(layer.weight_scale).all()`; identity-trick decode against `ref_decode` |
| NaN only with `--load-format dummy` | random scale bytes | expected; sanitize the scales (§3.3) |
| GSM8K drop > 3 | numerics bug in a path the kernel tests don't hit (likely MoE GEMM2 routed-weight multiply, or SiLU vs ReLU²) | §8.2 on real weights; MoE test with `top_k = 9` and `M ≥ 333`; compare `MUL_ROUTED_WEIGHT` handling with `TritonWNA16Experts` |
| JIT monitor fires after startup | a runtime int specialized (M, EM, num_valid_tokens), or a new pointer alignment | add the int to `do_not_specialize`. For alignment: make sure the profile run uses the same view types, or use `do_not_specialize_on_alignment` only for pointers that don't need vector loads |
| Startup much slower than the draft | too many kernel variants compiled | fewer M buckets; same config for every M > 64; `do_not_specialize` |
| `vllm serve` fails "mamba cache blocks" / max_num_seqs | memory budget at util 0.27 | keep `--max-num-seqs 256`, or raise `--gpu-memory-utilization` if free memory allows |
| OOM at startup | other tenants grew | §4.1; wait, or lower `--max-num-seqs` / add `--max-model-len 32768` for test runs only |
| `n_spills > 0` | tile too big | §7.3 last row |
| Bench numbers jump ±20% between runs | shared GPU | interleave A and B; take the median of 3; note utilization from `nvidia-smi` |
| Rebase conflict in `modelopt_block_quant.py` | upstream changed the file | take upstream's version, then re-apply only §6.1. Don't re-apply the draft's lines |
| `TP > 1` fails with an alignment error | Bolt's I = 2304 / 2 = 1152 isn't a multiple of 256 | expected and not this PR's problem; Bolt runs at TP=1 (it fits one GPU). Say so in the docs |
| Tiny e2e prompt 0 differs between eager and compiled | vLLM-side, also happens with `--bf16` | compare against the control (§8.3) |

---

## Appendix A: IQ2_XXS grid (ggml `iq2xxs_grid`)

Byte `j` of entry `i` (little-endian) = magnitude of weight `j` in group pattern `i`.

The SHA-256 of `np.array(GRID, dtype='<u8').tobytes()` is `05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54`. It was checked equal to `gguf.quants.IQ2_XXS.grid` on 2026-10-22. The same table is in `kernel-tests/ref_decode.py`.

```python
_IQ2XXS_GRID = (
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
)
```

## Appendix B: reference decoder for tests

`kernel-tests/ref_decode.py: ref_decode()`, pure torch. Checked bit-exact against gguf-py on 2026-10-22, including zero, negative, subnormal and 60000 scales. Copy the function verbatim into the test file. Self-check: `python ref_decode.py` (needs `gguf`).

## Appendix C: Bolt shapes at TP=1 (benchmark and test list)

Dense. Weight is `[N, K]`, out = x @ Wᵀ.

| Layer | Codec | N | K | Count | Packed MB each |
|---|---|---|---|---|---|
| `qkv_proj` (q+k+v merged by vLLM) | Q8_0 | 4608 | 4096 | 8 | 20.1 |
| `o_proj` | Q8_0 | 4096 | 4096 | 8 | 17.8 |
| `fc1_latent_proj` | Q8_0 | 1024 | 4096 | 40 | 4.5 |
| `fc2_latent_proj` | Q8_0 | 4096 | 1024 | 40 | 4.5 |
| `shared_experts.up_proj` | IQ2_XXS | 5376 | 4096 | 40 | 5.7 |
| `shared_experts.down_proj` | IQ2_XXS | 4096 | 5376 | 40 | 5.7 |
| `lm_head` (M = number of sequences) | Q8_0 | 131072 | 4096 | 1 | 570 |

MoE, IQ2_XXS, ReLU² non-gated, K_latent = 1024, top-9. `w13` per expert is `[I, 1024]`, `w2` is `[1024, I]`.

| E | I | Layers |
|---|---|---|
| 240 | 1536 | 1 |
| 240 | 2048 | 7 |
| 240 | 2304 | 4 |
| 304 | 2048 | 2 |
| 304 | 2304 | 3 |
| 304 | 2560 | 1 |
| 368 | 2048 | 2 |
| 368 | 2304 | 4 |
| 368 | 2560 | 3 |
| 432 | 2304 | 4 |
| 432 | 2560 | 6 |
| 496 | 2560 | 3 |

Benchmark at least (240, 2048), (432, 2560) and (496, 2560).

Bytes read per decode token at M = 1, from this work's kernels:
- routed experts ≈ 0.44 GB
- shared experts ≈ 0.45 GB
- latent ≈ 0.36 GB
- attention ≈ 0.30 GB
- `lm_head` 0.57 GB
- **total ≈ 2.1 GB**

The NVFP4 mamba projections (Marlin) add ≈ 2.5 GB. At 4.8 TB/s that's a weight-read floor of about 1 ms per token. Use these numbers to sanity-check the end-to-end tok/s.

## Appendix D: shared-memory / register / PTX report

Checked at home on Triton 3.8. Call it after a kernel has run once; it reads Triton's cache.

```python
import triton
from triton.backends.compiler import GPUTarget

SMEM_LIMIT = {80: 163 * 1024, 86: 99 * 1024, 89: 99 * 1024, 90: 227 * 1024,
              100: 227 * 1024, 120: 99 * 1024}


def kernel_report(jit_fn):
    """Native regs/spills plus cross-compiled shared memory for every cached variant."""
    for cache in jit_fn.device_caches.values():
        for ck in cache[0].values():
            ptx = ck.asm.get("ptx", "")
            print(f"{jit_fn.__name__} {ck.src.constants} warps={ck.metadata.num_warps} "
                  f"stages={ck.metadata.num_stages}: regs={ck.n_regs} spills={ck.n_spills} "
                  f"smem={ck.metadata.shared / 1024:.1f}KB wgmma={'wgmma' in ptx} "
                  f"v4_loads={ptx.count('ld.global.v4')}")
            for cc, limit in SMEM_LIMIT.items():
                k = triton.compile(ck.src, target=GPUTarget("cuda", cc, 32),
                                   options={"num_warps": ck.metadata.num_warps,
                                            "num_stages": ck.metadata.num_stages})
                print(f"    sm_{cc}: {k.metadata.shared / 1024:.1f} KB "
                      f"{'OK' if k.metadata.shared <= limit else 'OVER'}")
```

- Run it on H200 for `_gemm_kernel`, `_gemv_kernel` and `_moe_gemm_kernel` after the benchmark has exercised every config.
- `spills` must be 0 on sm_90, and every line must say `OK`.
- At home, the draft's `_iq2_gemm_kernel` showed `regs=255 spills=6` natively on sm_120. That's the pattern to avoid.

## Appendix E: work-box command sheet

```bash
# env
cd ~/vllm && source .venv/bin/activate
MODEL=/home/seigyo/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
# GPU check (always first)
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
# serve (never port 8000, never name gemma-4-31B)
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --port 8080 --served-model-name bolt-test \
  --trust-remote-code --max-num-seqs 256 --gpu-memory-utilization 0.27 --reasoning-parser nemotron_v3
# quick smoke request
curl -s localhost:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"bolt-test","prompt":"2+2=","max_tokens":8,"temperature":0}'
# tests
.venv/bin/python -m pytest tests/kernels/quantization/test_modelopt_block_triton.py -v
.venv/bin/python -m pytest tests/quantization/test_modelopt.py -k "block or iq2 or mixed_precision or triton" -v
# GSM8K
python tests/evals/gsm8k/gsm8k_eval.py --port 8080 --model bolt-test --num-questions 1319 --num-shots 5 --max-tokens 512 --save-results out.json
# kernel benchmark
.venv/bin/python benchmarks/kernels/benchmark_modelopt_block_triton.py [--sweep]
# lint
pre-commit run --files $(git diff --name-only upstream/main)
```
