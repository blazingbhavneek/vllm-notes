# 23 · Sources, open questions and risks

**What you'll learn:** where every fact in these notes came from; what nobody knows yet and how to find out; what could go wrong and what to do about it; and who to talk to.

Everything was read on **19 Oct 2026**, against vLLM `main` at `3e96f9cf63`. Hugging Face repos and GitHub PRs change, so re-check anything that matters before acting on it.

---

## 1. Sources

### Models (Hugging Face)

| Repo | What | Notes |
|---|---|---|
| [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | QAD NVFP4 experts, slab CSF scales | 05, 10, 11 |
| [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) | QAD NVFP4 experts, `csf_32x2_v1` scales, `tensors/native-QB` | 06, 10, 11 |
| [local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) | the original QAD GLM; branches `main`, `qad2-kda-w8a16`, `qad2-kda-w8a16-csf` | 06, 10 |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | vanilla PTQ NVFP4 | 08 |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | vanilla PTQ NVFP4 | 08 |
| [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | GGUF builds (IQ2 … Q2_0) | 09 |
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | BF16 and FP8 originals | 05 |
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8), [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) | GLM originals | 06 |

How they were read: model cards, `config.json`, `hf_quant_config.json`, index files and safetensors **headers**, plus about 6 MB of single tensors fetched with HTTP range requests (`scripts/st_header.py`, `st_tensor.py`, `st_slice.py`). No full download.

### Papers

- QAD: arXiv [2601.20088](https://arxiv.org/abs/2601.20088), "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery" (NVIDIA). Note 10.
- GSQ: arXiv [2604.18556](https://arxiv.org/abs/2604.18556), code [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ). Note 09.
- RCO: arXiv [2605.00649](https://arxiv.org/abs/2605.00649), code [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO). Note 09.
- Formats: OCP Microscaling (MX) specification v1.0 (MXFP4 / MXFP8, UE8M0); NVIDIA's NVFP4 description in the Blackwell / ModelOpt docs. Notes 02, 03.

### Code

| Code | Where | Notes |
|---|---|---|
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm), local clone `/mnt/common/Code/vllmServer/vllm` | 15, 16 |
| Humming | [vllm-project/humming](https://github.com/vllm-project/humming), pinned `humming-kernels[cu13]==0.1.17` | 07 |
| Marlin (in vLLM) | `csrc/quantization/marlin/`, `quantization/utils/marlin_utils_fp4.py` | 07 |
| b12x | now in FlashInfer, `flashinfer/experimental/b12x` ([FlashInfer #5767](https://github.com/flashinfer-ai/flashinfer/pull/5767)); old standalone repo archived 6 Oct 2026 | 14 |
| LIL vLLM fork | container `ghcr.io/local-inference-lab/vllm:release-10152026`; DGX Spark: `eugr/spark-vllm-b12x:latest` (read from the model cards, not pulled) | 10, 14 |
| GGUF plugin | [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin) | 09 |
| vLLM docs | `docs/features/quantization/b12x.md`, `docs/contributing/profiling.md`, `docs/contributing/incremental_build.md`, `AGENTS.md` | 14, 20, 21 |

### GitHub PRs and issues (state on 19 Oct 2026)

| # | Kind | State | What | Notes |
|---|---|---|---|---|
| [#60646](https://github.com/vllm-project/vllm/pull/60646) | PR | **draft**, by stecasta, updated 8 Oct | slab CSF decode at load, subfolder-shard fallback | 11, 16, 20 |
| [#38728](https://github.com/vllm-project/vllm/pull/38728) | PR | open | NVFP4 → FP8 weight conversion at load on Hopper (+62 % throughput, more memory) | 20 |
| [#49070](https://github.com/vllm-project/vllm/issues/49070) | issue | open | NVFP4 MoE garbage output on H100/H200 with Marlin | 07, 16 |
| [#48624](https://github.com/vllm-project/vllm/pull/48624) | PR | open | Marlin fix for unaligned sizes | 07, 16 |
| [#48895](https://github.com/vllm-project/vllm/issues/48895) | issue | open | wrong top-k weights in `moe_wna16_marlin_gemm` | 16 |
| [#57748](https://github.com/vllm-project/vllm/issues/57748) | issue | open | GLM NVFP4 on 4 × H100 (no logs) | 16 |
| [FlashInfer #5767](https://github.com/flashinfer-ai/flashinfer/pull/5767) | PR | merged | b12x moves into FlashInfer | 14 |

### Measured here (scripts in `scripts/`, how to rerun: note 21, section 8)

| Fact | Script | Note |
|---|---|---|
| FP4 × E4M3 exact in 73.2 % of cases, worst 4.76 % | `fp4_scale_exactness.py` | 17 |
| Slab CSF decodes; 53.4 % of raw size; NVIDIA's encoder reproduced | `csf_slab_decode.py` | 11 |
| `csf_32x2_v1` decodes to LIL's uncompressed scale; 54.1 %; encoder reproduced | `csf32_decode.py`, `csf32_vectorized.py` | 11 |
| W4A8 error: 2.11 % weight, 3.04 % with FP8 activations; QAD vs BF16 9.56 % | `w4a8_error_study.py` | 17 |
| FP4-Local scales differ from the vanilla PTQ scales (it's retrained, not recompressed) | `csf_slab_decode.py` | 10, 12 |

### Not verified at all

- **Nothing has been run on a GPU.** No vLLM server, no H200 run, no kernel test. Every "this should work on Hopper" is ⚠️ until T0 / T5.
- `scripts/kld_topk.py dump` (needs vLLM + GPU).
- Model-card accuracy numbers (📄): copied, not reproduced.
- Memory numbers in note 19 are computed from headers, not measured.

## 2. Open questions

Each one: what we don't know, how to find out, and which task needs the answer.

| # | Question | How to answer | Blocks |
|---|---|---|---|
| Q1 | What is Humming's scale layout for NVFP4 on sm_90 after `convert_to_humming_moe_kernel_format`? Can a precomputed index map produce it? | Read `humming/transform.py`; prototype bit-exact like T8; ask jinzhen-lin | T8 / T10 for Humming (Marlin goes first) |
| Q2 | Does Humming accept the small-N MXFP8 layers (Gated DeltaNet `in_proj_a/b`, N = 48), or do they fall back? | Read the startup log in T0 / T5; a unit test with that shape | T5 |
| Q3 | Does `maybe_fuse_shared_experts` stay off when the shared expert (MXFP8) and the routed experts (NVFP4) use different formats? | Read the code path; check in T5 that shared-expert weights load as MXFP8 | T5 / T6 |
| Q4 | How does vLLM count KV cache tokens for GLM (DSA + KDA, `index_kpool = 4`)? The card's "1.8 M tokens" doesn't match note 19's estimate. | Read the `GPU KV cache size` log line in T0 | note 19 budget |
| Q5 | Does `vllm-gguf-plugin` support `qwen4_exp` (hyper-connections, PLE, QSA, MTP)? | Load the smallest GGUF build; read the plugin's model list | GGUF comparison only (not on the main path) |
| Q6 | Can one packed weight layout feed both W4A16 and W4A8 Humming kernels on sm_90? | Read Humming's packing for both configs; ask jinzhen-lin | T12 (D2) |
| Q7 | Is the W4A8 speedup big enough (D0 gate)? | T11 benchmark | T12 |
| Q8 | Does routed-only decode on Hopper cost what note 18 estimates? | T9 benchmark | T10 |
| Q9 | Do the community containers run on sm_100 (B200)? | Not needed for Hopper; skip unless someone asks | — |
| Q10 | Does the MTP draft (its experts are `W4A16_NVFP4` with raw scales, note 10) keep a normal acceptance rate on Hopper? | In T6, run with MTP on and compare the spec-decode acceptance metrics with the Phase 0 run | T6 |

## 3. Risks

| Risk | Why it matters | What to do |
|---|---|---|
| **Duplicate work** | #60646 already covers the slab decoder; someone may start `csf_32x2_v1` at any time | Run the duplicate searches in note 20 before every PR; talk to stecasta first |
| **Licenses** | LIL License 1.0 forbids re-uploads, even small slices | Use NVIDIA repos in tests and docs; generate synthetic CSF with the scripts' encoders; never commit checkpoint bytes |
| **Nothing tested on Hopper yet** | A model-level bug (attention, hyper-connections, MTP) could block everything before quantization matters | Do T0 first, with the vanilla checkpoints. If they don't run, fix or report that before any CSF work |
| **Open Marlin MoE bugs on Hopper** (#49070, #48624, #48895) | Wrong output that looks like a CSF or accuracy bug | Compare Humming vs Marlin on every run; KLD between them should be ≈ 0 |
| **Kernel layout changes** | Phase C's index map depends on Marlin / Humming internals that can change in any release | Build the map from the same function the kernel's own post-load step uses, and test against it in CI (bit-exact) |
| **Humming changes need its owner** | Phase D needs a new kernel mode in a separate repo | Open an issue with jinzhen-lin before writing code; accept "no" |
| **QAD is tuned for exact FP4 math** | W4A8 rounds the fused scales (about 2–3 % error); QAD's careful accuracy recovery may be partly lost | Phase D stays opt-in; measure KLD and GSM8K on both models; stop if the gain is small (D0) |
| **Memory plans are estimates** | Note 19's numbers are from headers; real allocator, CUDA graphs and activations add more | Measure in T0 / T5; plan with a 5–10 % margin |
| **Long downloads, big disks** | 95 GiB + 165 GiB per checkpoint; slow re-runs | One shared `HF_HOME`; never re-download per test |
| **Agent PRs rejected** | vLLM rejects pure agent PRs and busywork | The human reviews every line and runs every test; follow note 20, Part 1 |

## 4. People

Use GitHub handles; be polite and specific, and include commands and results when asking for something.

| Who | Role here | When to contact |
|---|---|---|
| **stecasta** | author of draft PR #60646 (slab CSF) | T1 (test results), T2 / T3 (offer to add the GLM pieces to their PR or as a follow-up) |
| **jinzhen-lin** | maintainer of Humming and vLLM's Marlin | Q1, Q6, T8 (layouts), T12 (new Humming mode) |
| **Tib-Gridello** | author of #38728 (NVFP4 → FP8 on Hopper) | before Phase D: compare approaches, share benchmarks |
| vLLM quantization / MoE maintainers | review of loader and MoE changes | the RFC in T8; code owners show up as reviewers on PRs |
| **Local Inference Lab** | made the original QAD GLM, b12x and the fork | questions on CSF intent or b12x behavior; respect their license |
| **IST-DASLab** | GSQ, RCO, the GGUF builds, original Marlin | GGUF questions only |
| NVIDIA ModelOpt team | made the FP4-Local checkpoints | format questions (for example a CSF spec), through the model card discussions |

## Key takeaways

- All facts come from model cards, headers, a few small tensors, papers and code reading. **Nothing ran on a GPU.**
- The biggest unknowns: Humming's scale layout (Q1) and whether W4A8 is worth it (Q7). Neither blocks Phase A.
- The biggest risks: duplicating #60646, licenses, and Hopper problems unrelated to quantization. T0 first, always.

---

You've reached the end. Go back to [01 · Start here](01-Start-Here.md) for the map, or to [20 · Implementation plan](20-Implementation-Plan-For-The-Agent.md) to start working.
