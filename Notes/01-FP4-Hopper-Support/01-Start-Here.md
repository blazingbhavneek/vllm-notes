# 01 · Start here

This folder explains one project from zero:

> **Make NVIDIA's "FP4-Local" checkpoints of Qwen3.8-Flash-Next and GLM-5.3-Flash run on Hopper GPUs (H100 / H200) in upstream vLLM, keep their small memory footprint, and later make them fast.**

The notes have two readers:

1. **You**, a person who is new to GPU kernels and quantization. Every idea is explained from the ground up, in order. Nothing assumes you already know it.
2. **An implementing agent** that will write the code. It needs exact file paths, function names, steps, tests and "done" criteria. Those are in notes 15 to 21.

Read the notes in order. Each one ends with a "Next" link. If a word is new, look it up in [22 · Glossary](22-Glossary.md).

---

## The story on one page

1. **Two big new models.** `Qwen/Qwen3.8-Flash-Next` (Alibaba) and `zai-org/GLM-5.3-Flash` (Z.ai) are large "Mixture-of-Experts" models. In 16-bit form they are 335 GiB and 599 GiB. That is too big for most machines.
2. **Smaller copies exist.** People store the model's numbers with fewer bits. This is called **quantization**. Three kinds matter here:
   - **Vanilla NVFP4** by NVIDIA (`nvidia/Qwen3.8-Flash-Next-NVFP4`): 4-bit experts, made quickly after training ("post-training quantization", PTQ).
   - **GGUF GSQ-RCO** by IST-DASLab (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`): 2 to 3.5 bits, each tensor gets its own format. Made for llama.cpp.
   - **FP4-Local** by NVIDIA together with the Local Inference Lab community (`nvidia/Qwen3.8-Flash-Next-FP4-Local`, `nvidia/GLM-5.3-Flash-FP4-Local`, and the original `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`). The model was *trained a bit more* so that its 4-bit form nearly matches the original ("quantization-aware distillation", QAD). Its scale numbers are also packed with a lossless trick called **CSF**.
3. **FP4-Local only runs on Blackwell today.** The model cards say "NVIDIA Blackwell" and "you must use one of the containers provided". The containers use a community vLLM fork and the **b12x** kernel library, which only targets the Blackwell chips sm_120 and sm_121.
4. **Hopper cannot do FP4 math, but it can still run FP4 models.** Hopper's tensor cores multiply 16-bit and 8-bit numbers, not 4-bit ones. But a kernel can *store* weights as 4-bit and *unpack* them inside the kernel, right before the multiply. vLLM already has kernels that do this (**Marlin** and **Humming**). So nothing in the hardware stops FP4-Local from running on Hopper. What's missing is software.
5. **Our job, in four phases:**
   - **Phase A, load it.** Upstream vLLM can't read the CSF-packed scales yet. Teach it.
   - **Phase B, run it correctly.** Check every layer type on Hopper and fix what breaks.
   - **Phase C, keep it small.** Keep the scales compressed in GPU memory and unpack them only when needed.
   - **Phase D, make long prompts fast.** Unpack the 4-bit weights to 8-bit (FP8) inside the kernel instead of 16-bit, so Hopper's 2× faster FP8 math can be used.

## The goal, said precisely

| Goal | What "done" means |
|---|---|
| G1 · Correct | Both FP4-Local models load and answer correctly on Hopper with upstream vLLM. Accuracy (GSM8K, and token-level KL divergence) matches the Blackwell numbers within noise. |
| G2 · Small | Weights stay 4-bit in GPU memory. Scales stay CSF-compressed (Phase C). The n-gram table of Qwen can live in CPU memory. Nothing is ever expanded to a full 16-bit or 8-bit copy in memory. |
| G3 · Fast | Decode (one token at a time) uses 4-bit weights × 16-bit activations (W4A16), which is already the best choice. Long prompts can use 4-bit weights × 8-bit activations (W4A8), if measurements show it helps. |

## Ten facts to remember

Each fact was checked on 19 Oct 2026. The legend is below the list.

1. ✅ **Sizes.** Qwen3.8-Flash-Next is 335.3 GiB in BF16, 172.8 GiB in FP8, 123.5 GiB in vanilla NVFP4 and **95.1 GiB in FP4-Local**. GLM-5.3-Flash is 598.5 GiB in BF16, 305.8 GiB in FP8, 190.4 GiB in vanilla NVFP4 and **165.2 GiB in FP4-Local**. (Sum of all tensor sizes in the checkpoint index files.)
2. ✅ **Only the routed experts are 4-bit "W4A4" NVFP4.** Attention is MXFP8 (8-bit). GLM's shared experts and sparse-attention projections are 128×128-block FP8. Small parts stay BF16. See notes 05, 06 and 10.
3. ✅ **Upstream vLLM has no CSF code at all.** A search of `main` at commit `3e96f9cf63` finds nothing. Loading the FP4-Local checkpoints fails on every GPU, Blackwell included, unless you use the community fork.
4. ✅ **There are two CSF formats.** Qwen FP4-Local uses the "slab" format (`weight_scale_encoding: "csf"`, 2 tensors per scale). NVIDIA's GLM FP4-Local uses `csf_32x2_v1` (5 tensors per scale). Draft PR [#60646](https://github.com/vllm-project/vllm/pull/60646) handles only the slab format.
5. ✅ **CSF is lossless and saves about half of the scale memory.** On real tensors, CSF is 53 to 54 % of the raw scale bytes. That saves 3.3 GiB on Qwen and 8.1 GiB on GLM. A decoder written for these notes reproduces NVIDIA's GLM scales byte for byte ([scripts/csf32_decode.py](scripts/csf32_decode.py)).
6. ✅ **On Hopper, vLLM picks Humming first, then Marlin.** For NVFP4 and MXFP8, vLLM moves the Humming kernel in front of Marlin, but only on sm_90 (`humming/priority.py:11`). These are the in-kernel unpack ("dequantize") kernels.
7. ✅ **4-bit × scale doesn't fit exactly in FP8.** A 4-bit value times its 8-bit scale needs up to 6 significant bits; FP8 E4M3 has 4. On a real expert matrix, 22 % of the weights would be rounded if unpacked to FP8. The added error is about 2 %, which is about the same as the error from FP8 activations (note 17).
8. ✅ **b12x already uses this rounding trick on Blackwell.** Its "W4A8 on NVFP4" recipe splits each scale into a power of two plus a remainder, and rounds the product to FP8 in registers (`b12x/moe/_shared/kernels/reference.py`, `decompose_nvfp4_scales_to_mx_residual`). vLLM's own Hopper INT4×FP8 kernel (`cutlass_w4a8`) also rounds weight × scale into FP8. So the idea has precedent.
9. ✅ **The model code itself already runs on Hopper.** Qwen4Exp has sm_90 GEMM plans and sm_90 attention tuning tables. GLM5Next's FlashKDA supports sm_90, and FlashMLA-sparse supports sm_90. The gaps are in the quantization layer. (Not yet run end to end: ⚠️.)
10. 📄 **FP4-Local accuracy is close to BF16.** Qwen: GPQA-Diamond 91.5 % vs 92.2 %. GLM: GPQA-Diamond 91.3 % vs 92.1 %. (From the model cards.)

### Legend: how sure are we?

| Mark | Meaning |
|---|---|
| ✅ | Checked by reading the code, or by running a script on real data, on 19 Oct 2026. |
| 📄 | Taken from a model card, paper, README or doc. Not re-measured. |
| ⚠️ | Expected from reading the code, but not yet run on a Hopper GPU. |
| ❓ | Open question. Someone has to find out. |

## Reading order

| # | Note | Read it to learn |
|---|---|---|
| 01 | Start here | This page. |
| 02 | [Numbers and bits](02-Numbers-And-Bits.md) | How computers store numbers. BF16, FP8, FP4. Why "exact" matters. |
| 03 | [What quantization is](03-What-Quantization-Is.md) | Scales, groups, W4A16 / W8A8 / W4A4, PTQ vs QAD, memory math. |
| 04 | [How a GPU does matrix math](04-How-A-GPU-Does-Matrix-Math.md) | GEMM, decode vs prefill, memory-bound vs compute-bound, tensor cores per GPU generation. |
| 05 | [Model: Qwen3.8-Flash-Next](05-Model-Qwen3.8-Flash-Next.md) | Its layers, its sizes, what each tensor is. |
| 06 | [Model: GLM-5.3-Flash](06-Model-GLM-5.3-Flash.md) | The same for GLM. |
| 07 | [Marlin and Humming](07-Marlin-And-Humming-Kernels.md) | The kernels that unpack 4-bit weights inside the GEMM. |
| 08 | [The vanilla NVFP4 checkpoint](08-The-Vanilla-NVFP4-Checkpoint.md) | What NVIDIA's plain 4-bit checkpoint quantizes, and how it runs on Hopper. |
| 09 | [GGUF and GSQ-RCO](09-GGUF-And-GSQ-RCO.md) | The llama.cpp-style quant of Qwen and how it differs. |
| 10 | [FP4-Local and QAD](10-FP4-Local-And-QAD.md) | How FP4-Local was made and what's inside it. |
| 11 | [CSF scale compression](11-CSF-Scale-Compression.md) | Both CSF formats, bit by bit, with checked decoders. |
| 12 | [Comparing the three quants](12-Comparing-The-Three-Quants.md) | One big comparison table. |
| 13 | [Why Blackwell only](13-Why-Blackwell-Only.md) | What is truly impossible on Hopper and what is only missing software. |
| 14 | [b12x explained](14-b12x-Explained.md) | The Blackwell kernel library behind the community containers. |
| 15 | [How vLLM loads and runs these](15-How-vLLM-Loads-And-Runs-Quantized-Models.md) | Code walk with file:line references. |
| 16 | [What happens on Hopper today](16-What-Happens-On-Hopper-Today.md) | Layer-by-layer: which kernel, and what breaks. |
| 17 | [The FP8 scale problem](17-The-FP8-Scale-Problem.md) | The core math of a Hopper FP8 path, with measurements. |
| 18 | [Design for Hopper support](18-Design-Hopper-Support.md) | Phases A to D in detail. |
| 19 | [VRAM budget on H200](19-VRAM-Budget-On-H200.md) | Memory tables per phase. |
| 20 | [Implementation plan for the agent](20-Implementation-Plan-For-The-Agent.md) | PR-sized tasks, files, steps, acceptance criteria, vLLM rules. |
| 21 | [Testing and benchmarking](21-Testing-And-Benchmarking.md) | Commands, evals, profilers, and how to reproduce every check in these notes. |
| 22 | [Glossary](22-Glossary.md) | Every term, in one line each. |
| 23 | [Sources, open questions and risks](23-Sources-Open-Questions-And-Risks.md) | Links, unknowns, licenses, people. |

**Short on time?** Read 01, 03, 13, 16, 18 and 20. That's the whole plan.

**You're the implementing agent?** Read everything once. Then work from [20](20-Implementation-Plan-For-The-Agent.md) and [21](21-Testing-And-Benchmarking.md), and come back to the theory notes when a step refers to them.

## Snapshot: what these notes were checked against

| Thing | Version |
|---|---|
| vLLM | `main` at `3e96f9cf63` (19 Oct 2026), clone at `/mnt/common/Code/vllmServer/vllm` |
| b12x | `2cc7f66` (archived 6 Oct 2026; continued inside FlashInfer as `experimental/b12x`) |
| Humming | `vllm-project/humming` at `8960158` (19 Oct 2026); vLLM pins `humming-kernels==0.1.17` |
| Checkpoints | `main` revisions of the Hugging Face repos on 19 Oct 2026; LIL GLM branches listed in note 10 |
| Hardware assumed | 1–2 × H200 (sm_90, 141 GB) at work; RTX 5060 Ti (sm_120, 16 GB) at home |

No model weights were downloaded. Only configs, READMEs, safetensors **headers**, and a handful of single tensors (one expert matrix and its scales per model, 5 MB in total, for the decoder and error checks) were fetched, with HTTP range requests.

Things change fast. Before starting any task, re-run the duplicate checks in note 20.

---

**Next:** [02 · Numbers and bits](02-Numbers-And-Bits.md)
