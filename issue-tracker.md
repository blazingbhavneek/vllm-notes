# vLLM Open Lanes

*vllm-project/vllm · Q4 2026 · checked on GitHub and Hugging Face, 19 Oct 2026*

A plain-English map of where vLLM is going, and where someone with an **H200** and an **RTX 5060 Ti** can make a real dent, with a focus on kernels and 4-bit models. Every claim was checked against `main` (commit `3e96f9c`), open PRs, and the model files on Hugging Face.

Online copy: <https://claude.ai/artifact/CDcmxwY9eUQ6yMSSRNLhfv>

**Contents:** [Short version](#the-short-version) · [**Order**](#do-these-in-this-order) · [How vLLM works](#how-vllm-works-in-five-minutes) · [Quantization 101](#quantization-in-plain-english) · [Other words](#other-words-youll-see-in-issues) · [Q4 plan](#the-q4-2026-plan-in-plain-english) · [Where vLLM is heading](#where-vllm-is-heading) · [**FP4 on Hopper**](#quant-track-fp4-models-on-hopper) · [Online quant](#online-quantization-and-llm-compressor-answered) · [**GGUF**](#gguf-track-make-popular-local-quants-fast-in-vllm) · [**Kernel track**](#kernel-track) · [Workflow](#working-between-the-h200-and-home)

---

## The short version

1. **You won't find kernel work in the issue list.** A sweep covered 239 open kernel, performance and quantization issues filed since 1 Sep. Only 42 were untouched (no PR, no assignee, two comments or fewer), and nearly all of those are AMD, Intel, CPU or giant-model bugs. NVIDIA kernel bugs usually get fixed by their reporter the same day. You get established by **producing evidence nobody else has** and by **doing breadth work maintainers have asked for**.
2. **The biggest open gap that fits you: 4-bit "mixed" checkpoints on Hopper.** 24 of NVIDIA's 40 newest NVFP4 checkpoints mix formats layer by layer. The new QAD-trained **FP4-Local** models (8 Oct) say "Blackwell only" and need a community container. Qwen3.8-Flash-Next-FP4-Local is 95 GiB, small enough for **one H200**; the FP8 version is 173 GiB and needs two. Nobody is working on Hopper support. See [Q1](#q1--run-nvidias-fp4-local-and-other-mixed-checkpoints-on-hopper).
3. **The other big gap: GGUF.** It's the most-downloaded quant format (unsloth's Qwen3.8-27B GGUF: 6.7 million downloads), but vLLM runs it through a quiet plugin with 2024-era kernels. Mixed-type files like unsloth UD and GSQ-RCO hit slow paths on almost every layer. See the [GGUF track](#gguf-track-make-popular-local-quants-fast-in-vllm).
4. **Your two GPUs are a rare pair.** sm_90 (H200) plus sm_120 (5060 Ti) covers the two architectures where kernel defaults are still undecided. When a lane needs a third architecture (B200, sm_100), rent one for an afternoon.
5. **Where to start:** quick wins first. Run the Q1 test matrix (1–3 days, no code), then fix the Hopper bugs it turns up, with a sanitizer sweep running overnight. See [Do these in this order](#do-these-in-this-order).

---

## Do these in this order

Ranked by how fast you can finish it, how much it matters, and how likely it is to get merged quickly. Low-impact work is left out. Merge odds come from the 139 quantization-labelled PRs merged since 15 Sep. In that sample, bug fixes merged in a median of **5 days**, performance PRs with numbers in **3 days**, and new quantization features or refactors in **25 days**. The GGUF plugin is a separate repo: outside PRs there merged in a median of **1.3 days**, but nothing has merged since 24 Sep and 28 PRs are waiting.

Merge odds: 🟢 likely · 🟡 slower · 🔴 stalled right now

| # | Task | Effort | Impact | Merge |
|---|---|---|---|---|
| 1 | **Q1 steps 1–2:** run the Hopper test matrix, open one issue, post test reports on #60646 and #48624 | 1–3 days, no code, at work | High | 🟢 n/a (issue + test reports) |
| 2 | **G1:** benchmark the GGUF kernel upgrade (plugin #141) on the 5060 Ti and H200 | 1–2 days, no code, at home | High | 🟢 n/a (test report) |
| 3 | **Q1 steps 3 + 5:** fix what the matrix finds, one PR per cause, plus a Hopper GSM8K config | Small–medium per PR | High | 🟢 likely |
| 4 | **G2:** stop per-step weight copies in mixed-type GGUF layers | Small, at home | High for GSQ-RCO and unsloth UD files | 🟡 fast once reviewed; queue paused |
| 5 | **K4:** sanitizer sweep, then fix each out-of-bounds bug | Easy to start, runs overnight | Medium–high per bug | 🟢 likely |
| 6 | **Q1 step 4:** decoder for the GLM CSF variant | Medium, CPU only, at home | High (GLM-5.3-Flash-FP4-Local on any GPU) | 🟡 slower |
| 7 | **K1:** per-architecture kernel defaults | Medium, 1–2 weeks | High (roadmap item) | 🟢 likely with data |
| 8 | **G3:** GGUF vs llama.cpp benchmark, tested-model list, docs | Medium, after #141 lands | High (moves GGUF out of "experimental") | 🟢 likely with data |
| 9 | **Q3:** online requantization leftover | Medium | Medium–high on Hopper | 🟡 slower (features ~25 days) |
| 10 | **K2:** fusion batch (GLM4-MoE, InternLM2, MiniCPM) | Easy | Medium | 🔴 stalled right now |
| 11 | **Q2:** measure FP8 activations for FP4 weights on Hopper | Days to measure; months if a kernel is needed | Potentially high | 🟡 unknown until measured |
| 12 | **K3:** GELU-tanh + FP8 kernel | Small–medium | Medium | 🔴 stalled right now |
| 13 | **K6:** DeepGEMM warmup | Medium | Medium | 🟡 needs 4–8 GPUs |

1. **Test matrix and issue.** Run the four mixed checkpoints on the H200, with defaults and with Marlin forced. Open one issue with the results. Post your H200 results on the two stalled PRs that Q1 depends on: [#60646](https://github.com/vllm-project/vllm/pull/60646) (CSF decoder, no reviews since 8 Oct) and [#48624](https://github.com/vllm-project/vllm/pull/48624) (a small MoE fix, open since July). **Why first:** every Q1 PR depends on it, and it puts your name in front of the quantization maintainers in week 1.
2. **GGUF kernel benchmark.** The biggest GGUF speedup is already written, but it has no reviews and no numbers from Hopper or consumer Blackwell. Build it, run the kernel tests and an end-to-end benchmark, and post before/after tables. **Why now:** it runs at home while #1 runs at work, and it puts you in front of both the plugin maintainer and the PR author.
3. **Fix PRs.** Put the Hopper GSM8K config for one small mixed checkpoint into the first fix so it stays fixed. **Why:** bug fixes with a repro merge fastest; dispatch fixes like #60337 (+18/−2) merged the same day. Each fix unblocks NVIDIA's mixed checkpoints on every Hopper GPU, not just yours.
4. **GGUF mixed-shard fix.** A small fix with a before/after tokens-per-second table. It's independent of #141 and touches nearly every layer of the files you care about. Plugin PRs merge fast once the maintainer looks, so after you post it, ask once in Slack.
5. **Sanitizer sweep.** Start it in the background in week 1. Every bug it finds becomes its own small fix PR, with the sanitizer log as proof. You're the reporter, so nobody races you. The GGUF plugin's kernels deserve a pass too; someone already found one out-of-bounds bug there (#127).
6. **GLM CSF decoder.** Its sibling #60646 has no reviews yet. Offer it to stecasta as part of that PR or as a follow-up, not as a separate competing loader. Do it after your H200 report on #60646, so both pieces get attention together.
7. **Per-architecture defaults.** Benchmark every kernel per format on sm_90 and sm_120, plus sm_100 on a B200 rented for an afternoon, then send the priority PR. Q1 teaches you the dispatch code this changes. Defaults are sensitive (#60315), so the works/fails column matters as much as speed.
8. **GGUF mainstream.** The same GGUF files in llama.cpp and vLLM, with 1 and 32 users, on both your GPUs. Then add a small accuracy test list of popular files and open a docs PR that softens the "highly experimental" warning.
9. **Requantization leftover.** Take "requantization requires a runnable backend for the source format". Open it early and let it wait in review while you work on other things.
10. **Fusion batch.** Three community batches (#58665, #59470, #61107) have had no maintainer review for 1–4 weeks; only #57514 (reviewed by mgoin) is moving. Wait until it merges and copy its pattern exactly.
11. **FP8-activation measurement.** Needs Q1 working first. If FP8 wins for big batches, it's a short perf PR. If not, publish the table and move on.
12. **GELU + FP8 kernel.** The closest existing PR (#59467) has had no review since 30 Sep. Ask toan-tq and mgoin first.
13. **DeepGEMM warmup.** Only if you get a multi-GPU node.

Left out on purpose: K5 (H200 MoE configs; low impact, similar PRs sit unreviewed) and G4 (GGUF convert-at-load; needs an RFC first).

### As a calendar

1. **Week 1:** full CUDA build on the H200 and #1 at work. At home, #2 (build the plugin with #141 and benchmark it). Start the #5 sanitizer sweep overnight.
2. **Weeks 2–3:** #3 fix PRs at work; #4 GGUF fix and #6 CSF decoder at home.
3. **Weeks 3–5:** #7 K1 (rent a B200 for an afternoon), and #8 once #141 has landed. Open #9 early so it can sit in review.
4. **Weeks 5–8:** #10 once #57514 has merged, then #11.

#3 and #5 count toward the 4-merged-PR gate in the main vLLM repo. Plugin PRs live in a separate repo and probably don't count. How many H200s you have at work decides whether you rent for GLM-5.3-Flash-FP4-Local (2×) and K6 (4–8×).

---

## How vLLM works, in five minutes

vLLM is a server that runs large language models fast. Almost every issue touches one of these stops on the path a request takes.

```
Frontend ──► Scheduler ──► KV cache ──► Model runner ──► Models & kernels
(API, text    (who gets     (memory of    (runs one step    (each model's code +
 → tokens)     GPU time)     past tokens)  on the GPU)       the GPU math) ← your area
```

| Stop | Folder | What it does |
|---|---|---|
| Frontend | `vllm/entrypoints`, `rust/` | The web API (`/v1/chat/completions`). Turns text into tokens. Python today; a Rust rewrite is becoming the default this quarter. |
| Scheduler | `vllm/v1/core/sched` | Like a restaurant host. Every step it decides which requests get GPU time and how much. |
| KV cache | `vllm/v1/core` | The model's short-term memory of what it has read. Stored in fixed-size pages called blocks. |
| Model runner | `vllm/v1/worker/gpu` | Runs the model on the GPU one step at a time. "V2" (MRV2) became default in Q3; V1 gets deleted in Q4. |
| Models & kernels | `model_executor`, `csrc` | Each model's code, plus the GPU programs (kernels) that do the actual math. **Your area.** |

Then the new tokens go back up the chain and stream out as text. Repeat a few thousand times per answer.

Inside "models & kernels", most of the GPU time goes to **matrix multiplies** (called GEMMs: every linear layer and every MoE expert is one) and **attention**. Making those cheaper with fewer bits per number is what quantization is about.

---

## Quantization in plain English

A model is billions of numbers called weights. Quantization stores each one with fewer bits. Fewer bits mean less GPU memory, less data to move, and (if the GPU has the right hardware) faster math.

| Term | What it means |
|---|---|
| **BF16, FP8, FP4** | BF16 is the normal 16-bit format. FP8 uses 8 bits (half the memory). FP4 uses 4 bits (a quarter). FP4's "E2M1" layout has only 16 possible values: ±0, 0.5, 1, 1.5, 2, 3, 4 and 6. Every FP4 value also exists in FP8, so FP4 → FP8 conversion is exact. |
| **Scales** | FP4 only counts to 6, so each small group of weights gets its own multiplier, called a scale. **NVFP4**: one FP8 scale per 16 weights, plus one FP32 scale per tensor. **MXFP4 / MXFP8**: one power-of-two scale per 32 values. **Block FP8** (`FP8_PB_WO`): one scale per 128×128 tile. |
| **W4A16, W8A8, W4A4** | W is how many bits the weights use; A is how many bits the activations use (the numbers flowing from layer to layer). W4A16 stores 4-bit weights and does 16-bit math. W8A8 does the math in FP8, which Hopper can. W4A4 does the math in FP4, which only Blackwell can. |
| **Prefill vs decode** | Decode writes the answer one token at a time; its speed is limited by how fast weights are read from memory. 4-bit weights mean 4× less to read than BF16, so decode gets faster even without FP4 hardware. Prefill reads the whole prompt at once; its speed is limited by the math, so on Hopper the 16-bit math sets the ceiling. |
| **PTQ vs QAD** | PTQ (post-training quantization) simply rounds the finished weights: fast to do, but some accuracy is lost. **QAD** (quantization-aware distillation) keeps training the 4-bit model to copy the full-precision model's outputs, so it learns to work around the rounding. NVIDIA reports near-BF16 accuracy with QAD ([paper](https://huggingface.co/papers/2601.20088)). |
| **Mixed precision** | Each layer can use its own format. ModelOpt checkpoints with `quant_algo: MIXED_PRECISION` list a format per layer in `quantized_layers` in `config.json`. For example, Qwen3.8-Flash-Next-FP4-Local has 438 MXFP8 layers (attention, shared experts), 48 NVFP4 expert groups, 58 W4A16_NVFP4 groups (the MTP draft experts) and one NVFP4 embedding. |
| **CSF** | A new lossless compression for the NVFP4 scale bytes, used by the FP4-Local checkpoints. It saves about 3.3 GiB of expert scales for Qwen3.8-Flash-Next. |

### What each GPU can do natively

| Arch | Examples | Fast math | Notes |
|---|---|---|---|
| sm_80 Ampere | A100 | BF16, INT8 | Where Marlin was designed |
| sm_89 Ada | L4, RTX 4090 | adds FP8 (older `mma` instructions) | Marlin's FP8-activation mode is allowed here |
| **sm_90 Hopper** | **H200 (yours)**, H100 | BF16 and FP8 through WGMMA (big warp-group instructions). **No FP4.** | H200 has 141 GB. FP8 math runs about twice as fast as BF16. |
| sm_100 Blackwell DC | B200, GB200 | adds FP4/FP6 with block scales in hardware (`tcgen05`) | Most new FP4 kernels are written for this first |
| **sm_120 / 121** | **RTX 5060 Ti (yours)**, RTX PRO 6000, DGX Spark (121) | FP4 and FP8 block-scaled `mma`, but no `tcgen05` | About 99 KB of shared memory per block, so big datacenter kernels may not fit |

### The kernel families you'll meet

| Name | What it is | Runs on |
|---|---|---|
| Marlin | The classic "unpack 4-bit weights, do 16-bit math" GEMM. Supports many formats. Uses the older `mma` instructions, so it doesn't use Hopper's WGMMA. | sm_75+ |
| Humming | Newer multi-format library by jinzhen-lin. Preferred over Marlin on sm_90 since #56997. | sm_75+ |
| Machete | Hopper-native mixed-input GEMM for INT4/INT8 weights. | sm_90 |
| CUTLASS | NVIDIA's C++ template library. vLLM's FP8 `scaled_mm`, W4A8 and FP4 GEMMs live in `csrc/`. | per kernel |
| FlashInfer | NVIDIA-maintained kernel library (TRTLLM, CuTe-DSL, CUTLASS kernels). Most FP4 MoE kernels. | mostly sm_100+ |
| b12x | Kernel library for sm_12x (RTX PRO 6000, DGX Spark). The FP4-Local containers rely on it. | sm_120/121 |
| DeepGEMM | DeepSeek's FP8 GEMM, compiled at runtime the first time it sees a shape. | sm_90, sm_100 |
| Triton | A Python-like GPU language. Used for MoE, attention and lots of small ops. | everywhere |

### How vLLM picks a kernel

1. `config.json` names the format, and a config class turns it into a "quant method" for each layer. ModelOpt formats live in `vllm/model_executor/layers/quantization/modelopt.py`.
2. For a linear layer, the method walks a per-GPU list of kernels for that format and takes the first one that says it can run the layer (`vllm/model_executor/kernels/linear/__init__.py`). MoE layers do the same through an "oracle" (`vllm/model_executor/layers/fused_moe/oracle/`).
3. On sm_90 only, `prioritize_humming` moves Humming to the front of those lists (`quantization/utils/humming/priority.py:11`). You can override the choice with `--linear-backend` and `--moe-backend`, which is very handy for testing.

---

## Other words you'll see in issues

| Term | What it means |
|---|---|
| KV cache / blocks | Memory of previous tokens, so the model doesn't re-read the whole conversation every step. Split into pages ("blocks") like a notebook. |
| Prefix caching | If two requests start with the same text, they share the same pages instead of computing them twice. Huge for agents and chat. |
| MoE / experts | Mixture of experts: each layer has many small MLPs ("experts") and a router picks a few per token. Most of a modern model's weights are experts, so expert kernels matter most for memory. |
| TP / DP / EP / PP | Ways to use many GPUs. TP splits each layer across GPUs. DP runs copies of the model. EP spreads the experts over GPUs. PP gives each GPU a different set of layers. |
| Speculative decoding / MTP | A small fast "drafter" guesses the next few tokens and the big model checks all of them at once. MTP is a drafter built into the model's own checkpoint. |
| Hybrid models (GDN, Mamba, KDA) | Some layers keep one running "state" instead of a KV cache. Qwen3.8 and GLM-5.3 are hybrids. |
| Fusion | Doing two small GPU steps in one kernel so the data never leaves the chip in between. For example, "SiLU then convert to FP8". |
| CUDA graphs / torch.compile | Record the GPU work once and replay it, to cut per-step overhead. Many "wrong only with graphs on" bugs live here. |
| Shared memory | The fast on-chip scratchpad each GPU core group gets. Kernels that ask for more than the GPU has simply fail to launch, which is the classic sm_120 problem. |
| SIG | Special Interest Group. A team that owns one area and meets weekly. Each SIG has its own Q4 goals. |

---

## The Q4 2026 plan, in plain English

From the pinned roadmap [#61024](https://github.com/vllm-project/vllm/issues/61024) (opened 10 Oct). The quarter's theme is "make fast agent serving easy to deploy".

| SIG | What they want by December | Where you fit |
|---|---|---|
| Quantization | New 3-bit format on Rubin. Production 4-bit KV cache. Convert any checkpoint to any format on load. One dispatch system (QuantKey) everywhere. Benchmark overlapping kernels and pick per-architecture defaults. | **Q1–Q3, K1** |
| Model Performance | Fast on day one for new models (DeepSeek V4.1, GLM 5.3, Qwen3.8). 4-bit KV cache on more models. Perf dashboard. | Q1 (Qwen3.8 / GLM 5.3 on Hopper), K2 |
| torch.compile | Replace compiler fusion passes with explicit fused ops in model code ("Flat Model"). | K2, K3 |
| CI | Test results in about 30 minutes. Auto-pick which tests to run. Handle flaky tests. | Hopper mixed-precision accuracy test (Q1 step 5), sanitizer job (K4) |
| Core | Delete the old model runner. Per-model KV-cache layouts. Rust frontend by default. | Not kernel work |
| Large Scale Serving | Agent cache hints, deployment recipes, recovery from failed GPUs. | K6 only, with 4–8 GPUs |
| Spec Decode | Drafters for any attention type, adaptive checking for GDN models. | Not kernel-first |
| AMD, Agentic API, Omni / RL | ROCm speed, multi-agent APIs, speech/diffusion, RL weight updates. | Needs other hardware or isn't kernels |

---

## Where vLLM is heading

Eight directions show up again and again in the roadmap, RFCs and recent PRs. Each says what is changing, gives an example, and says what it means for you.

1. **Fusions are moving out of the compiler and into model code.**
   A GPU is fastest when it does several small steps in one trip to memory. After an MLP's activation (SiLU), the result is converted to FP8 for the next matrix multiply. Done separately, the GPU writes the 16-bit result out and reads it back. Fused, it never leaves the chip. vLLM used `torch.compile` "passes" that spotted these patterns and glued them together automatically. That proved fragile: a small change in model code silently disables the fusion, and it's hard to debug.
   The new plan, RFC [#42770](https://github.com/vllm-project/vllm/issues/42770) ("Flat Model"), is for model files to call the fused op directly. For example, `vllm/model_executor/models/llama.py:120` does `maybe_fused_act_quant(self.act_fn, x, self.down_proj)`. The sequence-parallel pass is already gone (#60517); tracker [#43224](https://github.com/vllm-project/vllm/issues/43224) lists the rest.
   **For you:** two kinds of work. Switch model families over by hand (K2), and write the fused kernels that don't exist yet (K3).
2. **One dispatch system for every quantized kernel.**
   vLLM supports about 20 quantization formats and dozens of kernels. Each format used to pick its kernel with its own if/else chain. Now a format is described by a "QuantKey" label (for example "FP8, one scale per 128 block, dynamic"), and shared code picks the kernel: per-GPU priority lists for linear layers, "oracles" for MoE layers. The leftover old code is being migrated ([#54959](https://github.com/vllm-project/vllm/issues/54959)), mostly by people who already claimed pieces.
   **For you:** reading this code is the fastest way to learn which kernel runs where. Every lane below touches it.
3. **Benchmarks decide the defaults.**
   For one format there are often three to six kernels (Marlin, Humming, CUTLASS, FlashInfer, b12x and so on). The fastest one depends on the GPU and the matrix shape. The Quantization SIG wants to "benchmark overlapping CuTe-DSL, b12x, FlashInfer TRTLLM, Marlin, Qutlass, and Humming paths to choose per-architecture defaults and retire superseded paths". Humming became the default over Marlin on Hopper ([#56997](https://github.com/vllm-project/vllm/pull/56997)). SM10x and SM12x are next, and nobody owns that.
   **For you:** K1. Numbers from two architectures are exactly what's missing.
4. **4-bit everywhere, mixed per layer.**
   New models now ship 4-bit checkpoints on launch day, and most new NVIDIA ones mix formats per layer (24 of the latest 40 NVFP4 releases). Hopper has no FP4 math, so these run through "unpack to 16-bit" kernels, and those paths get far less testing than the Blackwell ones.
   **For you:** Q1 and Q2. This is the biggest gap that fits your H200.
5. **Convert formats while loading.**
   "Online requantization" loads a checkpoint in one format and converts it to another during loading, for example NVFP4 → FP8 so Hopper's FP8 units can do the math. Linear layers landed in #55684 and MoE in #60276. Several follow-ups are listed and unclaimed.
   **For you:** Q3.
6. **Community forks ship first; upstream catches up.**
   The FP4-Local checkpoints only run in containers from the "Local Inference Lab" community fork, which has the b12x kernels and a loader for compressed scales. Their model cards say "upstream vLLM support for this format has not yet landed". Bringing such features upstream cleanly, with tests, is valued work. Draft [#60646](https://github.com/vllm-project/vllm/pull/60646) (CSF scale decoder) is the first piece.
   **For you:** Q1 step 4 (the second CSF variant, which nobody handles).
7. **Frontier models get hand-made kernels.**
   MonoKernels ([#60904](https://github.com/vllm-project/vllm/issues/60904)), Cake kernels via FlashInfer ([#59725](https://github.com/vllm-project/vllm/issues/59725)) and the DeepSeek-V4.1 specialized decode ([#61868](https://github.com/vllm-project/vllm/pull/61868)) fuse a whole layer, or a whole decode step, into one kernel for one specific model. NVIDIA, AMD and model-vendor engineers with early model access write them.
   **For you:** watch and learn; don't compete here yet.
8. **Hybrid-model kernels and the new model runner are busy.**
   Triton kernels for GDN, Mamba and KDA layers, "batch-invariant" kernels (identical numbers regardless of batch, for RL), and the sampler/logprob kernels in `vllm/v1/worker/gpu/` all see a lot of activity. Bugs get fixed fast, usually by whoever reports them, and issues often get two or three competing PRs (#60095 has three).
   **For you:** skip for now.

---

## Quant track: FP4 models on Hopper

This is the lane you asked about, and the one with the most room. Three steps: make the checkpoints **run correctly** on Hopper (Q1), make them **fast** (Q2), and offer a **convert-at-load** option (Q3).

### Q1 · Run NVIDIA's FP4-Local and other mixed checkpoints on Hopper
🟢 Nobody working on it · Medium–large, splits into PRs · 1× H200 (Qwen), 2× H200 (GLM) · ⭐ flagship

- **In plain English:** On 8 Oct NVIDIA released two "FP4-Local" models ([Qwen3.8-Flash-Next](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) and [GLM-5.3-Flash](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)), trained with QAD so the 4-bit version nearly matches the original. They are the smallest good versions of these models. The model cards list only Blackwell GPUs and say you must use a community container: upstream vLLM can't load them yet, and nobody has tried to make them run on Hopper.
  Hopper can't do FP4 math, but it can store FP4 weights and unpack them inside the kernel; Marlin and Humming already do that. So nothing in the hardware stops this from working. You'd make it work, prove it with numbers, and keep it working with a CI test.
- **Why it's worth it:**

  | Model | BF16 | FP8 | NVFP4 | FP4-Local | H200s needed |
  |---|---:|---:|---:|---:|---|
  | Qwen3.8-Flash-Next | 335 GiB | 173 GiB | 124 GiB | **95 GiB** | **1** (FP8 needs 2) |
  | GLM-5.3-Flash | 306 GiB | — | 190 GiB | **165 GiB** | 2, with more room left for KV cache |

  One H200 holds about 131 GiB. Sizes are the weight files on Hugging Face. The same fixes also help the other 22 MIXED_PRECISION checkpoints NVIDIA released recently.
- **What breaks today** (from reading the code; confirm each on your H200):
  1. **Compressed scales.** FP4-Local stores its NVFP4 scales with CSF compression, and `main` has no decoder, so loading fails on every GPU. Draft [#60646](https://github.com/vllm-project/vllm/pull/60646) (stecasta, 8 Oct, no reviews yet) decodes the Qwen variant (`*.nvfp4_csf_fixed` + `*.nvfp4_csf_exceptions`). GLM uses a second variant, `csf_32x2_v1`, with five streams per tensor (`csf32_page_bases`, `csf32_bases`, `csf32_codes`, `csf32_offsets`, `csf32_exceptions`). Nothing in vLLM handles it.
  2. **Hopper runs a path nobody validated.** `ModelOptMixedPrecisionConfig` claims support from SM75 and says it was "validated end-to-end on a Tesla T4 (SM75) and A100 (SM80)" (`modelopt.py:1640`). Those GPUs run Marlin. On sm_90 only, Humming jumps ahead of Marlin for every linear and MoE format (`humming/priority.py:11`, `kernels/linear/__init__.py:419`, `fused_moe/oracle/nvfp4.py:213`). So an H200 runs kernels the mixed format was never tested with. This has broken model startup in CI before (#60315; the cause was fixed in #60337 and #61556).
  3. **Known Marlin FP4-MoE bugs on Hopper.** [#49070](https://github.com/vllm-project/vllm/issues/49070): MiniMax-M3-NVFP4 gives garbage and crashes on H100/H200. One fix merged (#48929); the unaligned-size fix #48624 is still open. [#48895](https://github.com/vllm-project/vllm/issues/48895): wrong top-k weights in `moe_wna16_marlin_gemm`, still open.
  4. **Nobody has posted logs.** [#57748](https://github.com/vllm-project/vllm/issues/57748) tried GLM-5.3-Flash-NVFP4 on 4× H100, was told "not a vLLM issue", and has no error output.

  The model code is already Hopper-ready. Qwen4Exp has sm_90 GEMM plans (`vllm/models/qwen4_exp/nvidia/low_latency_gemm.py:72`) and sm_90 attention configs, and GLM5Next's FlashKDA supports sm_90. The gaps are in the quantization layer, which is your area.
- **Plan:**
  1. **Test matrix (1–2 days on the H200).** Go from small to big so each run is quick: `nvidia/Qwen3.6-35B-A3B-NVFP4` (22 GiB, FP8 + W4A16_NVFP4) → `nvidia/Qwen3.8-27B-NVFP4` (20 GiB, FP8 + NVFP4) → `nvidia/Qwen3.8-Flash-Next-NVFP4` (124 GiB, no CSF) → `nvidia/Qwen3.8-Flash-Next-FP4-Local` (95 GiB, with #60646 and #48624 applied). Run each twice: with defaults, then with `--moe-backend marlin --linear-backend marlin`. If Marlin works and the default doesn't, the problem is Humming on sm_90. Record the error, the failing layer, and a GSM8K score. Post your results on #60646 and #48624 too; a hardware test report helps stalled PRs get merged.
  2. **Open one issue with the matrix**, titled something like "[Bug][Hopper] ModelOpt MIXED_PRECISION / FP4-Local checkpoints on SM90". That's evidence nobody else has, and it makes the follow-up PRs easy to review.
  3. **One PR per root cause**, each with before/after output.
  4. **Decoder for the GLM CSF variant** (`csf_32x2_v1`). Plain Python/torch that you can write and unit-test on CPU at home. Ask stecasta on #60646 whether to add it there or as a follow-up. For reference implementations, see the Local Inference Lab vLLM fork and TensorFold (its #1021 merged today). Check their licenses before borrowing any code.
  5. **Keep it working.** Add a Hopper GSM8K config for one small mixed checkpoint. The Humming accuracy jobs run on L4, A100 and H100, but none of them uses a ModelOpt MIXED_PRECISION checkpoint.
- **Later:** #60646 decodes the scales at load time, so the GPU still holds them uncompressed. Keeping them compressed in VRAM (about 3.3 GiB for Qwen3.8-Flash-Next) needs kernels that read CSF directly. That's a kernel project for after the basics work.
- **People:** stecasta (CSF loader; mixed-precision NVFP4 detection in #56050), jinzhen-lin (Humming), mgoin (quantization lead).

### Q2 · Faster prompts for FP4 models on Hopper: FP8 activations
🟢 Unowned · Measure first; big kernel later · H200

- **In plain English:** On Hopper, FP4 weights are unpacked to 16-bit and multiplied in 16-bit (W4A16). For decode that's fine, because reading the weights is the bottleneck and 4-bit weights are 4× smaller. For long prompts (prefill), the math is the bottleneck. Hopper's FP8 units do about twice the work per second of BF16 (H100/H200 SXM: about 990 vs 1,980 dense TFLOPS). Every FP4 value fits in FP8 exactly, so unpacking to FP8 instead of BF16 loses nothing. The catch is the scales.
- **Today:**
  - Humming can use FP8 activations on sm_90, but picks 16-bit for group-scaled weights (`humming/schema.py:356`: "prefer wna16 for groupwise weight or groupwise input").
  - Marlin's FP8-activation mode refuses sm_90 ("It is slower than Marlin W4A16 on other devices", `marlin_utils.py:731`), because Marlin doesn't use Hopper's WGMMA instructions.
  - The only Hopper-native W4A8 kernel, CUTLASS W4A8 (`csrc/quantization/cutlass_w4a8/`), accepts only INT4 weights with one scale per 128 (`kernels/linear/mixed_precision/cutlass.py:36-60`).
- **The hard part:** NVFP4 has a new scale every 16 weights, but Hopper's FP8 multiply works on chunks of 32. The kernel must apply two different scales inside one chunk, which costs extra work. That's probably why the existing kernels fall back to 16-bit.
- **What you'd do:**
  1. **Measure.** On the H200, compare Humming W4A16 with Humming FP8-activation for NVFP4 on real shapes (Qwen3.8 expert sizes, batch 1 to 8192). If FP8 wins for big batches, open a PR that switches based on batch size, backed by the table.
  2. **If nothing fast exists:** a Hopper kernel for FP4 weights × FP8 activations with per-16 scales, starting from `cutlass_w4a8`. That's a multi-month project, so talk to jinzhen-lin and mgoin before starting.
- **Honest note:** the measurement may show W4A16 is already the best choice. Publishing that with data is still useful, and it only costs a few days.

### Q3 · Convert at load time: online requantization leftovers
🟡 Owner welcomes standalone PRs · Medium · H200

- **In plain English:** Instead of running a 4-bit checkpoint with 4-bit kernels, vLLM can convert it while loading, for example to FP8, so Hopper's fast FP8 path does the math. The model gets bigger (FP8 is twice the size of FP4), so this is the **speed** option for people with spare memory. Q1 is the **memory** option.
- **What's going on:** Linear layers landed in #55684 and MoE in #60276. #60276 lists leftovers nobody has claimed:
  - "Requantization unnecessarily requires a runnable backend for the source format". This is the one that matters on Hopper.
  - Fail early when the source method has no `dequantize_weight`.
  - Convert weight by weight instead of loading everything first. This saves peak memory.
  - Requantize to `nvfp4_per_token`.
- **Taken:** FP8 `dequantize_weight` (waynehacking8). Ask fxmarty-amd on [#59062](https://github.com/vllm-project/vllm/issues/59062) which leftovers are free.

---

## Online quantization and llm-compressor, answered

- **Does vLLM ship a model compressor?** Yes: [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) (about 3,900 stars, release 0.12.0.2 on 9 Oct). It works **offline**: it runs calibration-based methods such as GPTQ and AWQ on sample data, then saves a "compressed-tensors" checkpoint that vLLM loads directly. Most `RedHatAI/*` checkpoints on Hugging Face are made this way. It doesn't write GGUF.
- **What is online quantization?** Conversion **while vLLM loads the model**, with no calibration data. Each weight is simply rounded to the nearest value in the target format. It's fast and convenient. At 8 bits it's close to calibrated quality; at 4 bits it usually loses more accuracy than llm-compressor's calibrated methods or NVIDIA's QAD.
- **Higher bit → lower bit?** **Yes, from BF16/FP16 today.** `--quantization` accepts `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `mxfp4`, `int8_per_channel_weight_only` (MoE only) and `nvfp4_per_token` (MoE only, Blackwell only). See `vllm/config/quantization.py` and `docs/features/quantization/online.md`. On your H200, `mxfp4` runs as 4-bit weights with 16-bit math.
- **From an already-quantized checkpoint?** That's "requantization", and it's limited today. Supported sources are ModelOpt MXFP8 linear layers and NVFP4 MoE layers (compressed-tensors, ModelOpt, Quark). Going **FP8 → 4-bit** needs FP8 `dequantize_weight`, which waynehacking8 is working on. GGUF isn't a supported source (see G4). The open pieces are in [Q3](#q3--convert-at-load-time-online-requantization-leftovers).

---

## GGUF track: make popular local quants fast in vLLM

GGUF is llama.cpp's file format and the most-downloaded way to get a quantized model. `unsloth/Qwen3.8-27B-GGUF` has **6.7 million** downloads; NVIDIA's FP4-Local Qwen has about 6,000. In June vLLM moved GGUF out of the main repo into [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin) (#39612), and the docs still call it "highly experimental and under-optimized". It's not on the Q4 roadmap, which is why it's quiet: high demand, low competition.

| Plugin fact | Value (19 Oct) |
|---|---|
| Size | 54 stars, one main maintainer (Isotr0py, who did 24 of the 43 merged PRs) |
| Review speed | Outside PRs merged in a median of 1.3 days, but nothing has merged since 24 Sep and 28 PRs are waiting. Isotr0py is busy in the main repo (55 PR reviews in the last 10 days). |
| CUDA kernels | Copied from llama.cpp build b2899 (May 2024) and never updated (#142) |
| Recent work | Model adapters (Gemma 4, Qwen3.5/3.6 + MTP, diffusion), small fixes, one big kernel upgrade waiting for review (#141) |

### What GSQ-RCO actually is

[ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) (1.3 million downloads) is **not a new file format**. It's standard GGUF in which each tensor gets its own quant type, picked by a search: GSQ learns better rounding, and RCO decides which tensor gets how many bits under a size budget. Its 3.5-bit file mixes IQ3_S, IQ4_XS, IQ3_XXS, Q4_K, IQ2_S and Q2_K (plus BF16/F32 for small tensors). The model card reports it task-lossless at 11.8 GB, against 53.8 GB in BF16. Unsloth's "UD" quants follow the same per-tensor idea. So "support new GGUF quants" mostly means **making mixed files fast**. There are also four genuinely new types the plugin can't run at all: Q1_0, Q2_0, MXFP4 and NVFP4 (#142).

### Why these files are slow in vLLM today

1. **Old kernels, and no batched kernel for IQ types.** GGUF has a one-token-at-a-time kernel (MMVQ) and a batched one (MMQ). The plugin's MMQ covers only the older Q and K types (`quantization/utils.py:73-75`). For IQ types, which make up most of GSQ-RCO and unsloth UD files, any step with more than 8–16 tokens (prompts, or more than a handful of users) **unpacks the whole weight to BF16 on every step** and then does a normal matmul (`quantization/linear.py:38-54`). That's exactly the batched case vLLM exists for.
2. **Mixed types inside fused layers.** vLLM fuses q/k/v into one matrix, and gate/up into another. When the parts have different GGUF types, the plugin pads them into one buffer, and on every forward pass it slices each part back out with `.contiguous()` (`quantization/linear.py:247-257`). That copies the narrower parts every step. In the GSQ-RCO 3.5-bit file, **40 of 64 layers** have different gate/up types and **all 16** attention layers mix q/k/v types (54 of 64 in the 2.5-bit file). By my estimate that's hundreds of MB of copying per step; measure it before you quote a number.

Related: vLLM main can already run GGML-style IQ2_XS, IQ2_XXS and Q8_0 blocks stored inside ModelOpt mixed checkpoints, but only through b12x on sm_12x (`modelopt_block_quant.py`). Your 5060 Ti qualifies; the H200 doesn't.

### G1 · Help the GGUF kernel upgrade land ([plugin #141](https://github.com/vllm-project/vllm-gguf-plugin/pull/141))

🟡 Claimed by Maxwell-Lyu: support it, don't compete · No code · 5060 Ti + H200

- **In plain English:** A contributor ported current llama.cpp CUDA kernels into the plugin. It adds batched kernels for every IQ type and the four new types, with a routing layer that picks the fastest kernel per shape. It's big (+7,337 lines) and has had no reviews since 29 Sep. The only benchmarks posted are from an L4 and a V100.
- **What you'd do:** Build it, run the plugin's kernel tests, and benchmark end to end. `Qwen3.8-27B-GSQ-RCO-IQ2_XS` (7.8 GiB) and the smaller unsloth UD quants fit your 5060 Ti; run the bigger ones on the H200. Post before/after tables for 1 and 32 users. On sm_120, check that the batched kernels fit the 99 KB shared-memory limit.
- **Why:** No merge needed. Your data covers the two architectures missing from the PR, and that's what helps a reviewer say yes.

### G2 · Stop per-step weight copies in mixed-type fused layers

🟢 No PR · Small · 5060 Ti at home

- **In plain English:** When a fused layer's parts use different GGUF types, the plugin re-copies some of the weights on every step instead of laying them out properly once at load time. Fix the layout once and the copies disappear.
- **The fix:** At load time, store each part as its own contiguous tensor at its natural width, so `apply()` never slices and copies. Prove it with a profile (the copy kernels gone) and tokens/s before and after on `Qwen3.8-27B-GSQ-RCO-IQ2_XS` and one unsloth UD quant.
- **Watch out:** #141 also edits `linear.py`, so keep the diff small and offer to rebase on whichever lands first. Read the other open mixed-type PRs first: #107, #131, #134 and #135 (MoE dispatch).

### G3 · Make GGUF mainstream: benchmark vs llama.cpp, tested models, docs

🟢 Nobody doing it · Medium, after #141 · H200 + 5060 Ti

- **In plain English:** The "highly experimental" warning stays until someone shows GGUF in vLLM is correct and competitive. llama.cpp is great for one user; vLLM's strength is many users at once. A fair table shows where each wins.
- **What you'd do:**
  1. Same GGUF files (an unsloth UD quant and a GSQ-RCO quant), llama.cpp vs vLLM, with 1 and 32 concurrent users, on both your GPUs.
  2. An accuracy check (GSM8K) on a short list of popular files, added to the plugin's tests. The plugin already documents tested models (#86).
  3. A docs PR in the main repo (`docs/features/quantization/gguf.md`) that softens the warning and links the table.

### G4 · Later: let GGUF use vLLM's convert-at-load

Idea, RFC first · Medium

- **In plain English:** If the GGUF method could hand back BF16 weights (`dequantize_weight`, which the plugin's `ggml_dequantize` kernel already does), vLLM's online requantization could convert high-bit GGUFs at load time. For example, Q8_0 or Q6_K → FP8 would use Hopper's fast FP8 path. It wouldn't help 2–3-bit files, because the converted model would be bigger.
- **First step:** A short RFC on the plugin repo, asking Isotr0py and fxmarty-amd (owner of online quantization) whether they want it.

---

## Kernel track

Steady PRs that build a kernel reputation. Each lane starts with a plain-English summary, then the details.

- **Have:** H200 (sm_90) at work, RTX 5060 Ti 16 GB (sm_120) at home.
- **Rent when needed:** B200 (sm_100) for an SM100 benchmark column · a second H200 for GLM-5.3-Flash-FP4-Local · an RTX PRO 6000 to compare FP4-Local against its "native" Blackwell path.

### K1 · Per-architecture kernel defaults: Humming vs Marlin (and friends)
🟢 No issue, no PR · Medium · Both your GPUs (+ B200 rental)

- **In plain English:** For 4-bit and 8-bit weights there are several competing kernels, and vLLM picks one per GPU from a hard-coded order. Only Hopper's order was tuned with data. Measuring every kernel on more GPUs and setting the order from the numbers is a roadmap item nobody owns.
- **What's going on:** The whole Humming-vs-Marlin decision is one constant: `_HUMMING_PREFERRED_CAPABILITIES = frozenset({90})` in `quantization/utils/humming/priority.py:11`. The roadmap says to extend it to SM10x/SM12x. Humming itself marks its FP8 path as slow on SM10x/SM11x (`humming/schema.py:333`), so a B200 needs its own answer.
- **Why evidence:** Changing this default broke things once already (#60315 and #56997). Maintainers need a **works / fails / speed** matrix per format and per architecture, not just "Humming is faster".
- **Deliver:**
  1. A benchmark that runs every registered kernel for one format through vLLM's kernel abstraction on the same shapes. Today `benchmarks/kernels/` has separate Marlin and Machete scripts and nothing for Humming.
  2. A results table: format (INT4 W4A16, FP8 W8A16, NVFP4 A16, MXFP4, MXFP8, INT8, plus the MoE versions) × architecture (sm_90, sm_120, and sm_100 on a rented B200).
  3. A PR that sets the preference per format and per architecture, backed by the table.
  4. Follow-up PRs that delete kernel paths that never win.
- **Also answer:** On Hopper, dense W4A16 tries Machete before Marlin and Humming (`kernels/linear/__init__.py:531`). Is that still right?
- **First step:** Open an issue such as "[Quantization] Per-architecture kernel defaults: benchmark matrix for sm_90 / sm_100 / sm_120". Tag jinzhen-lin and mgoin, and post the plan before writing the benchmark.

### K2 · Manual activation+quant fusion for the next model families
🟢 Free per model family · Small–medium · H200

- **In plain English:** The fused "SiLU, then convert to FP8/FP4" kernel exists, but each model file has to call it explicitly. Only three models do so far. You switch more model families over, test one model per family, and show the speedup. It's small, repeatable PRs that a maintainer explicitly asked for, and it teaches you the fusion contract every future fusion will use.
- **What's going on:** `maybe_fused_act_quant(self.act_fn, x, self.down_proj)` (`llama.py:120`) does the activation and the next layer's input quantization in one kernel. On `main` only `llama.py`, `nemotron_h.py` and `pixtral.py` use it.
- **In flight:** DeepSeekV2 / Mistral / Qwen2 / Qwen3-MoE (#57514), Granite / EXAONE 4 / Seed-OSS (#58665), Muse-Glimmer / LFM2 (#59470), Gemma (#59467), Qwen MoE shared MLP (#61107).
- **Free families:** `glm4_moe` (dense layers and shared experts use `Glm4MoeMLP`), `internlm2`, `minicpm`, `ernie45_moe`, `hy_v3`, `step3_text` / `step3p5`, `longcat_flash`, `mimo_v2`, `bailing_moe`, `olmo_hybrid`, `exaone` (v1), `cohere2_moe`, `granitemoeshared`, `lfm2_moe`, `falcon_h1`.
- **The ask:** On #43501: "Remaining work is to apply the manual pass to all the applicable models. This should be done in batches of PRs, where some representative model is tested for the changed architecture."
- **Test on H200:** Use a block-FP8 checkpoint; GLM-4.5-Air-FP8 fits on one H200. Show identical output (greedy text plus GSM8K) and the speedup, with a profile. Plain `--quantization fp8` uses per-token scales on an H200 (`online/fp8.py:192`), and that fused kernel is still in review (#58664, #49828), so test with static or block-128 checkpoints.
- **Merge outlook:** Stalled right now. #58665, #59470 and #61107 have had no maintainer review for 1–4 weeks; only #57514 (reviewed by mgoin) is moving. Wait for it to merge, then copy its pattern.
- **First step:** Once #57514 merges, comment on [#43501](https://github.com/vllm-project/vllm/issues/43501) naming your batch (for example GLM4-MoE + InternLM2 + MiniCPM) and the checkpoint you'll test.

### K3 · GELU-tanh + FP8 fused kernels (your first CUDA kernel)
🟢 No CUDA PR · Small–medium · H200, then sm_120

- **In plain English:** Gemma models use GELU-tanh instead of SiLU. There's a fused "SiLU then FP8" kernel but no GELU version, so on Hopper Gemma makes an extra round trip to memory in every MLP. The kernel template already exists. You plug in a different activation function, register it, and test it. It's a well-scoped first CUDA kernel.
- **What's going on:** The fused registry (`layers/fusion/fused_act_quant.py:182-193`) covers only SiLU (static FP8, block FP8, NVFP4) and ReLU². Gemma's only fused path is NVFP4 (#59467), which needs SM100+. There's no GELU + FP8 kernel anywhere in `csrc/`.
- **How:** `csrc/quantization/activation_kernels.cu` has `act_and_mul_quant_kernel<scalar_t, ACT_FN, fp8_type>` (line 44). `silu_and_mul_quant` (line 592) is just a launcher that plugs in `vllm::silu_kernel`. You'd add a new launcher, an op binding in `csrc/torch_bindings.cpp`, a registry entry, a kernel test and a benchmark. The block-128 version follows `silu_and_mul_per_block_quant`.
- **Merge outlook:** Stalled right now: the closest PR, #59467 (GELU + NVFP4), has had no review since 30 Sep.
- **Check first:** Ask toan-tq (#59467) and mgoin whether they want FP8 GELU variants, and which Gemma FP8 checkpoints use static or block scales. Needs a full CUDA build.

### K4 · Kernel memory-check sweep: find your own bugs
🟢 Nobody runs it in CI · Ongoing · H200 or 5060 Ti

- **In plain English:** `compute-sanitizer` is like valgrind for the GPU: it catches kernels that read or write outside their buffers. Those bugs cause random crashes and garbage output that are very hard to trace later. CI never runs it, so the bugs are sitting there. Bugs you find yourself are yours to fix, and nobody races you.
- **Proof it works:** One contributor found three out-of-bounds kernel bugs this way in one day (#61800, #61802, #61803, 15 Oct).
- **How:** `PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python -m pytest <test> -k <small cases>`. Start with `tests/kernels/quantization`, `moe`, `mamba` and `attention`. It's slow, so pick small cases. Running the same tests on sm_90 and sm_120 doubles your coverage.
- **Later:** Propose a nightly sanitizer job to SIG CI.

### K5 · Missing H200 tuned MoE configs
🔴 Low priority: skip unless idle · H200

- **In plain English:** The Triton MoE kernel reads a small JSON file of tile sizes tuned for each GPU and matrix shape. If the file is missing, it uses generic sizes and runs slower. Many shapes have an H100 file but no H200 file, and vLLM doesn't fall back to the H100 one.
- **Details:** Unquantized MoE on Hopper uses Triton by default (`fused_moe/oracle/unquantized.py:78`). H200 has 51 config files and H100 has 62. 36 H100 shapes have no H200 twin, and `fused_moe.py:1097` only normalizes H200 names. Tune with `benchmarks/kernels/benchmark_moe.py --tune` for models people actually run. Reviews are slow (#60504, #60849, #61059), so treat this as filler between bigger PRs.

### K6 · DeepGEMM warmup gaps on Hopper ([#56684](https://github.com/vllm-project/vllm/issues/56684))
🟢 1 comment, no PR · 4–8 H200s (or rent)

- **In plain English:** DeepGEMM compiles a kernel the first time it sees a new matrix size. If that happens while serving, every user waits. The fix is to compile all the sizes the server can reach at startup.
- **Details:** DeepSeek-V4.1-Flash stalls for 3.4 s at a time when a new prefill chunk size appears. An H200 user reported a similar stall on sm_90. The pattern to copy is #60567 (GLM). Needs a multi-GPU node.

### Kernel areas to skip for now

| Area | Why |
|---|---|
| Per-token SiLU fused quant | Three PRs already (#58664, #49828, #44018) |
| MRV2 sampler / logprob kernels | 2–3 PRs per issue (#60095, #61998 claimed) |
| Triton attention tiling | Driven by their reporters (#61029, #61095) |
| GDN / KDA / Mamba kernel bugs | Reporters open fixes the same day (#61800–#61803, #62153) |
| MoE oracle migrations (#54959) | Mostly claimed: Humming #57538, Quark INT8 #58672, CT MXFP4 #56306, packed MXFP4 #56523 |
| Frontier-model kernels | MonoKernels, Cake and the DSv4.1 decode are vendor-owned |
| QuTLASS NVFP4 migration | Gated to SM10x (#61671); only with a rented B200 |

---

## Working between the H200 and home

1. **Kernel changes need a real build.** On the H200 machine, follow `docs/contributing/incremental_build.md`. Python-only changes (CSF decoder, dispatch fixes) can use `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`. Always use `uv`, never bare `pip`.
2. **Moving work home:** commit on a branch at work, then run `git format-patch origin/main..HEAD -o patches/`. Bring the patch files home and apply them with `git am patches/*.patch`. If allowed, pushing the branch to your GitHub fork from work is simpler.
3. **Bring the evidence too.** Keep benchmark output (JSON/CSV), test logs, GSM8K scores and profiler traces next to the patch. PRs need them.
4. **Re-test at home on sm_120** for anything architecture-specific (K1, K3, K4).
5. **Check your employer's open-source policy first.** The `Signed-off-by` line (DCO) states that you have the right to submit the code, and work hardware or work time can affect that.

### Before you open a PR

1. **Run the duplicate checks** that `AGENTS.md` requires:
   ```
   gh issue view <N> --repo vllm-project/vllm --comments
   gh pr list --repo vllm-project/vllm --state open --search "<N> in:body"
   gh pr list --repo vllm-project/vllm --state open --search "<area keywords>"
   ```
2. **Comment before coding:** what you reproduced, which piece you'll do, and your approach in two lines.
3. **PR format:** a title prefix like `[Bugfix]`, `[Kernel]` or `[Quantization]`, a `Signed-off-by:` line, why it isn't a duplicate, the test commands with results, and a clear note if AI helped. You must be able to defend every line.
4. **Expect a red ❌ on your first PRs.** Until you have 4 merged PRs, `pre-run-check` fails until a committer adds the `ready` or `verified` label (`.github/workflows/pre-commit.yml`, line 41). Ask once in the `#pr-reviews` Slack channel.
5. **No tiny one-off PRs** (a single typo or one-line style fix); the repo rejects them.

---

*Checked on 19 Oct 2026 against `main` at `3e96f9c`, open PRs and issue timelines on GitHub, and model configs, weight-index files and model cards on Hugging Face. Sources: 239 open kernel, performance and quantization issues, the roadmap trackers #61024, #43224, #43501, #54959, #59062 and #60276, and the configs of NVIDIA's 40 newest NVFP4 checkpoints. Merge times come from the 139 quantization-labelled PRs merged since 15 Sep and the 43 merged PRs of vllm-gguf-plugin. GGUF findings come from the plugin at commit `e2b8ad5` (24 Sep), its open issues and PRs, and the GSQ-RCO tensor-allocation files. "What breaks today" in Q1 comes from reading the code and hasn't been run on Hopper; the test matrix is how you confirm it. Things change by the hour, so re-run the duplicate checks before you start anything.*
