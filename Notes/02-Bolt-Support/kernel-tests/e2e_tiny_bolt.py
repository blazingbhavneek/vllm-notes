"""Tiny Nemotron-H with ModelOpt IQ2_XXS / Q8_0 blocks, end to end through LLM().

IQ2_XXS goes on attention q/k/v/o (dense layers that torch.compile traces, which is
where review B2/B4 bite) and on the routed experts. Q8_0 goes on shared experts,
latent projections, lm_head and embed_tokens. Dummy weights; block scales are made
finite after load so outputs are meaningful.

On H200 / A100 / Ada the non-b12x path is chosen automatically (draft or new code).
On SM12x with b12x installed, pass --triton (new code only: forces the Triton path).

usage:
    python e2e_tiny_bolt.py eager [--triton]
    python e2e_tiny_bolt.py compiled [--triton]
    python e2e_tiny_bolt.py graphs [--triton]
    python e2e_tiny_bolt.py compare
Add --bf16 to any of these for a control run of the same model without quantization.

Known (2026-10-22, home sm_120): even the --bf16 control has prompt 0 diverge from
step 0 between eager and compiled, while prompts 1-2 match. That is vLLM-side, not the
block-quant code. Judge the quantized run against the control's pattern, not 100%.
"""

import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "tiny_bolt")
MODE = sys.argv[1]
TAG = "_bf16" if "--bf16" in sys.argv else ""  # control run: same model, no quantization

if MODE == "compare":
    # Same rule as vLLM's tests/models/utils.py:check_logprobs_close: sequences may
    # diverge, but at the first divergence each side's token must be in the other's
    # top-5. Exact equality is too strict: kernels differ by token count.
    runs = {m: json.load(open(os.path.join(HERE, f"tokens_{m}{TAG}.json")))
            for m in ("eager", "compiled", "graphs")
            if os.path.exists(os.path.join(HERE, f"tokens_{m}{TAG}.json"))}
    ok = True
    for m, run in runs.items():
        if m == "eager":
            continue
        for i, (a, b) in enumerate(zip(runs["eager"], run)):
            div = next((j for j, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]), None)
            close = div is None or (str(a[div][0]) in b[div][1] and str(b[div][0]) in a[div][1])
            ok &= close
            print(f"{m} vs eager, prompt {i}: first divergence at "
                  f"{'none' if div is None else div}/{len(a)}, close={close}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def recipe(codec):
    return {"quant_algo": codec.upper(), "group_size": 256 if codec == "iq2_xxs" else 32,
            "block_payload_bytes": 66 if codec == "iq2_xxs" else 34, "packing": "ggml"}


layers = {f"model.layers.1.mixer.{p}": recipe("iq2_xxs")
          for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
layers["model.layers.2.mixer.experts"] = recipe("iq2_xxs")
for p in ("shared_experts.up_proj", "shared_experts.down_proj",
          "fc1_latent_proj", "fc2_latent_proj"):
    layers[f"model.layers.2.mixer.{p}"] = recipe("q8_0")
layers["lm_head"] = recipe("q8_0")
layers["model.embed_tokens"] = recipe("q8_0")

os.makedirs(MODEL, exist_ok=True)
with open(os.path.join(MODEL, "config.json"), "w") as f:
    json.dump({
        "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h",
        "torch_dtype": "bfloat16", "vocab_size": 4096, "hidden_size": 4096,
        "hybrid_override_pattern": "M*E", "num_hidden_layers": 3,
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        "max_position_embeddings": 4096, "intermediate_size": 8192,
        "mamba_num_heads": 64, "mamba_head_dim": 64, "ssm_state_size": 64,
        "n_groups": 1, "conv_kernel": 4, "chunk_size": 128, "expand": 1,
        "mamba_hidden_act": "silu", "use_conv_bias": True, "use_bias": False,
        "mlp_bias": False, "mlp_hidden_act": "relu2", "layer_norm_epsilon": 1e-5,
        "n_routed_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 1024,
        "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 2048,
        "moe_latent_size": 1024, "routed_scaling_factor": 2.5, "n_group": 1,
        "topk_group": 1, "norm_topk_prob": True, "tie_word_embeddings": False,
        "eos_token_id": 2, "bos_token_id": 1, "pad_token_id": 0,
        **({} if TAG else {"quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None, "ignore": [], "quantized_layers": layers}}),
    }, f, indent=1)

from vllm import LLM, SamplingParams  # noqa: E402

kw = dict(model=MODEL, load_format="dummy", skip_tokenizer_init=True, max_model_len=256,
          max_num_seqs=4, max_num_batched_tokens=256, gpu_memory_utilization=0.3, seed=0)
if "--triton" in sys.argv:
    kw.update(linear_backend="triton", moe_backend="triton")
if MODE == "eager":
    kw["enforce_eager"] = True
elif MODE == "compiled":
    kw["compilation_config"] = {"cudagraph_mode": "NONE"}
else:
    kw["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4]}
llm = LLM(**kw)


def sanitize_scales(model):
    """Dummy weights give random fp16 block scales (some NaN/Inf). Make them finite.

    Handles raw blocks (uint8 [..., 66|34], the draft) and repacked fp16 `*weight_scale`
    tensors (the new layout). Writes in place, so captured CUDA graphs see the change.
    """
    report = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if TAG and p.dtype == torch.bfloat16 and p.dim() >= 2:
                p.mul_(40)  # dummy +-1e-3 weights give a degenerate model; widen them
                continue
            raw = p.dtype == torch.uint8 and p.dim() >= 2 and p.shape[-1] in (66, 34)
            if not raw and not (p.dtype == torch.float16 and name.endswith("weight_scale")):
                continue
            shape = p.shape[:-1] if raw else p.shape
            g = torch.Generator(device=p.device).manual_seed(len(name))
            d = (torch.rand(shape, device=p.device, generator=g) * 1e-3 + 2e-4).half()
            if raw:
                p.data[..., :2] = d.unsqueeze(-1).view(torch.uint8)
            else:
                p.data.copy_(d)
            report[name] = "raw" if raw else "repacked"
        paths = {n: getattr(m.quant_method, "use_triton", getattr(m.quant_method, "sm90_fallback", None))
                 for n, m in model.named_modules()
                 if "BlockQuant" in type(getattr(m, "quant_method", None)).__name__}
    return report, paths


(report, paths), = llm.apply_model(sanitize_scales)
print(f"[{MODE}] block-quant layers: {len(paths)}; non-b12x path on all: {all(paths.values())}")
for n, v in paths.items():
    print(f"    {n}: non-b12x={v}")
print(f"[{MODE}] sanitized {len(report)} scale tensors ({set(report.values())})")

prompts = [{"prompt_token_ids": list(range(3 + i, 40 + 7 * i))} for i in range(3)]
sp = SamplingParams(temperature=0.0, max_tokens=12, logprobs=5, ignore_eos=True)

if MODE == "compiled":
    from torch.profiler import ProfilerActivity, profile
    llm.generate(prompts[:1], sp, use_tqdm=False)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([{"prompt_token_ids": [5]}],
                     SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True),
                     use_tqdm=False)
    names = sorted({e.key for e in prof.key_averages()
                    if any(s in e.key for s in ("iq2", "q8", "block", "gemv", "gemm", "moe_pass", "combine"))})
    print(f"[{MODE}] block kernels seen during a 1-token decode: {names}")

outs = llm.generate(prompts, sp, use_tqdm=False)
toks = [list(o.outputs[0].token_ids) for o in outs]
steps = [[(t, {str(k): v.logprob for k, v in lp.items()})
          for t, lp in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)] for o in outs]
finite = all(all(lp is not None and all(abs(v.logprob) < 1e9 for v in lp.values())
                 for lp in o.outputs[0].logprobs) for o in outs)
print(f"[{MODE}] tokens: {toks}")
print(f"[{MODE}] logprobs finite: {finite}")
json.dump(steps, open(os.path.join(HERE, f"tokens_{MODE}{TAG}.json"), "w"))
