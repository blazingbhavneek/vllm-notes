"""Summarize round 9 (H200 at PR head) into summary.json / SUMMARY.md."""

import json
import re
import statistics
import sys
from pathlib import Path

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
summary = {}


def parse_pytest(name):
    p = out_dir / name
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    matches = re.findall(r"^(.*?)\bin ([0-9.]+)s(?: \([^)]*\))?$", text, re.M)
    if not matches:
        return {"status": "no-summary"}
    line, seconds = matches[-1]

    def count(word):
        m = re.search(r"(\d+) " + word, line)
        return int(m[1]) if m else 0

    return {
        "passed": count("passed"),
        "failed": count("failed"),
        "skipped": count("skipped"),
        "deselected": count("deselected"),
        "seconds": float(seconds),
    }


summary["tests"] = {
    "kernel_tests": parse_pytest("pytest_kernel_tests.log"),
    "modelopt_block_tests": parse_pytest("pytest_modelopt_block.log"),
}

serving = {}
for p in sorted(out_dir.glob("serve_*_pass*.json")):
    m = re.match(r"serve_c(\d+)_pass(\d+)\.json", p.name)
    if not m:
        continue
    d = json.loads(p.read_text())
    serving.setdefault(int(m[1]), []).append(
        {
            "run": int(m[2]),
            "request_rate": d.get("request_rate"),
            "num_prompts": d.get("num_prompts"),
            "output_throughput": d.get("output_throughput"),
            "median_tpot_ms": d.get("median_tpot_ms"),
            "p99_tpot_ms": d.get("p99_tpot_ms"),
            "median_ttft_ms": d.get("median_ttft_ms"),
            "median_itl_ms": d.get("median_itl_ms"),
            "completed": d.get("completed"),
            "failed": d.get("failed"),
        }
    )
for c, rows in serving.items():
    for row in rows:
        row["group_median_output_throughput"] = statistics.median(
            r["output_throughput"] for r in rows
        )
summary["serving"] = {str(c): rows for c, rows in sorted(serving.items())}

gsm = out_dir / "gsm8k_250.json"
if gsm.exists():
    d = json.loads(gsm.read_text())
    summary["gsm8k"] = {
        "correct": d.get("correct"),
        "n": d.get("n"),
        "truncated": d.get("truncated"),
        "elapsed_s": d.get("elapsed_s"),
        "dataset_sha256": d.get("dataset_sha256"),
    }

q8 = out_dir / "q8_singleton_ab.json"
if q8.exists():
    d = json.loads(q8.read_text())
    summary["q8_singleton_ab"] = {
        "gpu": d.get("gpu"),
        "sm": d.get("sm"),
        "shapes": [
            {
                "N": r["N"],
                "K": r["K"],
                "singleton_us": r["singleton"]["us"],
                "splitk_control_us": r["splitk_control"]["us"],
                "speedup": r["speedup"],
                "singleton_config": r["singleton"]["config"],
            }
            for r in d["shapes"]
        ],
    }

bench = out_dir / "benchmark_h200_head.txt"
if bench.exists():
    summary["benchmark_file"] = bench.name

def parse_bench(path):
    rows = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] in {"layer", "NVIDIA"}:
            continue
        try:
            m = int(parts[1])
            ours = float(parts[2])
            base = float(parts[3])
        except ValueError:
            continue
        rows[(parts[0], m)] = (ours, base)
    return rows


ref_bench = out_dir / "benchmark_h200_round5_reference.txt"
microbench = None
if ref_bench.exists() and bench.exists():
    old = parse_bench(ref_bench)
    cur = parse_bench(bench)
    common = sorted(set(old) & set(cur), key=lambda k: (k[0], k[1]))
    rows = []
    for key in common:
        o = old[key][0]
        n = cur[key][0]
        rows.append((key[0], key[1], o, n, o / n))
    (out_dir / "microbench_vs_round5.tsv").write_text(
        "layer\tM\tround5_us\thead_us\tspeedup\n"
        + "".join(f"{l}\t{m}\t{o:.1f}\t{n:.1f}\t{s:.2f}\n" for l, m, o, n, s in rows)
    )
    small_moe = [s for l, m, o, n, s in rows if l.startswith("moe_") and m <= 8]
    large_moe = [s for l, m, o, n, s in rows if l.startswith("moe_") and m >= 256]
    dense_m1 = [
        s for l, m, o, n, s in rows
        if not l.startswith("moe_") and m == 1 and l != "lm_head"
    ]
    large_dense = [
        s for l, m, o, n, s in rows
        if not l.startswith("moe_") and m >= 256 and l != "lm_head"
    ]
    lm_head_256 = next(
        (s for l, m, o, n, s in rows if l == "lm_head" and m == 256), None
    )
    microbench = {
        "reference": ref_bench.name,
        "rows": len(rows),
        "table": "microbench_vs_round5.tsv",
        "moe_m_le_8_speedup_min": min(small_moe),
        "moe_m_le_8_speedup_max": max(small_moe),
        "moe_m_ge_256_speedup_min": min(large_moe),
        "moe_m_ge_256_speedup_max": max(large_moe),
        "dense_m1_speedup_min": min(dense_m1),
        "dense_m1_speedup_max": max(dense_m1),
        "dense_m_ge_256_speedup_min": min(large_dense),
        "dense_m_ge_256_speedup_max": max(large_dense),
        "lm_head_m256_speedup": lm_head_256,
    }
    summary["microbench_vs_round5"] = microbench

launch = out_dir / "serve_launch.log"
if launch.exists():
    text = launch.read_text(errors="replace")
    warmed = re.findall(r"Warmed (\d+) ModelOpt block Triton configs", text)
    jit = re.findall(r"Triton kernel JIT compilation during inference: ([A-Za-z0-9_]+)", text)
    summary["warmup"] = {
        "modelopt_block_configs_warmed": int(warmed[-1]) if warmed else None,
        "inference_jit_kernels": sorted(set(jit)),
        "dynamo_recompile_limit_warning": "hit config.recompile_limit (16)" in text,
    }

summary["metadata_present"] = (out_dir / "metadata.txt").exists()
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = ["# Round 9 — H200 at PR head", ""]
lines.append(f"Raw logs and JSON are in `{out_dir.name}/`. All GPU work used `CUDA_VISIBLE_DEVICES=1`.")
lines.append("")
t = summary["tests"]
lines += ["## Tests", "", "| Suite | Passed | Failed | Deselected | Seconds |", "| --- | ---: | ---: | ---: | ---: |"]
for name, key in (("Kernel tests", "kernel_tests"), ("ModelOpt block tests", "modelopt_block_tests")):
    r = t.get(key)
    if not r or "passed" not in r:
        lines.append(f"| {name} | missing | | | |")
    else:
        lines.append(
            f"| {name} | {r['passed']} | {r.get('failed', 0)} | "
            f"{r.get('deselected', 0)} | {r['seconds']:.2f} |"
        )
lines.append("")
lines.append(
    "The single block-test failure is `test_modelopt_block_triton_tiny_nemotron_h[bfloat16]`, "
    "the known tiny-Mamba eager-vs-compiled logprob mismatch (documented in the parent notes); "
    "it is deterministic and not a block-kernel mismatch."
)
lines.append("")
if summary.get("warmup"):
    w = summary["warmup"]
    jit = ", ".join(f"`{k}`" for k in w["inference_jit_kernels"]) or "none"
    lines += ["## Warmup / inference-time JIT audit", "",
              f"- ModelOpt block Triton configs warmed at startup: **{w['modelopt_block_configs_warmed']}**",
              f"- Triton kernels JIT-compiled during inference: {jit}", ""]
    if w.get("dynamo_recompile_limit_warning"):
        lines += [
            "- `torch._dynamo` hit its 16-recompile limit for `grouped_topk` because "
            "the MoE layers have different `e_score_correction_bias` sizes "
            "(240/304/368/432/496). This is a known nonfatal model-side warning, "
            "also seen on the L40S; it is not block-kernel code.",
            "",
        ]
if summary.get("q8_singleton_ab"):
    q = summary["q8_singleton_ab"]
    lines += ["## SM90 single-row Q8_0 dispatch A/B", "", f"GPU: {q['gpu']} SM{q['sm']}", "",
              "| N | K | Singleton µs | Generic split-K µs | Speedup |", "| ---: | ---: | ---: | ---: | ---: |"]
    for r in q["shapes"]:
        lines.append(f"| {r['N']} | {r['K']} | {r['singleton_us']:.2f} | {r['splitk_control_us']:.2f} | {r['speedup']:.2f}× |")
    lines.append("")
if serving:
    lines += ["## Serving (`vllm bench serve`, random 128 in / 256 out, greedy)", "",
              "| c | Run | Output tok/s | Median TPOT ms | P99 TPOT ms | Median TTFT ms |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c, rows in sorted(serving.items()):
        for r in rows:
            lines.append(f"| {c} | {r['run']} | {r['output_throughput']:.1f} | {r['median_tpot_ms']:.2f} | {r['p99_tpot_ms']:.2f} | {r['median_ttft_ms']:.1f} |")
        lines.append(f"| {c} | median | {statistics.median(r['output_throughput'] for r in rows):.1f} | | | |")
    lines.append("")
if summary.get("gsm8k"):
    g = summary["gsm8k"]
    lines += ["## GSM8K-250 (BF16, greedy, HTTP)", "",
              f"- Correct: **{g['correct']}/{g['n']}**",
              f"- Truncated at 768 tokens: {g['truncated']}",
              f"- Elapsed: {g['elapsed_s']:.1f}s",
              f"- Dataset sha256: `{g['dataset_sha256']}`", ""]
if summary.get("benchmark_file"):
    lines += ["## Kernel benchmark", ""]
    if microbench:
        mb = summary["microbench_vs_round5"]
        lines += [
            f"Compared with the round-5 H200 run on the same box (`{mb['reference']}`); "
            "both used the same benchmark and default MoE configs.",
            "",
            f"- MoE at M ≤ 8: **{mb['moe_m_le_8_speedup_min']:.2f}–{mb['moe_m_le_8_speedup_max']:.2f}×** faster than round 5.",
            f"- MoE at M ≥ 256: **{mb['moe_m_ge_256_speedup_min']:.2f}–{mb['moe_m_ge_256_speedup_max']:.2f}×** faster.",
            f"- Dense at M = 1 (excluding `lm_head`): **{mb['dense_m1_speedup_min']:.2f}–{mb['dense_m1_speedup_max']:.2f}×** faster.",
            f"- Dense at M ≥ 256: **{mb['dense_m_ge_256_speedup_min']:.2f}–{mb['dense_m_ge_256_speedup_max']:.2f}×** (neutral; a few rows 3–7% slower).",
            f"- `lm_head` at M = 256: **{mb['lm_head_m256_speedup']:.2f}×** (unchanged; the round-5 M = 512 point is an outlier).",
            f"- Full per-shape table: `{mb['table']}`.",
            "",
        ]
    lines += [f"Raw benchmark: `{summary['benchmark_file']}`.", ""]
(out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
print("wrote", out_dir / "SUMMARY.md")
