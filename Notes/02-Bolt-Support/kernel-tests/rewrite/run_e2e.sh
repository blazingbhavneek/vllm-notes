#!/usr/bin/env bash
# Tiny Nemotron-H end to end, Triton vs bf16 control, eager / compiled / graphs.
# Usage: run_e2e.sh <out_dir>   (python = the env with the branch installed)
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?out dir}; mkdir -p "$out"; cd "$out"
for m in eager compiled graphs; do
  python "$here/e2e_tiny_bolt.py" $m --triton > log_$m.txt 2>&1; echo "$m exit $?"
done
python "$here/e2e_tiny_bolt.py" compare > log_compare.txt 2>&1; echo "compare exit $?"
for m in eager compiled graphs; do
  python "$here/e2e_tiny_bolt.py" $m --bf16 > log_${m}_bf16.txt 2>&1; echo "$m bf16 exit $?"
done
python "$here/e2e_tiny_bolt.py" compare --bf16 > log_compare_bf16.txt 2>&1; echo "compare bf16 exit $?"
# The bf16 control legitimately JITs upstream's fused_moe_kernel; only the
# Triton path is in scope for the §5.3 "no JIT during inference" check.
grep -H 'jit_monitor.py:140' log_eager.txt log_compiled.txt log_graphs.txt \
  && echo "JIT DURING INFERENCE (see 05 §5.3)" || echo "no JIT during inference"
