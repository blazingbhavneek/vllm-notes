#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
exec /usr/local/cuda-13.1/bin/ncu \
  --clock-control none --cache-control none \
  --kernel-name regex:probe --launch-count 1 \
  --section SpeedOfLight --section Occupancy --section SchedulerStats \
  --section WarpStateStats --section MemoryWorkloadAnalysis \
  --force-overwrite --export /tmp/bolt-round6-sm-counter-probe \
  /tmp/bolt-cuda-round6-probe
