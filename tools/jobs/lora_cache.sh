#!/bin/bash
# Cache the frozen detector inputs (vision backbone last level + text features) for the LoRA run.
#   tools/locked.sh lora-cache bash tools/jobs/lora_cache.sh [extra cache_detector_inputs.py args]
# Env: LORA_DEVICE (mps|cpu, default mps)  LORA_FRAMES (train frames per video, default 64)
#      LORA_HELDOUT_FRAMES (default 24)    LORA_VIDEOS (comma list, default the whole split)
#      LORA_CACHE_DIR (default fixtures/lora_cache, git-ignored)  LORA_LOG_DIR (default bench/results/lora)
# Disk: 2.65 MB per frame in fp16; defaults are 15*64 + 5*24 = 1080 frames = 2.8 GB. Check `df -h ~` first.
set -uo pipefail
cd "$(dirname "$0")/../.."
device=${LORA_DEVICE:-mps}
log_dir=${LORA_LOG_DIR:-bench/results/lora}
mkdir -p "$log_dir"
export SAM3_DEVICE=$device PYTHONWARNINGS=ignore PYTORCH_ENABLE_MPS_FALLBACK=1
args=(--device "$device" --out "${LORA_CACHE_DIR:-fixtures/lora_cache}"
      --frames-per-video "${LORA_FRAMES:-64}" --heldout-frames-per-video "${LORA_HELDOUT_FRAMES:-24}")
[ -n "${LORA_VIDEOS:-}" ] && args+=(--videos "$LORA_VIDEOS")
df -h ~ | tail -1
uptime
tools/reference/.venv/bin/python -s tools/reference/cache_detector_inputs.py "${args[@]}" "$@" 2>&1 | tee "$log_dir/cache.log"
s=${PIPESTATUS[0]}
uptime
exit $s
