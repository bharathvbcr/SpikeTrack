#!/bin/bash
# Train the detector LoRA on the cached features, then evaluate zero-shot vs LoRA on the held-out videos.
#   tools/locked.sh lora-train bash tools/jobs/lora_train.sh [extra train_lora.py args]
# Env: LORA_DEVICE (mps|cpu, default mps)  LORA_RUN (default visem-lora-r16)  LORA_STEPS (default 600)
#      LORA_BATCH (default 2)  LORA_ACCUM (default 4)  LORA_CACHE_DIR (default fixtures/lora_cache)
# Adapters and the resumable checkpoint go to fixtures/lora_runs/$LORA_RUN (git-ignored); the loss log and
# eval tables to bench/results/lora/$LORA_RUN. Resume an interrupted run with `... lora_train.sh --resume`.
# Memory: the encoder's activations are checkpointed (upstream's own mechanism); read `rss_gb` and
# `mps_driver_gb` in train_log.json. A process over 32 GiB gets SIGTERM (exit 143): lower LORA_BATCH if so.
set -uo pipefail
cd "$(dirname "$0")/../.."
device=${LORA_DEVICE:-mps}
run=${LORA_RUN:-visem-lora-r16}
mkdir -p "bench/results/lora/$run"
export SAM3_DEVICE=$device PYTHONWARNINGS=ignore PYTORCH_ENABLE_MPS_FALLBACK=1
df -h ~ | tail -1
uptime
tools/reference/.venv/bin/python -s tools/reference/train_lora.py --device "$device" --run "$run" \
  --cache "${LORA_CACHE_DIR:-fixtures/lora_cache}" --steps "${LORA_STEPS:-600}" \
  --batch "${LORA_BATCH:-2}" --accum "${LORA_ACCUM:-4}" "$@" 2>&1 | tee -a "bench/results/lora/$run/train.log"
s=${PIPESTATUS[0]}
uptime
exit $s
