#!/bin/bash
# Drive the LoRA run as bounded locked jobs so one training run does not hold the shared
# mac_heavy lock for hours: verify the cache (no recompute), then the --steps schedule in
# chunks of CHUNK steps, each its own locked job continuing with --resume; the last chunk runs
# the held-out evaluation. Stops at the first failed step. Run it outside the lock:
#   nohup bash tools/jobs/lora_train_chunked.sh > lora_chunks.log 2>&1 &
# Env: LORA_STEPS (default 600), CHUNK (default 100); the rest as in lora_train.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
steps=${LORA_STEPS:-600}
chunk=${CHUNK:-100}
export LORA_STEPS=$steps
tools/locked.sh lora-cache-verify bash tools/jobs/lora_cache.sh || { echo "cache verify failed; training not started"; exit 1; }
done_steps=0
i=1
while [ "$done_steps" -lt "$steps" ]; do
  resume=()
  [ "$done_steps" -gt 0 ] && resume=(--resume)
  tools/locked.sh "lora-train-$i" bash tools/jobs/lora_train.sh "${resume[@]}" --stop-after "$chunk" || { echo "chunk $i failed at or after step $done_steps"; exit 1; }
  done_steps=$((done_steps + chunk))
  i=$((i + 1))
done
echo "all $steps steps trained; evaluation ran in the last chunk"
