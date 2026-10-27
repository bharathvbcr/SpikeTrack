#!/bin/bash
# End-to-end smoke of the LoRA path on this Mac's MPS GPU, small enough to finish in minutes:
#   tools/locked.sh lora-smoke bash tools/jobs/lora_smoke.sh
# 1. lora_selftest.py on MPS: what MPS supports (complex RoPE ops, float64, SDPA at 5184 tokens), then the
#    head-only detector, LoRA, loss, backward and adapter round trip on random features (step time, memory);
# 2. the cache for 3 videos (2 train x 1 frame, 1 held-out x 2 frames) + parity against upstream's full forward_grounding;
# 3. 20 training steps and the held-out evaluation on that tiny cache.
# Each stage runs even if an earlier one failed (the cache needs only stage 1's facts to be informative).
set -uo pipefail
cd "$(dirname "$0")/../.."
log=bench/results/lora/smoke
mkdir -p "$log"
export SAM3_DEVICE=mps PYTHONWARNINGS=ignore PYTORCH_ENABLE_MPS_FALLBACK=1
py=tools/reference/.venv/bin/python
df -h ~ | tail -1
uptime
echo "== stage 1: selftest on mps"
$py -s tools/reference/lora_selftest.py --device mps --batch 2 2>&1 | tee "$log/selftest.log"; echo "stage 1 exit ${PIPESTATUS[0]}"
echo "== stage 2: cache"
LORA_DEVICE=mps LORA_VIDEOS=11,12,22 LORA_FRAMES=1 LORA_HELDOUT_FRAMES=2 LORA_CACHE_DIR=fixtures/lora_cache_smoke LORA_LOG_DIR=$log \
  bash tools/jobs/lora_cache.sh --verify 2; c=$?; echo "stage 2 exit $c"
echo "== stage 3: train + eval"
if [ -f fixtures/lora_cache_smoke/meta.json ]; then
  LORA_DEVICE=mps LORA_RUN=smoke LORA_STEPS=20 LORA_BATCH=2 LORA_ACCUM=1 LORA_CACHE_DIR=fixtures/lora_cache_smoke \
    bash tools/jobs/lora_train.sh --save-every 10 --log-every 1 --eval-batch 2 --warmup 5 --out fixtures/lora_runs/smoke --results "$log"; echo "stage 3 exit $?"
else
  echo "stage 3 skipped: no cache"
fi
uptime
