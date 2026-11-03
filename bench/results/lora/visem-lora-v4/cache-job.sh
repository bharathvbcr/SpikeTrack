#!/bin/bash
# visem-lora-v4 step 1 (PLAN.md): rebuild the frozen-feature cache at 128 frames per training video
# (v3 used 64), 24 per held-out video, with the default 2-frame upstream verify. Writes
# fixtures/lora_cache (git-ignored); the log goes here, not over ../cache.log.
# usage: tools/locked.sh lora-v4-cache bash bench/results/lora/visem-lora-v4/cache-job.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
date -u +"start %Y-%m-%dT%H:%M:%SZ"
LORA_FRAMES=128 LORA_HELDOUT_FRAMES=24 LORA_LOG_DIR=bench/results/lora/visem-lora-v4 \
  bash tools/jobs/lora_cache.sh
s=$?
date -u +"end %Y-%m-%dT%H:%M:%SZ"
exit $s
