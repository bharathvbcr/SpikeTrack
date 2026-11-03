#!/bin/bash
# visem-lora-v4 held-out gate, one run (../../lora/visem-lora-v4/PLAN.md, 2026-11-03): v4 step 300 +
# exact inverse-temperature G = 3.6 (selected on train-5 8f: 0.5937 vs v3 0.5802, 4 of 5 videos up;
# G 4.2 extension 0.5927), prompt sperm, --boxes detector, cap 64, first 32 frames of the 5 held-out
# videos. Pre-stated gate: v4 replaces v3 only if HOTA >= 0.6025 + 0.010 = 0.6125.
# usage: tools/locked.sh held32-lorav4 bash bench/results/track/held32-lorav4-s300-it360/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
OUT=bench/results/track/held32-lorav4-s300-it360
{ date -u +"date %Y-%m-%dT%H:%M:%SZ"; echo "HEAD $(git rev-parse HEAD)"; git status --short | wc -l | xargs echo "dirty files"; } > "$OUT/provenance.txt"
SPIKETRACK_TRACK_OUT="$OUT" SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_MAX_OBJECTS=64 \
  SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-v4/adapter_step300.safetensors \
  SPIKETRACK_WEIGHT_EDIT=bench/results/lora/visem-lora-v4/exact_it_3p6.json \
  bash tools/jobs/track_visem.sh 32
