#!/bin/bash
# The one held-out gate run for the train-5-selected box-size correction (2026-11-03):
# LoRA v3 + lorav3_exact_it_3p00.json + --post-box-delta 0,0,0.16,0.16, prompt sperm, --boxes detector,
# cap 64, first 32 frames of the 5 held-out videos. Selection: ../../weight-space/box-delta-train5
# (train-5 8f HOTA 0.5802 -> 0.5913, gains on dense 54/13, losses on 14/30/19). Pre-stated gate:
# adopt only if HOTA > 0.6025 + 0.005 = 0.6075 (v3 + edit, 32 frames, ../../weight-space/README.md).
# usage: tools/locked.sh held32-boxdelta bash bench/results/track/held32-lorav3-it300-boxdelta016/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
OUT=bench/results/track/held32-lorav3-it300-boxdelta016
{ date -u +"date %Y-%m-%dT%H:%M:%SZ"; echo "HEAD $(git rev-parse HEAD)"; git status --short | wc -l | xargs echo "dirty files"; } > "$OUT/provenance.txt"
SPIKETRACK_TRACK_OUT="$OUT" SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_MAX_OBJECTS=64 \
  SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
  SPIKETRACK_WEIGHT_EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json \
  SPIKETRACK_POST_BOX_DELTA=0,0,0.16,0.16 \
  bash tools/jobs/track_visem.sh 32
