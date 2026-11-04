#!/bin/bash
# LoRA v3 + the 768-weight exact inverse-temperature edit over the first 300 frames of the five
# held-out VISEM videos (1,500 frames), prompt "sperm", --boxes detector, cap 64, the default
# CoreML fp32 trunk: the configuration whose 32-frame HOTA is 0.6025
# (../../weight-space/eval/held32_lorav3_it300.txt), at the length of the v1 headline (0.4922,
# ../held300-lora600-sperm-detector-f32-max64). One run; no noise estimate.
# usage: tools/locked.sh held300-lorav3 bash bench/results/track/held300-lorav3-it300-sperm-detector-max64/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
OUT=bench/results/track/held300-lorav3-it300-sperm-detector-max64
{
  date -u +%FT%TZ
  git rev-parse HEAD
  git status --short
  git diff | shasum -a 256
} > "$OUT/provenance.txt"
# The trunk is pinned: since 2026-11-04 track's Metal default is the fp16 trunk + fp16 memory
# attention (../held300-coreml), and this run used the coreml-fp32 default of 2026-11-03.
SPIKETRACK_TRUNK=coreml-fp32 \
SPIKETRACK_TRACK_OUT="$OUT" \
SPIKETRACK_PROMPT=sperm \
SPIKETRACK_BOXES=detector \
SPIKETRACK_MAX_OBJECTS=64 \
SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
SPIKETRACK_WEIGHT_EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json \
  bash tools/jobs/track_visem.sh 300
status=$?
shasum -a 256 target/release/spiketrack >> "$OUT/provenance.txt"
exit $status
