#!/bin/bash
# Inside the mac_heavy lock: `track --profile ops` with run 2's binary, the A/B's settings (VISEM 82,
# --max-objects 16), 12 frames so frames 8-11 attend the full 7-frame memory (36,400 keys). Every op
# is synced, so per-call times are the op's own; the shapes column separates memory depths.
#   tools/locked.sh mattn-ops bash bench/results/ab-coreml-tracker/ops-job.sh
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
cd "$M" || exit 1
D=$M/bench/results/ab-coreml-tracker
BIN=$D/run2/spiketrack
OUT=$D/run2-ops
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
echo "binary $(shasum -a 256 "$BIN" | cut -c1-16); $(uptime)" | tee "$OUT/versions.txt"
"$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
  --max-frames 12 --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
  --max-objects 16 --profile ops --out "$OUT" > "$OUT/track.log" 2>&1
s=$?
rm -rf "$OUT/masks"
uptime
exit $s
