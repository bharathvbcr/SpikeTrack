#!/bin/bash
# A demo clip: `spiketrack track` on Metal over the first N frames of one VISEM-Tracking video, with
# the overlay written as an H.264 .mp4 (masks, boxes and track ids). Results: bench/results/demo/
#   <id>-<prompt>[-lora].mp4, <id>.json, <id>.run.json, logs/<id>-<prompt>.log
# usage: tools/locked.sh demo-clip bash tools/jobs/demo_clip.sh VIDEO [N=49] [PROMPT=sperm]
# Env: SPIKETRACK_LORA=FILE (default: the step-600 VISEM adapter; set it empty for zero-shot),
#      SPIKETRACK_BOXES=mask|detector (default detector with an adapter, mask without).
set -uo pipefail
cd "$(dirname "$0")/../.."
VID=${1:?usage: $0 VIDEO [N] [PROMPT]}
N=${2:-49}
PROMPT=${3:-sperm}
case "$VID$N" in *[!0-9]*) echo "VIDEO and N must be numbers" >&2; exit 2 ;; esac
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
LORA=${SPIKETRACK_LORA-fixtures/lora_runs/visem-lora-r16/adapter.safetensors}
FRAMES=data/visem/VISEM_Tracking_Train_v4/Train/$VID/images
OUT=bench/results/demo
BIN=target/release/spiketrack
[ -d "$FRAMES" ] || { echo "missing $FRAMES" >&2; exit 2; }
args=(--prompt "$PROMPT" --backend metal --max-frames "$N")
tag="$VID-$PROMPT"
if [ -n "$LORA" ]; then
  [ -f "$LORA" ] || { echo "missing adapter $LORA" >&2; exit 2; }
  args+=(--lora "$LORA" --boxes "${SPIKETRACK_BOXES:-detector}")
  tag="$tag-lora"
else
  args+=(--boxes "${SPIKETRACK_BOXES:-mask}")
fi
mkdir -p "$OUT/logs"
rm -f "$OUT/$tag.mp4"
cargo build --release -p spiketrack 2>&1 | tail -3
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
uptime
/usr/bin/time -l "$BIN" track "$FRAMES" "${args[@]}" --out "$OUT/run-$tag" --overlay-video "$OUT/$tag.mp4" 2>&1 | tee "$OUT/logs/$tag.log"
exit "${PIPESTATUS[0]}"
