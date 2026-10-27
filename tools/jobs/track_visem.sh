#!/bin/bash
# `spiketrack track` on Metal over the 5 held-out VISEM-Tracking videos (tools/reference/visem_split.json),
# the first N frames of each (prompt $SPIKETRACK_PROMPT, default "cell", forward from frame 0), then `spiketrack eval-visem` (HOTA,
# sperm class, the same N frames) on the predictions. Results: bench/results/track/
#   run/<id>.json, run/<id>.run.json, run/masks/   the predictions and per-frame RLE masks
#   logs/<id>.log                                  per-frame progress and timings
#   eval.txt, eval.json                            the HOTA table
# Frames are tracked as a clip of N frames (the session is told the clip ends there), so numbers
# are for "the first N frames", not for the whole video. Zero-shot (no LoRA, no tuning) unless
# SPIKETRACK_LORA=FILE names a detector adapter, which every video then runs with (`track --lora`).
# SPIKETRACK_BOXES=mask|detector picks `track --boxes` (default mask, upstream's boxes).
# usage: tools/locked.sh track-visem bash tools/jobs/track_visem.sh [N=300]
# (SPIKETRACK_TRACK_OUT=DIR writes somewhere else, for a smoke run with a small N)
set -uo pipefail
cd "$(dirname "$0")/../.."
N=${1:-300}
case "$N" in '' | *[!0-9]* | 0) echo "usage: $0 [N frames, a positive integer; default 300]" >&2; exit 2 ;; esac
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM="${SPIKETRACK_VISEM:-data/visem/VISEM_Tracking_Train_v4/Train}"
SPLIT=tools/reference/visem_split.json
OUT="${SPIKETRACK_TRACK_OUT:-bench/results/track}"
BIN=target/release/spiketrack
BOXES=${SPIKETRACK_BOXES:-mask}
PROMPT=${SPIKETRACK_PROMPT:-cell}
case "$BOXES" in mask | detector) ;; *) echo "SPIKETRACK_BOXES=$BOXES is not mask or detector" >&2; exit 2 ;; esac
LORA_ARGS=()
if [ -n "${SPIKETRACK_LORA:-}" ]; then
  [ -f "$SPIKETRACK_LORA" ] || { echo "SPIKETRACK_LORA=$SPIKETRACK_LORA is not a file" >&2; exit 2; }
  LORA_ARGS=(--lora "$SPIKETRACK_LORA")
fi

for p in "$SPIKETRACK_WEIGHTS" "$SPIKETRACK_MODEL_DIR" "$VISEM" "$SPLIT"; do
  [ -e "$p" ] || { echo "missing $p" >&2; exit 2; }
done

# The held-out ids from the split file (a plain id list per line would also do for eval-visem, but
# `track` needs them one by one).
IDS=$(python3 -c 'import json,sys; print(" ".join(str(i) for i in json.load(open(sys.argv[1]))["held_out"]))' "$SPLIT") || exit 2
[ -n "$IDS" ] || { echo "no held_out videos in $SPLIT" >&2; exit 2; }

mkdir -p "$OUT/logs"
rm -rf "$OUT/run"
cargo build --release -p spiketrack 2>&1 | tail -5
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }

uptime
failed=""
for id in $IDS; do
  echo "== video $id: first $N frames on Metal, --boxes $BOXES, prompt $PROMPT${SPIKETRACK_LORA:+, LoRA $SPIKETRACK_LORA}"
  # ${LORA_ARGS[@]+...}: bash 3.2 (macOS) treats an empty array as unbound under set -u.
  "$BIN" track "$VISEM/$id/images" --prompt "$PROMPT" --backend metal --max-frames "$N" --boxes "$BOXES" --out "$OUT/run" ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} 2>&1 | tee "$OUT/logs/$id.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed $id"
done
uptime

if [ -n "$failed" ]; then
  echo "track failed for:$failed (see $OUT/logs); no evaluation of a partial set" >&2
  exit 1
fi
"$BIN" eval-visem --pred "$OUT/run" --split "$SPLIT" --set held_out --max-frames "$N" --visem "$VISEM" --out "$OUT/eval.json" 2>&1 | tee "$OUT/eval.txt"
[ "${PIPESTATUS[0]}" -eq 0 ]
