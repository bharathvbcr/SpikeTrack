#!/bin/bash
# `spiketrack track` on Metal over the 5 held-out VISEM-Tracking videos (tools/reference/visem_split.json),
# the first N frames of each (prompt $SPIKETRACK_PROMPT, default "cell", forward from frame 0), then `spiketrack eval-visem` (HOTA,
# sperm class, the same N frames) on the predictions. Results: bench/results/track/
#   run/<id>.json, run/<id>.run.json, run/masks/   the predictions and per-frame RLE masks
#   logs/<id>.log                                  per-frame progress and timings, then /usr/bin/time -l's
#                                                  resource summary (max RSS, peak memory footprint)
#   eval.txt, eval.json                            the HOTA table
# Frames are tracked as a clip of N frames (the session is told the clip ends there), so numbers
# are for "the first N frames", not for the whole video. Zero-shot (no LoRA, no tuning) unless
# SPIKETRACK_LORA=FILE names a detector adapter, which every video then runs with (`track --lora`).
# SPIKETRACK_WEIGHT_EDIT=FILE passes `track --weight-edit FILE` (a spiketrack-weight-edit-v1 manifest).
# SPIKETRACK_POST_BOX_DELTA=dx,dy,dw,dh passes `track --post-box-delta` (logit shift after the decoder).
# SPIKETRACK_BOXES=mask|detector picks `track --boxes` (default mask, upstream's boxes).
# SPIKETRACK_MAX_OBJECTS=N passes `track --max-objects N` (default 64, track's default; 16 is upstream's builder cap).
# SPIKETRACK_TRUNK=P passes `track --trunk-precision P` (default: track's: coreml-fp16 with fp16 memory
# attention on Metal since 2026-11-04, coreml-fp32 from 2026-10-29, the Ojas f32 trunk before then.
# Set SPIKETRACK_TRUNK=coreml-fp32 to reproduce a result recorded before 2026-11-04).
# SPIKETRACK_NECKS, SPIKETRACK_MEM_ATTN, SPIKETRACK_MEM_ENCODER and SPIKETRACK_SEG_HEAD=P pass
# `track --necks|--mem-attn|--mem-encoder|--seg-head P` (ojas, coreml-fp32 or coreml-fp16; track's
# default: coreml-fp16 memory attention, memory encoder (since E32) and necks (pipelined trunk only)
# beside the default fp16 trunk, ojas otherwise; the segmentation head is always ojas by default).
# SPIKETRACK_TRUNK_PIPELINE=on|off passes `track --trunk-pipeline`.
# SPIKETRACK_DET_ENCODER=P passes `track --det-encoder P` (opt-in; its package carries the --lora adapter).
# usage: tools/locked.sh track-visem bash tools/jobs/track_visem.sh [N=300]       all videos, then eval
#        tools/locked.sh track-visem-21 bash tools/jobs/track_visem.sh N 21 [ID...]  only those videos, no
#            eval (one lock hold per video, so a long run leaves the shared lock to others in between)
#        bash tools/jobs/track_visem.sh N --eval    eval-visem over all held-out videos' predictions (CPU)
# (SPIKETRACK_TRACK_OUT=DIR writes somewhere else, for a smoke run with a small N)
set -uo pipefail
cd "$(dirname "$0")/../.."
N=${1:-300}
case "$N" in '' | *[!0-9]* | 0) echo "usage: $0 [N frames, a positive integer; default 300] [--eval | ID...]" >&2; exit 2 ;; esac
[ $# -gt 0 ] && shift
MODE=all
ONLY=""
if [ "${1:-}" = "--eval" ]; then
  MODE=eval
elif [ $# -gt 0 ]; then
  MODE=some
  for id in "$@"; do
    case "$id" in '' | *[!0-9]*) echo "video id $id is not a number" >&2; exit 2 ;; esac
  done
  ONLY="$*"
fi
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM="${SPIKETRACK_VISEM:-data/visem/VISEM_Tracking_Train_v4/Train}"
SPLIT=tools/reference/visem_split.json
OUT="${SPIKETRACK_TRACK_OUT:-bench/results/track}"
BIN=target/release/spiketrack
BOXES=${SPIKETRACK_BOXES:-mask}
PROMPT=${SPIKETRACK_PROMPT:-cell}
MAX_OBJECTS=${SPIKETRACK_MAX_OBJECTS:-64}
case "$BOXES" in mask | detector) ;; *) echo "SPIKETRACK_BOXES=$BOXES is not mask or detector" >&2; exit 2 ;; esac
LORA_ARGS=()
if [ -n "${SPIKETRACK_LORA:-}" ]; then
  [ -f "$SPIKETRACK_LORA" ] || { echo "SPIKETRACK_LORA=$SPIKETRACK_LORA is not a file" >&2; exit 2; }
  LORA_ARGS=(--lora "$SPIKETRACK_LORA")
fi
TRUNK_ARGS=()
[ -n "${SPIKETRACK_TRUNK:-}" ] && TRUNK_ARGS=(--trunk-precision "$SPIKETRACK_TRUNK")
[ -n "${SPIKETRACK_NECKS:-}" ] && TRUNK_ARGS+=(--necks "$SPIKETRACK_NECKS")
[ -n "${SPIKETRACK_MEM_ATTN:-}" ] && TRUNK_ARGS+=(--mem-attn "$SPIKETRACK_MEM_ATTN")
[ -n "${SPIKETRACK_SEG_HEAD:-}" ] && TRUNK_ARGS+=(--seg-head "$SPIKETRACK_SEG_HEAD")
[ -n "${SPIKETRACK_TRUNK_PIPELINE:-}" ] && TRUNK_ARGS+=(--trunk-pipeline "$SPIKETRACK_TRUNK_PIPELINE")
[ -n "${SPIKETRACK_DET_ENCODER:-}" ] && TRUNK_ARGS+=(--det-encoder "$SPIKETRACK_DET_ENCODER")
[ -n "${SPIKETRACK_MEM_ENCODER:-}" ] && TRUNK_ARGS+=(--mem-encoder "$SPIKETRACK_MEM_ENCODER")
EDIT_ARGS=()
if [ -n "${SPIKETRACK_WEIGHT_EDIT:-}" ]; then
  [ -f "$SPIKETRACK_WEIGHT_EDIT" ] || { echo "SPIKETRACK_WEIGHT_EDIT=$SPIKETRACK_WEIGHT_EDIT is not a file" >&2; exit 2; }
  EDIT_ARGS=(--weight-edit "$SPIKETRACK_WEIGHT_EDIT")
fi
[ -n "${SPIKETRACK_POST_BOX_DELTA:-}" ] && EDIT_ARGS+=(--post-box-delta "$SPIKETRACK_POST_BOX_DELTA")

for p in "$SPIKETRACK_WEIGHTS" "$SPIKETRACK_MODEL_DIR" "$VISEM" "$SPLIT"; do
  [ -e "$p" ] || { echo "missing $p" >&2; exit 2; }
done

# The held-out ids from the split file (a plain id list per line would also do for eval-visem, but
# `track` needs them one by one).
IDS=$(python3 -c 'import json,sys; print(" ".join(str(i) for i in json.load(open(sys.argv[1]))["held_out"]))' "$SPLIT") || exit 2
[ -n "$IDS" ] || { echo "no held_out videos in $SPLIT" >&2; exit 2; }

evaluate() {
  "$BIN" eval-visem --pred "$OUT/run" --split "$SPLIT" --set held_out --max-frames "$N" --visem "$VISEM" --out "$OUT/eval.json" 2>&1 | tee "$OUT/eval.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ]
}

if [ "$MODE" = eval ]; then
  [ -x "$BIN" ] || { echo "$BIN is not built" >&2; exit 1; }
  for id in $IDS; do
    [ -s "$OUT/run/$id.json" ] || { echo "no predictions for video $id in $OUT/run; no evaluation of a partial set" >&2; exit 1; }
  done
  evaluate
  exit $?
fi
if [ "$MODE" = some ]; then
  for id in $ONLY; do
    case " $IDS " in *" $id "*) ;; *) echo "video $id is not in the held-out set of $SPLIT ($IDS)" >&2; exit 2 ;; esac
  done
fi

mkdir -p "$OUT/logs"
if [ "$MODE" = all ]; then
  rm -rf "$OUT/run"
else
  # Only these videos' earlier (possibly partial) outputs go.
  for id in $ONLY; do
    rm -f "$OUT/run/$id.json" "$OUT/run/$id.run.json" "$OUT/logs/$id.log"
    rm -rf "$OUT/overlay/${id}_frame_"*.png
    find "$OUT/run/masks" -name "${id}_frame_*.json" -delete 2>/dev/null
  done
  IDS=$ONLY
fi
cargo build --release -p spiketrack 2>&1 | tail -5
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }

uptime
failed=""
for id in $IDS; do
  echo "== video $id: first $N frames on Metal, --boxes $BOXES, --max-objects $MAX_OBJECTS, prompt $PROMPT${SPIKETRACK_LORA:+, LoRA $SPIKETRACK_LORA}${SPIKETRACK_WEIGHT_EDIT:+, weight edit $SPIKETRACK_WEIGHT_EDIT}${SPIKETRACK_POST_BOX_DELTA:+, post-box-delta $SPIKETRACK_POST_BOX_DELTA}${SPIKETRACK_TRUNK:+, trunk $SPIKETRACK_TRUNK}"
  # ${LORA_ARGS[@]+...}: bash 3.2 (macOS) treats an empty array as unbound under set -u.
  /usr/bin/time -l "$BIN" track "$VISEM/$id/images" --prompt "$PROMPT" --backend metal --max-frames "$N" --boxes "$BOXES" --max-objects "$MAX_OBJECTS" --out "$OUT/run" ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} ${EDIT_ARGS[@]+"${EDIT_ARGS[@]}"} ${TRUNK_ARGS[@]+"${TRUNK_ARGS[@]}"} 2>&1 | tee "$OUT/logs/$id.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed $id"
done
uptime

if [ -n "$failed" ]; then
  echo "track failed for:$failed (see $OUT/logs); no evaluation of a partial set" >&2
  exit 1
fi
[ "$MODE" = all ] || exit 0
evaluate
