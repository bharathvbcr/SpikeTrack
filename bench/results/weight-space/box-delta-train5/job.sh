#!/bin/bash
# Box-size calibration sweep for LoRA v3 + lorav3_exact_it_3p00.json (2026-11-03), selected on train-5 only.
# Why: ../diag-train5/buckets.txt (32 frames of 14 30 19 54 13) has 62.5% of FP as localization and 63.7%
# of FN as near-misses, and both matched and near-miss boxes are smaller than the labels (median
# ln(gt/pred) w/h +0.061/+0.054 matched, +0.335/+0.385 near-miss; centers within 0.03 of a box size).
# Sweep: --post-box-delta 0,0,d,d (logit shift on w and h after the decoder, boxRPB untouched) for
# d in {0, 0.08, 0.16, 0.24, 0.32}, d = 0 run twice as the noise floor; prompt sperm, --boxes detector,
# cap 64, first 8 frames of train-5 (the protocol of ../README.md's selection table).
# Pre-stated rule: take the d with the highest HOTA. Adopt it only if it beats both d = 0 runs by at least
# max(0.005, the gap between the two d = 0 runs); if it is d = 0.32, extend by 0.40 first. Then ONE held-out
# 32-frame run (track_visem.sh 32), adopted only if HOTA > 0.6025 + 0.005 = 0.6075 (v3 + edit at 32 frames).
# usage: tools/locked.sh box-delta bash bench/results/weight-space/box-delta-train5/job.sh [d ...]
set -uo pipefail
cd "$(dirname "$0")/../../../.."
OUT=bench/results/weight-space/box-delta-train5
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM=data/visem/VISEM_Tracking_Train_v4/Train
BIN=target/release/spiketrack
VIDEOS="14 30 19 54 13"
cargo build --release -p spiketrack 2>&1 | tail -2
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
mkdir -p "$OUT"
printf '%s\n' $VIDEOS > "$OUT/train5.txt"
[ $# -gt 0 ] && tags=("$@") || tags=(0a 0.08 0.16 0.24 0.32 0b)
status=0
for tag in "${tags[@]}"; do
  d=${tag%[a-z]}  # a trailing letter only names a repeat (0a, 0b, 0c)
  run="$OUT/run-$tag"
  rm -rf "$run"
  mkdir -p "$run/logs"
  for id in $VIDEOS; do
    "$BIN" track "$VISEM/$id/images" --prompt sperm --backend metal --max-frames 8 --boxes detector \
      --max-objects 64 --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
      --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json --post-box-delta "0,0,$d,$d" \
      --out "$run" > "$run/logs/$id.log" 2>&1 || { echo "track failed: d=$d video $id" >&2; status=1; }
  done
  "$BIN" eval-visem --pred "$run" --split "$OUT/train5.txt" --max-frames 8 --visem "$VISEM" \
    --out "$OUT/eval-$tag.json" > "$OUT/eval-$tag.txt" 2>&1 || { echo "eval failed: d=$d" >&2; status=1; }
  echo "== d=$tag"; tail -3 "$OUT/eval-$tag.txt"
done
exit $status
