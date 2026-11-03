#!/bin/bash
# Error diagnosis for a weight-space edit (2026-11-03): LoRA v3 + lorav3_exact_it_3p00.json, prompt
# "sperm", --boxes detector, cap 64, first 32 frames of the train-5 selection videos (14 30 19 54 13;
# never the held-out set). The predictions are bucketed against VISEM's labels (classes 0/1/2) by
# bucket_errors.py to decide whether any error class is a target for a direction edit.
# usage: tools/locked.sh ws-diag bash bench/results/weight-space/diag-train5/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
OUT=bench/results/weight-space/diag-train5
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM=data/visem/VISEM_Tracking_Train_v4/Train
BIN=target/release/spiketrack
cargo build --release -p spiketrack 2>&1 | tail -3
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
mkdir -p "$OUT/logs"
rm -rf "$OUT/run"
status=0
for id in 14 30 19 54 13; do
  "$BIN" track "$VISEM/$id/images" --prompt sperm --backend metal --max-frames 32 --boxes detector \
    --max-objects 64 --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
    --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json \
    --out "$OUT/run" > "$OUT/logs/$id.log" 2>&1 || { echo "track failed on $id" >&2; status=1; }
done
exit $status
