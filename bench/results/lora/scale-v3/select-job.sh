#!/bin/bash
# Scaled v3 task vector, rules 1-2 of PLAN.md: for each lambda, write v3 with alpha * lambda
# (tools/reference/lora_soup.py v3 zero lambda), then track the first 8 frames of train-5 with it and
# lorav3_exact_it_3p00.json. Default grid 0.8 0.9 1.1 1.2.
# usage: tools/locked.sh lora-scale bash bench/results/lora/scale-v3/select-job.sh [lambda ...]
set -uo pipefail
cd "$(dirname "$0")/../../../.."
OUT=bench/results/lora/scale-v3/select
ADAPTERS=fixtures/lora_runs/scale-v3
V3=fixtures/lora_runs/visem-lora-v3/adapter.safetensors
EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM=data/visem/VISEM_Tracking_Train_v4/Train
BIN=target/release/spiketrack
VIDEOS="14 30 19 54 13"
cargo build --release -p spiketrack 2>&1 | tail -2
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
mkdir -p "$OUT" "$ADAPTERS"
printf '%s\n' $VIDEOS > "$OUT/train5.txt"
[ $# -gt 0 ] && lams=("$@") || lams=(0.8 0.9 1.1 1.2)
status=0
for l in "${lams[@]}"; do
  adapter="$ADAPTERS/l${l/./p}.safetensors"
  tools/reference/.venv/bin/python -s tools/reference/lora_soup.py "$V3" zero "$l" "$adapter" \
    | tee -a "$OUT/scale.log" || { echo "scale failed: lambda=$l" >&2; status=1; continue; }
  run="$OUT/run-$l"
  rm -rf "$run"; mkdir -p "$run/logs"
  for id in $VIDEOS; do
    "$BIN" track "$VISEM/$id/images" --prompt sperm --backend metal --max-frames 8 --boxes detector \
      --max-objects 64 --lora "$adapter" --weight-edit "$EDIT" --out "$run" > "$run/logs/$id.log" 2>&1 \
      || { echo "track failed: lambda=$l video $id" >&2; status=1; }
  done
  "$BIN" eval-visem --pred "$run" --split "$OUT/train5.txt" --max-frames 8 --visem "$VISEM" \
    --out "$OUT/eval-$l.json" > "$OUT/eval-$l.txt" 2>&1 || { echo "eval failed: lambda=$l" >&2; status=1; }
  echo "== lambda=$l"; rg '^(14|30|19|54|13|all) ' "$OUT/eval-$l.txt"
done
exit $status
