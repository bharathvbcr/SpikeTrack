#!/bin/bash
# v3/v4 adapter soup, rules 0-3 of PLAN.md: build each soup with tools/reference/lora_soup.py, then
# track the first 8 frames of train-5 with it and the exact inverse-temperature edit at G.
# Configs are c:G (default: the parity check 1.0:3.0, then 0.75:3.0 0.5:3.0 0.25:3.0).
# usage: tools/locked.sh lora-soup bash bench/results/lora/soup-v3v4/select-job.sh [c:G ...]
set -uo pipefail
cd "$(dirname "$0")/../../../.."
D=bench/results/lora/soup-v3v4
OUT=$D/select
SOUPS=fixtures/lora_runs/soup-v3v4
V3=fixtures/lora_runs/visem-lora-v3/adapter.safetensors
V4=fixtures/lora_runs/visem-lora-v4/adapter_step300.safetensors
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM=data/visem/VISEM_Tracking_Train_v4/Train
BIN=target/release/spiketrack
VIDEOS="14 30 19 54 13"
cargo build --release -p spiketrack 2>&1 | tail -2
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
mkdir -p "$OUT" "$SOUPS"
printf '%s\n' $VIDEOS > "$OUT/train5.txt"
edit_for() {
  case "$1" in
    3.0) echo bench/results/weight-space/lorav3_exact_it_3p00.json ;;
    *) echo "bench/results/lora/visem-lora-v4/exact_it_${1/./p}.json" ;;
  esac
}
[ $# -gt 0 ] && cfgs=("$@") || cfgs=(1.0:3.0 0.75:3.0 0.5:3.0 0.25:3.0)
status=0
for cfg in "${cfgs[@]}"; do
  c=${cfg%%:*} g=${cfg##*:}
  soup="$SOUPS/c${c/./p}.safetensors" edit=$(edit_for "$g")
  [ -f "$edit" ] || { echo "missing $edit" >&2; status=1; continue; }
  if [ ! -f "$soup" ]; then
    tools/reference/.venv/bin/python -s tools/reference/lora_soup.py "$V3" "$V4" "$c" "$soup" \
      | tee -a "$OUT/soup.log" || { echo "soup failed: c=$c" >&2; status=1; continue; }
  fi
  run="$OUT/run-$c-$g"
  rm -rf "$run"; mkdir -p "$run/logs"
  for id in $VIDEOS; do
    "$BIN" track "$VISEM/$id/images" --prompt sperm --backend metal --max-frames 8 --boxes detector \
      --max-objects 64 --lora "$soup" --weight-edit "$edit" --out "$run" > "$run/logs/$id.log" 2>&1 \
      || { echo "track failed: $cfg video $id" >&2; status=1; }
  done
  "$BIN" eval-visem --pred "$run" --split "$OUT/train5.txt" --max-frames 8 --visem "$VISEM" \
    --out "$OUT/eval-$c-$g.json" > "$OUT/eval-$c-$g.txt" 2>&1 || { echo "eval failed: $cfg" >&2; status=1; }
  echo "== $cfg"; rg '^(14|30|19|54|13|all) ' "$OUT/eval-$c-$g.txt"
done
exit $status
