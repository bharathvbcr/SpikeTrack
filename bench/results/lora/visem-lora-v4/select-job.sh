#!/bin/bash
# visem-lora-v4 step 3 (PLAN.md): selection on train-5 only. Each checkpoint (100/200/300/400 = adapter)
# with the exact inverse-temperature edit at G in {2.4, 3.0, 3.6}; prompt sperm, --boxes detector, cap 64,
# first 8 frames of 14 30 19 54 13. Highest HOTA wins; a winner at a G edge extends the grid one step.
# v3's value on this protocol: 0.5802 at G = 3.0 (reproduced 2026-11-03, ../../weight-space/box-delta-train5).
# usage: tools/locked.sh lora-v4-select bash bench/results/lora/visem-lora-v4/select-job.sh [ckpt:G ...]
set -uo pipefail
cd "$(dirname "$0")/../../../.."
D=bench/results/lora/visem-lora-v4
OUT=$D/select
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
VISEM=data/visem/VISEM_Tracking_Train_v4/Train
BIN=target/release/spiketrack
VIDEOS="14 30 19 54 13"
cargo build --release -p spiketrack 2>&1 | tail -2
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
mkdir -p "$OUT"
printf '%s\n' $VIDEOS > "$OUT/train5.txt"
edit_for() { case "$1" in 3.0) echo bench/results/weight-space/lorav3_exact_it_3p00.json ;; *) echo "$D/exact_it_${1/./p}.json" ;; esac; }
adapter_for() { [ "$1" = 400 ] && echo fixtures/lora_runs/visem-lora-v4/adapter.safetensors || echo "fixtures/lora_runs/visem-lora-v4/adapter_step$1.safetensors"; }
if [ $# -gt 0 ]; then cfgs=("$@"); else cfgs=(); for c in 100 200 300 400; do for g in 2.4 3.0 3.6; do cfgs+=("$c:$g"); done; done; fi
status=0
for cfg in "${cfgs[@]}"; do
  c=${cfg%%:*} g=${cfg##*:}
  edit=$(edit_for "$g") adapter=$(adapter_for "$c")
  [ -f "$edit" ] && [ -f "$adapter" ] || { echo "missing $edit or $adapter" >&2; status=1; continue; }
  run="$OUT/run-$c-$g"
  rm -rf "$run"; mkdir -p "$run/logs"
  for id in $VIDEOS; do
    "$BIN" track "$VISEM/$id/images" --prompt sperm --backend metal --max-frames 8 --boxes detector \
      --max-objects 64 --lora "$adapter" --weight-edit "$edit" --out "$run" > "$run/logs/$id.log" 2>&1 \
      || { echo "track failed: $cfg video $id" >&2; status=1; }
  done
  "$BIN" eval-visem --pred "$run" --split "$OUT/train5.txt" --max-frames 8 --visem "$VISEM" \
    --out "$OUT/eval-$c-$g.json" > "$OUT/eval-$c-$g.txt" 2>&1 || { echo "eval failed: $cfg" >&2; status=1; }
  echo "== $cfg $(rg '^all ' "$OUT/eval-$c-$g.txt")"
done
exit $status
