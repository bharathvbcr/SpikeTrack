#!/bin/bash
# Diagnosis after E36 (not a gate, no adoption follows from it): which of the three placements moves
# held-out video 47's HOTA over 300 frames? One arm per placement on its own, plus a base rerun to show
# the reference reproduces, each scored alone with eval-visem. Shipped configuration, SpikeTrack-e34
# binary (as E36).
# usage (from SpikeTrack-e34): tools/locked.sh e36-diag47 bash ../SpikeTrack/bench/results/e36-combined/diag47.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$HERE/../../.." && pwd)
P=$(cd "$M/../SpikeTrack-e34" && pwd)
OUT=$HERE/diag47
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
BIN=$P/target/release/spiketrack
echo 47 > "$OUT/split.txt"
for arm in base dt16 sh16 ni16; do
  case $arm in
    base) args=() ;;
    dt16) args=(--det-transformer coreml-fp16) ;;
    sh16) args=(--seg-head coreml-fp16) ;;
    ni16) args=(--inter-neck coreml-fp16) ;;
  esac
  d=$OUT/$arm
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/47/images" --prompt sperm --backend metal \
    --max-frames 300 --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    ${args[@]+"${args[@]}"} --out "$d" > "$d.log" 2>&1 || { echo "$arm FAILED"; tail -3 "$d.log"; continue; }
  find "$d/masks" -type f -delete 2>/dev/null; rmdir "$d/masks" 2>/dev/null
  "$BIN" eval-visem --pred "$d" --split "$OUT/split.txt" --max-frames 300 \
    --visem "$M/data/visem/VISEM_Tracking_Train_v4/Train" --out "$d.eval.json" > "$d.eval.txt" 2>&1
  echo "== $arm: $(rg '^47 ' "$d.eval.txt")"
done
