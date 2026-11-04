#!/bin/bash
# E25: whole-frame time of the CoreML placements on the shipped configuration (LoRA v3 +
# lorav3_exact_it_3p00.json, prompt sperm, --boxes detector, cap 64), held-out VISEM 36 (dense:
# 27-46 sperm, so up to 3 tracker states), first 32 frames, no profiling (push_frame wall time).
# Arms (one binary):
#   A default   : --trunk-precision coreml-fp32 (the shipped default), necks and memory attention on Ojas
#   B necks32   : A + --necks coreml-fp32
#   C mattn16   : B + --mem-attn coreml-fp16
#   D fast16    : --trunk-precision coreml-fp16 --necks coreml-fp16 --mem-attn coreml-fp16
# 3 rounds, order rotated. Summary: tools/analysis/e25_frame.py -> summary.md (steady frames >= 8).
# usage: tools/locked.sh e25-frame bash bench/results/e25-coreml-frame/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=$PWD
OUT=bench/results/e25-coreml-frame
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
FRAMES=32
cargo build --release -j 2 -p spiketrack 2>&1 | tail -2
BIN=$OUT/spiketrack
cp target/release/spiketrack "$BIN"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1)"
  echo "tessl $(git -C ../tessl log --oneline -1)"
} > "$OUT/versions.txt"
args_for() {
  # Pinned 2026-11-04: the defaults these arms ran on (coreml-fp32 trunk, no pipeline, Ojas necks) have since moved.
  case $1 in
    default) echo "--trunk-precision coreml-fp32 --trunk-pipeline off --necks ojas" ;;
    necks32) echo "--trunk-precision coreml-fp32 --trunk-pipeline off --necks coreml-fp32" ;;
    mattn16) echo "--trunk-precision coreml-fp32 --trunk-pipeline off --necks coreml-fp32 --mem-attn coreml-fp16" ;;
    fast16) echo "--trunk-precision coreml-fp16 --trunk-pipeline off --necks coreml-fp16 --mem-attn coreml-fp16" ;;
  esac
}
arm() { # name round
  local d="$OUT/$1-r$2"
  rm -rf "$d"
  # shellcheck disable=SC2046
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" --prompt sperm --backend metal \
    --max-frames "$FRAMES" --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    $(args_for "$1") --out "$d" > "$OUT/$1-r$2.log" 2>&1
  echo "== $1 r$2 exit $? load $(uptime | sed 's/.*averages: //')"
}
for r in 1 2 3; do
  case $r in
    1) arm default 1; arm necks32 1; arm mattn16 1; arm fast16 1 ;;
    2) arm fast16 2; arm default 2; arm necks32 2; arm mattn16 2 ;;
    3) arm mattn16 3; arm fast16 3; arm default 3; arm necks32 3 ;;
  esac
done
python3 -I tools/analysis/e25_frame.py "$OUT" "$VID"
