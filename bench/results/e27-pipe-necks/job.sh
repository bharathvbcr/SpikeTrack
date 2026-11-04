#!/bin/bash
# E27: the CoreML necks on the trunk pipeline's worker on the shipped configuration (LoRA v3 +
# lorav3_exact_it_3p00.json, prompt sperm, --boxes detector, cap 64), held-out VISEM 36 (dense:
# 27-46 sperm, so up to 3 tracker states), first 32 frames, no profiling (push_frame wall time).
# Arms (one binary; --trunk-pipeline is on by default with a CoreML trunk from this build on):
#   default   : the shipped default (coreml-fp32 trunk, pipelined), everything else on Ojas
#   m16       : --mem-attn coreml-fp16
#   m16-n32   : m16 + --necks coreml-fp32 (both necks on the trunk's worker)
#   fast16    : --trunk-precision coreml-fp16 --mem-attn coreml-fp16
#   fast16-n16: fast16 + --necks coreml-fp16
# E27_SRC=DIR builds the binary from another checkout (the change under test was written in a copy).
# 3 rounds, order rotated. Summary: tools/analysis/e25_frame.py -> summary.md (steady frames >= 8).
# usage: tools/locked.sh e27-necks bash bench/results/e27-pipe-necks/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=$PWD
OUT=bench/results/e27-pipe-necks
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
FRAMES=32
SRC=${E27_SRC:-$M}; (cd "$SRC" && cargo build --release -j 2 -p spiketrack 2>&1 | tail -2)
BIN=$OUT/spiketrack
cp "$SRC/target/release/spiketrack" "$BIN"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1)"
  echo "tessl $(git -C ../tessl log --oneline -1)"
  echo "built from $SRC"
} > "$OUT/versions.txt"
# The fp32-trunk arms name their trunk: since 2026-11-04 track's Metal default is the fp16 trunk with
# fp16 memory attention (fast16 here), and these arms were run on the coreml-fp32 default.
args_for() {
  case $1 in
    default) echo "--trunk-precision coreml-fp32" ;;
    m16) echo "--trunk-precision coreml-fp32 --mem-attn coreml-fp16" ;;
    m16-n32) echo "--trunk-precision coreml-fp32 --mem-attn coreml-fp16 --necks coreml-fp32" ;;
    fast16) echo "--trunk-precision coreml-fp16 --mem-attn coreml-fp16 --necks ojas" ;;
    fast16-n16) echo "--trunk-precision coreml-fp16 --mem-attn coreml-fp16 --necks coreml-fp16" ;;
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
ARMS=(default m16 m16-n32 fast16 fast16-n16)
for r in 1 2 3; do
  for i in 0 1 2 3 4; do arm "${ARMS[$(( (i + r - 1) % 5 ))]}" $r; done
done
python3 -I tools/analysis/e25_frame.py "$OUT" "$VID" "${ARMS[@]}"
