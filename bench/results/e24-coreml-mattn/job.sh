#!/bin/bash
# E24: propagation's memory attention as one CoreML program (--mem-attn), against the Ojas path.
# Arms: ojas, coreml-fp32, coreml-fp16. Same settings as ../ab-coreml-tracker (VISEM 82, 26 frames,
# prompt sperm, r16 LoRA, --boxes detector, --max-objects 16, --profile stages), one binary for all
# arms, 3 rounds with the order rotated. Summary: tools/analysis/e24_mattn.py -> summary.md.
# usage: tools/locked.sh e24-mattn bash bench/results/e24-coreml-mattn/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=$PWD
OUT=bench/results/e24-coreml-mattn
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
FRAMES=26
cargo build --release -j 2 -p spiketrack 2>&1 | tail -2
BIN=$OUT/spiketrack
cp target/release/spiketrack "$BIN"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1) (dirty files: $(git -C ../ojas status --short | wc -l | tr -d ' '))"
  echo "tessl $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"
  for p in fp32 fp16; do
    pkg="$SPIKETRACK_MODEL_DIR/coreml/sam3.1_mattn_$p.mlpackage"
    echo "mattn $p: $(cat "$pkg"/../sam3.1_mattn_$p.mlpackage.stat-*.sha256 2>/dev/null | head -c 16 || echo 'hash recorded on first load')"
  done
} > "$OUT/versions.txt"
arm() { # name round
  local d="$OUT/$1-r$2"
  rm -rf "$d"
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames "$FRAMES" --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --profile stages --mem-attn "$1" --out "$d" > "$OUT/$1-r$2.log" 2>&1
  echo "== $1 r$2 exit $? load $(uptime | sed 's/.*averages: //')"
}
for r in 1 2 3; do
  case $r in
    1) arm ojas 1; arm coreml-fp32 1; arm coreml-fp16 1 ;;
    2) arm coreml-fp32 2; arm coreml-fp16 2; arm ojas 2 ;;
    3) arm coreml-fp16 3; arm ojas 3; arm coreml-fp32 3 ;;
  esac
done
python3 -I tools/analysis/e24_mattn.py "$OUT"
