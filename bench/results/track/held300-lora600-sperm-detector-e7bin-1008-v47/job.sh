#!/bin/bash
# Inside the mac_heavy lock: attribution check for E7 (Fable). Video 47 alone at --input-size 1008,
# --max-objects 64, with the speed/e7-geometry binary (45e5d63, ojas 8b88328). The pipeline is
# deterministic: it reproduces the committed 1008 row (HOTA 0.5149, ojas cbd5565, pre-e7 binary) or
# that baseline is stale. Scored over the 300-frame window.
set -uo pipefail
cd /Users/bharath/Code/research/SpikeTrack-e7 || exit 1
M=/Users/bharath/Code/research/SpikeTrack
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
echo "V47 SpikeTrack-e7 $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\? \.devmap' | wc -l | tr -d ' ')"
BIN=$PWD/target/release/spiketrack
OUT=$M/bench/results/track/held300-lora600-sperm-detector-e7bin-1008-v47
mkdir -p "$OUT/run" "$OUT/logs"
{ echo "SpikeTrack-e7 $(git log --oneline -1)"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1)"; uptime; } > "$OUT/versions.txt"
"$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/47/images" --prompt sperm --backend metal \
  --max-frames 300 --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
  --max-objects 64 --input-size 1008 --out "$OUT/run" > "$OUT/logs/47.log" 2>&1 || { echo "V47 track failed"; tail -5 "$OUT/logs/47.log"; exit 1; }
rm -rf "$OUT/run/masks"
"$BIN" eval-visem --pred "$OUT/run" --split /private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad/split47.txt \
  --visem "$M/data/visem/VISEM_Tracking_Train_v4/Train" --max-frames 300 --out "$OUT/eval.json" 2>&1 | tee "$OUT/eval.txt"
echo "V47 done"
