#!/bin/bash
# Repin check for ojas 286a444 -> 59adac6 (the kernel branch merge, 2026-11-03), two questions:
# 1. The Metal bf16 backbone gate failed after the merge (worst cosine 0.99812 <= 0.999,
#    ../module-gates/repin-59adac6.log). Did it fail before? Run the same test against ojas 286a444
#    (A: /Users/bharath/Code/repin-check, detached worktrees + a copy of this tree's sources) and
#    against 59adac6 (B: this tree), back to back.
# 2. Are the shipped predictions unchanged? Rerun v3 + lorav3_exact_it_3p00.json on train-5, 8 frames
#    (tag 0c), compared byte for byte with the pre-merge run-0a.
# usage: tools/locked.sh repin-check bash bench/results/repin-59adac6/job.sh
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
A=/Users/bharath/Code/repin-check/SpikeTrack
OUT=$M/bench/results/repin-59adac6
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_FIXTURES="$M/fixtures/backbone"
t() { cargo test --release -p st-model --test backbone_parity -- --ignored --nocapture --test-threads=1 --exact backbone_matches_pytorch_fp32_on_five_images_on_metal_bf16; }
for side in A B; do
  dir=$M; [ "$side" = A ] && dir=$A
  echo "== $side: $dir, ojas $(git -C "$dir/../ojas" rev-parse --short HEAD)"
  (cd "$dir" && t) > "$OUT/bf16-backbone-$side.log" 2>&1
  echo "exit $?"; rg "worst cosine" "$OUT/bf16-backbone-$side.log"
done
cd "$M" || exit 1
bash bench/results/weight-space/box-delta-train5/job.sh 0c > "$OUT/train5-0c.log" 2>&1
echo "train5 0c exit $?"
for id in 14 30 19 54 13; do
  cmp -s bench/results/weight-space/box-delta-train5/run-0a/$id.json bench/results/weight-space/box-delta-train5/run-0c/$id.json \
    && echo "$id predictions identical" || echo "$id predictions DIFFER"
done
