#!/bin/bash
# Per-op wall-time breakdown of one frame (trunk + detector + tracker step) on Metal; single
# run, contended machine. Results to $PROFILE_OUT (default bench/results/profile/).
#   tools/locked.sh profile bash tools/jobs/frame_profile.sh [metal|cpu]
set -uo pipefail
cd "$(dirname "$0")/../.."
which=${1:-metal}
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_TRACKER_FIXTURES="$PWD/fixtures/tracker/judo.safetensors"
out=${PROFILE_OUT:-bench/results/profile}
mkdir -p "$out"
{
  echo "SpikeTrack: $(git log --oneline -1)"
  echo "ojas-st:    $(git -C ../ojas-st log --oneline -1)"
  echo "tessl:      $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"
  echo "load:       $(uptime)"
} | tee "$out/versions-$which.txt"
cargo test --release -p st-model --test frame_profile -- --ignored --nocapture --test-threads=1 --exact "frame_profile_$which" 2>&1 | tee "$out/frame-$which.log"
exit "${PIPESTATUS[0]}"
