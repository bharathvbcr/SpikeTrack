#!/bin/bash
# Does SAM 3.1 run on tessl? Ojas's Metal backend (ojas-metal on tessl) against the same
# upstream fixtures and gates as the CPU runs: text encoder, detector, then the backbone
# (expected to stop at rope_interleaved_forward until the Metal rope commit lands on ojas main).
# metal-smoke worktree only (path dependencies on ../ojas; the ../ojas-st it once used is gone). Under the lock:
#   tools/locked.sh metal-smoke bash tools/jobs/metal_smoke.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
MAIN=/Users/bharath/Code/research/SpikeTrack
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_TEXT_FIXTURES="$MAIN/fixtures/text/prompts.safetensors"
export SPIKETRACK_DETECTOR_FIXTURES="$MAIN/fixtures/detector"
export SPIKETRACK_FIXTURES="$MAIN/fixtures/backbone"
out=bench/results/metal-smoke
mkdir -p "$out"
{
  echo "ojas: $(git -C ../ojas log --oneline -1)"
  echo "tessl:   $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"
  echo "SpikeTrack: $(git log --oneline -1)"
} | tee "$out/versions.txt"
uptime
status=0
run() {
  local name=$1; shift
  echo "== $name"
  "$@" 2>&1 | tee "$out/$name.log"
  local s=${PIPESTATUS[0]}
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; status=1; }
}
run build cargo build --release --tests -p st-model
run text cargo test --release -p st-model --test text_parity -- --ignored --nocapture --test-threads=1 text_encoder_matches_upstream_on_metal
run detector cargo test --release -p st-model --test detector_parity -- --ignored --nocapture --test-threads=1 on_metal
run backbone cargo test --release -p st-model --test backbone_parity -- --ignored --nocapture --test-threads=1 on_metal
uptime
exit $status
