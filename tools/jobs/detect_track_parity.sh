#!/bin/bash
# The text-prompted video session (tracker/session.rs) against upstream's builder-path model on
# the first 32 frames of VISEM video 11 with the prompt "cell" (fixtures/detect_track, made by
# tools/jobs/detect_track_ref.sh), CPU. Prints the per-visit and per-yield parity tables.
# usage: tools/locked.sh dt-parity bash tools/jobs/detect_track_parity.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_DT_FIXTURES="$PWD/fixtures/detect_track"
if [ ! -f "$SPIKETRACK_DT_FIXTURES/${SPIKETRACK_DT_VIDEO:-11}.safetensors" ]; then
  echo "missing $SPIKETRACK_DT_FIXTURES/${SPIKETRACK_DT_VIDEO:-11}.safetensors: run tools/locked.sh dt-ref bash tools/jobs/detect_track_ref.sh first" >&2
  exit 2
fi
mkdir -p bench/results/parity-tracker
uptime
cargo test --release -p st-model --lib -- --test-threads=2 tracker::state tracker::session 2>&1 | tee bench/results/parity-tracker/dt-unit.log
s0=${PIPESTATUS[0]}
cargo test --release -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 2>&1 | tee bench/results/parity-tracker/cpu-f32-detect-track.log
s1=${PIPESTATUS[0]}
uptime
[ "$s0" -eq 0 ] && [ "$s1" -eq 0 ]
