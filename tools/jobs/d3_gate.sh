#!/bin/bash
# D3 gate end to end (image -> backbone -> text -> detector) on 20 VISEM frames, CPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_DETECTOR_FIXTURES="$PWD/fixtures/detector"
mkdir -p bench/results/d3-gate
uptime
cargo test --release -p st-model --test d3_gate -- --ignored --nocapture --test-threads=1 2>&1 | tee bench/results/d3-gate/cpu-f32.log
s=${PIPESTATUS[0]}
uptime
exit $s
