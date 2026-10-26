#!/bin/bash
# Tracker module parity (CPU) against fixtures/tracker/judo.safetensors.
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_TRACKER_FIXTURES="$PWD/fixtures/tracker/judo.safetensors"
mkdir -p bench/results/parity-tracker
uptime
cargo test --release -p st-model --test tracker_parity -- --ignored --nocapture --test-threads=1 2>&1 | tee bench/results/parity-tracker/cpu-f32.log
s=${PIPESTATUS[0]}
exit $s
