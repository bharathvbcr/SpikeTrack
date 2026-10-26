#!/bin/bash
# Detector parity only (needs fixtures/detector from tools/reference/dump_detector_fixtures.py).
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_DETECTOR_FIXTURES="$PWD/fixtures/detector"
mkdir -p bench/results/parity-detector
uptime
cargo test --release -p st-model --test detector_parity -- --ignored --nocapture --test-threads=1 2>&1 | tee bench/results/parity-detector/cpu-f32.log
s=${PIPESTATUS[0]}
uptime
exit $s
