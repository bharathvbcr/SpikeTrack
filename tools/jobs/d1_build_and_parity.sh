#!/bin/bash
# D1 job: build, unit tests, full-checkpoint load, backbone parity (CPU). Run under the lock:
#   tools/locked.sh d1 bash tools/jobs/d1_build_and_parity.sh
set -uo pipefail
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_FIXTURES="$PWD/fixtures/backbone"
out=bench/results/parity-backbone
mkdir -p "$out"
status=0
step() { echo "== $*"; "$@" || { echo "== FAILED ($?): $*"; status=1; }; }
uptime
step cargo build --release --tests -p st-weights -p st-model -p st-eval
step cargo test --release -p st-weights -- --test-threads=2
step cargo test --release -p st-eval -- --nocapture --test-threads=2
step cargo test --release -p st-weights -- --ignored --nocapture --test-threads=1
# Riskiest (largest footprint) last.
cargo test --release -p st-model --test backbone_parity -- --ignored --nocapture --test-threads=1 \
  2>&1 | tee "$out/cpu-f32.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "== FAILED: backbone_parity"; status=1; }
uptime
exit $status
