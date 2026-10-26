#!/bin/bash
# st-model unit tests and association parity (exact), the multiplex mask decoder's module parity, then the VOS driver end to end on the tracker
# fixture (judo, 8 frames) stage by stage against upstream's stored state, CPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_TRACKER_FIXTURES="$PWD/fixtures/tracker/judo.safetensors"
mkdir -p bench/results/parity-tracker
uptime
cargo test --release -p st-model --lib --test assoc_parity -- --nocapture --test-threads=2 2>&1 | tee bench/results/parity-tracker/assoc.log
s0=${PIPESTATUS[0]}
cargo test --release -p st-model --test tracker_parity -- --ignored --nocapture --test-threads=1 multiplex_mask_decoder 2>&1 | tee bench/results/parity-tracker/cpu-f32-mask-decoder.log
s1=${PIPESTATUS[0]}
cargo test --release -p st-model --test vos_stages -- --ignored --nocapture --test-threads=1 2>&1 | tee bench/results/parity-tracker/cpu-f32-vos-stages.log
s2=${PIPESTATUS[0]}
uptime
[ "$s0" -eq 0 ] && [ "$s1" -eq 0 ] && [ "$s2" -eq 0 ]
