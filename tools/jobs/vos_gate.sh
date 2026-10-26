#!/bin/bash
# D5 gate: semi-supervised VOS J&F on DAVIS clips, ours vs upstream, CPU.
# usage: tools/jobs/vos_gate.sh [CLIPS]   (comma-separated; default judo,shooting,gold-fish)
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_VOS_FIXTURES="$PWD/fixtures/vos"
export SPIKETRACK_VOS_CLIPS="${1:-judo,shooting,gold-fish}"
mkdir -p bench/results/vos-gate
uptime
cargo test --release -p st-model --test vos_gate -- --ignored --nocapture --test-threads=1 2>&1 | tee "bench/results/vos-gate/cpu-f32-${SPIKETRACK_VOS_CLIPS//,/-}.log"
s=${PIPESTATUS[0]}
uptime
exit $s
