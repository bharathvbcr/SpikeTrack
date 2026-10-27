#!/bin/bash
# Event gating (gp-event-gating), CPU: st-gate unit and cost tests, the refactored backbone's unit tests and
# the CPU backbone parity gate (trunk() now runs through the shared block code, so its fp32 gate is re-run),
# then gating_parity: the synthetic gated trunk, and with weights and VISEM frames the exactness of the gated
# trunk (CPU Exact numerics asserted, Fast reported) and recompute fraction and wall time per policy on ten
# consecutive VISEM frames. Run it under the lock: tools/locked.sh gating bash tools/jobs/gating_parity.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_FIXTURES="$PWD/fixtures/backbone"
out=bench/results/gating
mkdir -p "$out"
uptime
cargo test --release -p st-gate -- --nocapture --test-threads=2 2>&1 | tee "$out/st-gate.log"
s0=${PIPESTATUS[0]}
cargo test --release -p st-model --lib backbone -- --nocapture --test-threads=2 2>&1 | tee "$out/backbone-unit.log"
s1=${PIPESTATUS[0]}
cargo test --release -p st-model --test gating_parity -- --nocapture --test-threads=1 2>&1 | tee "$out/synthetic-cpu-exact.log"
s2=${PIPESTATUS[0]}
cargo test --release -p st-model --test backbone_parity -- --ignored --nocapture --test-threads=1 --exact backbone_matches_pytorch_fp32_on_five_images 2>&1 | tee "$out/backbone-parity-cpu.log"
s3=${PIPESTATUS[0]}
cargo test --release -p st-model --test gating_parity -- --ignored --nocapture --test-threads=1 2>&1 | tee "$out/gating-parity-cpu.log"
s4=${PIPESTATUS[0]}
uptime
echo "statuses: st-gate $s0, backbone-unit $s1, synthetic $s2, backbone-parity $s3, gating-parity $s4"
[ "$s0" -eq 0 ] && [ "$s1" -eq 0 ] && [ "$s2" -eq 0 ] && [ "$s3" -eq 0 ] && [ "$s4" -eq 0 ]
