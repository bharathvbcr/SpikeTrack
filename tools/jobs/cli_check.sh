#!/bin/bash
# The spiketrack binary end to end: crate tests, a release build, `spiketrack image` on a VISEM
# frame where upstream keeps detections (visem 22 frame 0: 28 at 0.5 in the detector fixture)
# on Metal and on CPU, then the no-Python proof on the real binary.
#   tools/locked.sh cli bash tools/jobs/cli_check.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
out=bench/results/cli
mkdir -p "$out"
uptime
status=0
step() { echo "== $*"; "$@" || { echo "== FAILED ($?): $*"; status=1; }; }
step cargo test --release -p spiketrack -- --test-threads=2
step cargo build --release -p spiketrack
img=data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg
for be in metal cpu; do
  step target/release/spiketrack image "$img" --prompt cell --backend "$be" --out "$out/visem22_f0_$be"
done
step bash scripts/no_python_check.sh target/release/spiketrack -- image "$img" --prompt cell --backend metal --out "$out/no_python_run"
uptime
exit $status
