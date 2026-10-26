#!/bin/bash
# D2/D3 job: build, tokenizer + text-encoder parity, detector parity. Under the lock:
#   tools/locked.sh d2 bash tools/jobs/d2_text_detector.sh
set -uo pipefail
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_TEXT_FIXTURES="$PWD/fixtures/text/prompts.safetensors"
export SPIKETRACK_DETECTOR_FIXTURES="$PWD/fixtures/detector"
out=bench/results/parity-detector
mkdir -p "$out"
status=0
step() { echo "== $*"; "$@" || { echo "== FAILED ($?): $*"; status=1; }; }
uptime
step cargo build --release --tests -p st-model
step cargo test --release -p st-model --lib -- --test-threads=2
step cargo test --release -p st-model --test text_parity -- --ignored --nocapture --test-threads=1
if [ -f "$SPIKETRACK_DETECTOR_FIXTURES/visem22_cell.safetensors" ]; then
  cargo test --release -p st-model --test detector_parity -- --ignored --nocapture --test-threads=1 2>&1 | tee "$out/cpu-f32.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "== FAILED: detector_parity"; status=1; }
else
  echo "== NOT RUN: detector_parity (no fixtures in $SPIKETRACK_DETECTOR_FIXTURES; run tools/reference/dump_detector_fixtures.py)"
  status=1
fi
uptime
exit $status
