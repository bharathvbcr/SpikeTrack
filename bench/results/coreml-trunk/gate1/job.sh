#!/bin/bash
# Inside the mac_heavy lock: E10 gate 1. Release build, workspace unit tests, the CoreML trunk +
# Ojas CPU necks parity tests (fp32 gate: rel L2 < 1e-4; fp16: D2 cosine > 0.999), then the VISEM 22
# 8-frame Metal check that coreml-fp32 tracks with the same ids as the Ojas trunk.
set -uo pipefail
cd /Users/bharath/Code/research/SpikeTrack || exit 1
OUT=bench/results/coreml-trunk/gate1
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_FIXTURES="$PWD/fixtures/backbone"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1)"; uptime; } | tee "$OUT/versions.txt"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test .* \.\.\. |^test result|worst|loaded in|trunk call|frame [0-9]: ojas" "$OUT/$name.log"
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step build cargo build --release -j 2 --workspace --tests --bins
step unit-tests cargo test --release -j 2 --workspace
step parity-fp32 cargo test --release -j 2 -p st-coreml --test trunk_parity -- --ignored --nocapture --test-threads=1 --exact coreml_fp32_trunk_and_ojas_necks_match_upstream
step parity-fp16 cargo test --release -j 2 -p st-coreml --test trunk_parity -- --ignored --nocapture --test-threads=1 --exact coreml_fp16_trunk_and_ojas_necks_keep_d2
step visem22-ids cargo test --release -j 2 -p spiketrack --bin spiketrack -- --ignored --nocapture --test-threads=1 --exact track_cmd::tests::visem_22_coreml_fp32_trunk_tracks_as_the_ojas_trunk
uptime
echo "G1 status $status"
exit $status
