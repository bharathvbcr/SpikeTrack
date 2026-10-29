#!/bin/bash
# Inside the mac_heavy lock: the coreml-fp32-default change. Release build, workspace unit tests,
# and the ignored test that a default Metal run without the CoreML package fails loud.
set -uo pipefail
cd /Users/bharath/Code/research/SpikeTrack || exit 1
OUT=bench/results/coreml-trunk/default
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; uptime; } | tee "$OUT/versions.txt"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test .* \.\.\. |^test result" "$OUT/$name.log"
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step build cargo build --release -j 2 --workspace --tests --bins
step unit-tests cargo test --release -j 2 --workspace
step missing-package cargo test --release -j 2 -p spiketrack --bin spiketrack -- --ignored --nocapture --test-threads=1 --exact track_cmd::tests::the_default_metal_trunk_without_its_package_is_an_error_not_a_fallback
echo "D status $status"
exit $status
