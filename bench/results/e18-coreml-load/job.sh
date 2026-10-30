#!/bin/bash
# Inside the mac_heavy lock: E18 (docs/SPEED_PLAN.md), the CoreML trunk's load split into hash /
# compile / model-load. Build, st-coreml unit tests, 3 interleaved rounds of the E15 binary and the
# new one on 2 frames of VISEM 22, then a 16-frame identity run.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$M" || exit 1
OUT=$M/bench/results/e18-coreml-load
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^error|^warning: unused" "$OUT/$name.log" | tail -6
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step build cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
step coreml-unit cargo test --release -j 2 -p st-coreml --lib
OLD=$SP/spiketrack-e15 NEW=$SP/spiketrack-e18
cp target/release/spiketrack "$NEW"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; for b in "$OLD" "$NEW"; do echo "$b $(shasum -a 256 "$b" | cut -c1-16)"; done; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
run() { # name bin frames
  local name=$1 bin=$2 n=$3
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  local load; load=$(sysctl -n vm.loadavg)
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/22/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  echo "$name (load $load): $(rg -o 'loaded in .*' "$d.log" | head -1) | $(rg -o '^model loaded in [0-9.]+ s' "$d.log")"
}
for r in 1 2 3; do
  run "old-$r" "$OLD" 2
  run "new-$r" "$NEW" 2
done
run new-16 "$NEW" 16
cmp -s "$M/bench/results/closing-e15/speed/v22-new-1/22.json" "$OUT/new-16/22.json" && echo "new-16: predictions byte-identical to closing-e15 v22-new-1" || { echo "== FAILED: new-16 predictions differ"; status=1; }
uptime
echo "P status $status"
exit $status
