#!/bin/bash
# Inside the mac_heavy lock: E18b (docs/SPEED_PLAN.md), the CoreML package hash recorded under a
# stat key. Build, st-coreml unit tests, the ignored load_cache test on a clone of the real package,
# then `track` on VISEM 22: the first new run (full hash, record written), 3 interleaved rounds
# against the E18 binary, and 16-frame identity runs on 82 and 22.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$M" || exit 1
OUT=$M/bench/results/e18b-stat-key
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^error|^warning|^first:|^second:" "$OUT/$name.log" | tail -8
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step build cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
step coreml-unit cargo test --release -j 2 -p st-coreml --lib
step load-cache cargo test --release -j 2 -p st-coreml --test load_cache -- --ignored --nocapture
OLD=$SP/spiketrack-e18 NEW=$SP/spiketrack-e18b
cp target/release/spiketrack "$NEW"
ls -la "$HOME/Code/research/models/sam3.1/coreml/" > "$OUT/coreml-dir-before.txt"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; for b in "$OLD" "$NEW"; do echo "$b $(shasum -a 256 "$b" | cut -c1-16)"; done; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
run() { # name bin video frames
  local name=$1 bin=$2 v=$3 n=$4
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  local load; load=$(sysctl -n vm.loadavg)
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  echo "$name (load $load): $(rg -o 'loaded in .*' "$d.log" | head -1) | $(rg -o '^model loaded in [0-9.]+ s' "$d.log")"
}
run new-first "$NEW" 22 2
ls -la "$HOME/Code/research/models/sam3.1/coreml/" > "$OUT/coreml-dir-after.txt"
for r in 1 2 3; do
  run "old-$r" "$OLD" 22 2
  run "new-$r" "$NEW" 22 2
  rg -q 'hash [0-9.]+ s recorded' "$OUT/new-$r.log" || { echo "== FAILED: new-$r did not use the recorded hash"; status=1; }
done
for v in 82 22; do
  run "new16-v$v" "$NEW" $v 16
  cmp -s "$M/bench/results/closing-e15/speed/v$v-new-1/$v.json" "$OUT/new16-v$v/$v.json" && echo "new16-v$v: predictions byte-identical to closing-e15 v$v-new-1" || { echo "== FAILED: new16-v$v predictions differ"; status=1; }
done
uptime
echo "P status $status"
exit $status
