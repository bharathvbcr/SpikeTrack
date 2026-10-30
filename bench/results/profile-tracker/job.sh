#!/bin/bash
# Inside the mac_heavy lock: E13 (docs/SPEED_PLAN.md), each tracker state's propagation split into
# phases. Build, unit tests and the Metal RPB test, then per video (82: two states; 22: one), 16
# frames, prompt "sperm", r16 LoRA, --boxes detector, --max-objects 64: the E12 B binary as the
# unprofiled control, the E13 binary unprofiled, `--profile stages`, the control again; then
# `--profile ops` on 8 frames of 82. The stage table is read only if its in-Session sum is within
# 10% of the controls, and predictions must be byte-identical to the control.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
cd "$M" || exit 1
OUT=$M/bench/results/profile-tracker
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked" "$OUT/$name.log" | tail -6
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step build cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
step unit-tests cargo test --release -j 2 -p spiketrack -p st-model --lib --bins
step rpb-metal cargo test --release -j 2 -p st-model --lib -- --ignored rpb
BIN=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad/spiketrack-e13
cp target/release/spiketrack "$BIN"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 $BIN | cut -c1-16))"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1)"; uptime; } > "$OUT/versions.txt"
CTRL=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad/spiketrack-e12b
echo "control $(basename $CTRL) $(shasum -a 256 $CTRL | cut -c1-16)" >> "$OUT/versions.txt"
run() { # name bin video frames [extra...]
  local name=$1 bin=$2 v=$3 n=$4; shift 4
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; print(f\"{sys.argv[3]}: trunk {r['trunk_precision']}, profile {r['profile']}, {r['tracks']} tracks, frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms (frame 0 {1e3*f[0]:.0f} ms)\")" "$d/$v.run.json" "$d.log" "$name"
}
for v in 82 22; do
  run "v$v-control-1" "$CTRL" $v 16
  run "v$v-e13" "$BIN" $v 16
  run "v$v-stages" "$BIN" $v 16 --profile stages
  run "v$v-control-2" "$CTRL" $v 16
done
run v82-ops "$BIN" 82 8 --profile ops
for v in 82 22; do
  for r in e13 stages control-2; do
    cmp -s "$OUT/v$v-control-1/$v.json" "$OUT/v$v-$r/$v.json" && echo "v$v $r: predictions byte-identical to control-1" || { echo "== FAILED: v$v $r predictions differ from control-1"; status=1; }
  done
done
uptime
echo "P status $status"
exit $status
