#!/bin/bash
# Inside the mac_heavy lock: where the production frame goes with the CoreML fp32 trunk (the Metal
# default). Build, the spiketrack and st-model unit tests, then per video (22: one tracker state;
# 82: two, 17-19 tracks from frame 2) the first 16 frames, prompt "sperm", r16 LoRA, --boxes
# detector, --max-objects 64: an unprofiled control, `--profile stages`, a second control. Then
# `--profile ops` on 8 frames of video 82.
# Pre-stated (docs/SPEED_PLAN.md E11): the stage table is read literally only if its in-Session
# sum per steady frame is within 10% of the controls' mean push_frame time.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
cd "$M" || exit 1
OUT=$M/bench/results/profile-coreml
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
BIN=$OUT/spiketrack
cp target/release/spiketrack "$BIN"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 $BIN | cut -c1-16))"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1)"; uptime; } > "$OUT/versions.txt"
run() { # name video frames [extra...]
  local name=$1 v=$2 n=$3; shift 3
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; print(f\"{sys.argv[3]}: trunk {r['trunk_precision']}, profile {r['profile']}, {r['tracks']} tracks, frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms (frame 0 {1e3*f[0]:.0f} ms)\")" "$d/$v.run.json" "$d.log" "$name"
}
for v in 22 82; do
  run "v$v-control-1" $v 16
  run "v$v-stages" $v 16 --profile stages
  run "v$v-control-2" $v 16
done
run v82-ops 82 8 --profile ops
for v in 22 82; do
  for r in stages control-2; do
    cmp -s "$OUT/v$v-control-1/$v.json" "$OUT/v$v-$r/$v.json" && echo "v$v $r: predictions byte-identical to control-1" || { echo "== FAILED: v$v $r predictions differ from control-1"; status=1; }
  done
done
uptime
echo "P status $status"
exit $status
