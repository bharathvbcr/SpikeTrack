#!/bin/bash
# Inside the mac_heavy lock: E12 (docs/SPEED_PLAN.md). Binary A = commit $A_SHA (exact blocked,
# threaded host::linear). Binary B = commit $B_SHA (A plus the device box-RPB broadcast).
# The control is the E11b binary (3ca1476), which is not rebuilt.
# Gate, pre-stated: predictions byte-identical to the control on 22 and 82, for each binary.
# Per video, interleaved: control, A stages, B stages, A, B, control, A, B.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
A_SHA=$1 B_SHA=$2
cd "$M" || exit 1
OUT=$M/bench/results/e12-host-decoder
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
BIN0=$SP/spiketrack-e11b BINA=$SP/spiketrack-e12a BINB=$SP/spiketrack-e12b
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^test .* \.\.\. " "$OUT/$name.log" | tail -8
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
[ "$(git rev-parse HEAD)" = "$B_SHA" ] && git diff --quiet HEAD -- crates || { echo "== FAILED: tree is not clean at $B_SHA"; exit 1; }
# Binary B and the tests at B.
step build-b cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
cp target/release/spiketrack "$BINB"
step unit-tests-b cargo test --release -j 2 -p spiketrack -p st-model --lib --bins
step rpb-metal-b cargo test --release -j 2 -p st-model --lib -- --ignored rpb
# Binary A: commit A's detector.rs (B touches only that file), then restore B's.
git show "$A_SHA:crates/st-model/src/detector.rs" > crates/st-model/src/detector.rs
step build-a cargo build --release -j 2 -p spiketrack
cp target/release/spiketrack "$BINA"
git show "$B_SHA:crates/st-model/src/detector.rs" > crates/st-model/src/detector.rs
git diff --quiet HEAD -- crates || { echo "== FAILED: detector.rs not restored"; exit 1; }
[ "$status" -eq 0 ] || exit 1
{ echo "SpikeTrack A $A_SHA, B $B_SHA"; for b in "$BIN0" "$BINA" "$BINB"; do echo "$(basename "$b") $(shasum -a 256 "$b" | cut -c1-16)"; done; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
run() { # name bin video [extra...]
  local name=$1 bin=$2 v=$3; shift 3
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames 16 \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; print(f\"{sys.argv[3]}: profile {r['profile']}, {r['tracks']} tracks, frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms (frame 0 {1e3*f[0]:.0f} ms)\")" "$d/$v.run.json" "$d.log" "$name"
}
for v in 82 22; do
  run "v$v-control-1" "$BIN0" $v
  run "v$v-a-stages" "$BINA" $v --profile stages
  run "v$v-b-stages" "$BINB" $v --profile stages
  run "v$v-a-1" "$BINA" $v
  run "v$v-b-1" "$BINB" $v
  run "v$v-control-2" "$BIN0" $v
  run "v$v-a-2" "$BINA" $v
  run "v$v-b-2" "$BINB" $v
done
for v in 82 22; do
  for r in a-stages b-stages a-1 b-1 control-2 a-2 b-2; do
    cmp -s "$OUT/v$v-control-1/$v.json" "$OUT/v$v-$r/$v.json" && echo "v$v $r: predictions byte-identical to control-1" || { echo "== FAILED: v$v $r predictions differ from control-1"; status=1; }
  done
done
uptime
echo "P status $status"
exit $status
