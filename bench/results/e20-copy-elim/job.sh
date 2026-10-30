#!/bin/bash
# Inside the mac_heavy lock: E20 (docs/SPEED_PLAN.md), the memory encode without duplicate host
# copies, built from the e20-copy-elim worktree into the main target dir. Build, st-model unit
# tests, tracker and detect-track parity (CPU and Metal, video 22), then per video (82, 22), 16
# frames: the E19 binary as the unprofiled control, E20 unprofiled, E20 `--profile stages`, E19 `--profile stages`, the
# control again. Predictions must be byte-identical to control-1.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
W=/Users/bharath/Code/research/SpikeTrack-e20
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$W" || exit 1
export CARGO_TARGET_DIR=$M/target
OUT=$M/bench/results/e20-copy-elim
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^error|^yielded" "$OUT/$name.log" | tail -8
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step build cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
BIN=$SP/spiketrack-e20 CTRL=$SP/spiketrack-e19
cp "$M/target/release/spiketrack" "$BIN"
step unit-tests cargo test --release -j 2 -p st-model --lib
step tracker-parity env SPIKETRACK_TRACKER_FIXTURES="$M/fixtures/tracker/judo.safetensors" \
  cargo test --release -j 2 -p st-model --test tracker_parity -- --ignored --nocapture --test-threads=1
step detect-track-metal env SPIKETRACK_DT_VIDEO=22 SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track" \
  cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_metal
step detect-track-cpu env SPIKETRACK_DT_VIDEO=22 SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track" \
  cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_visem_11
# The tables, with each frame line's wall-clock time masked (it is a timing, not a result).
table() { sed -n '/^| f0/,/^yielded/p' "$1" | sed -E 's/(frame [0-9]+): [0-9.]+ s,/\1: T s,/'; }
for p in "detect-track-metal" "detect-track-cpu"; do
  lines=$(table "$OUT/$p.log" | wc -l | tr -d ' ')
  if [ "$lines" -lt 100 ]; then
    echo "== FAILED: $p table has $lines lines"; status=1
  elif diff <(table "$M/bench/results/e16-bisect-parity/$p.log") <(table "$OUT/$p.log") > "$OUT/$p.diff"; then
    echo "$p: visit and yield tables ($lines lines) identical to E16's apart from frame times"
  else
    echo "== FAILED: $p tables differ from E16's ($(wc -l < "$OUT/$p.diff" | tr -d ' ') diff lines)"; status=1
  fi
done
{ echo "SpikeTrack-e20 $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"; echo "control $(basename "$CTRL") $(shasum -a 256 "$CTRL" | cut -c1-16)"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
run() { # name bin video frames [extra...]
  local name=$1 bin=$2 v=$3 n=$4; shift 4
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; print(f\"{sys.argv[3]}: profile {r['profile']}, {r['tracks']} tracks, frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms (frame 0 {1e3*f[0]:.0f} ms)\")" "$d/$v.run.json" "$d.log" "$name"
}
for v in 82 22; do
  run "v$v-control-1" "$CTRL" $v 16
  run "v$v-e20" "$BIN" $v 16
  run "v$v-stages" "$BIN" $v 16 --profile stages
  run "v$v-e19-stages" "$CTRL" $v 16 --profile stages
  run "v$v-control-2" "$CTRL" $v 16
done
for v in 82 22; do
  for r in e20 stages e19-stages control-2; do
    cmp -s "$OUT/v$v-control-1/$v.json" "$OUT/v$v-$r/$v.json" && echo "v$v $r: predictions byte-identical to control-1" || { echo "== FAILED: v$v $r predictions differ from control-1"; status=1; }
  done
done
uptime
echo "P status $status"
exit $status
