#!/bin/bash
# Inside the mac_heavy lock: E21 (docs/SPEED_PLAN.md), exact host threading of the sigmoid fill and the area rule,
# built from branch e21-host-threads in the main tree. Build, st-model unit
# tests, tracker and detect-track parity (CPU and Metal, video 22), then per video (82, 22), 16
# frames: E20 rebuilt (own target dir) as the unprofiled control, E21 unprofiled, E21 `--profile stages`, E20 `--profile stages`, the
# control again. Predictions must be byte-identical to control-1.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
W=$M
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$W" || exit 1
export CARGO_TARGET_DIR=$M/target
OUT=$M/bench/results/e21-host-threads
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
# Xcode 27.2 reinstalled the Metal toolchain (27B47) after the E20 binary was built, so the control
# is E20 (the e20-copy-elim worktree, cd237ab) rebuilt here with the same toolchain as E21. The old
# E20 binary runs once per video as a toolchain check, reported but not an E21 gate.
# The control builds into its own target dir: the two checkouts of one workspace get the same unit
# hashes, so a shared target dir judged the worktree's older sources fresh against E21's st-model
# and linked E21 into the "E20" binary (attempt 2).
E20W=$M-e20
step build-e20 env -C "$E20W" CARGO_TARGET_DIR="$E20W/target" cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
CTRL=$SP/spiketrack-e20tc OLD=$SP/spiketrack-e20
cp "$E20W/target/release/spiketrack" "$CTRL"
step build cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
BIN=$SP/spiketrack-e21
cp "$M/target/release/spiketrack" "$BIN"
# Which code each binary holds: E21's fill_sigmoid_planes in E21 only, E20's slot assembly in both.
for b in "$CTRL:0" "$BIN:1" "$OLD:0"; do
  f=${b%:*} want=${b##*:}
  e21=$(nm "$f" | rg -c fill_sigmoid_planes); e21=$([ "${e21:-0}" -gt 0 ] && echo 1 || echo 0)
  e20=$(strings "$f" | rg -c 'occupied slot with no object planes')
  if [ "$e21" = "$want" ] && [ "${e20:-0}" -gt 0 ]; then echo "binary $(basename "$f"): E21 code $e21, E20 code yes, as expected"
  else echo "== FAILED: binary $(basename "$f") has E21 code $e21 (want $want), E20 code ${e20:-0}"; exit 1; fi
done
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
    if [ "$p" = detect-track-metal ]; then
      # Attribute it: E20 under the new toolchain, the same test.
      step detect-track-metal-e20tc env -C "$E20W" SPIKETRACK_DT_VIDEO=22 SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track" \
        cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_metal
      diff <(table "$OUT/detect-track-metal-e20tc.log") <(table "$OUT/$p.log") > "$OUT/$p-vs-e20tc.diff" \
        && echo "detect-track-metal: E21's table equals E20's under the new toolchain (the E16 difference is the toolchain)" \
        || echo "== detect-track-metal: E21's table also differs from E20's under the new toolchain"
    fi
  fi
done
{ echo "SpikeTrack $(git branch --show-current) $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"; echo "control $(basename "$CTRL") $(shasum -a 256 "$CTRL" | cut -c1-16) from $(git -C "$E20W" log --oneline -1 | cut -c1-60)"; echo "old E20 $(shasum -a 256 "$OLD" | cut -c1-16)"; echo "metal toolchain $(xcodebuild -showComponent MetalToolchain 2>&1 | rg 'Build Version')"; echo "tessl $(git -C ../tessl log --oneline -1)"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
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
  run "v$v-e21" "$BIN" $v 16
  run "v$v-stages" "$BIN" $v 16 --profile stages
  run "v$v-e20-stages" "$CTRL" $v 16 --profile stages
  run "v$v-control-2" "$CTRL" $v 16
  run "v$v-old-e20" "$OLD" $v 16
done
for v in 82 22; do
  for r in e21 stages e20-stages control-2; do
    cmp -s "$OUT/v$v-control-1/$v.json" "$OUT/v$v-$r/$v.json" && echo "v$v $r: predictions byte-identical to control-1" || { echo "== FAILED: v$v $r predictions differ from control-1"; status=1; }
  done
  cmp -s "$OUT/v$v-control-1/$v.json" "$OUT/v$v-old-e20/$v.json" && echo "v$v toolchain check: the old E20 binary's predictions equal the rebuilt E20's" || echo "== v$v toolchain check: the old E20 binary's predictions differ from the rebuilt E20's (not an E21 gate)"
done
uptime
echo "P status $status"
exit $status
