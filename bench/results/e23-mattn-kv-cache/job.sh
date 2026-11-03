#!/bin/bash
# Inside the mac_heavy lock: E23's gates and the old/new A/B (README.md, pre-stated).
#   tools/locked.sh e23 bash bench/results/e23-mattn-kv-cache/job.sh
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
cd "$M" || exit 1
OUT=$M/bench/results/e23-mattn-kv-cache
OLD=$M/bench/results/ab-coreml-tracker/run2/spiketrack
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_TRACKER_FIXTURES="$M/fixtures/tracker/judo.safetensors"
export SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track"
export SPIKETRACK_DT_VIDEO=22
status=0
step() { # name cmd...: log to $OUT/name.log, report failures, keep going
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^error" "$OUT/$name.log" | tail -6
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
  return "$s"
}

# Gate 5 as narrowed before run 2 (README): lint every target without -D, so another agent's
# uncommitted finding cannot stop the lint before it reaches these files, then fail on any
# finding in a file this change touches.
step clippy cargo clippy -j 2 -p st-model -p spiketrack --all-targets || exit 1
OURS='crates/st-model/src/tracker/(memory_attention|mod|state)\.rs|crates/st-model/tests/memory_attention_cache\.rs|crates/spiketrack/src/track_profile\.rs'
RIPGREP_CONFIG_PATH= rg -o -- '--> [^ ]+' "$OUT/clippy.log" | sort | uniq -c > "$OUT/clippy-findings.txt"
echo "clippy findings by location:"; cat "$OUT/clippy-findings.txt"
if RIPGREP_CONFIG_PATH= rg -q -e "$OURS" "$OUT/clippy-findings.txt"; then
  echo "== FAILED: clippy finding in a file this change touches"; exit 1
fi
step unit-tests cargo test --release -j 2 -p st-model --lib -- --test-threads=2 tracker::
step cache-cpu cargo test --release -j 2 -p st-model --test memory_attention_cache -- --test-threads=2
metal_bitwise=1
step cache-metal cargo test --release -j 2 -p st-model --test memory_attention_cache \
  cached_equals_uncached_on_metal -- --exact --ignored --nocapture --test-threads=1 || metal_bitwise=0
step tracker-parity cargo test --release -j 2 -p st-model --test tracker_parity \
  -- --ignored --nocapture --test-threads=1
step detect-track-parity-metal cargo test --release -j 2 -p st-model --test detect_track_parity \
  session_matches_upstream_on_metal -- --exact --ignored --nocapture --test-threads=1
step build cargo build --release -j 2 -p spiketrack || exit 1
NEW=$OUT/spiketrack
cp target/release/spiketrack "$NEW"
{
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"
  echo "old binary $(shasum -a 256 "$OLD" | cut -c1-16) (ab-coreml-tracker/run2), new $(shasum -a 256 "$NEW" | cut -c1-16)"
  echo "ojas $(git -C ../ojas log --oneline -1) (dirty files: $(git -C ../ojas status --short | wc -l | tr -d ' '))"
  echo "tessl $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"
  echo "metal bitwise (gate 3): $metal_bitwise"
  uptime
} > "$OUT/versions.txt"
cat "$OUT/versions.txt"

track() { # name binary frames [extra...]
  local name=$1 bin=$2 n=$3; shift 3
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  step "$name" "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames "$n" --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --out "$d" "$@" || return
  rm -rf "$d/masks"
}
track old-control "$OLD" 34
track new-control "$NEW" 34
track old-r1 "$OLD" 34 --profile stages; track new-r1 "$NEW" 34 --profile stages
track new-r2 "$NEW" 34 --profile stages; track old-r2 "$OLD" 34 --profile stages
track old-r3 "$OLD" 34 --profile stages; track new-r3 "$NEW" 34 --profile stages
track new-ops "$NEW" 12 --profile ops

for r in new-control old-r1 new-r1 old-r2 new-r2 old-r3 new-r3; do
  cmp -s "$OUT/old-control/82.json" "$OUT/$r/82.json" && echo "$r: predictions byte-identical to old-control" \
    || { echo "== $r: predictions DIFFER from old-control"; status=1; }
done
python3 -I tools/analysis/e23_summary.py "$OUT" || status=1
uptime
echo "E23 status $status"
exit $status
