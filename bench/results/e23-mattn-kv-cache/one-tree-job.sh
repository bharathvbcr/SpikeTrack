#!/bin/bash
# Inside the mac_heavy lock: E23's one-tree confirmation (README, "One-tree confirmation", pre-stated).
#   tools/locked.sh e23-1t bash bench/results/e23-mattn-kv-cache/one-tree-job.sh
# ../SpikeTrack-e23ctl built spiketrack-ctl (38e0a35 + control_tree.py). This job puts state.rs back
# to 38e0a35 and builds the cached path into the same target directory.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
CTL_TREE=/Users/bharath/Code/research/SpikeTrack-e23ctl
OUT=$M/bench/results/e23-mattn-kv-cache
CTL=$OUT/spiketrack-ctl
NEW=$OUT/spiketrack-1t
REV=38e0a35
cd "$M" || exit 1
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
step() { # name cmd...: log to $OUT/name.log, report failures, keep going
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
  return "$s"
}

git show "$REV:crates/st-model/src/tracker/state.rs" > "$CTL_TREE/crates/st-model/src/tracker/state.rs" || exit 1
mismatch=0
while IFS= read -r f; do
  git show "$REV:$f" | cmp -s - "$CTL_TREE/$f" || { echo "== tree differs from $REV: $f"; mismatch=1; }
done < <(git ls-tree -r --name-only "$REV" crates Cargo.toml Cargo.lock)
[ "$mismatch" -eq 0 ] || { echo "== refusing: $CTL_TREE is not $REV"; exit 1; }
echo "== $CTL_TREE matches $REV ($(git ls-tree -r --name-only "$REV" crates Cargo.toml Cargo.lock | wc -l | tr -d ' ') files)"

step 1t-build cargo build --release -j 2 --manifest-path "$CTL_TREE/Cargo.toml" -p spiketrack || exit 1
cp "$CTL_TREE/target/release/spiketrack" "$NEW"
{
  echo "1t-new $(shasum -a 256 "$NEW" | cut -c1-16) ($REV, $CTL_TREE/target), control $(shasum -a 256 "$CTL" | cut -c1-16) ($REV + control_tree.py, same target)"
  uptime
} > "$OUT/one-tree-versions.txt"
cat "$OUT/one-tree-versions.txt"

track() { # name binary [extra...]
  local name=$1 bin=$2; shift 2
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  step "$name" "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames 34 --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --out "$d" "$@" || return
  rm -rf "$d/masks"
}
track 1t-new-control "$NEW"
track 1t-ctl-r1 "$CTL" --profile stages; track 1t-new-r1 "$NEW" --profile stages
track 1t-new-r2 "$NEW" --profile stages; track 1t-ctl-r2 "$CTL" --profile stages
track 1t-ctl-r3 "$CTL" --profile stages; track 1t-new-r3 "$NEW" --profile stages

for r in 1t-new-control 1t-ctl-r1 1t-new-r1 1t-ctl-r2 1t-new-r2 1t-ctl-r3 1t-new-r3; do
  [ -s "$OUT/$r/82.json" ] || { echo "== $r: no predictions"; status=1; continue; }
  cmp -s "$OUT/new-control/82.json" "$OUT/$r/82.json" && echo "$r: byte-identical to new-control" \
    || { echo "$r: DIFFERS from new-control"; status=1; }
done
python3 -I tools/analysis/e23_summary.py "$OUT" --one-tree || status=1
uptime
echo "E23 one-tree status $status"
exit $status
