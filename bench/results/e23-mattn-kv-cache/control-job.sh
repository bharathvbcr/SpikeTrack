#!/bin/bash
# Inside the mac_heavy lock: E23's same-tree control (README, "Same-tree control", pre-stated).
#   tools/locked.sh e23-ctl bash bench/results/e23-mattn-kv-cache/control-job.sh
# ../SpikeTrack-e23ctl is `git archive 38e0a35` with control_tree.py applied: only
# compute_propagation's call goes back to the uncached fwd on the concatenated memory.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
CTL_TREE=/Users/bharath/Code/research/SpikeTrack-e23ctl
OUT=$M/bench/results/e23-mattn-kv-cache
NEW=$OUT/spiketrack
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
step ctl-build cargo build --release -j 2 --manifest-path "$CTL_TREE/Cargo.toml" -p spiketrack || exit 1
CTL=$OUT/spiketrack-ctl
cp "$CTL_TREE/target/release/spiketrack" "$CTL"
{
  echo "new $(shasum -a 256 "$NEW" | cut -c1-16) (38e0a35), control $(shasum -a 256 "$CTL" | cut -c1-16) (38e0a35 + control_tree.py)"
  uptime
} > "$OUT/control-versions.txt"
cat "$OUT/control-versions.txt"

track() { # name binary [extra...]
  local name=$1 bin=$2; shift 2
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  step "$name" "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames 34 --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --out "$d" "$@" || return
  rm -rf "$d/masks"
}
track ctl-control "$CTL"
track ctl-r1 "$CTL" --profile stages; track cnew-r1 "$NEW" --profile stages
track cnew-r2 "$NEW" --profile stages; track ctl-r2 "$CTL" --profile stages
track ctl-r3 "$CTL" --profile stages; track cnew-r3 "$NEW" --profile stages

for r in ctl-control ctl-r1 cnew-r1 ctl-r2 cnew-r2 ctl-r3 cnew-r3; do
  [ -s "$OUT/$r/82.json" ] || { echo "== $r: no predictions"; status=1; continue; }
  for ref in new-control old-control; do
    cmp -s "$OUT/$ref/82.json" "$OUT/$r/82.json" && echo "$r: byte-identical to $ref" || echo "$r: differs from $ref"
  done
done
python3 -I tools/analysis/e23_summary.py "$OUT" --control || status=1
uptime
echo "E23 control status $status"
exit $status
