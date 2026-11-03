#!/bin/bash
# Inside the mac_heavy lock: E23's paired speed run (README, "Paired speed run", pre-stated).
#   tools/locked.sh e23-paired bash bench/results/e23-mattn-kv-cache/paired-job.sh
# A = spiketrack-ctl (38e0a35 + control_tree.py), B = spiketrack-1t (38e0a35); one tree, one target.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
E23=$M/bench/results/e23-mattn-kv-cache
OUT=$E23/paired
A=$E23/spiketrack-ctl
B=$E23/spiketrack-1t
LOAD_MAX=5
WAIT_MAX_S=600
cd "$M" || exit 1
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
[ "$(shasum -a 256 "$A" | cut -c1-16)" = e69a0a02fcecb4b8 ] || { echo "== refusing: $A is not e69a0a02fcecb4b8"; exit 1; }
[ "$(shasum -a 256 "$B" | cut -c1-16)" = 596ae93d57261abf ] || { echo "== refusing: $B is not 596ae93d57261abf"; exit 1; }
mkdir -p "$OUT"
status=0

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }

track() { # name binary frames [extra...]: one run, its load sampled every 5 s into name.load
  local name=$1 bin=$2 n=$3; shift 3
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  (while :; do echo "$(date +%s) $(load1)"; sleep 5; done) > "$OUT/$name.load" &
  local sampler=$!
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames "$n" --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --out "$d" "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  kill "$sampler"; wait "$sampler" 2> "$OUT/.sampler-wait"
  rm -f "$OUT/.sampler-wait"
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; return "$s"; }
  rm -rf "$d/masks"
}

quiet() { # pair: wait up to WAIT_MAX_S for the 1-minute load to reach LOAD_MAX; flag the pair if not
  local p=$1 t=0
  while awk -v l="$(load1)" -v m="$LOAD_MAX" 'BEGIN{exit !(l > m)}'; do
    [ "$t" -ge "$WAIT_MAX_S" ] && { echo "== pair $p flagged: load $(load1) after ${t}s"; touch "$OUT/p$p.flagged"; return; }
    sleep 15; t=$((t + 15))
  done
  echo "== pair $p starts at load $(load1) after ${t}s"
}

{ echo "A $(shasum -a 256 "$A" | cut -c1-16), B $(shasum -a 256 "$B" | cut -c1-16)"; uptime; } > "$OUT/versions.txt"
cat "$OUT/versions.txt"

# Mechanism: --profile ops, 20 frames, A B B A.
track ops-1-ctl "$A" 20 --profile ops; track ops-2-new "$B" 20 --profile ops
track ops-3-new "$B" 20 --profile ops; track ops-4-ctl "$A" 20 --profile ops

# Wall clock: 16 pairs in the pre-stated order.
order=(AB AA BA BB AB BA AA AB BB BA AB AA BA BB AB BA)
arm() { [ "$1" = A ] && echo "ctl $A" || echo "new $B"; }
rm -f "$OUT"/p*.flagged
for i in "${!order[@]}"; do
  p=$(printf %02d $((i + 1)))
  quiet "$p"
  for j in 1 2; do
    read -r name bin <<< "$(arm "${order[$i]:$((j - 1)):1}")"
    track "p$p-$j-$name" "$bin" 34 --profile stages
  done
done

for r in "$OUT"/p*-*-*/; do
  r=$(basename "$r")
  [ -s "$OUT/$r/82.json" ] || { echo "== $r: no predictions"; status=1; continue; }
  cmp -s "$E23/new-control/82.json" "$OUT/$r/82.json" || { echo "== $r: DIFFERS from new-control"; status=1; }
done
echo "34-frame predictions checked against new-control"
for r in ops-2-new ops-3-new ops-4-ctl; do
  cmp -s "$OUT/ops-1-ctl/82.json" "$OUT/$r/82.json" || { echo "== $r: DIFFERS from ops-1-ctl"; status=1; }
done
echo "20-frame ops predictions checked against each other"
python3 -I tools/analysis/e23_summary.py "$OUT" --paired 19 || status=1
uptime
echo "E23 paired status $status"
exit $status
