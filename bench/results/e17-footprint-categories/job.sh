#!/bin/bash
# Inside the mac_heavy lock: E17 (docs/SPEED_PLAN.md). Which memory category E15's footprint creep
# is in, and whether it levels off: E13 over 300 frames of held-out video 47 (the control) and E15
# over all 1,470, each with `footprint` and `vmmap -summary` snapshots of the live process at set
# frames. Frame times from these runs are not read (the snapshots pause the process).
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$M" || exit 1
OUT=$M/bench/results/e17-footprint-categories
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
REF=$M/bench/results/track/held300-lora600-sperm-detector-coreml-fp32-max64/run/47.json
status=0
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; for b in e13 e15; do echo "$b $(shasum -a 256 "$SP/spiketrack-$b" | cut -c1-16)"; done; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; sw_vers; uptime; } > "$OUT/versions.txt"
run() { # build frames snapshot-frames...
  local b=$1 n=$2; shift 2
  local d=$OUT/v47-$b-$n
  rm -rf "$d"; mkdir -p "$d/snap"
  echo "== v47-$b-$n load $(sysctl -n vm.loadavg)"
  "$SP/spiketrack-$b" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/47/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --max-objects 64 --out "$d" --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    > "$d.log" 2>&1 &
  local pid=$!
  echo "pid $pid"
  for f in "$@"; do
    while kill -0 "$pid" 2>/dev/null && ! rg -q "^frame $f \(" "$d.log"; do sleep 2; done
    kill -0 "$pid" 2>/dev/null || break
    footprint --pid "$pid" -f bytes > "$d/snap/footprint-f$f.txt" 2>&1
    vmmap -summary "$pid" > "$d/snap/vmmap-f$f.txt" 2>&1
    echo "v47-$b-$n frame $f: $(rg -m1 'Footprint|footprint' "$d/snap/footprint-f$f.txt" | tr -s ' ')"
  done
  wait "$pid" || { echo "== FAILED: v47-$b-$n"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  if [ "$n" -eq 300 ]; then
    cmp -s "$REF" "$d/47.json" && echo "v47-$b-$n: predictions byte-identical to the committed held-out 47" || { echo "== FAILED: v47-$b-$n predictions differ"; status=1; }
  fi
}
run e13 300 30 150 290
run e15 1470 30 150 290 700 1100 1460
uptime
echo "P status $status"
exit $status
