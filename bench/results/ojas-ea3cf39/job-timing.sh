#!/bin/bash
# Inside the mac_heavy lock: hold 2 of 2 for the ojas pin bump. Interleaved A/B of what ships, on
# 49 frames of VISEM 22 (prompt "sperm", r16 LoRA, --boxes detector, --max-objects 64):
#   old = SpikeTrack-e7 45e5d63 binary built on ojas 8b88328, default mem cap (at 1008 it is
#         output-identical to main before the bump);
#   new = SpikeTrack main + this change, ojas ea3cf39, 4 GiB mem cap.
# Rounds old/new, new/old, old/new. Records load and GPU utilization before each arm.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
P=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
OUT=$M/bench/results/ojas-ea3cf39
mkdir -p "$OUT"
cp "$M/target/release/spiketrack" "$P/spiketrack-main-ojasea3cf39"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator | rg -o '"Device Utilization %"=[0-9]+' | head -1 | rg -o '[0-9]+$'; }
arm() { # round label binary
  local d=$OUT/track-$2-$1
  rm -rf "$d"; mkdir -p "$d"
  echo "round $1 $2: load $(sysctl -n vm.loadavg), GPU $(gpu)%" | tee -a "$OUT/timing.txt"
  "$3" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/22/images" --prompt sperm --backend metal --max-frames 49 \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" > "$OUT/track-$2-$1.log" 2>&1 || { echo "arm $2 round $1 failed"; tail -5 "$OUT/track-$2-$1.log"; exit 1; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"round {sys.argv[2]} {sys.argv[3]}: {r['mean_seconds_per_frame']:.4f} s/frame mean over {r['frames']} frames, {r['tracks']} tracks, mem cap {r.get('metal_mem_cap_bytes')}\")" "$d/22.run.json" "$1" "$2" | tee -a "$OUT/timing.txt"
}
OLD=$P/spiketrack-e7-ojas8b88328
NEW=$P/spiketrack-main-ojasea3cf39
: > "$OUT/timing.txt"
arm 1 old "$OLD"; arm 1 new "$NEW"
arm 2 new "$NEW"; arm 2 old "$OLD"
arm 3 old "$OLD"; arm 3 new "$NEW"
for r in 1 2 3; do cmp -s "$OUT/track-old-$r/22.json" "$OUT/track-new-$r/22.json" && echo "round $r: predictions byte-identical" || echo "round $r: predictions differ"; done | tee -a "$OUT/timing.txt"
for r in 2 3; do cmp -s "$OUT/track-new-1/22.json" "$OUT/track-new-$r/22.json" && echo "new rounds 1 and $r byte-identical" || echo "new rounds 1 and $r differ"; done | tee -a "$OUT/timing.txt"
uptime
echo "BT done"
