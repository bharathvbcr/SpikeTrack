#!/bin/bash
# Inside the mac_heavy lock: E10 interleaved A/B/C on 49 frames of VISEM 22 (prompt "sperm", r16 LoRA,
# --boxes detector, --max-objects 64), as bench/results/ojas-ea3cf39's timing. One frozen binary,
# three arms by --trunk-precision: f32 (what ships), coreml-fp32, coreml-fp16. Each arm takes each
# position once over the three rounds. Records load and GPU utilization before each arm.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
P=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
BIN=$P/spiketrack-e10
OUT=$M/bench/results/coreml-trunk/timing
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
{ echo "SpikeTrack $(git -C $M log --oneline -1) (binary $(shasum -a 256 $BIN | cut -c1-16))"; echo "ojas-st $(git -C $M/../ojas-st log --oneline -1)"; echo "tessl $(git -C $M/../tessl log --oneline -1)"; uptime; } > "$OUT/versions.txt"
gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator | rg -o '"Device Utilization %"=[0-9]+' | head -1 | rg -o '[0-9]+$'; }
arm() { # round precision
  local d=$OUT/track-$2-$1
  rm -rf "$d"; mkdir -p "$d"
  echo "round $1 $2: load $(sysctl -n vm.loadavg), GPU $(gpu)%" | tee -a "$OUT/timing.txt"
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/22/images" --prompt sperm --backend metal --max-frames 49 \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --trunk-precision "$2" --out "$d" > "$OUT/track-$2-$1.log" 2>&1 || { echo "arm $2 round $1 failed"; tail -5 "$OUT/track-$2-$1.log"; exit 1; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"round {sys.argv[2]} {sys.argv[3]}: {r['mean_seconds_per_frame']:.4f} s/frame mean over {r['frames']} frames, {r['tracks']} tracks, load {r['model_load_seconds']:.1f} s, external {r.get('external_trunk')}\")" "$d/22.run.json" "$1" "$2" | tee -a "$OUT/timing.txt"
}
: > "$OUT/timing.txt"
arm 1 f32; arm 1 coreml-fp32; arm 1 coreml-fp16
arm 2 coreml-fp32; arm 2 coreml-fp16; arm 2 f32
arm 3 coreml-fp16; arm 3 f32; arm 3 coreml-fp32
for p in coreml-fp32 coreml-fp16; do for r in 1 2 3; do
  cmp -s "$OUT/track-f32-$r/22.json" "$OUT/track-$p-$r/22.json" && echo "round $r: $p predictions byte-identical to f32" || echo "round $r: $p predictions differ from f32"
done; done | tee -a "$OUT/timing.txt"
for p in f32 coreml-fp32 coreml-fp16; do for r in 2 3; do
  cmp -s "$OUT/track-$p-1/22.json" "$OUT/track-$p-$r/22.json" && echo "$p rounds 1 and $r byte-identical" || echo "$p rounds 1 and $r differ"
done; done | tee -a "$OUT/timing.txt"
uptime
echo "T10 done"
