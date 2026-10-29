#!/bin/bash
# usage: hota_cml_inlock.sh CAP PRECISION   (one cap and precision per lock hold, so peers interleave)
# Inside the mac_heavy lock: E10's HOTA gate. Held-out VISEM HOTA with --trunk-precision PRECISION
# (coreml-fp16 or coreml-fp32), same setup as bench/results/track/held300-lora600-sperm-detector-max64:
# 300 frames per video, prompt "sperm", --boxes detector, the r16 LoRA, --max-objects CAP.
# Pre-stated (docs/SPEED_PLAN.md E10): fp16 ships as --fast only if HOTA >= 0.47 and within 0.005
# of the Ojas f32 run at the same cap (0.4868 at 64, 0.4695 at 16).
# Binary: the frozen copy of main's build (spiketrack-e10), so a rebuild mid-run cannot change it.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
P=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
BIN=$P/spiketrack-e10
cap=$1; prec=$2
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
OUT=$M/bench/results/track/held300-lora600-sperm-detector-$prec-max$cap
mkdir -p "$OUT/run" "$OUT/logs"
{ echo "SpikeTrack $(git -C $M log --oneline -1) (binary $(shasum -a 256 $BIN | cut -c1-16))"; echo "ojas-st $(git -C $M/../ojas-st log --oneline -1)"; echo "tessl $(git -C $M/../tessl log --oneline -1)"; uptime; } > "$OUT/versions.txt"
for v in 21 22 36 47 82; do
  echo "H10 $prec cap $cap video $v load $(sysctl -n vm.loadavg)"
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal \
    --max-frames 300 --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects "$cap" --trunk-precision "$prec" --out "$OUT/run" > "$OUT/logs/$v.log" 2>&1 || { echo "H10 track $v failed"; tail -5 "$OUT/logs/$v.log"; exit 1; }
  rm -rf "$OUT/run/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"H10 {r['video']}: {r['mean_seconds_per_frame']:.3f} s/frame, {r['tracks']} tracks, trunk {r['trunk_precision']}, max_objects {r['max_objects']}\")" "$OUT/run/$v.run.json"
done
"$BIN" eval-visem --pred "$OUT/run" --split "$M/tools/reference/visem_split.json" --set held_out \
  --visem "$M/data/visem/VISEM_Tracking_Train_v4/Train" --max-frames 300 --out "$OUT/eval.json" 2>&1 | tee "$OUT/eval.txt"
uptime
echo "H10 done"
