#!/bin/bash
# E33 smoke (not a gate): on the E32 default, does the interactive neck on the trunk's worker
# (--inter-neck coreml-fp16) or the existing CoreML segmentation head (--seg-head coreml-fp16) or both
# shorten the frame? Video 36, first 8 frames, shipped configuration (LoRA v3 + lorav3_exact_it_3p00.json,
# prompt sperm, --boxes detector, cap 64). One binary from the build copy.
# usage (from the build copy): tools/locked.sh e33-smoke bash ../SpikeTrack/bench/results/e33-smoke/job.sh
set -uo pipefail
P=$PWD
M=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$M/bench/results/e33-smoke
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
cargo build --release -j 2 -p spiketrack > "$OUT/build.log" 2>&1 || { echo "build FAILED"; tail -20 "$OUT/build.log"; exit 1; }
BIN=$P/target/release/spiketrack
status=0
run() { # name args...
  local name=$1; shift
  local d=$OUT/$name
  [ -e "$d" ] && { echo "$d exists"; status=1; return; }
  local l0; l0=$(sysctl -n vm.loadavg | awk '{print $2}')
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/36/images" --prompt sperm --backend metal \
    --max-frames 8 --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "$name FAILED"; tail -5 "$d.log"; status=1; return; }
  find "$d/masks" -type f -delete; rmdir "$d/masks"
  python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; s=f[2:]; print(f\"{sys.argv[3]}: load {sys.argv[4]}, inter-neck {r.get('inter_neck')} ({'set' if r.get('external_inter_neck') else 'none'}), seg-head {r.get('seg_head')}, {r.get('tracks')} tracks, frames 2.. {' '.join(f'{x:.2f}' for x in s)} (mean {1e3*sum(s)/len(s):.0f} ms)\")" "$d/36.run.json" "$d.log" "$name" "$l0"
}
run base
run ni16 --inter-neck coreml-fp16
run sh16 --seg-head coreml-fp16
run both --inter-neck coreml-fp16 --seg-head coreml-fp16
run base-2
echo "smoke status $status"
exit $status
