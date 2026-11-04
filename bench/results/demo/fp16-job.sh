#!/bin/bash
# Demo clips on the shipped default (since E30, 2026-11-04: CoreML fp16 trunk pipelined one frame ahead,
# CoreML fp16 necks on its worker,
# CoreML fp16 memory attention), replacing the v3 clips, which ran on the coreml-fp32 trunk. Same
# configuration as v3-job.sh otherwise: held-out VISEM 36 (27-46 sperm per frame), first 200
# frames, cap 64, written at 25 fps (about half speed):
#   36-sperm-lora-edit-fp16.mp4  LoRA v3 + the 768-weight edit, prompt "sperm", --boxes detector
#   36-cell-zeroshot-fp16.mp4    zero-shot SAM 3.1, prompt "cell", upstream's mask boxes
#   36-before-after-fp16.mp4     the two side by side (left zero-shot "cell", right adapter "sperm")
#   ../../docs/figures/visem36-frame30-sperm-fp16.png  frame 30 of the adapter clip, for the README
# usage: tools/locked.sh demo-fp16 bash bench/results/demo/fp16-job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=bench/results/demo
status=0
SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
SPIKETRACK_WEIGHT_EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json \
SPIKETRACK_MAX_OBJECTS=64 SPIKETRACK_OVERLAY_FPS=25 SPIKETRACK_DEMO_TAG=fp16 \
  bash tools/jobs/demo_clip.sh 36 200 sperm || status=1
SPIKETRACK_LORA= SPIKETRACK_MAX_OBJECTS=64 SPIKETRACK_OVERLAY_FPS=25 SPIKETRACK_DEMO_TAG=zeroshot-fp16 \
  bash tools/jobs/demo_clip.sh 36 200 cell || status=1
[ "$status" -eq 0 ] || exit 1
rm -rf "$OUT"/run-36-*-fp16/masks
ffmpeg -hide_banner -loglevel error -y -i "$OUT/36-cell-zeroshot-fp16.mp4" -i "$OUT/36-sperm-lora-edit-fp16.mp4" \
  -filter_complex hstack -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/36-before-after-fp16.mp4" || status=1
ffmpeg -hide_banner -loglevel error -y -i "$OUT/36-sperm-lora-edit-fp16.mp4" -vf "select=eq(n\,30)" -frames:v 1 \
  docs/figures/visem36-frame30-sperm-fp16.png || status=1
for r in "$OUT"/run-36-*-fp16/36.run.json; do
  python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[1], r['trunk_precision'], r.get('mem_attn'), r['frames'], 'frames', r['tracks'], 'tracks', round((r['total_seconds']-r['model_load_seconds'])/r['frames'], 3), 's/frame')" "$r"
done
uptime
exit $status
