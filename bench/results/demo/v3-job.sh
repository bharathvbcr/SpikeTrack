#!/bin/bash
# Demo clips for the submission video, on the dense held-out VISEM video 36 (27-46 sperm per
# frame), first 200 frames, cap 64, written at 25 fps (about half speed):
#   36-sperm-lora-edit-v3.mp4   LoRA v3 + the 768-weight edit, prompt "sperm", --boxes detector
#                               (the configuration of the README's headline HOTA)
#   36-cell-zeroshot.mp4        zero-shot SAM 3.1, prompt "cell", upstream's mask boxes
# usage: tools/locked.sh demo-v3 bash bench/results/demo/v3-job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
status=0
SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
SPIKETRACK_WEIGHT_EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json \
SPIKETRACK_MAX_OBJECTS=64 SPIKETRACK_OVERLAY_FPS=25 SPIKETRACK_DEMO_TAG=v3 \
  bash tools/jobs/demo_clip.sh 36 200 sperm || status=1
SPIKETRACK_LORA= SPIKETRACK_MAX_OBJECTS=64 SPIKETRACK_OVERLAY_FPS=25 SPIKETRACK_DEMO_TAG=zeroshot \
  bash tools/jobs/demo_clip.sh 36 200 cell || status=1
exit $status
