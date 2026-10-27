#!/bin/bash
# Every module parity gate on CPU and on Metal (Ojas's Metal backend on tessl), against the
# same upstream fixtures and thresholds: text encoder, detector, backbone, and the four
# tracker modules. Records the ojas/tessl commits built.
#   tools/locked.sh module-gates bash tools/jobs/module_gates.sh [cpu|metal|both]
set -uo pipefail
cd "$(dirname "$0")/../.."
which=${1:-both}
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_TEXT_FIXTURES="$PWD/fixtures/text/prompts.safetensors"
export SPIKETRACK_DETECTOR_FIXTURES="$PWD/fixtures/detector"
export SPIKETRACK_FIXTURES="$PWD/fixtures/backbone"
export SPIKETRACK_TRACKER_FIXTURES="$PWD/fixtures/tracker/judo.safetensors"
out=bench/results/module-gates
mkdir -p "$out"
{
  echo "SpikeTrack: $(git log --oneline -1)"
  echo "ojas-st:    $(git -C ../ojas-st log --oneline -1)"
  echo "tessl:      $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"
} | tee "$out/versions.txt"
uptime
status=0
run() {
  local name=$1; shift
  echo "== $name"
  "$@" 2>&1 | tee "$out/$name.log"
  local s=${PIPESTATUS[0]}
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; status=1; }
}
t() { cargo test --release -p st-model --test "$1" -- --ignored --nocapture --test-threads=1 --exact "$2"; }
run build cargo build --release --tests -p st-model
if [ "$which" != metal ]; then
  run cpu-text t text_parity text_encoder_matches_upstream
  run cpu-detector t detector_parity detector_matches_upstream_stage_by_stage
  run cpu-backbone t backbone_parity backbone_matches_pytorch_fp32_on_five_images
  for m in multiplex_mask_decoder_matches_upstream memory_encoder_matches_upstream memory_attention_matches_upstream interactive_decoder_matches_upstream_on_the_conditioning_frame; do
    run "cpu-$m" t tracker_parity "$m"
  done
fi
if [ "$which" != cpu ]; then
  run metal-text t text_parity text_encoder_matches_upstream_on_metal
  run metal-detector t detector_parity detector_matches_upstream_stage_by_stage_on_metal
  run metal-backbone t backbone_parity backbone_matches_pytorch_fp32_on_five_images_on_metal
  for m in multiplex_mask_decoder_matches_upstream memory_encoder_matches_upstream memory_attention_matches_upstream interactive_decoder_matches_upstream_on_the_conditioning_frame; do
    run "metal-$m" t tracker_parity "${m}_on_metal"
  done
fi
if [ "$which" != metal ]; then
  run cpu-vit-layer-norm-eps t backbone_parity vit_layer_norm_eps_matches_upstream
fi
if [ "$which" != cpu ]; then
  run metal-d3 t d3_gate image_mode_masks_match_upstream_on_20_visem_frames_on_metal
  run metal-bf16-d3 t d3_gate image_mode_masks_match_upstream_on_20_visem_frames_on_metal_bf16
  run metal-bf16-backbone t backbone_parity backbone_matches_pytorch_fp32_on_five_images_on_metal_bf16
fi
grep -h "== FAILED" "$out"/*.log 2>/dev/null
uptime
exit $status
