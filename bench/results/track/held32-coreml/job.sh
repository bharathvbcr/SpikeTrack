#!/bin/bash
# G2 of ../../e25-coreml-frame/RULES.md (2026-11-04): held-out HOTA, first 32 frames of the 5 held-out
# videos, for the CoreML placements on the shipped configuration (LoRA v3 + lorav3_exact_it_3p00.json,
# prompt sperm, --boxes detector, cap 64). One subfolder per config:
#   default  : the fp32 reference, coreml-fp32 trunk with everything else on Ojas: the shipped default
#              when these runs were made, pinned explicitly since t16m16 became the default on
#              2026-11-04 (the baseline must reproduce 0.6025)
#   m16      : --mem-attn coreml-fp16
#   seg32    : --seg-head coreml-fp32
#   fast16   : --trunk-precision coreml-fp16 --mem-attn coreml-fp16 --seg-head coreml-fp16
#   t16m16   : --trunk-precision coreml-fp16 --mem-attn coreml-fp16 (added 2026-11-04 after E27: the
#              combination that passes G4 placement by placement; the CoreML segmentation head failed G4)
#   t16m16n16: t16m16 + --necks coreml-fp16 (E30 lever A, ../../e30-quiet/RULES.md)
# Each config runs with --trunk-pipeline on, except default; E26 checks P1, that the pipeline does not
# change a byte. Gates (pre-stated in RULES.md):
#   fp32 configs: within +-0.002 of 0.6025, no video moving > 0.005;
#   fp16 configs: >= 0.6005, no video dropping > 0.01.
# usage: tools/locked.sh held32-coreml bash bench/results/track/held32-coreml/job.sh [CONFIG ...]
set -uo pipefail
cd "$(dirname "$0")/../../../.."
BASE=bench/results/track/held32-coreml
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
for cfg in "${@:-default m16 seg32 fast16}"; do
  for c in $cfg; do
    OUT=$BASE/$c
    mkdir -p "$OUT"
    { date -u +"date %Y-%m-%dT%H:%M:%SZ"; echo "HEAD $(git rev-parse HEAD)"; git status --short | wc -l | xargs echo "dirty files"; echo "config $c"; } > "$OUT/provenance.txt"
    unset SPIKETRACK_TRUNK SPIKETRACK_TRUNK_PIPELINE SPIKETRACK_MEM_ATTN SPIKETRACK_SEG_HEAD SPIKETRACK_NECKS
    case $c in
      default) export SPIKETRACK_TRUNK=coreml-fp32 ;;
      m16) export SPIKETRACK_TRUNK_PIPELINE=on SPIKETRACK_MEM_ATTN=coreml-fp16 ;;
      seg32) export SPIKETRACK_TRUNK_PIPELINE=on SPIKETRACK_SEG_HEAD=coreml-fp32 ;;
      fast16) export SPIKETRACK_TRUNK=coreml-fp16 SPIKETRACK_TRUNK_PIPELINE=on SPIKETRACK_MEM_ATTN=coreml-fp16 SPIKETRACK_SEG_HEAD=coreml-fp16 ;;
      t16m16) export SPIKETRACK_TRUNK=coreml-fp16 SPIKETRACK_TRUNK_PIPELINE=on SPIKETRACK_MEM_ATTN=coreml-fp16 SPIKETRACK_NECKS=ojas ;;
      t16m16n16) export SPIKETRACK_TRUNK=coreml-fp16 SPIKETRACK_TRUNK_PIPELINE=on SPIKETRACK_MEM_ATTN=coreml-fp16 SPIKETRACK_NECKS=coreml-fp16 ;;
      *) echo "unknown config $c" >&2; exit 2 ;;
    esac
    SPIKETRACK_TRACK_OUT="$OUT" SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_MAX_OBJECTS=64 \
      SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
      SPIKETRACK_WEIGHT_EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json \
      bash tools/jobs/track_visem.sh 32
    echo "== $c exit $?"
  done
done
