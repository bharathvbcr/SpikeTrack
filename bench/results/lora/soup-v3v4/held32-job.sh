#!/bin/bash
# v3/v4 adapter soup, rule 4 of PLAN.md: ONE held-out run, first 32 frames of 21 22 36 47 82, with the
# soup and edit chosen on train-5. Adopt only if HOTA >= 0.6125 and at least 4 of 5 videos are above
# v3's held-out per-video scores (21 0.6744, 22 0.6799, 36 0.5752, 47 0.5596, 82 0.5552).
# usage: tools/locked.sh held32-soup bash bench/results/lora/soup-v3v4/held32-job.sh SOUP EDIT
set -uo pipefail
cd "$(dirname "$0")/../../../.."
SOUP=${1:?usage: $0 SOUP EDIT} EDIT=${2:?usage: $0 SOUP EDIT}
[ -f "$SOUP" ] && [ -f "$EDIT" ] || { echo "missing $SOUP or $EDIT" >&2; exit 2; }
OUT=bench/results/lora/soup-v3v4/held32
mkdir -p "$OUT"
{ date -u +"date %Y-%m-%dT%H:%M:%SZ"; echo "HEAD $(git rev-parse HEAD)"; git status --short | wc -l | xargs echo "dirty files"
  echo "soup $SOUP sha256 $(shasum -a 256 "$SOUP" | cut -d' ' -f1)"; echo "edit $EDIT"; } > "$OUT/provenance.txt"
SPIKETRACK_TRACK_OUT="$OUT" SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_MAX_OBJECTS=64 \
  SPIKETRACK_LORA="$SOUP" SPIKETRACK_WEIGHT_EDIT="$EDIT" bash tools/jobs/track_visem.sh 32
