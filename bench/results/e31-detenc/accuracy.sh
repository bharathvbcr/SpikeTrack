#!/bin/bash
# E31 accuracy gates (RULES.md): E-G2 (first 32 frames of the 5 held-out videos) or E-G3 (1,500
# frames), for e16 = today's default + --det-encoder coreml-fp16, on the build copy's binary. The
# references are t16m16n16 in ../track/held32-coreml and ../track/held300-coreml.
# usage (from SpikeTrack-pipe): tools/locked.sh e31-g2 bash ../SpikeTrack/bench/results/e31-detenc/accuracy.sh 32
#                               tools/locked.sh e31-g3 bash ../SpikeTrack/bench/results/e31-detenc/accuracy.sh 300
set -uo pipefail
N=${1:?frames per video: 32 or 300}
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$HERE/../../.." && pwd)
P=$(cd "$M/../SpikeTrack-pipe" && pwd)
case $N in
  32) OUT=$M/bench/results/track/held32-coreml/t16m16n16e16 ;;
  300) OUT=$M/bench/results/track/held300-coreml/t16m16n16e16 ;;
  *) echo "N must be 32 or 300" >&2; exit 2 ;;
esac
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
{ date -u +"date %Y-%m-%dT%H:%M:%SZ"; echo "HEAD $(git -C "$M" rev-parse HEAD) + build copy $P"; echo "config t16m16n16e16"; } > "$OUT/provenance.txt"
cd "$P"
unset SPIKETRACK_TRUNK SPIKETRACK_TRUNK_PIPELINE SPIKETRACK_MEM_ATTN SPIKETRACK_SEG_HEAD SPIKETRACK_NECKS
SPIKETRACK_TRUNK=coreml-fp16 SPIKETRACK_TRUNK_PIPELINE=on SPIKETRACK_MEM_ATTN=coreml-fp16 SPIKETRACK_NECKS=coreml-fp16 \
  SPIKETRACK_DET_ENCODER=coreml-fp16 \
  SPIKETRACK_TRACK_OUT="$OUT" SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_MAX_OBJECTS=64 \
  SPIKETRACK_LORA="$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
  SPIKETRACK_WEIGHT_EDIT="$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
  bash tools/jobs/track_visem.sh "$N"
echo "== t16m16n16e16 N=$N exit $?"
