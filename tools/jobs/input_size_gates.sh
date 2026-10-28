#!/bin/bash
# The smaller-input (E7) gates: the generated trunk RoPE tables against the checkpoint's, and the
# trunk + three necks at 672 pixels (48 x 48 tokens) against upstream's own code built for that
# size, on CPU and on Metal. The 672 fixture is made once, offline:
#   SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_backbone_size.py \
#     --size 672 --image data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg \
#     --out fixtures/backbone672/visem_22_f0.safetensors
# Under the lock:
#   tools/locked.sh input-size-gates bash tools/jobs/input_size_gates.sh [cpu|metal|both]
set -uo pipefail
cd "$(dirname "$0")/../.."
which=${1:-both}
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_FIXTURES_672="$PWD/fixtures/backbone672"
out=bench/results/input-size-gates
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
t() { cargo test --release -j 2 -p st-model --test backbone_parity -- --ignored --nocapture --test-threads=1 --exact "$1"; }
run build cargo build --release -j 2 --tests -p st-model
run rope-tables t generated_rope_tables_equal_the_checkpoints
if [ "$which" != metal ]; then run cpu-backbone-672 t backbone_matches_pytorch_fp32_at_672; fi
if [ "$which" != cpu ]; then run metal-backbone-672 t backbone_matches_pytorch_fp32_at_672_on_metal; fi
grep -h "== FAILED" "$out"/*.log 2>/dev/null
uptime
exit $status
