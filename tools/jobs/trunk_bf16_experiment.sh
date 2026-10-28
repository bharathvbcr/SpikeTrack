#!/bin/bash
# The bf16-trunk-linears experiment on Metal (Backbone::with_trunk_linears_bf16):
# 1. the detect-track session against upstream fp32 on VISEM 22, gated like the f32 session (every
#    yielded frame's id set equal, mean mask IoU >= 0.95);
# 2. only if that passes, a paired timing of `spiketrack track` on VISEM 22's first N frames, f32
#    against bf16-linears with the same binary, alternating per round (order flipped each round),
#    3 rounds each, the per-run mean s/frame from <video>.run.json, load and GPU utilization
#    recorded before each run.
# Results: bench/results/trunk-bf16/
# usage: tools/locked.sh trunk-bf16 bash tools/jobs/trunk_bf16_experiment.sh [N=49]
set -uo pipefail
cd "$(dirname "$0")/../.."
N=${1:-49}
OUT=bench/results/trunk-bf16
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_DT_FIXTURES="$PWD/fixtures/detect_track" SPIKETRACK_DT_VIDEO=22
mkdir -p "$OUT"
util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'; }
{ echo "SpikeTrack $(git log --oneline -1)"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } | tee "$OUT/versions.txt"
cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_with_bf16_trunk_linears_matches_upstream_on_metal 2>&1 | tee "$OUT/session-parity-visem22.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "the bf16-linears session missed the gate; no timing" | tee -a "$OUT/session-parity-visem22.log"; exit 1; }
cargo build --release -j 2 -p spiketrack 2>&1 | tail -2
BIN=target/release/spiketrack
FRAMES=data/visem/VISEM_Tracking_Train_v4/Train/22/images
: > "$OUT/timing.txt"
for round in 1 2 3; do
  if [ $((round % 2)) -eq 1 ]; then order="f32 bf16-linears"; else order="bf16-linears f32"; fi
  for p in $order; do
    d="$OUT/run-$p-$round"
    rm -rf "$d"
    echo "round $round $p: load $(sysctl -n vm.loadavg), GPU $(util)%" | tee -a "$OUT/timing.txt"
    "$BIN" track "$FRAMES" --prompt cell --backend metal --max-frames "$N" --trunk-precision "$p" --out "$d" > "$d.log" 2>&1 || { echo "track failed ($p, round $round)"; exit 1; }
    python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"round {sys.argv[2]} {r['trunk_precision']}: {r['mean_seconds_per_frame']:.4f} s/frame mean over {r['frames']} frames, {r['tracks']} tracks\")" "$d/22.run.json" "$round" | tee -a "$OUT/timing.txt"
  done
done
python3 - "$OUT/timing.txt" <<'EOF' | tee -a "$OUT/timing.txt"
import re, statistics, sys
v = {}
for line in open(sys.argv[1]):
    m = re.match(r"round (\d+) (\S+): ([0-9.]+) s/frame", line)
    if m:
        v.setdefault(m[2], []).append(float(m[3]))
f, b = v.get("f32", []), v.get("bf16-linears", [])
print(f"f32: {f}; bf16-linears: {b}")
if f and b:
    print(f"median f32 {statistics.median(f):.4f}, bf16-linears {statistics.median(b):.4f} s/frame; ratio bf16/f32 {statistics.median(b)/statistics.median(f):.3f}; spread f32 {max(f)/min(f):.3f}, bf16 {max(b)/min(b):.3f}")
EOF
uptime | tee -a "$OUT/timing.txt"
