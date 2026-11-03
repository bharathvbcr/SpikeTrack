#!/bin/bash
# Speed baseline on the current pins, one mac_heavy hold, before any optimization work:
#   1. GPU occupancy: a 16-frame `track` (f32, 16 objects) with "Device Utilization %" sampled
#      every second and the process's CPU time, so idle GPU time between commits shows up;
#   2. the per-op frame profile, f32 and whole-model bf16 (tools/jobs/frame_profile.sh);
#   3. the CoreML port under CPU_AND_GPU, ALL and CPU_AND_NE (tools/baselines/coreml_bench.py).
# Utilization from other sessions is sampled before each step and recorded, not gated.
# The track run folds the held-out LoRA adapter (prompt "sperm" finds nothing zero-shot, which
# leaves the tracker idle). STEPS picks steps (default "occupancy profile coreml").
# Results: bench/results/speed-baseline/.
#   tools/locked.sh speed-baseline bash tools/jobs/speed_baseline.sh
#   STEPS="occupancy profile" tools/locked.sh speed-baseline bash tools/jobs/speed_baseline.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=bench/results/speed-baseline
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'; }
idle() { for _ in 1 2 3 4 5; do printf '%s ' "$(util)"; sleep 1; done; }
sample() { while :; do echo "$(date -u +%H:%M:%S) $(util)"; sleep 1; done; }
{
  echo "SpikeTrack $(git log --oneline -1)"
  echo "ojas $(git -C ../ojas log --oneline -1)"
  echo "tessl $(git -C ../tessl log --oneline -1)"
  sysctl -n machdep.cpu.brand_string; uptime
} | tee "$OUT/versions.txt"
status=0
STEPS=${STEPS:-occupancy profile coreml}
LORA=fixtures/lora_runs/visem-lora-r16/adapter.safetensors

echo "== build"
cargo build --release -j 2 -p spiketrack 2>&1 | tail -2
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "build failed"; exit 1; }

if [[ " $STEPS " == *" occupancy "* ]]; then
echo "== occupancy"
d="$OUT/occupancy-f32"
rm -rf "$d"
echo "GPU before: $(idle)(load $(sysctl -n vm.loadavg))" | tee "$OUT/occupancy.txt"
sample > "$OUT/occupancy-util.txt" &
sampler=$!
/usr/bin/time -l target/release/spiketrack track data/visem/VISEM_Tracking_Train_v4/Train/22/images \
  --prompt sperm --lora "$LORA" --backend metal --max-frames 16 --max-objects 16 --trunk-precision f32 --out "$d" > "$d.log" 2> "$d.time"
r=$?
kill "$sampler"
rm -rf "$d/masks"
[ "$r" -eq 0 ] || { echo "track failed"; status=1; }
python3 - "$d/22.run.json" "$OUT/occupancy-util.txt" "$d.time" <<'EOF' | tee -a "$OUT/occupancy.txt"
import json, re, statistics, sys
r = json.load(open(sys.argv[1]))
u = [int(l.split()[1]) for l in open(sys.argv[2]) if len(l.split()) == 2]
t = open(sys.argv[3]).read()
real = float(re.search(r"([\d.]+) real", t).group(1)); user = float(re.search(r"([\d.]+) user", t).group(1)); sys_ = float(re.search(r"([\d.]+) sys", t).group(1))
print(f"track f32, {r['frames']} frames, {r['tracks']} tracks: {r['mean_seconds_per_frame']:.4f} s/frame mean")
print(f"GPU Device Utilization %: median {statistics.median(u)}, mean {statistics.mean(u):.1f}, min {min(u)}, max {max(u)}, {len(u)} samples (includes model load)")
print(f"process: real {real:.1f} s, user {user:.1f} s, sys {sys_:.1f} s, CPU/real {(user + sys_) / real:.2f} cores")
m = re.search(r"(\d+)\s+peak memory footprint", t)
print(f"peak memory footprint {int(m.group(1)) / 2**30:.2f} GiB" if m else "peak memory footprint: not reported")
EOF
fi

[[ " $STEPS " == *" profile "* ]] && for p in metal metal_bf16; do
  echo "== profile $p (GPU before: $(idle))"
  PROFILE_OUT="$OUT/profile" bash tools/jobs/frame_profile.sh "$p" > /dev/null 2>&1 || { echo "profile $p failed"; status=1; }
  rg "unprofiled|recorded .* op calls|test result" "$OUT/profile/frame-$p.log"
done

[[ " $STEPS " == *" coreml "* ]] && for units in CPU_AND_GPU ALL CPU_AND_NE; do
  echo "== coreml $units (GPU before: $(idle))"
  sample > "$OUT/coreml-util-$units.txt" &
  sampler=$!
  tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py -n 5 --units "$units" 2>&1 | tail -3 | tee "$OUT/coreml-$units.log"
  kill "$sampler"
done
# coreml_bench writes into bench/results/baselines; copy this run's json beside the rest.
if [[ " $STEPS " == *" coreml "* ]]; then
  cp bench/results/baselines/coreml.json "$OUT/coreml-CPU_AND_GPU.json"
  cp bench/results/baselines/coreml-ALL.json bench/results/baselines/coreml-CPU_AND_NE.json "$OUT/"
fi
uptime | tee -a "$OUT/versions.txt"
echo "speed-baseline exit $status"
exit $status
