#!/bin/bash
# CoreML baseline timing under the mac_heavy lock (which excludes every compliant heavy job).
# The GPU is not gated idle: other sessions and the window server keep "Device Utilization %"
# around 20 even with no heavy job running, so a 60 s under-10% gate never passed (2026-10-27).
# Instead the utilization is sampled every 2 s before and during the run and recorded next to
# the timings as a contention note; the bench reports min alongside median.
#   tools/locked.sh coreml bash tools/jobs/coreml_bench.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
out=bench/results/baselines
mkdir -p "$out"
util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'; }
uptime
samples="$out/coreml_gpu_util.txt"
: > "$samples"
for _ in 1 2 3 4 5; do echo "before $(date -u +%H:%M:%S) $(util)" >> "$samples"; sleep 2; done
( while :; do echo "during $(date -u +%H:%M:%S) $(util)" >> "$samples"; sleep 2; done ) &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT
tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py -n 5
kill $sampler 2>/dev/null || true
awk '{n[$1]++; s[$1]+=$3; if ($3>m[$1]) m[$1]=$3} END {for (k in n) printf "GPU utilization %s: mean %.1f%%, max %d%% over %d samples\n", k, s[k]/n[k], m[k], n[k]}' "$samples"
uptime
