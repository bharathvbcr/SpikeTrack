#!/bin/bash
# CoreML baseline timing; run via tessl target-gp/gpu_idle_locked.sh (GPU idle 60 s, then the mac_heavy lock).
set -euo pipefail
cd "$(dirname "$0")/../.."
uptime
ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1
tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py -n 5
uptime
