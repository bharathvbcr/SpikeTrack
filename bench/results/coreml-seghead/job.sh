#!/bin/bash
# Segmentation head as a CoreML ML Program: torch mirror vs upstream's visem22_cell fixture, then
# fp32 and fp16 export + parity + predict time (CPU_AND_GPU, n 20).
# usage: tools/locked.sh cml-seg bash bench/results/coreml-seghead/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=tools/baselines/.venv-export/bin/python
S=tools/baselines/coreml_seghead.py
$PY -s $S torch-check || exit 1
for p in fp32 fp16; do
  $PY -s $S export --precision $p 2>&1 | tail -2
  $PY -s $S check --precision $p -n 20 2>&1 | tail -2
done
