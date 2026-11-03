#!/bin/bash
# visem-lora-v4 step 2 (PLAN.md): warm start from v3 (step 160), 400 steps on the 128-frame cache, v3's
# flags plus --box-loss-weights 7.5 4.0 (decided from ../../weight-space/diag-train5/buckets.txt before
# training). Checkpoints every 100 steps; no held-out evaluation here (--no-eval). Refuses to start
# unless selftest.log (lora_selftest.py, including the box-weight check) ended with ALL PASS.
# usage: tools/locked.sh lora-v4-train bash bench/results/lora/visem-lora-v4/train-job.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."
D=bench/results/lora/visem-lora-v4
grep -q "^ALL PASS" "$D/selftest.log" || { echo "selftest.log has no ALL PASS; not training" >&2; exit 1; }
export SAM3_DEVICE=mps PYTHONWARNINGS=ignore PYTORCH_ENABLE_MPS_FALLBACK=1
date -u +"start %Y-%m-%dT%H:%M:%SZ"
tools/reference/.venv/bin/python -s tools/reference/train_lora.py --device mps --cache fixtures/lora_cache \
  --run visem-lora-v4 --init-adapter fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
  --include-heads --adapt-class-tokens --o2m --o2m-weight 2.0 --neg-query-weight 1.5 --hard-neg-frac 0.65 \
  --lr 7e-5 --token-lr 2.5e-4 --warmup 16 --steps 400 --save-every 100 --no-act-ckpt --no-eval \
  --box-loss-weights 7.5 4.0 2>&1 | tee "$D/train.log"
s=${PIPESTATUS[0]}
date -u +"end %Y-%m-%dT%H:%M:%SZ"
exit $s
