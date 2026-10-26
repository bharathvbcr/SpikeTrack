#!/bin/bash
# usage: tools/jobs/ref.sh SCRIPT [ARGS...]: run an offline reference script on CPU (under tools/locked.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
export SAM3_DEVICE=cpu PYTHONWARNINGS=ignore
exec tools/reference/.venv/bin/python -s "$@"
