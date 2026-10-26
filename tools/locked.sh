#!/bin/bash
# usage: tools/locked.sh LABEL CMD [ARGS...]: run CMD from the repo root under the machine-wide
# mac_heavy lock (Lappi-decision/tools/mac_heavy.sh; -j 2, footprint and free-disk guards).
set -euo pipefail
export MAC_HEAVY_MAX_LOAD=${MAC_HEAVY_MAX_LOAD:-60}
export MAC_HEAVY_MAX_WAIT_S=${MAC_HEAVY_MAX_WAIT_S:-43200}
unset OJAS_ALLOW_NO_GPU
cd "$(dirname "$0")/.."
label=$1; shift
exec bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh "spiketrack-$label" "$@"
