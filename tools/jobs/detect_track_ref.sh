#!/bin/bash
# Upstream's text-prompted detect -> track loop on the first VISEM frames (CPU oracle).
# usage: tools/jobs/detect_track_ref.sh [VIDEO] [FRAMES] [PROMPT]
set -uo pipefail
cd "$(dirname "$0")/../.."
uptime
bash tools/jobs/ref.sh tools/reference/detect_track_reference.py --video "${1:-11}" --frames "${2:-32}" --prompt "${3:-cell}"
s=$?
uptime
exit $s
