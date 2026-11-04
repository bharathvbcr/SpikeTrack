#!/bin/bash
# E34 (RULES.md, written before the run). H-speed: base (SpikeTrack-pipe binary) / e34 (SpikeTrack-e34
# binary), held-out VISEM 36, first 32 frames, 3 rounds, order rotated, each arm started below load 6.
# Then H-same: both binaries on the first 32 frames of the other 4 held-out videos, predictions
# compared with cmp (video 36 is compared in every speed round).
# usage: tools/locked.sh e34 bash bench/results/e34-host-loops/job.sh   (from the main tree)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$HERE/../../.." && pwd)
OUT=$HERE/${E34_RUN:-run1}
[ -e "$OUT" ] && { echo "$OUT exists; set E34_RUN" >&2; exit 2; }
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
UNTIL=$(TZ=America/Chicago date -j -f "%Y-%m-%d %H:%M" "2026-11-04 16:00" +%s)
for t in pipe e34; do
  (cd "$M/../SpikeTrack-$t" && cargo build --release -j 2 -p spiketrack > "$OUT/build-$t.log" 2>&1) || { echo "build $t FAILED"; tail -20 "$OUT/build-$t.log"; exit 1; }
done
bin() { case $1 in base) echo "$M/../SpikeTrack-pipe/target/release/spiketrack" ;; e34) echo "$M/../SpikeTrack-e34/target/release/spiketrack" ;; esac; }
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for a in base e34; do echo "$a $(shasum -a 256 "$(bin $a)" | cut -c1-16)"; done
} > "$OUT/versions.txt"
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
status=0
run() { # arm video name
  local d="$OUT/$3"
  "$(bin "$1")" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$2/images" --prompt sperm --backend metal \
    --max-frames 32 --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    --out "$d" > "$d.log" 2>&1
  local s=$?
  find "$d/masks" -type f -delete 2>/dev/null
  rmdir "$d/masks" 2>/dev/null
  [ $s = 0 ] || { echo "== $3 FAILED (exit $s)"; status=1; }
}
arm() { # name round
  until awk -v l="$(load1)" 'BEGIN { exit !(l < 6) }' || [ "$(date +%s)" -ge "$UNTIL" ]; do sleep 20; done
  [ "$(date +%s)" -ge "$UNTIL" ] && { echo "== $1 r$2 not run: no load below 6 before the cutoff: NOT READ"; return; }
  local start; start=$(load1)
  run "$1" 36 "$1-r$2"
  local end; end=$(load1)
  local read=read
  awk -v a="$start" -v b="$end" 'BEGIN { exit !(a < 8 && b < 8) }' || read="NOT READ"
  echo "== $1 r$2 load start $start end $end: $read"
}
ARMS=(base e34)
for r in 1 2 3; do
  for ((i = 0; i < 2; i++)); do arm "${ARMS[$(( (i + r - 1) % 2 ))]}" $r; done
done
python3 -I "$M/tools/analysis/e25_frame.py" "$OUT" 36 "${ARMS[@]}"
for r in 1 2 3; do
  cmp -s "$OUT/base-r$r/36.json" "$OUT/e34-r$r/36.json" && echo "H-same video 36 round $r: byte-identical" || { echo "H-same video 36 round $r: DIFFERS"; status=1; }
done
for v in 21 22 47 82; do
  run base $v "same-base-$v"
  run e34 $v "same-e34-$v"
  cmp -s "$OUT/same-base-$v/$v.json" "$OUT/same-e34-$v/$v.json" && echo "H-same video $v: byte-identical" || { echo "H-same video $v: DIFFERS"; status=1; }
done
echo "E34 status $status"
exit $status
