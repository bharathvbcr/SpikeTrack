#!/bin/bash
# E35 speed rounds (RULES.md, written before the run): base (the E32 default) / dt16 (+ --det-transformer
# coreml-fp16), held-out VISEM 36, first 32 frames, 3 rounds, order rotated. One binary, built in the
# second copy (SpikeTrack-e34, which carries the hook) and run from its target/release. Data, the
# adapter and the weight edit come from this checkout. The job waits once, up to 30 minutes, for the
# 1-minute load to fall below 8, and stops ("not read") if it does not; an arm whose start or end
# load is above 8 is "not read".
# usage (from SpikeTrack-e34): tools/locked.sh e35-speed bash ../SpikeTrack/bench/results/e35-dettr/job.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(cd "$HERE/../../.." && pwd)
P=$(cd "$M/../SpikeTrack-e34" && pwd)
OUT=$HERE${E31_RUN:+/$E31_RUN}
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
FRAMES=32
(cd "$P" && cargo build --release -j 2 -p spiketrack 2>&1 | tail -1)
BIN=$P/target/release/spiketrack
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git -C "$M" log --oneline -1) + build copy $P (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C "$M/../ojas" log --oneline -1), dirty $(git -C "$M/../ojas" status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"
  echo "tessl $(git -C "$M/../tessl" log --oneline -1)"
} > "$OUT/versions.txt"
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
quiet() { awk -v l="$(load1)" 'BEGIN { exit !(l < 8) }'; }
args_for() {
  case $1 in
    base) ;;
    dt16) echo "--det-transformer coreml-fp16" ;;
  esac
}
waited=0
until quiet || [ "$waited" -ge 1800 ]; do sleep 30; waited=$((waited + 30)); done
quiet || { echo "== E35 speed NOT READ: load $(load1) after ${waited}s"; SKIP=1; }
echo "== waited ${waited}s for a quiet machine"
arm() { # name round
  # run2 onward: wait (up to 10 minutes) for a quiet machine before every arm, not only once.
  # run3 onward: E31_START_MAX (e.g. 6) starts an arm only below that load, so its own threads do
  # not carry the end load past 8; E31_UNTIL (epoch seconds) stops waiting and marks the rest not read.
  local w=0
  if [ -n "${E31_START_MAX:-}" ]; then
    until awk -v l="$(load1)" -v m="$E31_START_MAX" 'BEGIN { exit !(l < m) }' || [ "$(date +%s)" -ge "${E31_UNTIL:-0}" ]; do sleep 20; done
    [ "$(date +%s)" -ge "${E31_UNTIL:-0}" ] && { echo "== $1 r$2 not run: no load below $E31_START_MAX before the deadline: NOT READ"; return; }
  elif [ -n "${E31_RUN:-}" ]; then
    until quiet || [ "$w" -ge 600 ]; do sleep 20; w=$((w + 20)); done
  fi
  local start; start=$(load1)
  local d="$OUT/$1-r$2"
  [ -e "$d" ] && { echo "$d exists; set E31_RUN to a new name" >&2; exit 2; }
  # shellcheck disable=SC2046
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" --prompt sperm --backend metal \
    --max-frames "$FRAMES" --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    $(args_for "$1") --out "$d" > "$OUT/$1-r$2.log" 2>&1
  local s=$? end; end=$(load1)
  find "$d/masks" -type f -delete 2>/dev/null
  rmdir "$d/masks" 2>/dev/null
  local read=read
  awk -v a="$start" -v b="$end" 'BEGIN { exit !(a < 8 && b < 8) }' || read="NOT READ"
  echo "== $1 r$2 exit $s load start $start end $end: $read"
}
ARMS=(base dt16)
[ "${SKIP:-0}" = 1 ] || for r in 1 2 3; do
  n=${#ARMS[@]}; for ((i = 0; i < n; i++)); do arm "${ARMS[$(( (i + r - 1) % n ))]}" $r; done
done
[ "${SKIP:-0}" = 1 ] || python3 -I "$M/tools/analysis/e25_frame.py" "$OUT" "$VID" "${ARMS[@]}"
