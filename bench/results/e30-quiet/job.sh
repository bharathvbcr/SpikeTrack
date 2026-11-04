#!/bin/bash
# E30 speed rounds (RULES.md, written before the run): t16m16 (the default) / t16m16n16 (+ fp16 CoreML
# necks) / o32 (+ --overlap-dispatches 32), held-out VISEM 36, first 32 frames, 3 rounds, order
# rotated. The binary runs from target/release (no fresh copy for macOS to assess mid-run). The job
# waits once, up to 30 minutes, for the 1-minute load to fall below 8, and stops ("not read") if it
# does not. The load at each arm's start and end is logged; an arm with either above 8 is "not read".
# Then clippy -D warnings on both repositories (ojas: its CI command; SpikeTrack: the workspace).
# usage: tools/locked.sh e30-speed bash bench/results/e30-quiet/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=$PWD
OUT=bench/results/e30-quiet${E30_RUN:+/$E30_RUN}
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
FRAMES=32
cargo build --release -j 2 -p spiketrack 2>&1 | tail -1
BIN=target/release/spiketrack
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1), dirty $(git -C ../ojas status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"
  echo "tessl $(git -C ../tessl log --oneline -1)"
} > "$OUT/versions.txt"
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
quiet() { awk -v l="$(load1)" 'BEGIN { exit !(l < 8) }'; }
args_for() {
  case $1 in
    t16m16) echo "--necks ojas" ;; # the default until E30 made fp16 necks the default
    t16m16n16) echo "--necks coreml-fp16" ;;
    o32) echo "--overlap-dispatches 32" ;;
  esac
}
waited=0
until quiet || [ "$waited" -ge 1800 ]; do sleep 30; waited=$((waited + 30)); done
quiet || { echo "== E30 speed NOT READ: load $(load1) after ${waited}s"; SKIP=1; }
echo "== waited ${waited}s for a quiet machine"
arm() { # name round
  local start; start=$(load1)
  local d="$OUT/$1-r$2"
  rm -rf "$d"
  # shellcheck disable=SC2046
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" --prompt sperm --backend metal \
    --max-frames "$FRAMES" --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    $(args_for "$1") --out "$d" > "$OUT/$1-r$2.log" 2>&1
  local s=$? end; end=$(load1)
  rm -rf "$d/masks"
  local read=read
  awk -v a="$start" -v b="$end" 'BEGIN { exit !(a < 8 && b < 8) }' || read="NOT READ"
  echo "== $1 r$2 exit $s load start $start end $end: $read"
}
read -r -a ARMS <<< "${E30_ARMS:-t16m16 t16m16n16 o32}"
[ "${SKIP:-0}" = 1 ] || for r in 1 2 3; do
  n=${#ARMS[@]}; for ((i = 0; i < n; i++)); do arm "${ARMS[$(( (i + r - 1) % n ))]}" $r; done
  if [ -d "$OUT/o32-r$r" ]; then
    cmp -s "$OUT/t16m16-r$r/$VID.json" "$OUT/o32-r$r/$VID.json" && echo "O1 o32 r$r: byte-identical to t16m16" || echo "O1 o32 r$r: DIFFERS from t16m16"
  fi
done
[ "${SKIP:-0}" = 1 ] || python3 -I tools/analysis/e25_frame.py "$OUT" "$VID" "${ARMS[@]}"
[ -n "${E30_RUN:-}" ] || cargo clippy --workspace --all-targets -j 2 -- -D warnings > "$OUT/clippy-spiketrack.log" 2>&1; echo "== clippy SpikeTrack exit $?"
[ -n "${E30_RUN:-}" ] || (cd ../ojas && cargo clippy --workspace --all-targets -j 2 -- -D warnings) > "$OUT/clippy-ojas.log" 2>&1; echo "== clippy ojas exit $?"
