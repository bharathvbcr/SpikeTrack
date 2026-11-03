#!/bin/bash
# Inside the mac_heavy lock: SpikeTrack's tracker step against the CoreML port's `track_step` at
# 16 objects, one tracker state, steady frames, interleaved in one hold. Pre-stated in
# bench/results/ab-coreml-tracker/README.md (arms, boundary, decision rule).
#   AB_RUN=run3 tools/locked.sh ab-coreml bash tools/jobs/ab_coreml_tracker.sh   (a new folder name)
# Arms: SpikeTrack (`track`, VISEM 82, --max-objects 16, --profile stages), the port's fp16
# tracker (+ its fp16 backbone), and the port's fp32 tracker when its package is present. Three
# rounds, the order rotated each round; unprofiled SpikeTrack controls before and after.
# Summary: tools/analysis/ab_coreml_tracker.py -> summary.md.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
cd "$M" || exit 1
OUT=$M/bench/results/ab-coreml-tracker/${AB_RUN:?set AB_RUN to the run folder name, e.g. run2}
[ -e "$OUT" ] && { echo "$OUT exists; pick a new AB_RUN"; exit 1; }
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
FRAMES=34   # frames 8..33 are steady (the memory bank holds 7 frames from frame 7 on)
STEPS=32    # timed port calls after 2 untimed ones: steps 2..33, steady 8..33
PY=$M/tools/baselines/.venv/bin/python
FP32_PKG=$HOME/Code/research/models/sam3.1-coreml/tracker/models/dense_sam3_trackstep_fp32.mlpackage
util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$'; }
status=0

step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg) gpu $(util)%"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
  return "$s"
}
step build cargo build --release -j 2 -p spiketrack || exit 1
step unit-tests cargo test --release -j 2 -p spiketrack --bins track_profile || exit 1
BIN=$OUT/spiketrack
cp target/release/spiketrack "$BIN"
{
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1) (dirty files: $(git -C ../ojas status --short | wc -l | tr -d ' '))"
  echo "tessl $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"
  echo "coremltools $("$PY" -c 'import coremltools; print(coremltools.__version__)')"
  echo "port fp32 tracker: $([ -d "$FP32_PKG" ] && echo present || echo ABSENT, arm not run)"
  sysctl -n machdep.cpu.brand_string; uptime
} > "$OUT/versions.txt"
cat "$OUT/versions.txt"

spiketrack() { # name [extra...]
  local name=$1; shift
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  step "$name" "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames "$FRAMES" --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --out "$d" "$@" || return
  rm -rf "$d/masks"
}
port() { # precision round
  local p=$1 r=$2 only=tracker
  [ "$p" = fp16 ] && only=both
  if [ "$p" = fp32 ] && [ ! -d "$FP32_PKG" ]; then echo "== coreml-fp32-r$r: not run, package absent"; return; fi
  step "coreml-$p-r$r" "$PY" -s tools/baselines/coreml_bench.py --only "$only" --tracker-precision "$p" -n "$STEPS" \
    --out "$OUT/coreml-$p-r$r.json"
}

spiketrack st-control-1
for r in 1 2 3; do
  case $r in
    1) port fp16 1; port fp32 1; spiketrack st-r1 --profile stages ;;
    2) spiketrack st-r2 --profile stages; port fp16 2; port fp32 2 ;;
    3) port fp32 3; spiketrack st-r3 --profile stages; port fp16 3 ;;
  esac
done
spiketrack st-control-2

for r in st-r1 st-r2 st-r3 st-control-2; do
  cmp -s "$OUT/st-control-1/82.json" "$OUT/$r/82.json" && echo "$r: predictions byte-identical to st-control-1" \
    || { echo "== FAILED: $r predictions differ from st-control-1"; status=1; }
done
python3 -I tools/analysis/ab_coreml_tracker.py "$OUT" || status=1
uptime
echo "AB status $status"
exit $status
