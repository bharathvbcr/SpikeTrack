#!/bin/bash
# E29: unwaited Metal commits every N dispatches (RULES.md, written before the run). Shipped
# configuration on held-out VISEM 36, first 32 frames, arms o256 (default) / o32 / o8 / o1, 3 rounds,
# order rotated. Checks first: the flag's refusal test and the ojas-metal overlap tests.
# usage: tools/locked.sh e29-overlap bash bench/results/e29-overlap/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=$PWD
OUT=bench/results/e29-overlap
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
FRAMES=32
status=0
{
  cargo build --release -j 2 -p spiketrack 2>&1 | tail -2
  cargo test --release -j 2 -p spiketrack --bin spiketrack -- bad_command_lines_fail_before_any_model_work 2>&1 | rg "^test |test result"
  (cd ../ojas && cargo test --release -j 2 -p ojas-metal --lib -- set_overlap_dispatches overlap_commits 2>&1 | rg "^test |test result")
} > "$OUT/checks.log" 2>&1
cat "$OUT/checks.log"
[ "$(rg -c "test result: ok" "$OUT/checks.log")" = 2 ] || { echo "== checks FAILED"; exit 1; }
BIN=$OUT/spiketrack
cp target/release/spiketrack "$BIN"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1), dirty $(git -C ../ojas status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"
  echo "tessl $(git -C ../tessl log --oneline -1)"
} > "$OUT/versions.txt"
args_for() {
  case $1 in
    o256) echo "" ;;
    o32) echo "--overlap-dispatches 32" ;;
    o8) echo "--overlap-dispatches 8" ;;
    o1) echo "--overlap-dispatches 1" ;;
  esac
}
arm() { # name round
  local d="$OUT/$1-r$2"
  rm -rf "$d"
  # shellcheck disable=SC2046
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" --prompt sperm --backend metal \
    --max-frames "$FRAMES" --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    $(args_for "$1") --out "$d" > "$OUT/$1-r$2.log" 2>&1
  local s=$?
  rm -rf "$d/masks"
  echo "== $1 r$2 exit $s load $(uptime | sed 's/.*averages: //')"
  [ "$s" -eq 0 ] || status=1
}
ARMS=(o256 o32 o8 o1)
for r in 1 2 3; do
  for i in 0 1 2 3; do arm "${ARMS[$(( (i + r - 1) % 4 ))]}" $r; done
  for a in o32 o8 o1; do
    cmp -s "$OUT/o256-r$r/$VID.json" "$OUT/$a-r$r/$VID.json" && echo "O1 $a r$r: byte-identical to o256" || echo "O1 $a r$r: DIFFERS from o256"
  done
done
python3 -I tools/analysis/e25_frame.py "$OUT" "$VID" "${ARMS[@]}"
echo "E29 status $status"
exit $status
