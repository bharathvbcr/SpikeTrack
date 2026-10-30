#!/bin/bash
# Inside the mac_heavy lock: the E11b-E15 series' closing job (docs/SPEED_PLAN.md, "closing").
# 1. Every st-model and spiketrack test target compiles (`--no-run`; the holds so far built --lib --bins only).
# 2. Cumulative speed: the E11b binary (3ca1476) against the E15 binary (66c3a58), interleaved, 3 rounds,
#    VISEM 82 and 22, 16 frames, load recorded per run.
# 3. The E15 code over the 5 held-out videos x 300 frames (tools/jobs/track_visem.sh, the committed
#    held300-lora600-sperm-detector-coreml-fp32-max64 settings), then HOTA, and a `cmp` of every
#    video's predictions against the committed ones.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$M" || exit 1
OUT=$M/bench/results/closing-e15
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
OLD=$SP/spiketrack-e11b NEW=$SP/spiketrack-e15
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^error|Finished|Executable" "$OUT/$name.log" | tail -30
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -30 "$OUT/$name.log"; status=1; }
}
step tests-compile cargo test --release -j 2 -p st-model -p spiketrack --tests --no-run
step build cargo build --release -j 2 -p spiketrack
[ "$status" -eq 0 ] || exit 1
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; for b in "$OLD" "$NEW" target/release/spiketrack; do echo "$b $(shasum -a 256 "$b" | cut -c1-16)"; done; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
run() { # name bin video
  local name=$1 bin=$2 v=$3
  local d=$OUT/speed/$name
  rm -rf "$d"; mkdir -p "$d"
  local load; load=$(sysctl -n vm.loadavg)
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames 16 \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  python3 -c "import sys; f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[1]) if l.startswith('frame ')]; print(f\"{sys.argv[2]}: frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms, load {sys.argv[3]}\")" "$d.log" "$name" "$load"
}
for r in 1 2 3; do
  for v in 82 22; do
    run "v$v-old-$r" "$OLD" $v
    run "v$v-new-$r" "$NEW" $v
  done
done
for v in 82 22; do
  for r in 1 2 3; do
    cmp -s "$OUT/speed/v$v-old-1/$v.json" "$OUT/speed/v$v-new-$r/$v.json" && echo "v$v new-$r: predictions byte-identical to old-1" || { echo "== FAILED: v$v new-$r differs from old-1"; status=1; }
  done
done
echo "== held-out 5 x 300 load $(sysctl -n vm.loadavg)"
SPIKETRACK_TRACK_OUT=$OUT/held300 SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_MAX_OBJECTS=64 \
  SPIKETRACK_LORA=$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors \
  bash tools/jobs/track_visem.sh 300 > "$OUT/held300.log" 2>&1 || { echo "== FAILED: held-out run"; tail -20 "$OUT/held300.log"; status=1; }
rm -rf "$OUT/held300/run/masks"
REF=$M/bench/results/track/held300-lora600-sperm-detector-coreml-fp32-max64
for id in 21 22 36 47 82; do
  cmp -s "$REF/run/$id.json" "$OUT/held300/run/$id.json" && echo "held-out $id: predictions byte-identical to the committed coreml-fp32 run" || { echo "== FAILED: held-out $id differs"; status=1; }
  echo "held-out $id: $(rg -o 'footprint [0-9]+ MiB' "$OUT/held300/logs/$id.log" | sed -n '16p;100p;200p;$p' | tr '\n' ' ')(frames 15, 99, 199, last)"
done
rg "HOTA|mean" "$OUT/held300/eval.txt" | head -12
uptime
echo "P status $status"
exit $status
