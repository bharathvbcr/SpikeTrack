#!/bin/bash
# Inside the mac_heavy lock: the gate for merging main (d9a137f, E10) into speed/e7-geometry.
# Build; workspace unit tests; the 1008 backbone parity on Metal; the 672 gates on Metal; the CoreML
# fp32 trunk parity; then 49 frames of VISEM 22 (E10's timing setup) with --trunk-precision f32 and
# with the default (coreml-fp32), whose 22.json must be byte-identical to the b11bb69 binary's
# (bench/results/coreml-trunk/timing/track-{f32,coreml-fp32}-1/22.json on main).
set -uo pipefail
E=/Users/bharath/Code/research/SpikeTrack-e7
M=/Users/bharath/Code/research/SpikeTrack
cd "$E" || exit 1
OUT=$E/bench/results/merge-e7-e10
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_FIXTURES="$E/fixtures/backbone"
export SPIKETRACK_FIXTURES_672="$E/fixtures/backbone672"
{ echo "SpikeTrack-e7 $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1)"; uptime; } | tee "$OUT/versions.txt"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test .* \.\.\. |^test result|worst" "$OUT/$name.log" | tail -12
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
t() { cargo test --release -j 2 -p "$1" --test "$2" -- --ignored --nocapture --test-threads=1 --exact "$3"; }
step build cargo build --release -j 2 --workspace --tests --bins
[ "$status" -eq 0 ] || { echo "M status $status"; exit 1; }
step unit-tests cargo test --release -j 2 --workspace
step metal-backbone-1008 t st-model backbone_parity backbone_matches_pytorch_fp32_on_five_images_on_metal
step metal-backbone-672 t st-model backbone_parity backbone_matches_pytorch_fp32_at_672_on_metal
step rope-tables t st-model backbone_parity generated_rope_tables_equal_the_checkpoints
step coreml-fp32-parity t st-coreml trunk_parity coreml_fp32_trunk_and_ojas_necks_match_upstream
for arm in f32 default; do
  d=$OUT/track-$arm
  rm -rf "$d"; mkdir -p "$d"
  extra=(); [ "$arm" = f32 ] && extra=(--trunk-precision f32)
  echo "== track $arm load $(sysctl -n vm.loadavg)"
  target/release/spiketrack track "$M/data/visem/VISEM_Tracking_Train_v4/Train/22/images" --prompt sperm --backend metal --max-frames 49 \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    ${extra[@]+"${extra[@]}"} --out "$d" > "$d.log" 2>&1 || { echo "== FAILED: track $arm"; tail -5 "$d.log"; status=1; continue; }
  rm -rf "$d/masks"
  ref=f32; [ "$arm" = default ] && ref=coreml-fp32
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"track {sys.argv[2]}: trunk {r['trunk_precision']}, input_size {r['input_size']}, {r['mean_seconds_per_frame']:.3f} s/frame, {r['tracks']} tracks\")" "$d/22.run.json" "$arm"
  if cmp -s "$d/22.json" "$M/bench/results/coreml-trunk/timing/track-$ref-1/22.json"; then
    echo "track $arm: 22.json byte-identical to the b11bb69 binary's $ref run"
  else
    echo "== FAILED: track $arm: 22.json differs from the b11bb69 binary's $ref run"; status=1
  fi
done 2>&1 | tee "$OUT/track.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1
RIPGREP_CONFIG_PATH= rg -q 'FAILED' "$OUT/track.txt" && status=1
uptime
echo "M status $status"
exit $status
