#!/bin/bash
# Inside the mac_heavy lock: E22 (docs/SPEED_PLAN.md), M1-M5 flash cross-attention + tile-register
# boxRPB + zero-copy host residency (load_host) + L1-tiled fused Mlp::host + single-allocation
# HostBuf readback + inference-only trunk/detector paths (no unused Taps) + row-chunked detector
# decoder host prep + pre-frame-0 Sam31Weights/Text drop and Metal pool trim.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
cd "$M" || exit 1
OUT=$M/bench/results/e22-flash-uma-ram
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_FIXTURES="$M/fixtures/backbone"
export SPIKETRACK_TEXT_FIXTURES="$M/fixtures/text/prompts.safetensors"
export SPIKETRACK_TRACKER_FIXTURES="$M/fixtures/tracker/judo.safetensors"
export SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track"
export SPIKETRACK_DT_VIDEO=22
BIN=$M/target/release/spiketrack
status=0

echo "== building release spiketrack"
cargo build --release -j 2 -p spiketrack || exit 1

{
  echo "SpikeTrack $(git branch --show-current) $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16), $(wc -c < "$BIN" | tr -d ' ') bytes)"
  echo "metal toolchain $(xcodebuild -showComponent MetalToolchain 2>&1 | rg 'Build Version')"
  echo "tessl $(git -C ../tessl log --oneline -1)"
  echo "ojas $(git -C ../ojas log --oneline -1)"
  memory_pressure | rg 'System-wide memory free percentage' || true
  uptime
} > "$OUT/versions.txt"

: > "$OUT/parity-gates.log"

gate_step() {
  local name=$1; shift
  echo "== gate: $name load $(sysctl -n vm.loadavg)"
  echo "=== GATE: $name ===" >> "$OUT/parity-gates.log"
  local tmp="$OUT/.gate-$name.tmp"
  "$@" > "$tmp" 2>&1
  local s=$?
  cat "$tmp" >> "$OUT/parity-gates.log"
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^error|^yielded|detection\(s\)" "$tmp" | tail -8
  if [ "$s" -ne 0 ]; then
    echo "== FAILED ($s): $name"
    tail -25 "$tmp"
    rm -f "$tmp"
    exit 1
  fi
  if RIPGREP_CONFIG_PATH= rg -q "test result: .* 0 passed;" "$tmp"; then
    echo "== FAILED (0 tests executed): $name"
    cat "$tmp"
    rm -f "$tmp"
    exit 1
  fi
  rm -f "$tmp"
}

gate_step adversarial cargo test --release -j 2 -p st-model \
  --test adversarial_m3_layers \
  --test adversarial_m3_roundtrip \
  --test adversarial_m5_boundary \
  -- --test-threads=1

gate_step rpb-bias-metal cargo test --release -j 2 -p st-model --lib \
  detector::tests::the_rpb_bias_matches_the_host_loop_bit_for_bit_on_metal \
  -- --ignored --exact --nocapture

gate_step backbone-parity-metal cargo test --release -j 2 -p st-model --test backbone_parity \
  backbone_matches_pytorch_fp32_on_five_images_on_metal \
  -- --exact --ignored --nocapture --test-threads=1

gate_step text-parity-metal cargo test --release -j 2 -p st-model --test text_parity \
  text_encoder_matches_upstream_on_metal \
  -- --exact --ignored --nocapture --test-threads=1

gate_step tracker-parity cargo test --release -j 2 -p st-model --test tracker_parity \
  -- --ignored --nocapture --test-threads=1

gate_step detect-track-parity-metal cargo test --release -j 2 -p st-model --test detect_track_parity \
  session_matches_upstream_on_metal \
  -- --exact --ignored --nocapture --test-threads=1

rm -rf "$OUT/image-smoke"
gate_step image-smoke "$BIN" image "$M/data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg" \
  --prompt sperm --backend metal \
  --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
  --out "$OUT/image-smoke"
[ -s "$OUT/image-smoke/detections.json" ] && [ -s "$OUT/image-smoke/overlay.png" ] || {
  echo "== FAILED: image-smoke outputs missing"
  exit 1
}
ls -la "$OUT/image-smoke" >> "$OUT/parity-gates.log"
rm -f "$OUT/image-smoke/overlay.png"
echo "== all parity gates PASS"

run() { # name video frames [extra...]
  local name=$1 v=$2 n=$3; shift 3
  local d=$OUT/$name
  rm -rf "$d"; mkdir -p "$d"
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$v/images" --prompt sperm --backend metal --max-frames "$n" \
    --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" --max-objects 64 \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -10 "$d.log"; status=1; return; }
  rm -rf "$d/masks"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; print(f\"{sys.argv[3]}: profile {r['profile']}, {r['tracks']} tracks, frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms (frame 0 {1e3*f[0]:.0f} ms)\")" "$d/$v.run.json" "$d.log" "$name"
}

for v in 82 22; do
  run "v$v-e22-1" $v 16
  run "v$v-stages" $v 16 --profile stages
  run "v$v-e22-2" $v 16
done

for v in 82 22; do
  cmp -s "$OUT/v$v-e22-1/$v.json" "$OUT/v$v-stages/$v.json" \
    && echo "v$v stages: predictions byte-identical to e22-1" \
    || { echo "== FAILED: v$v stages predictions differ from e22-1"; status=1; }
  cmp -s "$OUT/v$v-e22-1/$v.json" "$OUT/v$v-e22-2/$v.json" \
    && echo "v$v e22-2: predictions byte-identical to e22-1" \
    || { echo "== FAILED: v$v e22-2 predictions differ from e22-1"; status=1; }
  python3 -c '
import json, sys
a = json.load(open(sys.argv[1]))["frames"]
b = json.load(open(sys.argv[2]))["frames"]
assert len(a) == len(b), (len(a), len(b))
id_eq = 0
inst = 0
max_box = 0.0
max_score = 0.0
for fa, fb in zip(a, b):
    oa = {o["id"]: o for o in fa["objects"]}
    ob = {o["id"]: o for o in fb["objects"]}
    if set(oa) == set(ob):
        id_eq += 1
        for k in oa:
            inst += 1
            max_score = max(max_score, abs(oa[k]["score"] - ob[k]["score"]))
            for xa, xb in zip(oa[k]["box_xyxy_norm"], ob[k]["box_xyxy_norm"]):
                max_box = max(max_box, abs(xa - xb))
print(f"v{sys.argv[3]} vs E21: id sets equal {id_eq}/{len(a)} frames ({inst} instances), max box_xyxy_norm diff = {max_box:.2e}, max score diff = {max_score:.2e}")
if id_eq != len(a):
    sys.exit(1)
' "$M/bench/results/e21-host-threads/v$v-e21/$v.json" "$OUT/v$v-e22-1/$v.json" "$v" || status=1
done

uptime
echo "E22 status $status"
exit $status
