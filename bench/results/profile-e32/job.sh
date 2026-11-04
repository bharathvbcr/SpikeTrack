#!/bin/bash
# Where the frame goes on the default after E32 (CoreML fp16 trunk, pipelined one
# frame ahead with the fp16 necks, fp16 memory attention and fp16 memory encoder; segmentation head on Ojas). Shipped
# configuration as E27: LoRA v3 + lorav3_exact_it_3p00.json, prompt sperm, --boxes detector, cap 64.
# Video 36 (held-out, dense: 27-46 sperm, up to 3 tracker states), first 32 frames: an unprofiled
# control, `--profile stages`, a second control. Then `--profile ops` on the first 8 frames.
# Pre-stated (as E11, bench/results/profile-coreml): the stage table is read literally only if its
# in-Session sum per steady frame is within 10% of the controls' mean frame time, and the profiled
# runs' predictions are byte-identical to control-1. With the trunk pipelined, the "trunk" stage is the
# wait for the worker's result, not the program's run time.
# usage: tools/locked.sh e32-profile bash bench/results/profile-e32/job.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
M=$PWD
OUT=bench/results/profile-e32
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
status=0
cargo build --release -j 2 -p spiketrack > "$OUT/build.log" 2>&1 || { echo "== FAILED: build"; tail -20 "$OUT/build.log"; exit 1; }
BIN=$OUT/spiketrack
cp target/release/spiketrack "$BIN"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ') (binary $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ojas $(git -C ../ojas log --oneline -1)"
  echo "tessl $(git -C ../tessl log --oneline -1)"
  uptime
} > "$OUT/versions.txt"
run() { # name frames [extra...]
  local name=$1 n=$2; shift 2
  local d=$OUT/$name
  [ -e "$d" ] && { echo "== $d exists"; status=1; return; }
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" --prompt sperm --backend metal \
    --max-frames "$n" --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    --out "$d" "$@" > "$d.log" 2>&1 || { echo "== FAILED: $name"; tail -5 "$d.log"; status=1; return; }
  find "$d/masks" -type f -delete; rmdir "$d/masks"
  python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); f=[float(l.split(': ')[1].split(' s')[0]) for l in open(sys.argv[2]) if l.startswith('frame ')]; print(f\"{sys.argv[3]}: trunk {r.get('trunk_precision')}, mem-attn {r.get('mem_attn')}, necks {r.get('necks')}, mem-enc {r.get('mem_encoder')}, profile {r.get('profile')}, {r.get('tracks')} tracks, frames 1.. mean {1e3*sum(f[1:])/len(f[1:]):.1f} ms (frame 0 {1e3*f[0]:.0f} ms)\")" "$d/$VID.run.json" "$d.log" "$name"
}
run control-1 32
run stages 32 --profile stages
run control-2 32
run ops 8 --profile ops
for r in stages control-2; do
  cmp -s "$OUT/control-1/$VID.json" "$OUT/$r/$VID.json" && echo "$r: predictions byte-identical to control-1" || { echo "== FAILED: $r predictions differ from control-1"; status=1; }
done
uptime
echo "profile-e32 status $status"
exit $status
