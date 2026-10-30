#!/bin/bash
# Inside the mac_heavy lock: E16 (docs/SPEED_PLAN.md). The footprint bisect over five saved
# binaries on held-out video 47 x 300 frames, then the tracker and detect-track parity suites run
# (CPU and Metal), not just compiled.
set -uo pipefail
M=/Users/bharath/Code/research/SpikeTrack
SP=/private/tmp/claude-501/-Users-bharath-Code-research-tessl/9d40c168-6abf-4804-ae5a-f4d6d42e9ece/scratchpad
cd "$M" || exit 1
OUT=$M/bench/results/e16-bisect-parity
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
REF=$M/bench/results/track/held300-lora600-sperm-detector-coreml-fp32-max64/run/47.json
status=0
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; for b in e15 e10 e12b e13 e11b; do echo "$b $(shasum -a 256 "$SP/spiketrack-$b" | cut -c1-16)"; done; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; uptime; } > "$OUT/versions.txt"
for b in e15 e10 e12b e13 e11b; do
  d=$OUT/v47-$b
  rm -rf "$d"; mkdir -p "$d"
  echo "== v47-$b load $(sysctl -n vm.loadavg)"
  "$SP/spiketrack-$b" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/47/images" --prompt sperm --backend metal --max-frames 300 \
    --boxes detector --max-objects 64 --out "$d" --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    > "$d.log" 2>&1 || { echo "== FAILED: v47-$b"; tail -5 "$d.log"; status=1; continue; }
  rm -rf "$d/masks"
  cmp -s "$REF" "$d/47.json" && echo "v47-$b: predictions byte-identical to the committed held-out 47" || { echo "== FAILED: v47-$b predictions differ"; status=1; }
  python3 -c "
import re,sys
def slope(f):
    xs=range(16,len(f)); mx=sum(xs)/len(xs); my=sum(f[16:])/len(f[16:])
    return sum((x-mx)*(f[x]-my) for x in xs)/sum((x-mx)**2 for x in xs)
L=[l for l in open(sys.argv[1]) if l.startswith('frame ')]
fp=[int(re.search(r'footprint (\d+) MiB',l).group(1)) for l in L]
ma=[int(re.search(r'metal allocated (\d+) MiB',l).group(1)) for l in L]
t=[float(l.split(': ')[1].split(' s')[0]) for l in L]
print(f'{sys.argv[2]}: {len(L)} frames, footprint slope {slope(fp):+.2f} MiB/frame (f16 {fp[16]}, f299 {fp[-1]}), metal allocated slope {slope(ma):+.3f} (f16 {ma[16]}, f299 {ma[-1]}), steady mean {1e3*sum(t[1:])/len(t[1:]):.0f} ms')
" "$d.log" "v47-$b"
done
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test result|FAILED|panicked|^yielded|^test " "$OUT/$name.log" | tail -20
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; tail -20 "$OUT/$name.log"; status=1; }
}
step tracker-parity env SPIKETRACK_TRACKER_FIXTURES="$M/fixtures/tracker/judo.safetensors" \
  cargo test --release -j 2 -p st-model --test tracker_parity -- --ignored --nocapture --test-threads=1
step detect-track-metal env SPIKETRACK_DT_VIDEO=22 SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track" \
  cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_metal
step detect-track-cpu env SPIKETRACK_DT_VIDEO=22 SPIKETRACK_DT_FIXTURES="$M/fixtures/detect_track" \
  cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_visem_11
for p in "metal detect-track-metal" "cpu detect-track-cpu"; do
  set -- $p
  if diff <(sed -n '/^| y0 /,/^yielded/p' "$M/bench/results/parity-tracker/$1-f32-detect-track-visem22.log") <(sed -n '/^| y0 /,/^yielded/p' "$OUT/$2.log") > "$OUT/$2.diff"; then
    echo "$2: yield table and summary identical to the committed visem22 log"
  else
    echo "$2: yield table differs from the committed visem22 log ($(wc -l < "$OUT/$2.diff" | tr -d ' ') diff lines, $2.diff)"
  fi
done
uptime
echo "P status $status"
exit $status
