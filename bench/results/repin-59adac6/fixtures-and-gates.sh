#!/bin/bash
# Repin check, part 2: fixtures/detector was missing, so the detector and D3 gates could not run after
# the ojas 59adac6 merge (../module-gates/repin-59adac6.log). Regenerate it with the manifest's own
# command, require every sha256 in fixtures/DETECTOR_MANIFEST.json to match (same fixtures as when the
# gates were set), then run the cpu/metal detector gates, the metal f32/bf16 D3 gates, and detect-track
# parity on video 22 (the only fixtures/detect_track video on disk).
# usage: tools/locked.sh repin-gates bash bench/results/repin-59adac6/fixtures-and-gates.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=bench/results/repin-59adac6
SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_detector_fixtures.py > "$OUT/dump-detector.log" 2>&1 \
  || { echo "dump_detector_fixtures.py failed"; tail -5 "$OUT/dump-detector.log"; exit 1; }
python3 -c "
import hashlib, json, sys
m = json.load(open('fixtures/DETECTOR_MANIFEST.json'))
bad = 0
for e in m['files'] + [m['gate']]:
    got = hashlib.sha256(open(e['file'], 'rb').read()).hexdigest()
    ok = got == e['sha256']
    bad += not ok
    print(('match ' if ok else 'MISMATCH ') + e['file'])
sys.exit(1 if bad else 0)
" | tee "$OUT/fixture-hashes.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || echo "== fixture hashes differ from the manifest: the gates below compare against new fixtures"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_DETECTOR_FIXTURES="$PWD/fixtures/detector"
t() { cargo test --release -p st-model --test "$1" -- --ignored --nocapture --test-threads=1 --exact "$2"; }
status=0
for g in "cpu-detector detector_parity detector_matches_upstream_stage_by_stage" \
         "metal-detector detector_parity detector_matches_upstream_stage_by_stage_on_metal" \
         "metal-d3 d3_gate image_mode_masks_match_upstream_on_20_visem_frames_on_metal" \
         "metal-bf16-d3 d3_gate image_mode_masks_match_upstream_on_20_visem_frames_on_metal_bf16"; do
  set -- $g
  t "$2" "$3" > "$OUT/$1.log" 2>&1; s=$?
  echo "== $1 exit $s"; [ $s -eq 0 ] || status=1
done
SPIKETRACK_DT_VIDEO=22 bash tools/jobs/detect_track_parity.sh > "$OUT/dt-parity-22.log" 2>&1; s=$?
echo "== dt-parity-22 exit $s"; [ $s -eq 0 ] || status=1
exit $status
