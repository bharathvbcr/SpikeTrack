#!/bin/bash
# Checks for the E31 port into the main tree (the opt-in --det-encoder flag): fmt on the changed
# files, clippy -D warnings, the spiketrack + st-model + st-coreml tests, then 2-frame runs that read
# back what each configuration chose (run.json "det_encoder") and the refusals.
# usage: tools/locked.sh e31-main-checks bash bench/results/e31-detenc/main-checks.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=bench/results/e31-detenc/main-checks
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
# st-model has unformatted code this change does not touch (tracker/mod.rs), so fmt checks the files
# E31 changed rather than the whole package.
rustfmt --edition 2021 --check crates/st-model/src/detector.rs crates/st-model/src/lib.rs crates/st-coreml/src/lib.rs \
  crates/spiketrack/src/track_cmd.rs > "$OUT/fmt.log" 2>&1
if rg -q "Diff in" "$OUT/fmt.log"; then
  # rustfmt follows `mod x;` into children: only diffs in the changed files count.
  if rg "Diff in" "$OUT/fmt.log" | rg -q "detector.rs|st-model/src/lib.rs|st-coreml/src/lib.rs|track_cmd.rs"; then
    echo "fmt FAILED"; status=1
  else
    echo "fmt ok (diffs only in files E31 did not change: $(rg 'Diff in' "$OUT/fmt.log" | awk '{print $3}' | sort -u | tr '\n' ' '))"
  fi
else
  echo "fmt ok"
fi
cargo clippy --workspace --all-targets -j 2 -- -D warnings > "$OUT/clippy.log" 2>&1 && echo "clippy ok" || { echo "clippy FAILED"; status=1; }
cargo test --release -j 2 -p spiketrack -p st-model -p st-coreml --lib --bins > "$OUT/tests.log" 2>&1 && echo "tests ok" || { echo "tests FAILED"; status=1; }
rg "^test result" "$OUT/tests.log" | awk '{p+=$4; f+=$6} END {print "test binaries " NR ", passed " p ", failed " f}'
rg -c "det_encoder" "$OUT/tests.log" | xargs echo "det_encoder test lines:"
cargo build --release -j 2 -p spiketrack > "$OUT/build.log" 2>&1 && echo "build ok" || { echo "build FAILED"; exit 1; }
BIN=target/release/spiketrack
IMG=data/visem/VISEM_Tracking_Train_v4/Train/36/images
LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors
EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json
run() { # name expect(ok|refuse) args...
  local name=$1 expect=$2; shift 2
  local d="$OUT/$name"
  [ -e "$d" ] && { echo "$d exists" >&2; status=1; return; }
  "$BIN" track "$IMG" --prompt sperm --backend metal --max-frames 2 --boxes detector --max-objects 64 "$@" \
    --out "$d" > "$OUT/$name.log" 2>&1
  local s=$?
  find "$d/masks" -type f -delete 2>/dev/null; rmdir "$d/masks" 2>/dev/null
  if [ "$expect" = ok ]; then
    [ $s = 0 ] || { echo "$name FAILED (exit $s)"; status=1; return; }
    python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], 'det_encoder', r.get('det_encoder'), '|', r.get('external_det_encoder'))" "$d/36.run.json" "$name"
  else
    [ $s != 0 ] && echo "$name refused: $(rg -m1 'det-encoder' "$OUT/$name.log")" || { echo "$name was NOT refused"; status=1; }
  fi
}
run default ok --lora "$LORA" --weight-edit "$EDIT"
run e16 ok --lora "$LORA" --weight-edit "$EDIT" --det-encoder coreml-fp16
run e16-no-lora-package refuse --det-encoder coreml-fp32 --lora "$LORA"
run e16-small-input refuse --det-encoder coreml-fp16 --lora "$LORA" --input-size 672
echo "checks status $status"
exit $status
