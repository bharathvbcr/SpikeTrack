#!/bin/bash
# Checks for the E36 port into the main tree (--det-transformer, --seg-head and --inter-neck defaulting
# to coreml-fp16 together): fmt, clippy -D warnings, the spiketrack + st-model + st-coreml tests, then
# 2-frame runs that read back what each configuration chose (run.json "det_transformer", "seg_head",
# "inter_neck", "detector_programs_default") and the refusals.
# usage: tools/locked.sh e36-main-checks bash bench/results/e36-combined/main-checks.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=bench/results/e36-combined/main-checks
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
cargo fmt -p st-model -p st-coreml -p spiketrack --check > "$OUT/fmt.log" 2>&1 && echo "fmt ok" || { echo "fmt FAILED"; status=1; }
cargo clippy --workspace --all-targets -j 2 -- -D warnings > "$OUT/clippy.log" 2>&1 && echo "clippy ok" || { echo "clippy FAILED"; status=1; }
rg "Checking (st-model|st-coreml|spiketrack) " "$OUT/clippy.log" | awk '{print $2}' | tr '\n' ' ' | xargs echo "clippy checked:"
cargo test --release -j 2 -p spiketrack -p st-model -p st-coreml --lib --bins > "$OUT/tests.log" 2>&1 && echo "tests ok" || { echo "tests FAILED"; status=1; }
rg "^test result" "$OUT/tests.log" | awk '{p+=$4; f+=$6} END {print "test binaries " NR ", passed " p ", failed " f}'
rg "the_detector_programs_default_together|the_transformer_takes_adapters|coreml_placements_refuse" "$OUT/tests.log"
cargo build --release -j 2 -p spiketrack > "$OUT/build.log" 2>&1 && echo "build ok" || { echo "build FAILED"; exit 1; }
BIN=target/release/spiketrack
IMG=data/visem/VISEM_Tracking_Train_v4/Train/36/images
LORA=fixtures/lora_runs/visem-lora-v3/adapter.safetensors
EDIT=bench/results/weight-space/lorav3_exact_it_3p00.json
FFN=$OUT/ffn-edit.json
printf '%s\n' '{"format": "spiketrack-weight-edit-v1", "edits": [{"op": "scale_rows", "param": "detector.transformer.decoder.layers.2.linear1.weight", "rows": [0, 8], "scale": 0.5}]}' > "$FFN"
run() { # name expect(ok|refuse) args...
  local name=$1 expect=$2; shift 2
  local d="$OUT/$name"
  "$BIN" track "$IMG" --prompt sperm --backend metal --max-frames 2 --boxes detector --max-objects 64 "$@" \
    --out "$d" > "$OUT/$name.log" 2>&1
  local s=$?
  find "$d/masks" -type f -delete 2>/dev/null; rmdir "$d/masks" 2>/dev/null
  if [ "$expect" = ok ]; then
    [ $s = 0 ] || { echo "$name FAILED (exit $s)"; status=1; return; }
    python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], '| transformer', r.get('det_transformer'), bool(r.get('external_det_transformer')), '| seg-head', r.get('seg_head'), bool(r.get('external_seg_head')), '| inter-neck', r.get('inter_neck'), bool(r.get('external_inter_neck')), '| det-encoder', r.get('det_encoder'), '| why:', r.get('detector_programs_default'))" "$d/36.run.json" "$name"
  else
    rmdir "$d" 2>/dev/null
    [ $s != 0 ] && echo "$name refused: $(rg -m1 'det-transformer|det-encoder' "$OUT/$name.log")" || { echo "$name was NOT refused"; status=1; }
  fi
}
run default ok --lora "$LORA" --weight-edit "$EDIT"
run det-encoder ok --lora "$LORA" --weight-edit "$EDIT" --det-encoder coreml-fp16
run no-pipeline ok --lora "$LORA" --weight-edit "$EDIT" --trunk-pipeline off
run ffn-edit ok --lora "$LORA" --weight-edit "$FFN"
run ffn-edit-explicit refuse --lora "$LORA" --weight-edit "$FFN" --det-transformer coreml-fp16
run fp32-missing-package refuse --lora "$LORA" --weight-edit "$EDIT" --det-transformer coreml-fp32
run both-encoders refuse --lora "$LORA" --weight-edit "$EDIT" --det-encoder coreml-fp16 --det-transformer coreml-fp16
echo "checks status $status"
exit $status
