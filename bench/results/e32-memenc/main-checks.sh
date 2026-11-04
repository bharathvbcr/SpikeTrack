#!/bin/bash
# Checks for the E32 port into the main tree (--mem-encoder, defaulting to coreml-fp16 beside the default
# fp16 trunk): fmt, clippy -D warnings, the spiketrack + st-model + st-coreml tests, then 2-frame runs that
# read back what each configuration chose (run.json "mem_encoder", "external_mem_encoder",
# "mem_encoder_calls") and the refusal.
# usage: tools/locked.sh e32-main-checks bash bench/results/e32-memenc/main-checks.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=bench/results/e32-memenc/main-checks
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
cargo fmt -p st-model -p st-coreml -p spiketrack --check > "$OUT/fmt.log" 2>&1 && echo "fmt ok" || { echo "fmt FAILED"; status=1; }
cargo clippy --workspace --all-targets -j 2 -- -D warnings > "$OUT/clippy.log" 2>&1 && echo "clippy ok" || { echo "clippy FAILED"; status=1; }
rg "Checking (st-model|st-coreml|spiketrack) " "$OUT/clippy.log" | awk '{print $2}' | tr '\n' ' ' | xargs echo "clippy checked:"
cargo test --release -j 2 -p spiketrack -p st-model -p st-coreml --lib --bins > "$OUT/tests.log" 2>&1 && echo "tests ok" || { echo "tests FAILED"; status=1; }
rg "^test result" "$OUT/tests.log" | awk '{p+=$4; f+=$6} END {print "test binaries " NR ", passed " p ", failed " f}'
rg "the_memory_encoder_defaults_as_memory_attention_does|coreml_placements_refuse" "$OUT/tests.log"
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
    python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], 'mem_encoder', r.get('mem_encoder'), '|', r.get('external_mem_encoder'), '| calls', r.get('mem_encoder_calls'), '| mem_attn', r.get('mem_attn'))" "$d/36.run.json" "$name"
  else
    [ $s != 0 ] && echo "$name refused: $(rg -m1 'mem-encoder' "$OUT/$name.log")" || { echo "$name was NOT refused"; status=1; }
  fi
}
run default ok --lora "$LORA" --weight-edit "$EDIT"
run ojas ok --lora "$LORA" --weight-edit "$EDIT" --mem-encoder ojas
run fp32-trunk ok --lora "$LORA" --weight-edit "$EDIT" --trunk-precision coreml-fp32
run small-input refuse --trunk-precision f32 --mem-encoder coreml-fp16 --input-size 672
echo "checks status $status"
exit $status
