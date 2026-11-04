#!/bin/bash
# Checks for the E30 default flip (--necks defaults to coreml-fp16 beside the default pipelined fp16
# trunk): fmt, clippy -D warnings, the spiketrack and st-model tests, then 2-frame runs that read back
# which necks each configuration chose (run.json "necks" and the package line in the log).
# usage: tools/locked.sh e30-checks bash bench/results/e30-quiet/default-checks.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=bench/results/e30-quiet/default-checks
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
status=0
cargo fmt -p spiketrack --check > "$OUT/fmt.log" 2>&1 && echo "fmt ok" || { echo "fmt FAILED"; status=1; }
cargo clippy --workspace --all-targets -j 2 -- -D warnings > "$OUT/clippy.log" 2>&1 && echo "clippy ok" || { echo "clippy FAILED"; status=1; }
cargo test --release -j 2 -p spiketrack -p st-model --lib --bins > "$OUT/tests.log" 2>&1 && echo "tests ok" || { echo "tests FAILED"; status=1; }
rg "^test result" "$OUT/tests.log" | awk '{p+=$4; f+=$6} END {print "test binaries " NR ", passed " p ", failed " f}'
cargo build --release -j 2 -p spiketrack > "$OUT/build.log" 2>&1 && echo "build ok" || { echo "build FAILED"; exit 1; }
BIN=target/release/spiketrack
IMG=data/visem/VISEM_Tracking_Train_v4/Train/36/images
for c in "default:" "fp32-trunk:--trunk-precision coreml-fp32" "no-pipeline:--trunk-pipeline off" "explicit-ojas:--necks ojas"; do
  name=${c%%:*}; extra=${c#*:}
  rm -rf "$OUT/$name"
  # shellcheck disable=SC2086
  "$BIN" track "$IMG" --prompt sperm --backend metal --max-frames 2 --boxes detector --max-objects 64 \
    --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
    --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json \
    $extra --out "$OUT/$name" > "$OUT/$name.log" 2>&1 || { echo "$name FAILED"; status=1; continue; }
  rm -rf "$OUT/$name/masks"
  python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], 'trunk', r.get('trunk_precision'), 'mem_attn', r.get('mem_attn'), 'necks', r.get('necks'))" "$OUT/$name/36.run.json" "$name"
done
echo "checks status $status"
exit $status
