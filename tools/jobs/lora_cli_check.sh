#!/bin/bash
# `spiketrack image --lora` end to end: one VISEM frame on Metal without and with a trained detector
# adapter. Checks that detections.json reports every target and tensor the adapter's metadata lists
# as merged (each attention module adapts q, k, v and out_proj: 4 targets in 2 tensors), and prints the
# detection counts and presence probability of both runs (a plumbing check, not an accuracy result:
# one frame, and the frame may be a training frame). Results: bench/results/lora/cli-check/.
# usage: tools/locked.sh lora-cli bash tools/jobs/lora_cli_check.sh ADAPTER [FRAME]
set -uo pipefail
cd "$(dirname "$0")/../.."
ADAPTER=${1:?usage: $0 ADAPTER [FRAME]}
FRAME=${2:-data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg}
export SPIKETRACK_WEIGHTS="${SPIKETRACK_WEIGHTS:-$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors}"
export SPIKETRACK_MODEL_DIR="${SPIKETRACK_MODEL_DIR:-$HOME/Code/research/models/sam3.1}"
OUT=bench/results/lora/cli-check
BIN=target/release/spiketrack
for p in "$ADAPTER" "$FRAME" "$SPIKETRACK_WEIGHTS" "$SPIKETRACK_MODEL_DIR"; do
  [ -e "$p" ] || { echo "missing $p" >&2; exit 2; }
done
cargo build --release -p spiketrack 2>&1 | tail -3
[ -x "$BIN" ] || { echo "$BIN was not built" >&2; exit 1; }
mkdir -p "$OUT"
"$BIN" image "$FRAME" --prompt cell --backend metal --out "$OUT/base" || exit 1
"$BIN" image "$FRAME" --prompt cell --backend metal --lora "$ADAPTER" --out "$OUT/lora" || exit 1
python3 - "$OUT" "$ADAPTER" <<'EOF' | tee "$OUT/summary.txt"
import json, struct, sys
out, adapter = sys.argv[1], sys.argv[2]
# The adapter's own module list (safetensors header: u64 length, then JSON).
with open(adapter, "rb") as f:
    header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
modules = json.loads(header["__metadata__"]["spiketrack_lora"])["modules"]
base = json.load(open(f"{out}/base/detections.json"))
lora = json.load(open(f"{out}/lora/detections.json"))
assert base["lora"] is None, base["lora"]
info = lora["lora"]
assert info is not None and info["targets"] == len(modules), (info, len(modules))
assert info["tensors"] == len({m["param"] for m in modules}), info
print(f"adapter: {adapter}; merged {info['targets']} targets into {info['tensors']} tensors; base_sha256 {info['base_sha256']}")
for name, r in (("base", base), ("lora", lora)):
    print(f"{name}: {len(r['detections'])} detections above {r['threshold']}, presence {r['presence_probability']:.4f}")
EOF
[ "${PIPESTATUS[0]}" -eq 0 ]
