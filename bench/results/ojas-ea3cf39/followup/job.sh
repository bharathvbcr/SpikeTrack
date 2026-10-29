#!/bin/bash
# Inside the mac_heavy lock: follow-ups to the ea3cf39 bump (SpikeTrack 4a13c09).
# 1. Is the cap test order-independent? Run the VISEM 22 smoke test (which opens the backend through
#    open_metal) and then the cap test in one process. If set_mem_cap on one backend moved the cap of
#    a later MetalBackend::new, the cap test's "ojas opens at 1 GiB" assertion fails here.
# 2. CPU parity at ea3cf39: the CPU text and four tracker-module gates. The range changed the CPU
#    softmax row kernel (82d8c2c, 8cb6c5d).
set -uo pipefail
cd /Users/bharath/Code/research/SpikeTrack || exit 1
OUT=bench/results/ojas-ea3cf39/followup
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_TEXT_FIXTURES="$PWD/fixtures/text/prompts.safetensors"
export SPIKETRACK_TRACKER_FIXTURES="$PWD/fixtures/tracker/judo.safetensors"
{ echo "SpikeTrack $(git log --oneline -1), dirty $(git status --short | rg -v '^\?\?' | wc -l | tr -d ' ')"; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1)"; uptime; } | tee "$OUT/versions.txt"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  RIPGREP_CONFIG_PATH= rg "^test .* \.\.\. |^test result" "$OUT/$name.log"
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; status=1; }
}
step cap-after-smoke cargo test --release -j 2 -p spiketrack --bin spiketrack -- --ignored --nocapture --test-threads=1 visem_22_frames_0_to_7_on_metal a_track_runs_metal_backend_has_the_raised_memory_cap
t() { cargo test --release -j 2 -p st-model --test "$1" -- --ignored --nocapture --test-threads=1 --exact "$2"; }
step cpu-text t text_parity text_encoder_matches_upstream
for m in multiplex_mask_decoder_matches_upstream memory_encoder_matches_upstream memory_attention_matches_upstream interactive_decoder_matches_upstream_on_the_conditioning_frame; do
  step "cpu-$m" t tracker_parity "$m"
done
uptime
echo "BF status $status"
exit $status
