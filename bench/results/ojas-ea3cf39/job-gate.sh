#!/bin/bash
# Inside the mac_heavy lock: the gate for SpikeTrack's ojas pin bump 8b88328 -> ea3cf39 plus the
# 4 GiB Metal memory cap in `spiketrack track`. Hold 1 of 2 (hold 2 is the timing A/B).
# Pre-stated gate: workspace unit tests green; the cap test and the VISEM 22 Metal smoke test
# green (run.json records the cap); session parity on VISEM 22 at the default cap and at 4 GiB:
# every yielded id set equal and mean per-id mask IoU >= 0.95 (8b88328 gave 0.99855); Metal
# module gates green (the detector gate cannot run: fixtures/detector was deleted for disk).
set -uo pipefail
cd /Users/bharath/Code/research/SpikeTrack || exit 1
OUT=bench/results/ojas-ea3cf39
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
export SPIKETRACK_DT_FIXTURES="$PWD/fixtures/detect_track"
export SPIKETRACK_DT_VIDEO=22
{ echo "SpikeTrack $(git log --oneline -1) + uncommitted:"; git status --short | rg -v '^\?\?'; echo "ojas-st $(git -C ../ojas-st log --oneline -1)"; echo "tessl $(git -C ../tessl log --oneline -1) (dirty files: $(git -C ../tessl status --short | wc -l | tr -d ' '))"; date -u +%FT%TZ; uptime; } | tee "$OUT/versions.txt"
status=0
step() {
  local name=$1; shift
  echo "== $name load $(sysctl -n vm.loadavg)"
  "$@" > "$OUT/$name.log" 2>&1
  local s=$?
  tail -4 "$OUT/$name.log"
  [ "$s" -eq 0 ] || { echo "== FAILED ($s): $name"; status=1; }
}
step fmt cargo fmt --all --check
step build cargo build --release -j 2 --workspace --tests --bins
step clippy cargo clippy --release -j 2 --workspace --all-targets -- -D warnings
step unit-tests cargo test --release -j 2 --workspace
step cap-test cargo test --release -j 2 -p spiketrack --bin spiketrack -- --ignored --nocapture --test-threads=1 --exact track_cmd::tests::a_track_runs_metal_backend_has_the_raised_memory_cap
step smoke-visem22 cargo test --release -j 2 -p spiketrack --bin spiketrack -- --ignored --nocapture --test-threads=1 --exact track_cmd::tests::visem_22_frames_0_to_7_on_metal
step session-parity-default cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_metal
step session-parity-cap4g cargo test --release -j 2 -p st-model --test detect_track_parity -- --ignored --nocapture --test-threads=1 --exact session_matches_upstream_on_metal_at_the_track_mem_cap
export SPIKETRACK_TEXT_FIXTURES="$PWD/fixtures/text/prompts.safetensors"
export SPIKETRACK_FIXTURES="$PWD/fixtures/backbone"
export SPIKETRACK_TRACKER_FIXTURES="$PWD/fixtures/tracker/judo.safetensors"
t() { cargo test --release -j 2 -p st-model --test "$1" -- --ignored --nocapture --test-threads=1 --exact "$2"; }
step metal-text t text_parity text_encoder_matches_upstream_on_metal
step metal-backbone t backbone_parity backbone_matches_pytorch_fp32_on_five_images_on_metal
for m in multiplex_mask_decoder_matches_upstream memory_encoder_matches_upstream memory_attention_matches_upstream interactive_decoder_matches_upstream_on_the_conditioning_frame; do
  step "metal-$m" t tracker_parity "${m}_on_metal"
done
rg -h "yielded .* frames; id-set misses" "$OUT"/session-parity-*.log
uptime
echo "BG status $status"
exit $status
