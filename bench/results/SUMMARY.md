# Results summary

Every number SpikeTrack reports, with the file it comes from and the command that produced it.
Machine: Apple M5 Pro, macOS 27.0.1, shared with other sessions (load averages are in the logs).
Ojas `cbd5565` and tessl as pinned in CONTRIBUTIONS.md unless a row says otherwise. Reference:
the official SAM 3.1 code run in fp32 on CPU (`tools/reference`).

Heavy jobs run under the machine lock: `tools/locked.sh <label> bash tools/jobs/<script>`.

## Parity with the official SAM 3.1

| What | Result | File | Command |
|---|---|---|---|
| Every module (text encoder, detector, PE ViT trunk + necks, mask decoder, memory encoder, memory attention, interactive decoder), CPU and Metal | worst cosine 1.0000000 on both backends | `module-gates-cbd5565/` | `tools/jobs/module_gates.sh both` |
| Image mode end to end, 20 VISEM frames, Metal f32 | kept sets identical on 20/20, mean mask IoU 1.0000, ~2.3 s/frame | `d3-gate/metal-f32.log` | `tools/jobs/module_gates.sh metal` (the `metal-d3` step; `tools/jobs/d3_gate.sh` is the CPU run) |
| Semi-supervised VOS, 3 DAVIS clips (judo, shooting, gold-fish), Metal | J&F 0.9107 / 0.9456 / 0.9514 vs upstream 0.9107 / 0.9456 / 0.9513; mask IoU vs upstream 1.0000 | `vos-gate/metal-f32-judo-shooting-gold-fish.log` | `tools/jobs/vos_gate.sh metal` |
| Text-prompted detect -> associate -> track session, VISEM 22, 32 frames, upstream's frame tensors | identical id sets on 32/32 frames (13-15 objects, births included); mask IoU 0.99854 (CPU), 0.99855 (Metal) | `parity-tracker/{cpu,metal}-f32-detect-track-visem22.log` | `SPIKETRACK_DT_VIDEO=22 tools/jobs/detect_track_parity.sh` (CPU); the Metal entry point `session_matches_upstream_on_metal` |
| JPEG decode vs Pillow 13.0.0 (libjpeg-turbo 3.2.0) | 6 VISEM frames byte-identical (5,529,600 of 5,529,600 bytes); 37 edge-case files byte-identical | crates/st-video/tests (`decode.rs`, `jpeg_pillow.rs`) | `cargo test -p st-video -- --include-ignored` |
| LoRA adapter merge, Rust vs PyTorch, on the real checkpoint | max abs diff 1.5e-8; 3/3 tests pass | `lora/merge-parity.log` | `cargo test -p st-weights --release --test lora_merge -- --ignored` |

## Accuracy on VISEM-Tracking (5 held-out videos: 21, 22, 36, 47, 82)

| What | Result | File | Command |
|---|---|---|---|
| Detector LoRA (rank 16, 983,040 params, 600 steps on this Mac), sperm detection at IoU 0.5 on 120 held-out frames | prompt "sperm": recall 0.000 -> 0.761 (precision 0.584) at score 0.3, 0.684 / 0.733 at 0.5; "cell" at 0.5: recall 0.212 -> 0.576, precision 0.122 -> 0.451 | `lora/visem-lora-r16/eval.md` | `tools/jobs/lora_train_chunked.sh` (evaluation runs in the last chunk) |
| Presence calibration and absent prompts | absent VISEM classes now give a detection on 16% of items (zero-shot 0); seen and unseen negatives stay at 0; presence ECE 0.177 -> 0.045, detection ECE 0.072 -> 0.110 | `lora/visem-lora-r16/eval.md` | as above |
| Tracking, first 8 frames of each held-out video (40 frames, 879 GT boxes) | HOTA 0.1838 zero-shot (prompt "cell", upstream mask boxes) -> 0.4694 (adapter, prompt "sperm", `--boxes detector`); 2x2 and per-video tables in `track/box-convention` | `track/smoke-8f-*` | `SPIKETRACK_MAX_OBJECTS=16 SPIKETRACK_PROMPT=.. SPIKETRACK_BOXES=.. SPIKETRACK_LORA=.. tools/jobs/track_visem.sh 8` (measured at upstream's cap of 16) |
| Tracking, first 300 frames of each held-out video (1,500 frames, 32,949 GT boxes), zero-shot (prompt "cell", upstream mask boxes) | HOTA 0.1549, DetA 0.0509, AssA 0.5942 (per video 0.059-0.223) | `track/held300-zeroshot-cell-mask/eval.txt` | `SPIKETRACK_MAX_OBJECTS=16 SPIKETRACK_PROMPT=cell SPIKETRACK_BOXES=mask SPIKETRACK_TRACK_OUT=.. tools/jobs/track_visem.sh 300 <id>` per video, then `.. 300 --eval` (measured at upstream's cap of 16) |
| Same, with the adapter, prompt "sperm", `--boxes detector` | HOTA 0.4695, DetA 0.3198, AssA 0.7250 (per video 0.326-0.629): 3.0x zero-shot | `track/held300-lora600-sperm-detector/eval.txt` | as above with `SPIKETRACK_PROMPT=sperm SPIKETRACK_BOXES=detector SPIKETRACK_LORA=fixtures/lora_runs/visem-lora-r16/adapter.safetensors` |
| Same, with `--max-objects 64`, all five videos rerun (2026-10-29; the earlier 0.4868 had copied video 21 from the cap-16 run, though it is capped too) | HOTA 0.4922, DetA 0.3734, AssA 0.6903 (21: 0.620 -> 0.642, 36: 0.326 -> 0.352, 82: 0.441 -> 0.486) | `track/held300-lora600-sperm-detector-f32-max64/eval.txt` | as above with `SPIKETRACK_MAX_OBJECTS=64 SPIKETRACK_TRUNK=f32` (64 is the default of `track` and `track_visem.sh` since 2026-10-28) |
| Same, with the ViT trunk as CoreML fp32 (`--trunk-precision coreml-fp32`, the Metal default since 2026-10-29) | HOTA 0.4922 at cap 64, every video equal to the Ojas trunk's; id sets equal on all 1,500 frames; frame 0.80x on 49 interleaved frames of VISEM 22 | `coreml-trunk/README.md` | as above with `SPIKETRACK_MAX_OBJECTS=64` |
| Same, with the detector's host DETR decoder blocked and threaded (E12, 5e13d74) | predictions byte-identical to 3ca1476 on VISEM 22 and 82 (16 frames, 14 of 14 runs); host decoder 337 -> 62 ms per frame on 82; frame 0.86x on 22 (min of 2, interleaved, load ~5); 82 is smaller but load-confounded; host threads make the gain load-dependent | `e12-host-decoder/README.md` | `e12-host-decoder/job.sh` |
| Same, with memory-attention inputs kept on the device (E15, 66c3a58) | predictions byte-identical to E13 on VISEM 22 and 82 (8 of 8); tracker tokens phase 64 / 50 -> 14 / 6 ms per state on 82; frame 0.97x on 22 (min of 2; both pairs bracket the expected 2.4%); 82 is faster in both pairs with per-run load unrecorded; footprint ~760 MiB lower | `e15-memory-tokens/README.md` | `e15-memory-tokens/job.sh` |
| Closing job: E15 (66c3a58) against E11b (3ca1476), and E15 over the held-out set | frame 0.843x on 82 and 0.825x on 22 (min of 3, interleaved, load 10-23, 6 of 6 pairs faster); predictions byte-identical to the committed CoreML fp32 run on all 1,500 held-out frames; HOTA 0.4922; Metal allocated flat, but the process footprint creeps 0.6-1.6 MiB per frame where E10's was flat (GAP-E15-HOST-FOOTPRINT-CREEP); every test target compiles (not run) | `closing-e15/README.md` | `closing-e15/job.sh` |

The 8-frame numbers are a smoke comparison: one run per configuration, no noise estimate. VISEM's
ground truth is sperm-head boxes; upstream's track boxes are whole-sperm mask extents, so every
VISEM number states its box source (`track/box-convention/README.md`, GAP-VISEM-BOX-CONVENTION).
Supervised state of the art on VISEM is HOTA 74.3 (YOLOv8 + DeepOCSORT); ours are zero-shot or a
detector-only LoRA, not head to head.

## Speed and footprint (Metal f32)

| What | Result | File | Command |
|---|---|---|---|
| CoreML port (tracker only, fp16, CPU+GPU), the speed reference | backbone 0.272 s/frame, multiplex-16 track step 0.0775 s/frame (median of 5); backbone output vs upstream cosine 0.99995 | `baselines/coreml.json`, `baselines/coreml_parity.txt` | `tools/jobs/coreml_bench.sh` |
| One frame, unprofiled, contended | detector run (trunk + neck + detector) 2511 ms; tracker frames incl. their trunk 2364 / 2572 / 2963 ms at F = 1 / 3 / 7 memory frames | `profile-cbd5565/frame-metal.log` | `tools/jobs/frame_profile.sh metal` (writes `profile/`; copied per Ojas pin) |
| Per-op profile, one frame | trunk 1430 ms (f32 linears 950 ms of it), detector neck + detector 386 ms, memory cross-attention 343 ms at F = 7 | `profile-cbd5565/frame-metal.log` | as above |
| bf16 (whole-model Autocast) | trunk 1430 -> 1202 ms profiled, but D2 misses (cosine 0.998) and D3 kept sets change: not used | `profile-cbd5565/frame-metal_bf16.log`, `module-gates-cbd5565/` | `tools/jobs/frame_profile.sh metal_bf16`; `module_gates.sh metal` (its bf16 steps) |
| Binary | 8.4 MB, 14 linked libraries, none outside the OS; `image` runs with no Python on PATH (3.0 s wall, max RSS 7.0 GiB, peak footprint 9.3 GiB) | `packaging/summary.txt`, `packaging/run.txt` | `scripts/no_python_check.sh target/release/spiketrack -- image ...` |

We are ~8x slower per frame than the CoreML port, which runs only the tracker; the D6 target is
restated to parity-or-better (HANDOFF.md). What CoreML does not do: text prompts, the detector,
births of new objects, parity evidence.

## Event gating (st-gate)

| What | Result | File | Command |
|---|---|---|---|
| Per-window change rates on 20 VISEM videos x 300 frames (no model) | at tau 8, stride 1: 54.6% of the 3x3 attention windows change per frame on average, 0.9% of patches | `gating-change-rates/summary.md` | `tools/reference/.venv/bin/python -s tools/analysis/window_change_rates.py` |
| Gated trunk, exact policy | bitwise equal to the ungated trunk when every window is recomputed; on a mosaic frame with 3 of 9 windows changed it recomputes 85.4% of the blocks (86.6% of the FLOPs) | `gating/gating-parity-cpu.log` | `tools/jobs/gating_parity.sh` |
| Gated trunk, approximate policy | same frame: 41.7% of the blocks (46.3% of the FLOPs); neck cosines >= 0.9998 | `gating/gating-parity-cpu.log` | as above |

On VISEM, moving sperm change most windows, so exact gating saves little; the approximate policy
is the one that skips work, at a measured accuracy cost.
