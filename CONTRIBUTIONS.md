# Contributions: what existed before the hackathon, and what was built during it

Hackathon window: Mon 2026-10-26 to Wed 2026-11-04, 8:00 PM CT.
How to read the Official Rules on prior work is still open (gaps.jsonl `GAP-OFFICIAL-RULES-UNREAD`).
This file records dates and commits so they can be classified once the rules are read.

## Pre-existing libraries (used as dependencies, not hackathon work)

| Library | What SpikeTrack uses | Pinned commit |
|---|---|---|
| [Ojas](https://github.com/bharathvbcr/ojas) | `ojas-io` safetensors reader; Backend trait and CPU ops | base `83d7a29` (ojas main on the morning of 2026-10-26, before any SpikeTrack op work). **The workspace builds against `ea3cf39`** (ojas main, 2026-10-29; moved from `8b88328` with Bharath's approval of 2026-10-28 after the gate in bench/results/ojas-ea3cf39: session parity unchanged, predictions byte-identical, frame time 0.930x with the 4 GiB memory cap; before that `8b88328`, moved from `cbd5565` after bench/results/ojas-8b88328: frame time 0.874x; before that `d5d33cc`, `4710bb2` and `d2922fb`): the base plus the Oct 26 SAM ops and the Oct 26-27 Metal ops (see below), through path dependencies on the `~/Code/research/ojas-st` worktree. |
| tessl (Metal 4 runtime, part of the Ojas stack) | Metal kernels under Ojas's Metal backend (ojas-metal reaches it as the sibling path `~/Code/research/tessl`, the live tree; every Metal run records the tessl commit it built, e.g. bench/results/metal-smoke/versions.txt) | `789a132` at the switch |
| [sparsl](https://github.com/bharathvbcr/sparsl) | bitpacked spikes, gather/scatter (planned for st-gate) | `44edd92` |

## Third-party reference code and assets (not ours)

| Item | Use | Version |
|---|---|---|
| facebookresearch/sam3 (SAM License) | Offline reference only (`tools/reference`), never linked or run by the binary | `f1647e8` plus local CPU patch `tools/reference/sam3-cpu.patch` |
| SAM 3.1 weights `facebook/sam3.1` (SAM License, gated) | Model weights; never committed | `sam3.1_multiplex.pt`, SHA-256 `0567debe…41cb6` |
| VISEM-Tracking (CC BY 4.0, Zenodo 7293726) | Evaluation data; never committed | v4 train split, 20 videos |

## Built during the hackathon (this repository)

Everything in this repository other than the planning files is hackathon work. The planning files
(HANDOFF.md, COMPONENTS.md, IMPROVEMENTS.md, tasks/, an empty gaps.jsonl) were written on
2026-10-25, before the start, and contain no code. They were committed unchanged as the first
commit on 2026-10-26.

| Date | Work | Where |
|---|---|---|
| 2026-10-26 | Repository, workspace, licences, this file | root |
| 2026-10-26 | CPU-runnable reference: shims, device patch, model builder, checkpoint→safetensors converter, fp32 backbone fixtures, independent fixture checks | `tools/reference/` |
| 2026-10-26 | Typed checkpoint loader with claim-every-tensor checks | `crates/st-weights/` |
| 2026-10-26 | HOTA (TrackEval definition), DAVIS J&F, VISEM-Tracking loader | `crates/st-eval/` |
| 2026-10-26 | SAM 3.1 on the Ojas `Backend`: PE ViT trunk and necks, CLIP tokenizer and text encoder, DETR detector and mask head | `crates/st-model/src/{backbone,text,tokenizer,detector}.rs` |
| 2026-10-26 | Multiplex tracker (memory attention, mask decoders, memory encoder, interactive decoder), VOS driver, NMS/association, per-frame bookkeeping, the text-prompted detect-track session | `crates/st-model/src/tracker/` |
| 2026-10-26 | Pillow-exact bilinear resize; ImageIO/AVFoundation decode (raw FFI) | `crates/st-video/` |
| 2026-10-26 | Event gating: per-window change maps, recompute policies, gated trunk | `crates/st-gate/`, `crates/st-model/src/backbone.rs` |
| 2026-10-26 | Per-op profiler over any backend | `crates/st-model/src/profile.rs` |
| 2026-10-26 | `spiketrack image` and `eval-visem`; the no-Python proof | `crates/spiketrack/`, `scripts/no_python_check.sh` |
| 2026-10-26 | LoRA pipeline (VISEM split, frozen-feature cache, PyTorch trainer on MPS) and the Rust adapter merge | `tools/reference/`, `crates/st-weights/src/lora.rs` |
| 2026-10-26 | Parity gates and job scripts (module gates on CPU and Metal, D3, D5, detect-track, profiles), CoreML baseline tooling | `crates/st-model/tests/`, `tools/jobs/`, `tools/baselines/` |
| 2026-10-27 | ViT LayerNorm eps fix; device-side memory assembly; Metal entry points for every gate | `crates/st-model/` |
| 2026-10-27 | `spiketrack track` (frames or mp4), `--lora`, `--boxes detector` (`tracker::associate::track_detections`), per-frame telemetry | `crates/spiketrack/`, `crates/st-model/src/tracker/` |
| 2026-10-27 | Pillow-exact JPEG decoder, a port of libjpeg-turbo 3.2.0's default decompression (IJG acknowledgement in NOTICE), and its fixtures | `crates/st-video/src/jpeg.rs`, `crates/st-video/tests/`, `tools/reference/jpeg_cases.py` |
| 2026-10-27 | H.264 `.mp4` writer on AVFoundation's `AVAssetWriter` (raw FFI, no crate), and `spiketrack track --overlay-video` for the demo video | `crates/st-video/src/encode.rs`, `crates/st-video/tests/encode.rs`, `crates/spiketrack/src/track_cmd.rs` |
| 2026-10-27 | LoRA training (600 steps), held-out evaluation, write-up; box-convention analysis | `bench/results/lora/`, `bench/results/track/box-convention/` |
| 2026-10-28 | Smaller model input (E7): one runtime geometry (`st_gate::Geometry::from_image_side`, a multiple of 336) threaded through trunk, necks, detector, tracker and CLI (`track --input-size`); the global blocks' RoPE tables generated in Rust for any grid; a 672 fixture dumper and gate | `crates/st-gate/src/change_map.rs`, `crates/st-model/`, `crates/spiketrack/src/track_cmd.rs`, `tools/reference/dump_backbone_size.py`, `tools/jobs/input_size_gates.sh` |
| 2026-10-28 | Speed plan and baselines: CoreML port on this Mac (GPU vs ANE), frame profiles, the Metal memory-cap A/B | `docs/SPEED_PLAN.md`, `bench/results/{baselines,speed-baseline,profile*}` |
| 2026-10-29 | ViT trunk as a CoreML ML Program exported from the checkpoint (E10), the Metal default; `track --profile stages\|ops` (E11) | `crates/st-coreml/`, `tools/baselines/coreml_trunk.py`, `crates/st-model/src/profile.rs` |
| 2026-10-29/30 | Host DETR decoder threading (E12), device-resident memory tokens (E15), footprint bisect (E16, E17), CoreML load record (E18b), memory-encode copy removal and exact host threading (E20, E21) | `crates/st-model/`, `crates/st-coreml/`, `bench/results/e1*`, `bench/results/e2[01]*` |
| 2026-10-31/11-01 | Architecture and test-suite docs; memory-attention head-count record and its negative control | `docs/`, `PROJECT.md`, `bench/results/head-count-control/` |
| 2026-11-02 | Zero-copy unified memory in `st-model` (host weight copies removed), tap-free inference paths, host-only weight residency, fused host MLP (E22); developer workflow scripts | `crates/st-model/`, `scripts/`, `docs/WORKFLOWS.md`, `bench/results/e22-flash-uma-ram/` |
| 2026-11-02 | Weight-space programming: `spiketrack weights` (inspection, SVD, token probes) and `--weight-edit` manifests; `2x2` tiled detection | `crates/st-weights/src/{inspect,edit}.rs`, `crates/st-model/src/tile.rs`, `bench/results/weight-space/` |
| 2026-11-03 | LoRA v2 and v3 (heads, one-to-many matching, BPE token rows, negative-query suppression) and their Rust merge; head-box NMS and head-bulb coasting under `--boxes detector`; the tracker step A/B against the CoreML port; memory-attention projection cache (E23, speed claim withdrawn) | `tools/reference/train_lora.py`, `crates/st-weights/src/lora.rs`, `crates/st-model/src/tracker/`, `bench/results/{lora,ab-coreml-tracker,e23-mattn-kv-cache}/` |
| 2026-11-03 | Error diagnosis on train-5 (localization 62.5% of FP, near-misses 63.7% of FN, concept 13.9%: no direction edit); a box-size calibration sweep that passed train-5 selection and failed the held-out gate (0.5017 vs 0.6025); LoRA v4 (128 cached frames per video, `--box-loss-weights 7.5 4.0`, +400 steps) that tied v3 on held-out (0.5995 vs 0.6025) and was not shipped; the v3 demo clips | `bench/results/weight-space/{diag-train5,box-delta-train5}/`, `bench/results/lora/visem-lora-v4/`, `tools/reference/{train_lora,lora_common,lora_selftest}.py`, `tools/jobs/track_visem.sh`, `bench/results/demo/` |
| 2026-11-04 | CoreML placements in the frame (E25-E27): memory attention, necks and segmentation head as fp16/fp32 ML Programs, the trunk one frame ahead on its own thread (byte-identical, default with a CoreML trunk), the necks optionally on that thread; the fp16 trunk + fp16 memory attention made the Metal default after held-out gates (HOTA 0.5694 vs 0.5700 on 1,500 frames, frame 0.57-0.60x) | `crates/st-coreml/`, `crates/st-model/src/detector.rs` (`ExternalSegHead`), `crates/spiketrack/src/track_cmd.rs`, `tools/baselines/coreml_{mattn,necks,seghead}.py`, `bench/results/{e25-coreml-frame,e26-pipeline,e27-pipe-necks,default-fp16}/`, `bench/results/track/{held32,held300}-coreml/` |

## Work in the pre-existing libraries during the window

Session ojas-78 is building SAM-oriented ops in Ojas from 2026-10-26: bidirectional and cross
attention, LayerNorm/GELU/bias, and conv/RoPE-2D/resize. That session logs the commit hashes and
timestamps. Whether that work counts as hackathon work or library work depends on the Official
Rules. It is listed here, with commits, before submission either way.

| Ojas range | Content | Dates | Classification |
|---|---|---|---|
| `ea3cf39..59adac6` (243 commits, most of them other Ojas lanes) | For SAM: the f32 `linear_bias_forward` adding its bias in place on the GEMM output, 11-13% faster per call with identical output bits (`aade8fa`, merge `59adac6`; ojas `bench/results/2026-11-03-linear-bias-inplace`), and a flash forward A/B at SpikeTrack's cross-attention that did not move it (`a40b82d`); tiled flash cross-attention with bias and key lengths (`d92a2e7`, `610921b`, merge `d74f0d8`) and its routing measurement (`c47986b`..`9d56bf0`); zero-copy mapped tensors and unified-memory payloads (`0e1c6bc`, `e74f1e0`, `7087da0`, `96c8412`); SAM training-side Backend methods and backwards (`b5abe74`, `f4b68da`, `5f34923`, `15ffe43`) | 2026-10-29 to 11-03 | pending the Official Rules |
| tessl `789a132..52bdd19` | `SharedBuf` / `SharedAlloc` zero-copy CPU/GPU buffers fenced on the commit timeline (`029e2f5`, `bee513c`, `37e686c`), `SharedBuf::to_gpu_buffer` (`80caf39`); a SAM trunk-linear GEMM bench lane (`daff7db`, `f535dd3`) | 2026-10-28 to 11-02 | pending the Official Rules |
| `83d7a29..d2922fb` | `attention_forward` (bidirectional/cross, bias, key_lens), `layer_norm`, `channel_layer_norm`, `gelu_erf`, `linear_bias`, `broadcast_*`, `conv2d`, `conv_transpose2d`, `max_pool2d`, bilinear interpolation, `rope_interleaved`, axial RoPE tables, window partition/merge, pixel shuffle (CPU; some Metal) | 2026-10-26 (commit times in ojas-78's report) | pending the Official Rules |
| `8b88328..ea3cf39` | `MetalBackend::set_mem_cap` / `mem_cap` as public API, and the resource plan sets aside the memory caps (1ca5286, ab36b24, fa1b328, 7b6152b); SAM ops dedup: one rank4 helper, one refusal helper, one softmax row kernel, the SAM Autocast policy in one test (b4a92af); CPU softmax row max (8cb6c5d); a test pinning the pre-cast bf16 weight route (017f5c0); a target-excluded cross-entropy merged and then reverted (net unchanged); unrelated ojas lanes (wgpu scratch, CUDA syntax check, gated RMSNorm backward, LoRA draw order, oracle NaN gates, Tape shape check), WIP preservation commits, and bench records | 2026-10-27/29 | pending the Official Rules |
| `cbd5565..8b88328` | Metal permute plans (row copies, a transpose tile; 170b22d) and a TensorOps flash cross-attention route for head dim 32 with no bias or key lengths (821c874, 2b14dca); the rest of the range is ojas bench records and rules | 2026-10-27/28 | pending the Official Rules |
| `d5d33cc..cbd5565` | Metal D32 attention routed to tessl `encoder_attn`; memory cross-attention (Tq < Tk) through a pad-q route or a split-K tiled kernel with bias (stage 1); `group_norm_forward`, `concat`, `slice`, `fill` on CPU and Metal; routing fallbacks, bilinear full-input scan, bitwise tests; SpikeTrack attention benches | 2026-10-27 | pending the Official Rules |
| `4710bb2..d5d33cc` | CPU `relu_forward`, two-pass f64 LayerNorm, SAM fixture parity and refusal fixes (gp/78-cpu-oracle); Metal `rope_interleaved`, unmasked D=64 self-attention routed to tessl `encoder_attn`, conv2d at any stride/padding, depthwise conv2d, bilinear (both align_corners), `relu` (gp/78-metal) | 2026-10-26/27 | pending the Official Rules |
| `d2922fb..4710bb2` | Metal `conv2d_forward` (im2col + GEMM) and `conv_transpose2d_forward` (kernel == stride) for the SAM backbone; a ViT attention bench; README acknowledgements | 2026-10-26 | pending the Official Rules |

Reproducibility (2026-11-03): the workspace builds against the sibling checkouts `../ojas` at
`59adac6` and `../tessl` at `52bdd19` (path dependencies). Neither commit is public yet: Ojas's
`origin/main` is `09e5ea9` (40 commits behind) and tessl's is `ce5cab1` (31 behind). The repin from
`286a444` to `59adac6` was checked before it was written here: every module gate passes as before,
the detector and D3 gates pass on fixtures whose sha256s match `fixtures/DETECTOR_MANIFEST.json`,
detect-track parity holds (video 22, id sets equal on 32 of 32 frames), and v3's train-5
predictions are byte-identical (`bench/results/repin-59adac6/`). Until both are
pushed, a fresh clone cannot build SpikeTrack. History before that: `ea3cf39` was on Ojas's local main only; Ojas's public `origin/main` is `170b22d` (pushed on Bharath's
instruction 2026-10-27; it contains `cbd5565`, the previous pin). The owner chose to keep main local for now (2026-10-28). Cargo.toml still uses path dependencies on local checkouts (`../ojas`, and tessl as
ojas-metal's sibling path), so the workspace builds on this machine only. Before submission, Cargo.toml must point at
the public repositories (Ojas by rev; tessl likewise once its commit is public).
