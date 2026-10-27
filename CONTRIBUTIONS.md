# Contributions: what existed before the hackathon, and what was built during it

Hackathon window: Mon 2026-10-26 to Wed 2026-11-04, 8:00 PM CT.
How to read the Official Rules on prior work is still open (gaps.jsonl `GAP-OFFICIAL-RULES-UNREAD`).
This file records dates and commits so they can be classified once the rules are read.

## Pre-existing libraries (used as dependencies, not hackathon work)

| Library | What SpikeTrack uses | Pinned commit |
|---|---|---|
| [Ojas](https://github.com/bharathvbcr/ojas) | `ojas-io` safetensors reader; Backend trait and CPU ops | base `83d7a29` (ojas main on the morning of 2026-10-26, before any SpikeTrack op work). **The workspace builds against `cbd5565`** (ojas main, 2026-10-27; moved from `d5d33cc` with Bharath's approval, after `4710bb2` and `d2922fb` before it): the base plus the Oct 26 SAM ops and the Oct 26-27 Metal ops (see below), through path dependencies on the `~/Code/research/ojas-st` worktree. |
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
| 2026-10-27 | LoRA training (600 steps), held-out evaluation, write-up; box-convention analysis | `bench/results/lora/`, `bench/results/track/box-convention/` |

## Work in the pre-existing libraries during the window

Session ojas-78 is building SAM-oriented ops in Ojas from 2026-10-26: bidirectional and cross
attention, LayerNorm/GELU/bias, and conv/RoPE-2D/resize. That session logs the commit hashes and
timestamps. Whether that work counts as hackathon work or library work depends on the Official
Rules. It is listed here, with commits, before submission either way.

| Ojas range | Content | Dates | Classification |
|---|---|---|---|
| `83d7a29..d2922fb` | `attention_forward` (bidirectional/cross, bias, key_lens), `layer_norm`, `channel_layer_norm`, `gelu_erf`, `linear_bias`, `broadcast_*`, `conv2d`, `conv_transpose2d`, `max_pool2d`, bilinear interpolation, `rope_interleaved`, axial RoPE tables, window partition/merge, pixel shuffle (CPU; some Metal) | 2026-10-26 (commit times in ojas-78's report) | pending the Official Rules |
| `d5d33cc..cbd5565` | Metal D32 attention routed to tessl `encoder_attn`; memory cross-attention (Tq < Tk) through a pad-q route or a split-K tiled kernel with bias (stage 1); `group_norm_forward`, `concat`, `slice`, `fill` on CPU and Metal; routing fallbacks, bilinear full-input scan, bitwise tests; SpikeTrack attention benches | 2026-10-27 | pending the Official Rules |
| `4710bb2..d5d33cc` | CPU `relu_forward`, two-pass f64 LayerNorm, SAM fixture parity and refusal fixes (gp/78-cpu-oracle); Metal `rope_interleaved`, unmasked D=64 self-attention routed to tessl `encoder_attn`, conv2d at any stride/padding, depthwise conv2d, bilinear (both align_corners), `relu` (gp/78-metal) | 2026-10-26/27 | pending the Official Rules |
| `d2922fb..4710bb2` | Metal `conv2d_forward` (im2col + GEMM) and `conv_transpose2d_forward` (kernel == stride) for the SAM backbone; a ViT attention bench; README acknowledgements | 2026-10-26 | pending the Official Rules |

Reproducibility: `cbd5565` is on Ojas's public `origin/main` (pushed on Bharath's instruction 2026-10-27; origin/main
is `170b22d`, which contains it). Cargo.toml still uses path dependencies on local worktrees (ojas-st, and tessl as
ojas-metal's sibling path), so the workspace builds on this machine only. Before submission, Cargo.toml must point at
the public repositories (Ojas by rev; tessl likewise once its commit is public).
