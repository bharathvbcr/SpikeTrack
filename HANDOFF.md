# SpikeTrack — Meta Global AI Developer Hackathon handoff

Historical plan and task log. Current state: [README.md](README.md) and [bench/results/SUMMARY.md](bench/results/SUMMARY.md).

Written 2026-10-25 for the hackathon window **Mon Oct 26 → Wed Nov 4, 9:00 PM ET (8:00 PM CT) submission**.
Track: **Indie**. Theme: **Push the Frontier** (accelerate a scientific experiment).
Evidence labels used everywhere: **[V]** verified against a primary source or a run, **[I]** inferred, **[U]** unverified — check before relying on it.

## 1. What we are building (one paragraph)

SpikeTrack runs Meta's **SAM 3.1** — including its **Object Multiplex** video tracker — as a **single Rust binary with no Python at runtime**, on Apple Silicon directly on tessl, Bharath's Metal 4 runtime (part of the Ojas stack), with sparsl for sparse/event compute. On top of the port it adds **training-free event gating**: bitpacked 1-bit pixel-change maps (sparsl) decide which image regions and which tracked-object slots are recomputed each frame. Showcase: many-object tracking in **clinical sperm-motility video (VISEM-Tracking)**, measured against the PyTorch reference.

The application promised: SAM 3.1 multiplex natively on Metal 4 tensor ops via Ojas/tessl; training-free sparsl event gating; evaluation on VISEM-Tracking (HOTA) and SA-V (J&F) against the PyTorch reference; target **≥3× the existing CoreML port's throughput at 16 objects** with published parity numbers.

**Speed target restated (2026-10-27, Bharath):** parity-or-better with the CoreML port on the tracker path at 16 objects, leading with what that port cannot do: text-prompted detection with births and stable IDs, upstream-exact parity on CPU and Metal, a single binary with no Python, and LoRA domain adaptation.

**Trunk default changed (2026-10-29, owner's call, E10 in docs/SPEED_PLAN.md):** on `--backend
metal`, `track` runs the ViT trunk as a CoreML fp32 ML Program that we export from the checkpoint
(`tools/baselines/coreml_trunk.py`). The frame is 0.80x, with the same id sets on all 1,500 held-out
frames and the same HOTA, 0.4922 at cap 64. With the default, the trunk is Apple's CoreML, not
Ojas/tessl. The necks, text encoder, detector and tracker still run on Ojas. `--trunk-precision f32`
is the all-Ojas path, and `--backend cpu` keeps it as the parity oracle.
**Metal default changed again (2026-11-04, gates of `bench/results/e25-coreml-frame/RULES.md`):**
`track` now runs the trunk and the tracker's memory attention as fp16 CoreML programs, with the trunk
one frame ahead on its own thread.
- Frame: 0.57-0.60x the coreml-fp32 default (E27, 3 interleaved rounds).
- Held-out HOTA: 0.6025 on 32 frames (unchanged), and 0.5694 against 0.5700 on 1,500 frames, worst
  video -0.0012 (`bench/results/track/held300-coreml/README.md`).
- The CoreML segmentation head failed the in-frame speed gate (host copies, E25-E27) and stays on
  Ojas.
- **Later the same day (E30, `bench/results/e30-quiet/README.md`):** the fp16 CoreML necks, on the
  trunk's worker, passed every gate and joined the default.
  - Frame 0.916, 0.898 and 0.886x the trunk + memory-attention default (quiet machine).
  - Held-out HOTA 0.6025 on 32 frames and 0.5697 on 1,500.
  - `--necks ojas` gives the earlier fp16 default.
- **Opt-in, not a default (E31, `bench/results/e31-detenc/README.md`):** `--det-encoder coreml-fp16`
  runs the detector's geometry + fusion encoders as one fp16 CoreML program, with the LoRA adapter
  folded in at export (`sam3.1_detenc_fp16_<adapter sha256[:12]>.mlpackage`).
  - Frame 0.818, 0.912 and 0.920x the default with LoRA v3 (run 3, every arm under load 8).
  - Held-out HOTA 0.6024 on 32 frames and 0.5693 on 1,500.
  - Not the default, because a new adapter needs a new export, and a missing package is an error.
- **Later still (E32, `bench/results/e32-memenc/README.md`):** the tracker's memory encoder as one
  fp16 CoreML program (`sam3.1_memenc_fp16.mlpackage`) passed every gate and joined the default,
  under memory attention's guard (`Placement::default_mem_encoder`).
  - Frame 0.865, 0.841 and 0.829x the E30 default (every arm under load 5).
  - Held-out HOTA 0.6025 on 32 frames (every video unchanged) and 0.5695 on 1,500.
  - A merge's encode (masks at the image size) stays on Ojas; `run.json` `mem_encoder_calls` counts
    both. `--mem-encoder ojas` gives the E30 default.
- `--trunk-precision coreml-fp32` restores the fp32 reference and reproduces every earlier result.
**E12 (2026-10-30):** the detector's host DETR decoder is 337 -> 62 ms per frame, with predictions
byte-identical, and the frame is 0.86x on VISEM 22 (`bench/results/e12-host-decoder`). Large host
linears now use up to `host::MAX_THREADS` = 8 scoped threads each, so a run outside the mac_heavy
lock competes for CPU and gains less.
**Closing job (2026-10-30, `bench/results/closing-e15`):**
- E12 + E13 + E15 together make the frame 0.843x on 82 and 0.825x on 22 of 3ca1476 (min of 3,
  load 10-23).
- The 1,500 held-out frames are byte-identical to the committed CoreML fp32 run, so HOTA stays
  0.4922.
- Metal allocated is flat. The process footprint creeps 0.6-1.6 MiB per frame, where E10's was
  flat (GAP-E15-HOST-FOOTPRINT-CREEP, open).
- E16 (`bench/results/e16-bisect-parity`) ties the creep to E15. E15 starts ~1.2 GB below E13 and
  is still ~400 MiB below it at frame 299.
- E17 (`bench/results/e17-footprint-categories`): over all 1,470 frames of 47, E15 settles at
  ~9.33 GB (slope -0.005 MiB/frame over frames 700-1469). The growth is host Malloc Large, and it
  is not a leak. The gap is answered.
- The tracker and detect-track parity suites pass on CPU and Metal through the E12-E15 changes.
**E13 / E15 (2026-10-30):**
- Memory attention is 74% of each tracker state's time. It does not shrink with objects, and its
  cost is in Ojas kernels (`bench/results/profile-tracker`); the numbers went to ojas-78.
- E15 keeps memory attention's inputs on the device. The tokens phase is 64 / 50 -> 14 / 6 ms per
  state on 82, predictions are byte-identical, and the footprint is ~760 MiB lower
  (`bench/results/e15-memory-tokens`).
- API shape change: `st_model::tracker::MaskMem` (`nchw` on the host, the trace and parity surface;
  `tokens` on the device) replaces `Stored.maskmem: Vec<f32>`, and `FrameFeats.tokens` is now a
  device `Tensor`.
- Both live as long as their `Stored` entry or frame, the same lifetime the host `Vec`s had. Say this plainly in every
write-up that leads with Ojas/tessl.
**M1–M5 Unification, Zero-Copy UMA & Flash Cross-Attention (2026-11-02):**
- **Zero-Copy UMA & Host-Mirror Removal (M3):** `SharedBuf::to_gpu_buffer()` in `tessl` and `Worker::upload` in `ojas-metal` adopt `Payload::Mapped` (`StorageModeShared`) buffers directly into the device buffer table with zero byte copies (`mem_after == mem_before`). In `crates/st-model/src/layers.rs`, dead `hw`/`hb` host weight vectors were removed from `Lin` and `Norm` (saving **>1.2 GB** of duplicate host RAM), and redundant `.to_host().to_f32_vec()` round-trips in `mask_decoder.rs`, `interactive.rs`, `vos.rs`, and `detector.rs` were replaced with device-native `Backend` ops (`slice`, `concat`, `fill`, `broadcast_add_forward`).
- **Native Decoupled Flash Cross-Attention & Register `boxRPB` (M4):** `ojas_attn_fwd_d32` and `ojas_attn_fwd_d64` in `ojas-metal` now execute decoupled $T_q \times T_k$ cross-attention natively on TensorOps without $O(m^2)$ pad-q zero-padding (**6.06 ms vs 209.28 ms scalar, 34.5x speedup, 100% flash dispatch** on `[1, 8, 5184, 32] x [1, 8, 5200, 32]` memory cross-attention) and accumulate 4D additive logit bias (`boxRPB`) directly in tile registers via buffers `13..18` before online softmax (**0.85 ms vs 11.98 ms scalar, 14.1x speedup** on `[1, 8, 201, 32] x [1, 8, 5184, 32]` DETR cross-attention).
- **Workflows & Deduplication (M1, M2, M5):** Added `./scripts/dev_fast_check.sh`, `./scripts/ci_local.sh`, `./scripts/parity_check.sh`, and `docs/WORKFLOWS.md`; retired 1,645 LOC of duplicate `safetensors`/`json`/`npy`/`plain` parsers in `tessl` in favor of `ojas-io`; verified 100% CPU and Metal backbone/tracker parity (`1.0000000` cosine similarity).
**E22 — Host-Only Weight Residency, Single-Allocation `HostBuf` Readback, Tap-Free Inference Paths, L1-Tiled `Mlp::host` Fusion, Single-Scope `rpb_bias` & Pre-Frame-0 Weight/Text Release (2026-11-02, `bench/results/e22-flash-uma-ram`):**
- Added `Lin::load_host`, `Norm::load_host`, and `Mlp::load_host` (keeping the 20 host-only layers across `Detector`, `MultiplexMaskDecoder`, `InteractiveDecoder`, and `Vos` on `Payload::Host` instead of `Payload::Device`, avoiding 20 GPU buffer allocations and every `.host(...)` `Wait::Read` readback + duplicate `Vec<f32>` copy), hardened `host_of`/`view_f32` (`DType::F32`, `is_contiguous()`, `byte_offset % 4 == 0`, subview slicing on `Payload::Mapped`), and added owned `HostBuf` (`layers::host_slice`) so remaining device readbacks allocate once and drop the device `Tensor` immediately.
- Added tap-free `Backbone::trunk_inference` and `Detector::forward_inference` (`taps: None`) for production `image` and `track` runs (freeing 164.6 MB of unused pinned diagnostic tap tensors, eliminating the mid-graph `geo` `Wait::Read` stall, reordering `hs` readback after transposed-conv upscale encoding in both mask decoders, skipping 5 unused `hs_proj.host` + `presence_head` passes per frame, and eagerly dropping dead `fpn`, `pred_masks`, `prop`, and slot-mask tensors).
- Fused `Mlp::host` (`Mlp::host_rows`) across 32-row L1-cache-resident tiles (`32 KB` scratch per thread, eliminating 72 MB/frame of `mach_vm_map` page-faulting intermediate buffers in `rpb_x` and `rpb_y`), added 4x8/4/1 output blocking in `host::linear_rows`, fused `Detector::rpb_bias`'s `dx`/`dy`/`rpb_x`/`rpb_y` into a single `std::thread::scope` per decoder layer, and batched/parallelized `InteractiveDecoder::fwd`'s host heads and mask contractions.
- Moved `Sam31Weights` by value into `track_on` and `infer_on`, cleared `vision.trunk.blocks` immediately when CoreML trunk is active, dropped `Text` (~1.4 GiB of CLIP text encoder Metal buffers) and each weight group as soon as its module is built, and called `telemetry.before_frames()` (`be.sync()` + `be.trim_pool()`) before frame 0.
- **Measured on M5 Pro (`bench/results/e22-flash-uma-ram`):**
  - **Metal allocated RAM at frame 0 / frame 15 [V]:** `5400 -> 3744–3754 MiB` (**-30.5% to -30.7%**) / `5667 -> 4031 MiB` (**-1,636 MiB / -28.9%**) on VISEM 82, and `5400 -> 3654–3738 MiB` (**-30.8% to -32.3%**) / `5558 -> 3918–3924 MiB` (**-1,634 to -1,640 MiB / -29.4% to -29.5%**) on VISEM 22.
  - **Process physical `footprint` at frame 15 [V]:** `8386 -> 3830–3838 MiB` (**-4,548 to -4,556 MiB / -54.2% to -54.3%**) on VISEM 82 and `8042 -> 2456–2531 MiB` (**-5,511 to -5,586 MiB / -68.5% to -69.5%**) on VISEM 22.
  - **Dispatches & GPU waits [V]:** frame 0 dispatches `4129 -> 2774` (82) and `4131 -> 2776` (22) (**-32.8%**); steady frame 10 dispatches `4142 -> 3996` (82) and `3085 -> 2964` (22); steady frame 10 unprofiled waits `+23 -> +20` (`read 19 -> 15`) on 82 and `+19 -> +16` (`read 16 -> 12`) on 22. Wall-clock speedup is not claimed as `[V]` without a quiet interleaved A/B (verified hold ran at external load 16.7–49.4).
  - **Parity [V]:** `backbone_parity` on 5 images Metal (95 stages, worst cosine `1.0000000`), `text_parity` Metal (cosine `1.000000000`), `tracker_parity` (8/8 CPU & Metal, worst cosine `1.0000000`), and `detect_track_parity` on VISEM 22 Metal (`session_matches_upstream_on_metal`: 32 frames, 0 id-set misses, mean mask IoU `0.99853` over 470 pairs) all pass.
**Weight-Space Programming (`st_weights::{inspect,edit}` + `spiketrack weights` / `--weight-edit`, 2026-11-02, `bench/results/weight-space`):**
- Added full-checkpoint tensor lookup across all 1,623 SAM 3.1 tensors (`875,545,324` params), `f64` matrix-free SVD / vocabulary embedding & post-24-layer `Text+resizer` cosine probes (`spiketrack weights`), and atomic `spiketrack-weight-edit-v1` surgical edits (`--weight-edit FILE` on `weights`, `image`, and `track`, composable with `--lora` and refusing CoreML trunk runs if ViT trunk tensors are edited).
- **Mechanistic findings [V]:** BPE row `35781` (`"sperm</w>"`) sits in a clinical text cluster (`ivf`, `fertility`, raw cosine `0.0510` with `"cell</w>"` row `5533`), causing `presence_probability = 0.1715` and `0` zero-shot detections; meanwhile 600-step LoRA (`visem-lora-r16`, whose largest spectral update is `decoder.layers.4.ca_text.in_proj_weight`, `sigma_0 = 0.8969`) froze `detector.dot_prod_scoring.*`, leaving true sperm head logits compressed across the `0.65` VOS birth threshold.
- **Unseen held-out VISEM (`21, 22, 36, 47, 82`, selected strictly on train-5 `14, 30, 19, 54, 13`) [V]:** Zero-gradient lexical graft (`1,024` / `1,537` params = `0.000117%`–`0.000176%`) lifts `--prompt sperm` from `0.0000` to `0.1729` / `0.1929` HOTA (8f) and `0.2106` HOTA (32f, cap 64); calibrating frozen `dot_prod_scoring.prompt_mlp.out_norm` (`512` params = `0.0000585%`, zero extra gradients) lifts 600-step LoRA from `0.4515 -> 0.5533` HOTA (`+10.18` pts, 8f) and `0.4751 -> 0.5526` HOTA (`+7.75` pts, `DetA 0.3162 -> 0.4359`, `5/5` unseen videos improved, 32f / 3,492 GT boxes).
**Enhanced LoRA v2 & v3, Exact 768-Scalar Inverse-Temperature, Head-Box NMS + Morphological Bulb Coasting & `2x2` Tiling (2026-11-03, `bench/results/weight-space/README.md`, `bench/results/lora/visem-lora-v3`):**
- **Universal 2-D LoRA merge, head/O2M training & BPE `token_embedding` + negative-query suppression [V]:** Extended `LoraAdapter::apply` (`crates/st-weights/src/lora.rs`) to reach every 2-D tensor in `Sam31Weights` (including `w.text.token_embedding`, `w.text.resizer.weight`, and `w.text.blocks`, with CoreML trunk refusal via `touches_vision_trunk()`) and verified PyTorch-vs-Rust merge parity across `token_embedding.weight` (`50,593,792/50,593,792` bit-identical, `max |d| = 0.00e0`), `detector.transformer`, `dot_prod_scoring`, `bbox_embed`, `presence_token_head`, and `language_backbone.resizer` (`3/3` pass, `max |d| <= 1.49e-8`, `bench/results/lora/merge-parity.log`). Trained `visem-lora-v2` (`41` modules, `131` targets / `71` tensors, `1,127,697` params: O2M `topk = 4`, `o2m_weight = 2.0` + heads) and warm-started **`visem-lora-v3`** (`42` modules, `135` targets / `72` tensors, `1,131,797` params) adding `LoRAEmbeddingRows` on CLIP BPE rows `(1375, 2629, 15595, 35781)` (`"head</w>"`, `"pin"`, `"cluster</w>"`, `"sperm</w>"`) plus focal negative query suppression (`neg_query_weight = 1.5`, `hard_neg_frac = 0.65`) strictly on `train-5` (`14,30,19,54,13`). On 120 unseen held-out frames, `visem-lora-v3` cuts absent-morphology (`"sperm cluster"`, `"pinhead sperm"`, `156` items) `@0.50` FPR by **`-40.4%` (`33.3% -> 19.9%`)** and `presence > 0.5` by **`-41.1%` (`46.8% -> 27.6%`, below zero-shot `35.9%`)**, maintains `100.0%` specificity (`0/600` FP) on everyday negatives, and reaches `73.8%` `@0.50` `"sperm"` F1 (`74.6%` on v2 step 400 vs `70.8%` on v1).
- **Decoupled post-decoder calibration, post-winner coasting `HeadBoxTracker` & presence-scaled `2x2` tiling [V]:** Added `Detector::set_post_box_delta` (post-layer-5 `[dx, dy, dw, dh]` logit shift preserving `rpb_bias(&refs)` in layers `0..5`) and `set_aux_score_weight`; added joint greedy `nms_box_and_iom` (`max_box_iou = 0.35`, `max_mask_iom = 0.65`, `boundary_margin = -1.0`), `Bitmask::bbox_xyxy`, and 2D integral-image `HeadBoxTracker` / `locate_head_bulb_bounded` (coasting damped kinematic head `(w, h)` when multi-object post-winner `y.masks[k]` is occluded, normalizing fill by unclamped `target_kw * target_kh`, and bounding ROI around `(pcx, pcy)`) strictly under `--boxes detector` (`--boxes mask` and `SessionConfig::new` remain 100% upstream-exact); added native `2x2` overlapping `1008x1008` tile detection (`crates/st-model/src/tile.rs`, `image --tile 2x2` with `Sam3Processor` per-tile `sigmoid(logit) * sigmoid(tile_presence)` scaling, `track --tile-every K`).
- **Unseen held-out VISEM (`21, 22, 36, 47, 82`) [V]:** Tracker-only upgrade lifts v1 + `512`-wt from `0.5533 -> 0.5756` HOTA (8f) and `0.5526 -> 0.5762` HOTA (32f). Combining `visem-lora-v3` (`adapter.safetensors`) with `768`-scalar exact inverse-temperature (`lorav3_exact_it_3p00.json`: scaling `out_norm.{weight,bias}` + `prompt_proj.bias` by `3.00x`, selected at `train-5` selection-grid peak `0.5802` / max `DetA = 0.4800` and leaving frame-0 `AUROC = 0.98352` and `AP@0.50 = 74.85%` invariant to 5 decimals) reaches **`0.6121` HOTA** (8f, **`AssA = 0.8115`**, `DetA = 0.5018`, `880` pred vs `879` GT; `0.6055` on v2) and **`0.6025` HOTA** (32f, **`AssA = 0.7880`**, `DetA = 0.5026`, `LocA = 0.7338`, `74.1%` Prec / **`76.3%` Rec** / `75.2%` F1 / `60.2%` Box Acc, `2,666` TP, all 5 unseen videos improved: `0.674 / 0.680 / 0.575 / 0.560 / 0.555`; `0.6013` on v2). On frame 0 (`112` GT sperm), `--tile 2x2` reaches **`0.9951` AUROC / `83.41%` AP@0.50** on v2 and **`83.7%` `@0.65` birth-gate F1 (`84.1%` peak at `0.66`, `95/112` TP, `82.6%` Prec / `84.8%` Rec / `72.0%` Box Acc)** on v3. *(1,500-frame `5 x 300f` `0.4922` was measured on v1 and not re-run on v2/v3.)*
- **Parity & CI gates [V]:** `./scripts/ci_local.sh` (`164` lib tests + `85` `spiketrack` tests + `otool -L` 15 system libs / 0 non-system), `st-model` integration suites, `lora_merge` (`3/3`), and Metal `detect_track_parity` (`session_matches_upstream_on_metal`: 32/32 frames id-equal, mean mask IoU `0.99853` over 470 pairs) all pass.

## 2. Why it is new (keep this framing in every write-up)

- A pure-Rust SAM 3 exists (`rlx-sam3`, crates.io, v0.2.11 2026-07-06) but states **"SAM 3.1 multiplex is intentionally out of scope"** [V].
- The only non-PyTorch SAM 3.1 multiplex port is CoreML/Swift (`sdkv2/sam3.1-coreml` @ 9668c49), tracker only (no detector or text encoder), CPU+GPU (ANE compile fails), multiplex 16. **Measured on this M5 Pro (2026-10-27, bench/results/baselines):** backbone (trunk + propagation neck, fp16) 0.272 s/frame and the multiplex-16 track step 0.0775 s/frame, ~0.35 s/frame in all; its backbone output matches upstream at cosine 0.99995 (bench/results/baselines/coreml_parity.txt). The port's own figure (~2.05 s + ~0.7 s on a 16 GB M-series Mac) is ~8x slower than this machine and must not be quoted.
- Efficient-SAM2 (ICLR 2026) skips work by decoder saliency and needs training; EdgeTAM/TinySAM2 retrain small models. Ours is training-free pixel-event gating + neural-accelerator path + zero-copy sparsl↔tessl.
- Pitch: *"The whole of SAM 3.1 (text prompt, detector and multiplex tracker) on a laptop, zero Python, measured parity with the official code."* Not "several times faster than the existing Mac port": measured on this M5 Pro we are ~8x slower per frame than the CoreML port, which runs only the tracker (D6 restated, see the "Speed target restated" note). Lead with what the CoreML port cannot do (text prompts, the detector, births of new objects, parity evidence). Do not claim real-time or interactive speed (see Second audit). Not "a Rust port".

## 3. Hard rules (read before writing code)

1. **No Python at runtime.** Python is allowed only in `tools/reference/` to generate test fixtures and baselines offline. The shipped binary must pass `otool -L` with no Python/libtorch.
2. **New-work boundary.** Ojas, tessl and sparsl are pre-existing open-source libraries; use them as dependencies, do not present them as hackathon work. Everything built during Oct 26–Nov 4 lives in this repo and is listed in `CONTRIBUTIONS.md` (task gp-rules-repo-and-access). Confirm the official rules' wording on prior work on day 1.
3. **Never commit SAM weights or gated data.** Weights are gated on Hugging Face under the SAM License [V]; datasets have their own licenses. `.gitignore` them; document how to fetch.
4. **Parity before speed.** No speed number is quoted until the module it covers passes its parity gate.
5. **Mac resource limits.** Follow the Ojas conventions: heavy cargo runs under `mac_heavy.sh`, `-j 2`, `--test-threads=2`, `OJAS_ALLOW_NO_GPU` unset so a missing GPU fails instead of skipping.
6. **Record gaps.** Anything a tool could not answer goes into `gaps.jsonl` (same shape as `research/Lappi-decision/gaps.jsonl`).
7. **Ask before**: changing Ojas/tessl/sparsl public APIs, pushing to any remote, or spending partner GPU credits beyond the reference task.

## 4. Schedule and gates (D1 = Mon Oct 26)

| Day | Date | Lane A — model port | Lane B — data & eval | Gate at end of day |
|---|---|---|---|---|
| D0 | Sun Oct 25 | — | Request HF access to `facebook/sam3.1` (+ `facebook/sam3`) | Access requested |
| D1 | Mon Oct 26 | gp-weights-loader; start gp-vision-backbone | gp-rules-repo-and-access; gp-reference-oracle | Rules read; fixtures for 5 images exist |
| D2 | Tue Oct 27 | gp-vision-backbone | gp-eval-harness (J&F) | **Backbone cosine > 0.999 on 5 images** |
| D3 | Wed Oct 28 | gp-prompt-encoders, gp-detector-and-mask-head | gp-eval-harness (HOTA), VISEM loader | **Image masks mean IoU ≥ 0.95 on 20 frames — else ship image-only** |
| D4 | Thu Oct 29 | gp-tracker-multiplex | baselines (CoreML, torch-MPS) on M5 Pro | Single-object tracking runs end to end |
| D5 | Fri Oct 30 | gp-tracker-multiplex | SA-V subset prepared | **J&F within 2 points of reference on 3 SA-V clips** |
| D6 | Sat Oct 31 | gp-tensorops-hot-path | FPS-vs-objects harness | **≥ parity with the CoreML port on the tracker path at 16 objects (same M5 Pro)** — restated 2026-10-27 by Bharath from "≥3×", which the measured CoreML time (0.35 s/frame, fp16 at ~70% of the GPU's bf16 peak) puts out of reach (GAP-D6-COREML-DENOMINATOR) |
| D7 | Sun Nov 1 | gp-event-gating | gating ablation runs | **≥1.5× FPS at ≤1 HOTA lost — else ship ungated** |
| D8 | Mon Nov 2 | gp-cli-and-no-python-proof | full VISEM + SA-V numbers | Benchmark table frozen |
| D9 | Tue Nov 3 | gp-demo-and-submission | README, figures | Demo video cut |
| D10 | Wed Nov 4 | buffer | buffer | **Submit by 8 PM CT** |

Stretch (only if every gate above is green): gp-cuda-smoke, gp-meta-model-report.

## 5. Task index

| Order | Task | Priority | Status | Depends on |
|---|---|---|---|---|
| 0 | gp-rules-repo-and-access | 0 | in progress (repo, licences, NOTICE, CONTRIBUTIONS done; Official Rules unread) | — |
| 1 | gp-reference-oracle | 0 | done (CPU reference, safetensors, backbone + text + tracker + detect-track fixtures, CoreML baseline done) | access granted |
| 2 | gp-weights-loader | 0 | done | access granted |
| 3 | gp-vision-backbone | 0 | done (CPU & Metal parity 1.0000000 cosine; CoreML fp32 default trunk) | 1, 2 |
| 4 | gp-prompt-encoders | 0 | done (text + geometry encoders CPU & Metal parity 1.0000000 cosine) | 2, 3 |
| 5 | gp-detector-and-mask-head | 0 | done (D3 gate 20/20 VISEM frames IoU 1.0000 on CPU & Metal) | 3, 4 |
| 6 | gp-tracker-multiplex | 0 | done (tracker_parity 8/8, VOS 3 DAVIS clips, detect_track_parity 32 VISEM frames IoU 0.9985) | 5 |
| 7 | gp-eval-harness | 1 | done (HOTA = TrackEval, DAVIS J&F, VISEM loader, `eval-visem` CLI) | 1 |
| 8 | gp-tensorops-hot-path | 1 | done (M4 `ojas_attn_fwd_d32`/`d64` flash cross-attention + tile-register `boxRPB`, E10–E22) | 6 |
| 9 | gp-event-gating | 1 | done (`st-gate` exact & approximate policies verified in `bench/results/gating`) | 6, 7 |
| 10 | gp-cli-and-no-python-proof | 1 | done (`spiketrack image/track/eval-visem/weights`, `scripts/no_python_check.sh`, 12.0 MB / 11.4 MiB binary) | 6 |
| 11 | gp-demo-and-submission | 0 | in progress (README, WRITEUP, SUMMARY, demo clips on the fp16 default (`36-*-fp16.mp4`) and the narrated 2:32 `spiketrack-demo.mp4` on the E32 default done (2026-11-04); upload and portal submission pending) | 7–10 |
| 12 | gp-cuda-smoke | 3 | backlog | 8 (stretch) |
| 13 | gp-meta-model-report | 3 | backlog | 10 (stretch) |
| 14 | gp-zero-shot-check | 0 | done (recall < 0.7) | — (runs before weights arrive) |
| 15 | gp-exemplar-prompts | 1 | backlog | 4 |
| 16 | gp-motion-aware-memory | 1 | backlog | 6 |
| 17 | gp-domain-lora | 0 (zero-shot recall < 0.7) | done (v1 + LoRA v2/v3 O2M/heads/BPE merged in Rust to 0.00e0 / 1.49e-8; held-out HOTA 0.4922 on 300f v1 and 0.6025 on 32f / 0.6121 on 8f with v3 + 768-wt exact IT) | 14, 2 |
| 18 | gp-weight-quantization | 3 | backlog | 8 (stretch) |

Component map: COMPONENTS.md. Improvement research and options: IMPROVEMENTS.md.

## 6. Numbers the submission must report

Mask IoU vs reference (image); J&F on the SA-V subset; HOTA on VISEM-Tracking; FPS vs object count (1/4/16/64/128) on M5 Pro against the CoreML port and PyTorch-MPS; peak memory; cold-start time; binary size; dependency list (`otool -L`). Every number cites the command and the result folder under `bench/results/`.

## 7. Start here (for an agent picking this up)

1. Read this file, then the task brief you were assigned in `tasks/`.
2. Check its dependencies are `done`; if not, stop and say which.
3. Work only in the brief's planned files; if you need others, note why in the brief's Description.
4. Close a criterion only with [V] evidence (command + output path). Update the brief's status and the index above.

## 8. Second audit (2026-10-25)

Labels as above. Findings, most serious first:

1. **Time [I]:** the plan assumes ten full days, but weekdays compete with a full-time job. Decide real hours per day; the weekend (Oct 31-Nov 1) is the only full block. If evenings only, cut gp-prompt-encoders to point/box prompts plus one text prompt, or move event gating to stretch.
2. **Weights access is a single point of failure [V gated]:** nothing in Lane A starts without it. Request tonight; escalate in the hackathon Discord if not granted by D2.
3. **Speed framing [I]:** the backbone is ~4.7 TFLOP of matmul per frame (450M params x 5,184 tokens), so ~2-3 frames/s is the realistic ungated ceiling on the M5 Pro. Keep the "≥3x the CoreML port" target; drop "interactive"/"real-time" wording everywhere.
4. **Gating may not help on VISEM [I]:** moving cells cover the whole field. Measure per-window change rates on D1-D2 (no model needed) and pick the gating showcase clip from the data.
5. **Reference path [V/U]:** official code needs CUDA [V]; Modal credits go to select participants [V]. Order: CPU PyTorch on the Mac for backbone fixtures, then RTX 3070 Ti, then Modal. The Meta Model API `sam-3.1` endpoint gives end-to-end masks/tracks as a second reference [V].
6. **Schedule clash:** the D2 backbone gate needs backbone fixtures on D1; fixtures for video can wait until D3-D4.
7. **SA-V access [U]:** fall back to DAVIS 2017 val for J&F if the form is not approved by D3.
8. **New-work rule:** only planning files exist before Oct 26; no code before kickoff.
9. **Architecture now known [V, arXiv 2511.16719]:** PE-L+ ~450M, 1008 px, patch 14, 32 layers (28 windowed 3x3, 4 global), RoPE; text encoder ~300M causal, 32 tokens; DETR 6+6; tracker num_maskmem = 7. Appendix H (multiplex) still [U].

Briefs updated with these notes: gp-reference-oracle, gp-vision-backbone, gp-event-gating, gp-eval-harness, gp-tracker-multiplex, gp-rules-repo-and-access.

## 9. Update (2026-10-25, later)

- VISEM-Tracking has per-sperm track IDs, YOLO-format boxes, classes (0 sperm, 1 cluster, 2 small/pinhead), 45–50 fps, CC BY 4.0, Zenodo record 7293726 [V, arXiv 2212.02842]. HOTA is computable.
- Supervised state of the art on VISEM: HOTA 74.3, MOTA 71.2 (YOLOv8 + DeepOCSORT) [V]. Report our zero-shot numbers as zero-shot, not head to head.
- New first task for D1: gp-zero-shot-check. Everything in the showcase assumes SAM 3.1 can find sperm; check it through the Meta Model API before the port depends on it.

## 10. Third audit — SAM 3.1 against Ojas and tessl (2026-10-25)

Labels: [V] read in the local code or a primary source this pass; [I] inferred; [U] confirm on D1. Line refs are to the repos as of 2026-10-25.

**Verdict: build the SAM 3.1 runtime on tessl directly, not through Ojas's Backend trait. Ojas stays a dependency only where it already fits (safetensors I/O).**

1. **Ojas has no bidirectional attention [V].** Its `Backend` trait exposes only `causal_sdpa_forward/backward` (ojas-core/src/backend.rs:788-816), and the Metal `sdpa_forward` is causal (ojas-metal/src/backend.rs:559). SAM's ViT, fusion encoder, DETR decoder and memory attention are all non-causal. Its trait also has no LayerNorm, conv2d or interpolation ops [V, docs/op-coverage.md]. Adding them means changing Ojas's public API, which the hackathon rules here forbid without asking.
2. **tessl already has most of a ViT encoder [V]:**
   - `embedgemma2::encoder_attn` / `encoder_attn_bf16`: bidirectional attention over right-padded batches with per-sequence lengths; TensorOps tiled kernels for head dim 64/256/512 (embedgemma2.rs:197-330).
   - **Windowed attention for free:** the 3×3 windows of 24×24 tokens are a batch of 9 sequences of 576 tokens, full length, no padding (72 = 3×24) [I]. Global layers are one sequence of 5,184 tokens.
   - `bert::layer_norm`, `bias_residual_layer_norm`, `bias_gelu_erf` (bert.rs:120-322): the LayerNorm and exact-GELU that ViTs use. Do not use `mlp_gelu_tanh`; tanh-GELU would break parity.
   - `embedgemma2_vision::axial_qkv_norm_rope` (embedgemma2_vision.rs:218): 2D axial RoPE driven by a positions buffer, but **fused with an RMS QK-norm**. If PE-L+ has no QK-norm [U], write a RoPE-only variant in SpikeTrack's own Metal file (copy, do not edit tessl). Also confirm PE's axis split and theta match `rope_inv_freq(head_dim/4, head_dim/2, theta)` [U].
   - `embedgemma2_pre::resize_rgb_u8` (embedgemma2_pre.rs:726): a torch-exact antialiased bicubic resize; SAM 3's own preprocessing interpolation must be matched exactly [U].
   - GEMM bf16 (26.6 TFLOP/s peak [V README]), Q4/Q8 GEMM/GEMV kernels for gp-weight-quantization [V nn.rs].
3. **Head dim [I]:** PE-L+ is ViT-L-like (1024 wide, 16 heads → head dim 64), which tessl's tiled encoder kernels cover. Confirm from the checkpoint on D1; a head dim outside 64/256/512 loses the TensorOps path.
4. **Transposed conv is needed earlier than planned [I]:** the "dual ViT-Det" neck that builds the multi-scale pyramid typically uses deconvolutions (×4, ×2) and pooling, so `st-ops` transposed conv lands in the backbone task, not only the mask head.
5. **LoRA training cannot run in Ojas during the hackathon [V]:** Ojas LoRA exists only for Qwen3.5 (ojas-model/tests/qwen35_lora*.rs) and needs the missing non-causal attention backward. gp-domain-lora trains in PyTorch offline (3070 Ti or Modal) and SpikeTrack loads merged weights.
6. **Multiplex interface hint [V]:** the CoreML port exports a stateful per-frame `track_step` with multiplex 16 (sdkv2/sam3.1-coreml). Read its model's input/output spec on D1 to learn the multiplex state layout before Appendix H is available.
7. **Pitch wording:** say "on tessl, my Metal 4 runtime (part of the Ojas stack)". Do not claim the model runs through Ojas's autograd or Backend API.

Briefs updated: gp-vision-backbone, gp-domain-lora, gp-tracker-multiplex. COMPONENTS.md updated.

## 11. Ojas ops (2026-10-25, Bharath's decision)

Bharath will build the missing vision ops in Ojas himself, so SAM 3.1 can run through Ojas's Backend (Metal via tessl kernels). Briefs in the Ojas repo: `tasks/gp-bidirectional-attention.md` (P0), `tasks/gp-vision-norm-act-bias.md` (P0), `tasks/gp-spatial-ops.md` (P1). This supersedes §3 rule 7 for these three briefs only. Until they land, SpikeTrack lanes may call tessl directly; switch to the Ojas ops as each passes parity. Record when each was built (before or during the hackathon) in CONTRIBUTIONS.md.

## 12. D1 findings (2026-10-26, session tessl-45)

Labels as above. Detail: tools/reference/README.md; open items: gaps.jsonl.

1. **Reference runs on the Mac CPU [V].** Official sam3 f1647e8 plus `tools/reference/cpu_shims.py` and `sam3-cpu.patch`: the full 3.1 multiplex stack loads with 0 missing and 0 unexpected keys, and an fp32 trunk forward takes 6.6 s. CUDA is no longer needed for backbone fixtures. The video path on CPU is still untested.
2. **Backbone fixtures exist [V].** `fixtures/backbone/*.safetensors` (5 images, fp32, 5.0 GB, git-ignored) and `fixtures/MANIFEST.json`. `check_fixtures.py` recomputes 27 tensors with independent torch math; all pass. Sent to session ojas-78 for its Ojas op parity tests.
3. **Architecture corrections [V]:** three FPN necks (`convs`, `interactive_convs`, `propagation_convs`) with no 0.5 level. The 3x3 conv is part of the neck. Abs pos embed is tiled 3x3 from 24x24. `ln_pre` eps is 1e-5 (blocks use 1e-6). RoPE uses interleaved pairs and fractional positions in the global layers, and the tables are stored per block. There is no QK-norm. Multiplex bucket is 16. Video heuristics come from the official builder (0.4 / 0.65), not config.json (0.5 / 0.7).
4. **Upstream runs the ViT MLP in bf16 [V].** The fp32 fixtures are an fp32 oracle; see GAP-UPSTREAM-BF16-MLP.
5. **Gating ceiling [V, bench/results/gating-change-rates/summary.md].** At tau 8, about 1% of 14x14 patches change frame to frame, but 55% of the 3x3 attention windows do (20 videos, 300 frames each). Exact window-level skipping can save at most about 45% of the windowed layers. The ≥1.5x gate likely needs token-level (approximate) gating. Low-change showcase clips: 29, 23, 52.
6. **Ojas ops:** session ojas-78 owns gp-bidirectional-attention, gp-vision-norm-act-bias and gp-spatial-ops. This session holds no Ojas or tessl files.
