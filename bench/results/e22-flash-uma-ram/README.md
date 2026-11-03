# E22: Flash cross-attention (`ojas_attn_fwd_d32`/`d64`), tile-register `boxRPB`, host-only weight residency (`load_host`), single-allocation `HostBuf` readback, tap-free inference paths, L1-tiled `Mlp::host` fusion, and pre-frame-0 weight/text release (2026-11-02)

Follows E21 (`docs/SPEED_PLAN.md`) and combines the M1–M5 Metal 4 TensorOps flash cross-attention (`ojas_attn_fwd_d32`/`d64`) + tile-register `boxRPB` kernels with six host-execution and Unified Memory (UMA) residency optimizations:

1. **Host-only weight residency (`Lin::load_host`, `Norm::load_host`, `Mlp::load_host`) & validated `view_f32` / `HostBuf` (`crates/st-model/src/layers.rs`, `detector.rs`, `tracker/{mask_decoder,interactive,vos}.rs`):**
   - Added `load_host` constructors for the 20 layers across `Detector`, `MultiplexMaskDecoder`, `InteractiveDecoder`, and `Vos` that execute exclusively on the host via `.host(...)`. Because `MetalBackend` wraps uploaded weights via `Tensor::from_device_reserved` (`Payload::Device`, not `Payload::Mapped`), calling `.host(...)` on `load()` weights previously fell back to `t.to_host(budget)?.to_f32_vec()?` — performing a synchronous GPU readback (`Wait::Read`) and a duplicate `Vec<f32>` copy on every call. Keeping those 20 layers on `Payload::Host` eliminates 20 GPU buffer allocations and every host-layer `Wait::Read` readback.
   - Hardened `view_f32` and `host_of` to validate `DType::F32`, `is_contiguous()`, `byte_offset % 4 == 0`, and subview window bounds `[start..start + len]` on `Payload::Mapped` tensors before exposing a slice.
   - Added owned `HostBuf` (`layers::host_slice`) over `Tensor::to_host`, eliminating the double host allocation in `t.to_host()?.to_f32_vec()?` across `backbone.rs`, `detector.rs`, `interactive.rs` (`[24, 32, 288, 288]` = 191.1 MiB on VISEM 82 frame 0), `mask_decoder.rs`, `session.rs` (`binary_input_masks`: `[24, 1, 1152, 1152]` = 121.5 MiB), `state.rs`, and `vos.rs`, while allowing immediate `drop` of the source device `Tensor`.
2. **Tap-free inference paths (`Backbone::trunk_inference`, `Detector::forward_inference`) & eager device buffer drops (`crates/st-model/src/{backbone,detector}.rs`, `tracker/{session,state,vos,interactive,mask_decoder}.rs`):**
   - Production `image` and `track` runs never read fixture diagnostic taps (`Taps` / `DetectorTaps`), yet `Backbone::trunk` and `Detector::forward` previously pinned `patch_embed` + `ln_pre` (**42.5 MB**) across all 32 ViT blocks and pinned 6 `enc_layers` + `seg_cross_attn` + `pixel_embed` (**122.1 MB**) across the DETR decoder, forced a mid-graph `host_of(&geo, be)` `Wait::Read` GPU stall before the 6 fusion encoder layers, and ran 5 unused `[200, 256 -> 2048 -> 256]` `hs_proj.host` + `presence_head` evaluations per frame.
   - `trunk_inference` and `forward_inference` (`taps: None`) release every intermediate activation immediately, remove the mid-graph `geo` `Wait::Read` (and reorder `hs` readback after transposed-conv upscale encoding in `InteractiveDecoder::fwd` and `MultiplexMaskDecoder::fwd`), and skip the 5 unused `hs_proj` + `presence_head` passes.
   - `Session::push_frame`, `round_neck`, `Vos::encode_mask_memory`, and `Vos::pointers_from_masks` eagerly drop `fpn` (111.5 MB), `pred_masks` (66.4 MB), `prop` (106.2 MB), and concatenated slot masks (~170 MB) as soon as their downstream consumers finish.
3. **L1-tiled row-chunked `Mlp::host` & `Mlp::host_rows` fusion (`crates/st-model/src/layers.rs`):**
   - Replaced sequential per-layer `Vec<f32>` allocations in `Mlp::host` (which allocated 14.75 MB per layer in `rpb_x`/`rpb_y` 12 times per frame = 72 MB/frame of `mach_vm_map` page-faulting buffers) with 32-row L1-cache-resident tiles (`32 KB` ping-pong scratch per thread) while preserving bit-identical IEEE `f32` reductions and empty/zero-row edge contracts.
4. **4x8 NEON-friendly `host::linear_rows` blocking & fused single-scope `rpb_bias` (`crates/st-model/src/host.rs`, `crates/st-model/src/detector.rs`):**
   - Added 4-row x 8/4/1 output blocking in `host::linear_rows` (with zero-input and ragged-slice guards in `linear_rows` and `layer_norm`), parallelized `sine_embed_boxes` and `layer_norm` above cache-amortized row thresholds, and fused `dx`, `dy`, `rpb_x.host_rows`, and `rpb_y.host_rows` into a single `std::thread::scope` per DETR decoder layer (eliminating 3 of 4 thread-pool spawns per layer = 144 spawns/frame).
5. **Batched/parallel `InteractiveDecoder::fwd` host head (`crates/st-model/src/tracker/interactive.rs`):**
   - Batched `hyper`, `iou_head`, and `obj_score_head` across all `N` prompts and parallelized the `[N * 4, 32] x [32, 82944]` mask contractions across `host::for_each_row_chunk_mut` with bit-identical row inner products.
6. **Pre-frame-0 `Sam31Weights` + `Text` encoder drop and Metal pool trim (`crates/spiketrack/src/{track_cmd,image_cmd}.rs`):**
   - `track_on` and `infer_on` consume `Sam31Weights` by value, clear `vision.trunk.blocks` immediately when CoreML trunk is active, drop `text_w` and `Text` (~1.4 GiB of CLIP text encoder Metal buffers) right after prompt encoding, drop each module's `TensorGroup` immediately upon construction, and call `telemetry.before_frames()` (`be.sync()` + `be.trim_pool()`) before frame 0.

## Verification & Parity Gates [V]

- **CI & static gates (`./scripts/ci_local.sh` + `--bins`):**
  - `rustfmt --check` across all workspace crates: pass.
  - `no_python_check.sh`: pass (`target/release/spiketrack` = 11,777,568 bytes at the E22 hold in `versions.txt`, 11,821,488 bytes / 11.8 MB / 11.3 MiB after `spiketrack weights` landed; 15 system frameworks/dylibs, 0 non-system, 0 Python/PyTorch/ONNX symbols).
  - `cargo clippy --workspace --all-targets -- -D warnings`: pass (0 warnings).
  - `cargo test --workspace --release --lib`: **153 passed** (150 at the E22 hold + 3 added `st-weights` tests), 0 failed, 1 Metal-only ignored (`st-coreml`: 7, `st-eval`: 10, `st-gate`: 29, `st-model`: 50, `st-video`: 35, `st-weights`: 22).
  - `cargo test -p spiketrack --release --bins -- --test-threads=1`: **80 passed** (79 at the E22 hold + 1 added modifier-order test), 0 failed, 8 ignored (`spiketrack`).
- **Adversarial M1–M5 & E22 tests (`parity-gates.log`):**
  - `adversarial_m3_layers`: **6 passed** (including `test_load_host_zero_waits_and_fused_mlp_host_bit_identity`, `mlp_host_panics_on_nonempty_input_when_n_is_zero`, and `view_f32_and_host_of_respect_subview_window_strided_and_bf16_on_metal`).
  - `adversarial_m3_roundtrip`: **3 passed**.
  - `adversarial_m5_boundary`: **7 passed**.
  - `detector::tests::the_rpb_bias_matches_the_host_loop_bit_for_bit_on_metal`: **1 passed** (`0.14 s`).
- **Upstream fixture parity gates (`parity-gates.log`, fail-closed via `gate_step` in `job.sh`):**
  - `backbone_parity` (`backbone_matches_pytorch_fp32_on_five_images_on_metal`): worst cosine **1.0000000** over 95 stages across 5 images (`gate > 0.999`).
  - `text_parity` (`text_encoder_matches_upstream_on_metal`): cosine **1.000000000**, `max abs / max ref = 1.835e-6` (`encoder.tokens`) and `8.130e-6` (`resizer.out`).
  - `tracker_parity` (**8 of 8**, CPU and Metal): `interactive_decoder`, `memory_attention`, `memory_encoder`, `multiplex_mask_decoder` all pass with worst cosine **1.0000000**.
  - `detect_track_parity` (`session_matches_upstream_on_metal`, 32 frames on VISEM 22): 32 yielded frames, **0 id-set misses**, mean per-id mask IoU **0.99853** over 470 `(frame, id)` pairs (`gate >= 0.95`).
  - `spiketrack image` smoke (`image-smoke/detections.json`): 12 detections for `"sperm"` above 0.5 on VISEM 22 frame 0 (`640x480`).
- **End-to-end repeatability & parity against E21 (`job.sh`):**
  - Within E22, `v82-e22-1`, `v82-stages`, and `v82-e22-2` produce **byte-identical** `82.json` files (`cmp -s`), and `v22-e22-1`, `v22-stages`, and `v22-e22-2` produce **byte-identical** `22.json` files (`cmp -s`).
  - Against E21 (`bench/results/e21-host-threads/`), E22 produces identical object ID sets in **16/16 frames** on both video 82 (278 object instances, `max box_xyxy_norm diff = 9.50e-7`, `max score diff = 3.54e-6`) and video 22 (184 object instances, `max box_xyxy_norm diff = 5.93e-6`, `max score diff = 6.50e-7`); the `1e-6` float differences come solely from M4's single-pass online-softmax flash cross-attention (`ojas_attn_fwd_d32`/`d64`) vs E21's unfused 2-pass softmax.

## Memory, Dispatch & GPU-Wait Reduction vs E21 [V]

All six 16-frame runs (`v82-{e22-1,stages,e22-2}` and `v22-{e22-1,stages,e22-2}`) were recorded in one `mac_heavy` hold (`versions.txt`: binary `de9b4e9f3fbdee33`, 58% system-wide memory free). Every metric below is deterministic or structural across load:

| Metric (16 frames, `--backend metal`) | Video 82 (E21) | Video 82 (E22, 3 runs) | Delta (82) | Video 22 (E21) | Video 22 (E22, 3 runs) | Delta (22) |
|---|---:|---:|---:|---:|---:|---:|
| `metal allocated` at frame 0 | 5400 MiB | **3744–3754 MiB** | **-1646 to -1656 MiB (-30.5% to -30.7%)** | 5400 MiB | **3654–3738 MiB** | **-1662 to -1746 MiB (-30.8% to -32.3%)** |
| `metal allocated` at frame 15 | 5667 MiB | **4031 MiB** | **-1636 MiB (-28.9%)** | 5558 MiB | **3918–3924 MiB** | **-1634 to -1640 MiB (-29.4% to -29.5%)** |
| `frame peak` Metal allocated (frame 0) | 9325 MiB | **7378–7393 MiB** | **-1932 to -1947 MiB (-20.7% to -20.9%)** | 8970 MiB | **6834–7240 MiB** | **-1730 to -2136 MiB (-19.3% to -23.8%)** |
| `frame peak` Metal allocated (steady frames 5..15) | 6459–6603 MiB | **5636–6281 MiB** | **-322 to -823 MiB** | 6353–6491 MiB | **5623–6204 MiB** | **-287 to -730 MiB** |
| Process physical `footprint` at frame 15 | 8386 MiB | **3830–3838 MiB** | **-4548 to -4556 MiB (-54.2% to -54.3%)** | 8042 MiB | **2456–2531 MiB** | **-5511 to -5586 MiB (-68.5% to -69.5%)** |
| Peak process `footprint` across frames 0..15 | 9141 MiB | **4074–4313 MiB** | **-4828 to -5067 MiB (-52.8% to -55.4%)** | 8716 MiB | **3266–3995 MiB** | **-4721 to -5450 MiB (-54.2% to -62.5%)** |
| Metal dispatches on frame 0 | 4129 | **2774** | **-1355 (-32.8%)** | 4131 | **2776** | **-1355 (-32.8%)** |
| Metal dispatches on steady frame 10 | 4142 | **3996** | **-146 (-3.5%)** | 3085 | **2964** | **-121 (-3.9%)** |
| Host waits on steady frame 10 (unprofiled) | +23 (upload 4, read 19) | **+20 (upload 5, read 15)** | **-3 waits (-4 read)** | +19 (upload 3, read 16) | **+16 (upload 4, read 12)** | **-3 waits (-4 read)** |

## Wall-Clock Timing & Load Contention Note [I — not claimed as an A/B speedup]

- **Why wall-clock speedup is not claimed as `[V]`:** `job.sh` ran without a rebuilt E21 control binary interleaved in the same hold (`GAP-SHARED-TARGET-DIR-AB`), and background processes outside `mac_heavy` drove system load averages to **16.68–49.44** during the verified hold (`versions.txt`). Under that contention, even the unchanged CoreML FP32 `trunk` stretched from `981.2 ms/frame` (E21's hold at load `4.4–15.9`) to **`1647.7 ms/frame`** on VISEM 82 (`v82-stages/82.profile-stages.md`) and **`1449.5 ms/frame`** on VISEM 22 (`v22-stages/22.profile-stages.md`), yielding contended steady-state means of `3976.7–4354.0 ms/frame` (`4241.3 ms` stage sum) on 82 and `3402.0–3456.0 ms/frame` (`4079.4 ms` stage sum) on 22.
- **Earlier single-arm E22 hold at lower load (`~6.7`, before the tap-free inference and `HostBuf` single-allocation pass) [I, cross-hold vs E21]:** steady-state unprofiled mean was `2143.3–2243.3 ms/frame` (`2156.7 ms` stage sum, `trunk 913.2 ms`, `decoder host prep 37.3 ms`, `decoder layer (device) 15.7 ms`, `segmentation head 103.7 ms`) on 82 and `1894.0–1964.0 ms/frame` on 22, against E21's `2416.0 ms/frame` (`2497.2 ms` stage sum) on 82 and `1849.3 ms/frame` (`1843.7 ms` stage sum) on 22. Establishing an exact `[V]` wall-clock delta requires a quiet-machine interleaved A/B hold building E21 and E22 in separate target directories.

Files: `v82-e22-1/`, `v82-stages/`, `v82-e22-2/`, `v22-e22-1/`, `v22-stages/`, `v22-e22-2/` (and their `.log` files), `image-smoke/detections.json`, `parity-gates.log`, `versions.txt`, and `job.sh`.
