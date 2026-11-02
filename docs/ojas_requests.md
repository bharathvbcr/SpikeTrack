# Ojas work SpikeTrack is waiting on

Ojas is a pre-existing library (owner: the ojas-78 session; API changes are its and Bharath's call).
This lists what SpikeTrack asked for, what ojas-78 committed to, and the SpikeTrack call sites
that switch when each lands. Update a row when the commit arrives; delete it once switched.

| Op | Status (updated 2026-11-02; all ops landed and wired in `ojas` and `SpikeTrack`) | SpikeTrack call sites | Notes |
|---|---|---|---|
| Metal `rope_interleaved_forward` | **landed & wired** (`d5d33cc`) | `backbone.rs` `block_forward`; `memory_attention.rs` | bit-identical to CPU |
| Metal conv2d any stride/padding (groups 1, dilation 1) | **landed & wired** (`d5d33cc`, plus 1x1 s1 p0 direct NN GEMM in `9ffe251`) | `memory_encoder.rs` mask downsampler (3x3 s2 p1); `backbone.rs` / `detector.rs` | 1x1 s1 p0 skips im2col (>2x speedup) |
| Metal depthwise conv2d (groups == C) | **landed & wired** (`d5d33cc`) | `memory_encoder.rs` ConvNeXt fuser (7x7 p3, 256 ch, 72^2) | direct kernel `ojas_dwconv` |
| Metal `interpolate_bilinear_forward` (both align_corners) | **landed & wired** (`d5d33cc`) | `memory_encoder.rs` (to 1152), `vos.rs` and `tracker/state.rs` (to 1008 and to video size) | matches CPU bit-for-bit |
| Metal `relu_forward` | **landed & wired** (`d5d33cc`) | `layers.rs` `relu` (`be.relu_forward`) | zero host round-trip |
| `group_norm_forward` (CPU f64 two-pass, Metal 3-pass multi-threadgroup) | **landed & wired** (`6a870f9`, `317d238`) | `detector.rs` `group_norm` (`be.group_norm_forward`) | 3-pass GPU kernel with large-mean residual correction |
| `concat` (one axis, same rank) and `slice` (narrow one axis) | **landed & wired** (`71fd64d`) | `memory_attention.rs`, `mask_decoder.rs`, `interactive.rs`, `vos.rs`, `tracker/state.rs` | bit-exact device-side assembly; no host round-trips |
| `fill(shape, dtype, value)` | **landed & wired** (`eddba6f`) | `memory_attention.rs` pointer-slot zero padding; `detector.rs` / `vos.rs` / `interactive.rs` | single GPU dispatch (`ojas_fill_words`), zero host upload |
| tessl `encoder_attn` routing & decoupled TensorOps flash cross-attention (`D=32`, `D=64`) | **landed & wired** (M4, `ojas_attn_fwd_d32` / `ojas_attn_fwd_d64`) | `ojas-metal/src/backend.rs` (`flash_cross_route`, `encoder_attn_route`) | `encoder_attn` for $T_q = T_k$; native decoupled $T_q \times T_k$ flash cross-attention without $O(m^2)$ pad-q (**6.06 ms vs 209.28 ms scalar, 34.5x speedup** on `5184 x 5200`) |
| Bias-capable tiled / flash attention kernel (`boxRPB`) | **landed & wired** (M4, register-level additive bias in `ojas_attn_fwd_d32`/`d64` + split-K `ojas_xattn_part_dD`) | `ojas-metal/src/backend.rs`, `detector.rs` | tile-register bias via buffers `13..18` (**0.85 ms vs 11.98 ms scalar, 14.1x speedup** on DETR cross-attention) |

Also landed in `ojas` (CPU & Metal): `max_pool2d_forward` / `max_pool2d_backward` (available for FPN neck / training; no inference call site in `SpikeTrack`).
