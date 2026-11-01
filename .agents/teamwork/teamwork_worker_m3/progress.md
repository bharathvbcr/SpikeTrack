# Progress — Milestone 3 Implementation

Last visited: 2026-10-31T23:54:30Z

## Status
- [x] Phase 1: In /Users/bharath/Code/research/tessl - Merged `gp/w34-nt-bias-round` into `main` (commit 48937fb). Verified `tests/gemm_nt_bf16_bias.rs` (5 of 5 passed under `mac_heavy` lock).
- [x] Phase 2: In /Users/bharath/Code/research/ojas - Merged all 5 branches cleanly into `main`:
  - `gp/ug-cpu-sam` (Features 19 & 20)
  - `gp/ug-metal-maxpool` (Feature 13)
  - `gp/ug-metal-spatial-bwd` (Feature 14)
  - `gp/ug-metal-encoder-bwd` (Feature 18)
  - `gp/ug-metal-attn-bwd` (Features 15 & 16)
- [x] Phase 3: Wire-up and Reconciliation in ojas-metal:
  - Reconciled `build.rs` SOURCES (6 active metal sources)
  - Reconciled `link.rs` with `Cmd::MaxPool2d`, `Cmd::MaxPool2dBwd`, `Cmd::Conv2dBwd`, `Cmd::DepthwiseConv2dBwd`, `Cmd::ConvTranspose2dBwd`, `Cmd::AttnBwd`, `Cmd::LinearBias`
  - Reconciled `device.rs` dynamic width kernel formatting `format!("ojas_attn_fwd_d{width}")` and `linear_bias`
  - Reconciled `backend.rs` `linear_bias_forward` to fused tessl GEMM, `ENCODER_ATTN_HEAD_DIMS` extended to include 256, `FLASH_CROSS_HEAD_DIMS` extended to `[32, 64, 128, 256]`
  - Verified `cargo check -p ojas-metal --features metal` compiles cleanly with 0 errors (commit bf100ea).
- [x] Phase 4: Wire remaining vision tape ops in ojas-autograd and ojas-core/cpu:
  - Added `relu_backward_dims` and `slice_backward_dims` to `ojas-core::shapes`.
  - Added `relu_backward` and `slice_backward` to `Backend` trait, `with_backend_methods!`, `every_call_reaches`, and `AutocastBackend`.
  - Implemented `relu_backward` and `slice_backward` on `CpuBackend`.
  - Added `Rec::RopeInterleaved`, `Rec::Concat`, `Rec::Slice`, `Rec::Relu` to `ojas-autograd::Tape`.
  - Tested in `sam_tape.rs` (8 passed, 0 failed) and `probe_forwarding.rs` (5 passed, 0 failed). Committed in ojas (commit eba1e29).
- [x] Phase 5: PyTorch Golden Fixtures Generation:
  - Generated `max_pool/golden.txt` (68 lines)
  - Generated `conv_bwd/golden.txt` (12 cases)
  - Generated `sam_bwd/golden.txt` (20 cases, torch 2.15.0)
  - Committed in ojas (commit f2c126d).
- [x] Phase 6: Quality Gates & Full Verification:
  - `cargo test --package ojas-metal --test max_pool -- --test-threads=1` -> PASS (11 passed, 0 failed)
  - `cargo test --package ojas-metal --test conv_backward -- --test-threads=1` -> PASS (10 passed, 0 failed)
  - `cargo test --package ojas-metal --test attention_backward_general -- --test-threads=1` -> PASS (14 passed, 0 failed)
  - `cargo test --package ojas-metal --test encoder_backward -- --test-threads=1` -> PASS (22 passed, 0 failed)
  - `cargo test --package ojas-autograd --test sam_tape` -> PASS (8 passed, 0 failed)
  - `cargo test --package ojas-autograd --test metal_encoder_tape -- --test-threads=1` -> PASS (3 passed, 0 failed)
  - `cargo test --package ojas-oracle --test e2e_tier4_production` -> PASS (4 passed, 0 failed)
  - SAM fixtures parity CPU & Metal -> PASS (all supported operations match torch goldens with cosine similarity 1.000000000000)
  - Pinned counts and unit tests reconciled across `gate_w11_adversarial.rs`, `autocast_sam_policy.rs`, `spatial_backward.rs`, `attention_dims.rs`, `attention_flash.rs`, `attention_general.rs`, `sam_ops_refused.rs`, and rustdoc links.
  - Formatted workspace with `./scripts/fmt_check.sh --fix`.
  - `./scripts/ci_local.sh` under `mac_heavy` lock completed with exit code 0 (`ci:local OK`).
