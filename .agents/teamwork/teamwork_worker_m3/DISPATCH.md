## 2026-10-31T22:03:15Z
You are teamwork_worker_m3, assigned to implement Milestone 3 (SAM 3.1 & Vision Kernels with PyTorch Parity) for Features 13, 14, 15, 16, 17, 18, 19, and 20.

Your working directory is:
/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m3
Your parent is:
7f1f02db-7cbf-40d1-8919-b2741643d465

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Authoritative Files:
- You MUST read /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md
- /Users/bharath/Code/research/SpikeTrack/PROJECT.md
- Explorer 1 Report: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m3_1/handoff.md
- Explorer 2 Report: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m3_2/handoff.md
- Explorer 3 Report: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m3_3/handoff.md
- Branch Audit Report: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_branches/handoff.md

Domain Skills to apply:
- /Users/bharath/.gemini/config/skills/core-engineering/SKILL.md
- /Users/bharath/.gemini/config/skills/systems/SKILL.md

Implementation Instructions:
Follow the exact synthesis of the 3 Explorer reports:

Phase 1: In /Users/bharath/Code/research/tessl
- Merge `gp/w34-nt-bias-round` into `main` (Feature 17: Fused NT BF16 GEMM with bias addition and BF16 rounding in registers).
- Verify tessl tests: `./scripts/mac_heavy.sh cargo test --test gemm_nt_bf16_bias -- --test-threads=1`.

Phase 2: In /Users/bharath/Code/research/ojas
- Merge branches cleanly into `main`:
  1. `gp/ug-cpu-sam` (Features 19 & 20: GroupNorm & Bilinear backward, tape records, PyTorch fixture generator).
  2. `gp/ug-metal-maxpool` (Feature 13: Metal max_pool2d forward and backward with zero-atomic inverse gather).
  3. `gp/ug-metal-spatial-bwd` (Feature 14: Metal conv2d and conv_transpose2d backward, exact f32 GEMM + permute for SAM $K=S=2$, and `ojas_col2im`).
  4. `gp/ug-metal-encoder-bwd` (Feature 18: Double-single 46-bit arithmetic for LayerNorm, GELU erf, and Softmax backward).
  5. `gp/ug-metal-attn-bwd` (Features 15 & 16: FlashAttention-2 general tiled backward for $D=256$, `boxRPB` additive logit bias).

Phase 3: Wire-up and Reconciliation in ojas-metal
- In `ojas-metal/build.rs`: Reconcile `const SOURCES` to include all 7 kernel sources:
  `["per_head_gate.metal", "causal_attn_bwd.metal", "attn_gen_bwd.metal", "ojas_backend.metal", "encoder_bwd.metal", "max_pool.metal", "spatial_bwd.metal"]`, and add `println!("cargo:rerun-if-changed=kernels/attn_bwd_tiled.h");`.
- In `ojas-metal/src/link.rs`: Reconcile command enums and geometries (`PoolGeom`, `Cmd::MaxPool2d`, `Cmd::MaxPool2dBwd`, `Cmd::Conv2dBwd`, `Cmd::DepthwiseConv2dBwd`, `Cmd::ConvTranspose2dBwd`).
- In `ojas-metal/src/device.rs`: Reconcile kernel registrations, dispatch handlers, and replace hardcoded `"ojas_attn_fwd_d32"` with dynamic `format!("ojas_attn_fwd_d{width}")`.
- In `ojas-metal/src/backend.rs`: Reconcile imports (`ConvGrad` alongside `AdamWScalars`), implement `Backend::max_pool2d_forward`, `max_pool2d_backward`, `conv2d_backward`, `conv_transpose2d_backward`, extend `ENCODER_ATTN_HEAD_DIMS` to include 256, extend `FLASH_CROSS_HEAD_DIMS` to `[32, 64, 128, 256]`, and wire fused `linear_bias_forward` to tessl's `gemm_nt_bf16_bias`.

Phase 4: Wire Remaining Vision Tape Ops in ojas-autograd
- Add `relu_backward` and `slice_backward` to `ojas-core::Backend` and `with_backend_methods!`.
- Implement on `ojas-cpu::backend`.
- Add `Rec::RopeInterleaved`, `Rec::Concat`, `Rec::Slice`, and `Rec::Relu` to `ojas-autograd/src/tape.rs` with forward and backward execution.

Phase 5: PyTorch Golden Fixtures Generation
- Run fixture generators using Python (`/Users/bharath/.venvs/torch/bin/python`):
  - `max_pool_fixture.py` -> `ojas-oracle/fixtures/max_pool/golden.txt`
  - `conv_backward_fixture.py` -> `ojas-oracle/fixtures/conv_bwd/golden.txt`
  - `sam_backward_fixture.py` -> `ojas-oracle/fixtures/sam_bwd/golden.txt`

Phase 6: Quality Gates & Full Verification
- Run:
  - `cargo test --package ojas-metal --test max_pool -- --test-threads=1`
  - `cargo test --package ojas-metal --test conv_backward -- --test-threads=1`
  - `cargo test --package ojas-metal --test attention_backward_general -- --test-threads=1`
  - `cargo test --package ojas-metal --test encoder_backward -- --test-threads=1`
  - `cargo test --package ojas-autograd --test sam_tape`
  - `cargo test --package ojas-oracle --test e2e_tier4_production tier4_scenario_a_sam31_video_tracker_multiplex_pipeline -- --nocapture`
  - Parity tests against `/Users/bharath/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors`
  - CI gates: `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`, `./scripts/check_tracked_targets.sh`, `./scripts/fmt_check.sh`, `./scripts/ci_local.sh`.

Update `progress.md` after each step with timestamps. When complete, write `handoff.md` with:
- Observation (all files touched, git commits merged)
- Logic Chain (soundness and verification proof)
- Caveats & Edge cases
- Conclusion & summary of verified tests
- Exact commands and verification results.
Send a message to your parent (7f1f02db-7cbf-40d1-8919-b2741643d465) when done.

## 2026-10-31T22:29:56Z
**Context**: Milestone 3 Implementation Status Check
**Content**: Heartbeat check — please report your current progress across Phase 4 (wiring vision tape ops in ojas-autograd and ojas-core/cpu), Phase 5 (fixture generation), and Phase 6 (verification).
**Action**: Reply with your current status and update progress.md.

## 2026-10-31T22:49:56Z
**Context**: Milestone 3 Phase 6 Verification Status Check
**Content**: Heartbeat check — please provide a brief update on your Phase 6 verification progress (which test suites and CI checks have run so far).
**Action**: Reply with your current test progress and update progress.md.

## 2026-10-31T23:10:02Z
**Context**: Milestone 3 Phase 6 Liveness Check
**Content**: Heartbeat check — it has been ~19 minutes since your last progress update during `encoder_backward` tests. Please report your current execution status and update progress.md.
**Action**: Reply immediately with your status.
