# BRIEFING — 2026-10-31T17:22:00Z

## Mission
Investigate SAM and Gemma embedding architectures, analyze existing Ojas/Tessl kernels, identify missing ops, and specify kernel designs with PyTorch parity and numerical exactness.

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: Kernel Specification Explorer
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: SAM & Gemma Embedding Kernel Investigation and Specification

## 🔒 Key Constraints
- Read-only investigation — do NOT implement in source code
- Safe Rust, bit-identical reduction under Numerics::Exact, no silent fallback
- Strict shape validation at boundaries (`*_dims`)
- 5-Component handoff report with verification method

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ojas/ojas-core/src/backend.rs` (95 trait methods in `with_backend_methods!`)
  - `ojas/ojas-cpu/src/backend.rs` & `ojas-cpu/src/attn/general.rs`
  - `ojas/ojas-metal/src/backend.rs` & `ojas_backend.metal`
  - `ojas/ojas-autograd/src/tape.rs` (autograd `Rec` variants)
  - `ojas/ojas-oracle/src/sam_fixtures.rs`, `parity.rs`, `bf16_operands.rs`
  - `tessl/src/embedgemma2.rs`, `embedgemma2_vision.rs`, `gemm.rs`, `bert.rs`
  - `tessl/kernels/encoder_attn.metal`, `attn_tiled.h`, `act.h`
  - `SpikeTrack/COMPONENTS.md`, `HANDOFF.md`, `docs/embedgemma_reuse.md`, `docs/ojas_requests.md`
  - GitPulse tasks (`ft-0f61c4e29a53fe886ac192979e4d571b`, `gp-bidirectional-attention`, `gp-spatial-ops`, `gp-vision-norm-act-bias`)
- **Key findings**:
  - SAM requires: 1) Metal `max_pool2d_forward`, 2) Generalized `conv_transpose2d` beyond stride=kernel, 3) D=256 attention routing to tiled Metal kernels (currently falls back to scalar kernel causing 74% of frame time in memory attention), 4) Additive logit bias attention (DETR boxRPB), 5) Fused QKV + 2D RoPE + window partition, 6) BF16 activation tier & LayerNorm, 7) GELU-Erf GEMM epilogue fusion, 8) Autograd Tape wiring for vision ops.
  - Gemma embedding requires: 1) `gelu_tanh` / `gated_gelu_tanh` (GeGLU) in Backend, 2) Symmetric inclusive non-causal sliding window attention (`|i - j| <= W`), 3) Sequence-level masked mean pooling (`segment_mean_rows`), 4) Standalone L2 normalization with Matryoshka prefix slicing (`l2_normalize_rows`), 5) Per-head V RMSNorm.
- **Unexplored areas**: None for kernel specification.

## Key Decisions Made
- Fully documented all 13 kernel specifications with inputs, outputs, shape validators, CPU exact algorithms, Metal GPU execution models, and PyTorch parity test suites.

## Artifact Index
- DISPATCH.md — Task dispatch log
- BRIEFING.md — Working memory
- progress.md — Liveness heartbeat
- handoff.md — Final structured report
