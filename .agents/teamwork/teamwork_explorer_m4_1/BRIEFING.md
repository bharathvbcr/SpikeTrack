# BRIEFING — 2026-11-01T01:12:00Z

## Mission
Investigate and map the CPU reference implementations, shape validators, autocast rules, and autograd tape wiring for Milestone 4 (Features 21–25: Gemma CPU & Autograd) in `ojas-ug-cpu-gemma` and `ojas`.

## 🔒 My Identity
- Archetype: explorer
- Roles: Gemma CPU & Autograd Explorer
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m4_1
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: Milestone 4 (Features 21–25: Gemma CPU & Autograd)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify production code
- Write only to /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m4_1/
- Self-contained handoff report following 5-component format
- Follow ojas invariants: exact bit reproducibility (Numerics::Exact), no silent fallback, shape validators in ojas-core, Backend trait wiring requirements

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ojas-ug-cpu-gemma` worktree and branch `gp/ug-cpu-gemma`
  - `ojas-core/src/gelu_tanh.rs`, `ojas-core/src/erf_exact.rs`, `ojas-core/src/shapes.rs`, `ojas-core/src/autocast.rs`, `ojas-core/src/backend.rs`, `ojas-core/src/lib.rs`
  - `ojas-cpu/src/gemma_ops.rs`, `ojas-cpu/src/backend.rs`, `ojas-cpu/src/attn/general.rs`, `ojas-cpu/src/lib.rs`
  - `ojas-autograd/src/tape.rs`, `ojas-autograd/tests/gelu_tanh_tape.rs`, `ojas-autograd/tests/attention_general.rs`
  - `ojas-cpu/tests/gemma_ops.rs`, `ojas-cpu/tests/attention_general.rs`, `ojas-cpu/tests/autocast_sam_policy.rs`
  - `ojas-oracle/tests/e2e_tier1_features.rs`, `ojas-oracle/tests/e2e_tier4_production.rs`
  - Sibling checkout `tessl/src/embedgemma2.rs` and `tessl/kernels/embed_pool.metal`
- **Key findings**:
  - Feature 21 (`gelu_tanh_forward` and `gelu_tanh_backward`): fully drafted in `ojas-ug-cpu-gemma` (`gelu_tanh.rs`, `gemma_ops.rs`, `backend.rs`, `tape.rs`, `shapes.rs`, `autocast.rs`). Passes 14 CPU tests and 4 tape tests. Uses `f64` arithmetic over `exp_neg(|u|)` from `erf_exact.rs`, zero libm `tanh` drift, bit-identical across thread counts.
  - Feature 22 (Symmetric non-causal sliding window attention): fully drafted in `ojas-ug-cpu-gemma` (`ojas-cpu/src/attn/general.rs`, `tape.rs`, `shapes.rs`). `opts.radius` ($|i - j| \le W$) and `opts.scale` implemented, handles ragged key lengths and inert padding rows ($T_q == T_k, i \ge \text{len}$).
  - Feature 23 (Masked sequence mean pooling): defined on `Backend` as `seq_mean_forward(&self, x: &Tensor, key_lens: &Tensor) -> Result<Tensor, OjasError>`. Matches `tessl`'s `segment_mean_rows_f32`. Verified bit-identical across threads and numerics in `gemma_ops.rs`.
  - Feature 24 (L2 normalization with Matryoshka prefix truncation): defined on `Backend` as `l2_normalize_forward(&self, x: &Tensor, eps: f32) -> Result<Tensor, OjasError>`. Matryoshka dimension truncation is cleanly achieved via `slice(&x, 1, 0, target_dim)` + `l2_normalize_forward(&truncated, eps)`.
  - Feature 25 (Per-head V RMSNorm): weightless RMSNorm formula $V / \sqrt{\text{mean}(V^2) + \epsilon}$ verified in `e2e_tier1_features.rs` and `tessl/src/embedgemma2.rs`. In Ojas, V norm can be executed via `rms_norm_forward` with unit scale or by adding `rms_qkv_norm_forward` to `Backend`.
  - Cargo lockfile collision in `ojas-ug-cpu-gemma` explained: `tessl/Cargo.toml` hardcodes path `../ojas/ojas-core`, which collides with `./ojas-core` when cargo is run from `ojas-ug-cpu-gemma`. When integrated onto `main` in `/Users/bharath/Code/research/ojas`, the path matches and resolves cleanly.
- **Unexplored areas**: none. All 5 features and their CPU/autograd mappings thoroughly examined.

## Key Decisions Made
- Mapped all 5 features to exact files, lines, shape validators, autocast rules, and autograd tape records.
- Prepared comprehensive 5-component handoff report.

## Artifact Index
- DISPATCH.md — Initial dispatch message
- BRIEFING.md — Situational awareness briefing
- progress.md — Liveness heartbeat
- handoff.md — Comprehensive handoff report
