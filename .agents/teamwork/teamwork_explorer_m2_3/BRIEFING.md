# BRIEFING — 2026-10-31T18:31:00Z

## Mission
Investigate and design the elimination of duplicated IO readers (Feature 11) and training loops (Feature 12) across tessl and ojas.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, synthesis
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_3
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: Milestone 2: Unified Architecture & Shared Memory Pools

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Reuse, unify, and shrink: Extending what exists is the default; One canonical owner per behavior; Replace, don't accumulate.
- No new dependency without asking.
- Cite evidence (file:line, verified/inferred/unverified).
- Write findings to handoff.md, message parent on completion.

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: 2026-10-31T18:31:00Z

## Investigation State
- **Explored paths**:
  - `tessl/src/safetensors.rs`, `tessl/src/json.rs`, `tessl/src/npy.rs`, `tessl/src/plain.rs`, `tessl/src/config_json.rs`, `tessl/src/qwen35_model.rs`, `tessl/src/loader.rs`, `tessl/src/embedgemma2*.rs`, `tessl/src/capi.rs`
  - `ojas/ojas-io/src/{lib.rs, safetensors.rs, json.rs, npy.rs, open.rs, posread.rs}`
  - `tessl/src/qwen35_train.rs`, `tessl/src/qwen35_adamw.rs`, `tessl/src/qwen35_bwd.rs`
  - `ojas/ojas-metal/src/{lib.rs, gpu.rs, device.rs}`
  - `ojas/ojas-model/src/{qwen35/step_provider.rs, qwen35/forward.rs, qwen35/optim.rs}`
  - `ojas/ojas-qwen35/src/{lib.rs, step.rs, names.rs}`
  - `ojas/ojas-core/src/{backend.rs, lib.rs, shapes.rs}`
- **Key findings**:
  - Feature 11: `tessl` duplicates `safetensors.rs` (951 lines), `json.rs` (517 lines), `npy.rs` (418 lines), and `plain.rs` (146 lines) = 2,032 lines. Canonical crate is `ojas-io`. `ojas-qwen35` was even importing `tessl::safetensors::CheckpointView` because sharded checkpointing was placed in tessl instead of `ojas-io`.
  - Feature 12: `ojas-metal/src/gpu.rs` (3,736 lines) is a dead toy-step training loop with only 1 active caller (`gate_dbias_threads`). `tessl/src/qwen35_train.rs` (4,726 lines) and `tessl/src/qwen35_adamw.rs` (1,519 lines) duplicate whole-step autograd and training orchestration that canonically belongs in `ojas-model::TapeStepProvider`. Total dead/duplicate code to retire: ~12,013 lines!
  - Full caller graphs, line counts, and 6-phase migration design mapped for Worker.
- **Unexplored areas**: None within Feature 11 & 12 scope.

## Key Decisions Made
- Confirmed acyclic dependency chain: `ojas-metal` -> `tessl` -> `ojas-io` -> `ojas-core`.
- Established `ojas-io` as canonical owner of single & sharded safetensors, strict JSON, and .npy IO.
- Established `ojas-model` as canonical owner of training step orchestration, while `tessl` retains low-level Metal execution kernels (`qwen35_bwd.rs`, `attn_train.rs`, `gdn_train.rs`).
- Consolidated AdamW scalar generation into `ojas-core::adamw_kernel_scalars`.

## Artifact Index
- DISPATCH.md — Record of dispatch instructions
- BRIEFING.md — Working memory and context index
- progress.md — Liveness heartbeat and step tracker
- handoff.md — Final 5-component handoff report
