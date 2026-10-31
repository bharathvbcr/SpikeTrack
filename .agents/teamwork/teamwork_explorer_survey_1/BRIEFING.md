# BRIEFING — 2026-10-31T17:25:00Z

## Mission
Investigate ojas and tessl architecture, memory management, and duplicate code to design a unified architecture with shared CPU/GPU memory pools without silent fallback.

## 🔒 My Identity
- Archetype: teamwork_explorer
- Roles: explorer, architecture_survey
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_1
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: survey_and_architecture_analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Target codebases: /Users/bharath/Code/research/ojas and /Users/bharath/Code/research/tessl
- Unified architecture & shared memory pools across CPU and GPU without silent fallback
- Tessl specializes in Metal / Mac inference optimizations; Ojas serves as universal system with focus on Rust training

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ojas/Cargo.toml`, `ojas-core/src/tensor.rs`, `ojas-core/src/backend.rs`, `ojas-metal/src/backend.rs`, `ojas-metal/src/device.rs`, `ojas-metal/build.rs`, `ojas-qwen35/src/lib.rs`, `ojas-io/src/safetensors.rs`, `ojas-io/src/json.rs`, `ojas-data/src/bpe.rs`, `ojas-model/src/qwen35/`
  - `tessl/Cargo.toml`, `tessl/src/lib.rs`, `tessl/src/tensor.rs`, `tessl/src/runtime.rs`, `tessl/src/safetensors.rs`, `tessl/src/json.rs`, `tessl/src/qwen35_train.rs`, `tessl/src/embedgemma2_tok.rs`
  - `ojas-ug-plan-doc/docs/unification-plan.md`, `ojas-ug-spike-uma/src/main.rs`, `tessl/BRANCH_REPORT.md`, `tessl/AUDIT_REPORT.md`
- **Key findings**:
  - Memory: `tessl` allocates all buffers with `MTLResourceOptions::StorageModeShared` (Apple Silicon unified memory), while `ojas-cpu` allocates `Vec<T>` on the heap, and `ojas-metal` proxies via an integer ID to an actor worker thread.
  - Zero-Copy & Silent Fallback: Zero-copy sharing is achievable via `MappedBuffer` and `HostLease` without violating explicit placement or silent fallback rules.
  - Dedup: 10k-11k lines of duplicate whole-step training orchestration, 2k lines of duplicate safetensors/json readers, duplicate AdamW scalar math, and duplicate kernel headers identified.
  - Architectural Seam: The Cargo dependency arrow points `ojas-metal -> tessl -> ojas-core`. `ojas-core` gains a minimal HAL trait layer (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`), `Budget` serves as the single authoritative ledger, and `tessl` specializes in Metal inference optimizations.
- **Unexplored areas**: Detailed kernel implementations for SAM & Gemma embedding (assigned to Explorer 2 & 3).

## Key Decisions Made
- Initiated architecture and memory exploration across ojas and tessl.
- Adopted the verified architecture from `unification-plan.md` (Proposal B + HAL traits in `ojas-core` with `ojas-metal -> tessl -> ojas-core` dependency flow).
- Structured the complete 5-component handoff report.

## Artifact Index
- DISPATCH.md — Task instructions and dispatch log
- BRIEFING.md — Situational awareness
- progress.md — Liveness heartbeat and steps log
- handoff.md — Final 5-component handoff report
