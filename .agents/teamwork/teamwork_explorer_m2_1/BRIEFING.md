# BRIEFING — 2026-10-31T18:30:30Z

## Mission
Investigate and design the exact HAL trait definitions and unified buffer integration between ojas-core and tessl (Milestone 2: Features 7 and 9).

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, architectural design, synthesis
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_1
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: Milestone 2: Unified Architecture & Shared Memory Pools

## 🔒 Key Constraints
- Read-only investigation — do NOT implement directly in source trees (except reports/artifacts in agent directory).
- Pure Rust HAL traits in ojas-core: `#![forbid(unsafe_code)]`, zero external dependencies.
- Exact type definitions, zero-copy guarantees on Apple Silicon, fence synchronization.

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: 2026-10-31T18:30:30Z

## Investigation State
- **Explored paths**:
  - `ojas-core/src/hal.rs`, `tensor.rs`, `lib.rs`, `error.rs`, `budget.rs`
  - `tessl/src/runtime.rs`, `tensor.rs`, `mtl_tensor.rs`
  - `ojas-metal/src/backend.rs`, `device.rs`, `Cargo.toml`
  - `ojas-ug-plan-doc/docs/unification-plan.md`
  - `ojas-ug-spike-uma/src/main.rs`
  - `SpikeTrack/PROJECT.md`, `TEST_INFRA.md`
- **Key findings**:
  - Pure Rust HAL traits exist partially in `ojas-core/src/hal.rs` (14 unit tests passing) with reflection test `the_hal_list_names_every_trait_method`.
  - Tessl allocates all pooled buffers with `MTLResourceOptions::StorageModeShared`. Host pointer `buffer.contents()` and GPU address `buffer.gpuAddress()` are stable for buffer lifetime.
  - Current `to_host` and `upload` perform costly double memcpy through `device.rs` and channel passing.
  - Zero-copy representation requires `Payload::Mapped(Arc<dyn MappedBuffer>)` on `Tensor`, timeline fence synchronization via `MTLSharedEvent` (`lifetime_event`), and RAII `HostLease` guards.
- **Unexplored areas**:
  - Milestone 3/4 kernel implementations (assigned to Explorer 2).

## Key Decisions Made
- Unify `MappedBuffer` to implement `DeviceBuffer` with `host_ptr()`, `gpu_address()`, `timeline()`, `host_read()`, and `host_write()`.
- Add `current_value()` and `wait_for(value, timeout)` to `Timeline` trait.
- Map `AllocError::Exhausted { requested, available }` and `AllocError::InvalidAlignment` cleanly to `OjasError::CapacityExceeded`.
- Implement zero-copy `to_host()` on `Payload::Mapped` via timeline wait without memcpy or re-reservation.

## Artifact Index
- DISPATCH.md — record of incoming dispatch messages
- BRIEFING.md — persistent working memory
- progress.md — liveness heartbeat
- handoff.md — final comprehensive report
