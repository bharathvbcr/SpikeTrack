# BRIEFING — 2026-10-31T19:53:00Z

## Mission
Analyze how to bridge `tessl::shared_buf::SharedBuf` to `ojas_core::hal::MappedBuffer` to enable `Tensor::from_mapped` wrapping real Metal unified memory buffers safely.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, synthesis
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_2/
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: Milestone 2 Iteration 2 (Remediation)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify source code
- Preserved #![forbid(unsafe_code)] in ojas-core and ojas-metal
- The only unsafe remains inside tessl::SharedBuf
- Follow 5-component handoff report structure

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: 2026-10-31T19:50:13Z (server restart resume acknowledged)

## Investigation State
- **Explored paths**:
  - `ojas-core/src/hal.rs` (MappedBuffer, DeviceBuffer, Timeline, HostLease, HostLeaseMut, AllocError, FenceError)
  - `ojas-core/src/tensor.rs` (Tensor::from_mapped, Payload::Mapped, host_view, to_host, device_buffer)
  - `tessl/src/shared_buf.rs` (SharedBuf, SharedAlloc, HostRead, HostWrite, GpuRole, unsafe impl Send/Sync)
  - `tessl/src/runtime/host_fence.rs` (HostFence, MTLSharedEvent, Timeline parity)
  - `tessl/src/shared.rs` (Shared, Timeline, RuntimePoison, commit clock)
  - `tessl/src/dispatch.rs` (bind_shared_ro, bind_shared_rw)
  - `ojas-metal/src/backend.rs` (MetalBuffer, MetalBackend::upload)
- **Key findings**:
  1. Rust orphan rules require `impl MappedBuffer for SharedBuf` to live in `tessl` where `SharedBuf` is defined (consistent with PROJECT.md:68).
  2. `MappedBuffer::host_read` returns `Box<dyn HostLease>`, requiring `'static + Send + Sync`. Existing `HostRead<'a>` borrows `&'a SharedBuf`. Refactoring `SharedBuf` to wrap `Arc<SharedBufInner>` cleanly solves this without unsafe code.
  3. `tessl::runtime::host_fence::HostFence` already implements all primitives required by `ojas_core::hal::Timeline`. Implementing `Timeline for HostFence` directly bridges `timeline()` on `MappedBuffer`.
  4. Retaining `_pooled: Option<Arc<PooledBuffer>>` in `SharedBufInner` ensures that the buffer pool will never prematurely recycle unified memory while an `ojas_core::Tensor` or `HostLease` is active.
- **Unexplored areas**: none (investigation scope complete).

## Key Decisions Made
- Bridge location: `tessl/src/shared_buf.rs` and `tessl/src/runtime/host_fence.rs`.
- Preserved `#![forbid(unsafe_code)]` in `ojas-core` and `ojas-metal`: zero unsafe added.

## Artifact Index
- /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_2/handoff.md — Final handoff report
