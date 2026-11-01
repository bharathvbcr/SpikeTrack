## 2026-10-31T18:22:58Z
You are Explorer 1 for Milestone 2: Unified Architecture & Shared Memory Pools.
Your working directory is: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_1/

MANDATORY FIRST STEPS:
1. Read /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md
2. Read /Users/bharath/Code/research/SpikeTrack/PROJECT.md (especially ## Architecture, ## Interface Contracts, and Features 7, 9)
3. Read /Users/bharath/Code/research/ojas/AGENTS.md and /Users/bharath/Code/research/tessl/AGENTS.md (or codebase conventions)

Your specific technical scope:
Investigate and design the exact HAL trait definitions and unified buffer integration between ojas-core and tessl:
1. Feature 7: HAL Traits in ojas-core:
   - Inspect ojas-core/src/tensor.rs, ojas-core/src/lib.rs, ojas-core/src/error.rs.
   - Design pure Rust HAL traits (#![forbid(unsafe_code)], zero external dependencies):
     - pub trait MappedBuffer: Any + Send + Sync + fmt::Debug: methods byte_len(&self) -> usize, host_ptr(&self) -> *mut u8, gpu_address(&self) -> u64, timeline(&self) -> &dyn Timeline, as_any(&self) -> &dyn Any.
     - pub trait HostLease: RAII guard for reading/writing mapped CPU buffers with fence synchronization.
     - pub trait Timeline: Send + Sync: current_value(&self) -> u64, wait_for(&self, value: u64, timeout: Option<Duration>) -> Result<(), AllocError>.
     - AllocError: typed enum (Exhausted { requested, available }, DeviceLost, InvalidAlignment) that cleanly maps into OjasError::CapacityExceeded.
2. Feature 9: Zero-copy memory representation on Tensor:
   - Inspect how ojas-core::Tensor currently holds memory (Payload::Owned, Payload::View, etc.).
   - Design Payload::Mapped(Arc<dyn MappedBuffer>).
   - Specify zero-copy operations on Apple Silicon (where MTLResourceOptions::StorageModeShared is coherent).
   - Design to_host() and upload() so that on mapped storage they become timeline fences rather than memory copies.
   - Specify how tessl::SharedBuf implements MappedBuffer.
3. Provide precise file paths, exact Rust struct and trait definitions, call sites, and step-by-step implementation instructions for Worker.

Output:
Write your structured findings to /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_1/handoff.md following standard handoff format (Observation, Logic Chain, Caveats, Conclusion, Implementation Plan).
Send a completion message back when done.
