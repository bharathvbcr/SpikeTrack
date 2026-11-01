## 2026-10-31T18:33:08Z
You are Worker for Milestone 2: Unified Architecture & Shared Memory Pools (Features 7, 8, 9, 10, 11, 12).
Your working directory is: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m2/

MANDATORY FIRST STEPS:
1. Read /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md
2. Read /Users/bharath/Code/research/SpikeTrack/PROJECT.md
3. Read the 3 Explorer handoff reports:
   - /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_1/handoff.md (HAL Traits & Zero-Copy Tensor)
   - /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_2/handoff.md (Budget Root Ledger, Trim Ladder, Placement)
   - /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_3/handoff.md (Retiring Duplicate IO Readers & Dead Training Code)
4. Read codebase guidelines: /Users/bharath/Code/research/ojas/AGENTS.md and user global rules (Reuse, unify, shrink; Safe Rust; Chesterton's fence).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

SCOPE & IMPLEMENTATION PLAN:

Phase 1: HAL Traits & Zero-Copy Tensor in ojas-core (Features 7 & 9)
1. In ojas-core/src/hal.rs:
   - Extend AllocError with InvalidAlignment { requested: usize, alignment: usize }, update Display, is_retryable, and From<AllocError> for OjasError.
   - Update Timeline trait: add current_value(&self) -> u64 and wait_for(&self, value: u64, timeout: Option<Duration>) -> Result<(), AllocError>.
   - Update MappedBuffer: add host_ptr(&self) -> *mut u8, gpu_address(&self) -> u64, timeline(&self) -> &dyn Timeline.
   - Update tests in hal.rs (default_calls) so that the compile-time reflection test the_hal_list_names_every_trait_method passes.
2. In ojas-core/src/tensor.rs:
   - Add Payload::Mapped(Arc<dyn MappedBuffer>).
   - Update Storage::len to support Payload::Mapped.
   - Implement Tensor::from_mapped and Tensor::from_mapped_reserved.
   - Add Tensor::mapped_buffer(&self) -> Option<&Arc<dyn MappedBuffer>>.
   - Implement Tensor::to_host(&self, budget: &Budget): on Payload::Mapped, wait on timeline fence without memcpy, returning zero-copy representation.
   - Implement Tensor::host_view(&self) -> Result<Box<dyn HostLease>, OjasError>.
   - Maintain pure Rust: #![forbid(unsafe_code)] in ojas-core, 0 external dependencies.

Phase 2: Budget Root Ledger, Trim Ladder & Placement (Features 8 & 10)
1. In ojas-core/src/budget.rs:
   - Define pub trait BudgetTrimHandler: Send + Sync { fn trim_memory(&self, needed_bytes: u64) -> u64; }.
   - In Inner: add trim_handler: std::sync::RwLock<Option<Arc<dyn BudgetTrimHandler>>>.
   - Implement set_trim_handler(&self, handler: Arc<dyn BudgetTrimHandler>) and trim_handler(&self).
   - In try_reserve: if charge fails with CapacityExceeded, query trim_handler, and if memory was freed, retry charge before returning CapacityExceeded.
2. In ojas-core/src/tensor.rs:
   - Enforce fail-closed placement on CPU accessors: host_window, host_data, f32_slice, bf16_slice, write_f32 return Err(OjasError::Placement) for any tensor with device placement (Payload::Device or Payload::Mapped with device backend).

Phase 3: Retire Duplicate IO Readers in Tessl (Feature 11)
1. In ojas-io:
   - Port CheckpointView, TensorSource, and INDEX_FILE_NAME into ojas-io/src/safetensors.rs.
   - Add in-place zero-copy readers in ojas-io::SafeTensors: read_f32_into, read_bf16_bits_into, read_u32_into.
   - Port widen_in_place into ojas-io/src/half.rs.
   - In ojas-io/src/npy.rs: add write_npy_f32, transpose_last2, read_npy_with_limit, NpyArray wrapper.
2. In tessl/Cargo.toml:
   - Add ojas-core = { path = "../ojas/ojas-core" } and ojas-io = { path = "../ojas/ojas-io" }.
3. In tessl/src/:
   - Rewrite tessl/src/config_json.rs to parse using ojas_io::parse_json and ojas_io::JsonValue.
   - Delete tessl/src/json.rs and tessl/src/plain.rs; remove their mod declarations from tessl/src/lib.rs.
   - Turn tessl/src/safetensors.rs and tessl/src/npy.rs into clean re-export facades over ojas_io.
   - In ojas-qwen35/src/names.rs: import CheckpointView directly from ojas_io.

Phase 4: BufferPool Child Budget, SharedBuf & Retiring dead gpu.rs (Features 8 & 12)
1. In tessl/src/:
   - Create tessl/src/shared_buf.rs implementing DeviceBuffer and MappedBuffer for MTLBuffer using StorageModeShared.
   - In tessl/src/runtime.rs: connect BufferPool to Budget child, implement BudgetTrimHandler on BufferPool to evict idle cached buffers on pressure.
2. In ojas-metal:
   - Move gate_dbias_threads into ojas-metal/src/device.rs.
   - Delete dead 3,736-line toy model in ojas-metal/src/gpu.rs (remove pub mod gpu; from ojas-metal/src/lib.rs).
   - In ojas-metal/src/device.rs: replace string scraping classify_alloc with pattern matching over typed AllocError.
   - In ojas-metal/src/backend.rs: wrap kernel outputs with Tensor::from_mapped_reserved; in upload(), short-circuit when tensor.mapped_buffer() is already on Metal.

Phase 5: AdamW Scalars & Parity Verification (Features 8 & 12)
1. In ojas-core/src/backend.rs: expose adamw_kernel_scalars(config, step, grad_scale) -> Result<[f32; 8], OjasError>.
2. Run full builds and test verification across all affected targets:
   - cargo test -p ojas-core
   - cargo test -p ojas-io
   - cargo check --manifest-path ../tessl/Cargo.toml --lib
   - cargo test --manifest-path ../tessl/Cargo.toml --test bert_model --test embedgemma2_model
   - cargo check -p ojas-metal
   - cargo test -p ojas-oracle --test e2e_tier4_production
   - Full 4-tier E2E suite: cargo test -p ojas-oracle --test e2e_tier1_features --test e2e_tier2_boundaries --test e2e_tier3_combinatorial --test e2e_tier4_production
   - scripts/ci_local.sh or dev_fast_check.sh
   - scripts/sitegen/main.go -check

Write a comprehensive handoff report to /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m2/handoff.md detailing:
- Observation (what changed, lines added/removed, commands run)
- Logic chain
- Caveats
- Conclusion
- Verification results and commands.
Send a message when complete.

## 2026-10-31T18:50:09Z
**Context**: User follow-up instruction & branch auditing
**Content**: User requested to check other branches across the repositories and continue their work. A dedicated Branch & Worktree Audit Explorer (conversation ID `9464a2f8-b032-412e-8d9a-ec1cbe6c8ab0`) has been dispatched to audit all branches, worktrees, and stashes across `ojas`, `tessl`, and `SpikeTrack` to identify in-flight work. Please continue executing your current Milestone 2 implementation phases as planned; the audit report will be shared for integrating relevant branch code into subsequent phases and milestones.
**Action**: Continue executing Milestone 2 phases.

## 2026-10-31T18:59:11Z
**Context**: Milestone 2 Implementation & Branch Integration
**Content**: The Branch Audit Explorer has delivered its report at `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_branches/handoff.md`. Complete, verified implementations for Milestone 2 exist on the following clean-merging branches:
In `tessl`:
- `ug/const-dedup`, `ug/alloc-error`, `ug/bucket-dedup`, `ug/tessl-shared`, `ug/sharedbuf`, `ug/pool-ledger`
In `ojas`:
- `gp/ug-alloc-error`, `gp/ug-bucket-dedup`, `gp/ug-const-dedup`, `gp/ug-adamw-scalars`, `gp/ug-adamw-msl`, `gp/ug-pool-ledger`, `gp/ug-sharedbuf`, `gp/ug-readback-clean`, `gp/ug-io-sink`, `gp/ug-tiny-step` (which cleanly retires 3,735 lines of legacy `gpu.rs`).

All 16 branches were simulated with `git merge-tree` and verified to merge with zero conflicts. Please inspect §4.1 Phase 1 of `handoff.md` and merge/integrate these branches to complete Milestone 2 Phases 3, 4, and 5.
**Action**: Integrate the Milestone 2 branches across `tessl` and `ojas`, run the full test suite, and deliver your handoff report.

## 2026-10-31T19:08:00Z
**Context**: Milestone 2 Implementation & Branch Integration
**Content**: Heartbeat status check. Please update your progress.md with your current progress across Phases 3, 4, and 5 and branch integration.
**Action**: Update progress.md and continue execution.

## 2026-10-31T19:09:55Z
**Context**: Milestone 2 Implementation Status
**Content**: Please send an immediate status update on your progress: Have you integrated the Phase 3/4 branches? What is your current build/test state? Please update progress.md.
**Action**: Update progress.md and respond with current status.
