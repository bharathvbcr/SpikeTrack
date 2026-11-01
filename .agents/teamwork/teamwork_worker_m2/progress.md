# Progress — Milestone 2 Worker

Last visited: 2026-10-31T19:23:00Z

## Status
- [x] Phase 1: HAL Traits & Zero-Copy Tensor in `ojas-core` (Features 7 & 9).
  - Extended `AllocError` with `InvalidAlignment`.
  - Added `Timeline` methods (`current_value`, `wait_for`).
  - Added `MappedBuffer` methods (`host_ptr`, `gpu_address`, `timeline`).
  - Added `Payload::Mapped`, zero-copy `to_host`, `host_view`, and fail-closed placement checks in `ojas-core/src/tensor.rs`.
  - Added unit tests for mapped tensor. All 206 unit & integration tests passing in `ojas-core`.
- [x] Phase 2: Budget Root Ledger, Trim Ladder & Placement (Features 8 & 10).
  - Implemented `BudgetTrimHandler`, `trim_handler` in `Inner`, `set_trim_handler`, `trim_handler` accessors, and memory pressure retry loop in `Budget::try_reserve`.
  - All 21 budget tests passing in `ojas-core`.
- [x] Phase 3: Retire Duplicate IO Readers in Tessl & Ojas Branch Merges (Feature 11).
  - Ported `CheckpointView`, `TensorSource`, `INDEX_FILE_NAME` and in-place zero-copy readers to `ojas-io/src/safetensors.rs`.
  - Ported `widen_in_place` to `ojas-io/src/half.rs`.
  - Added `write_npy_f32`, `transpose_last2`, `read_npy_with_limit`, `NpyArray` to `ojas-io/src/npy.rs`.
  - Re-exported facades in `tessl/src/safetensors.rs` and `tessl/src/npy.rs`.
  - All 6 `tessl` branches merged: `ug/const-dedup`, `ug/alloc-error`, `ug/bucket-dedup`, `ug/tessl-shared`, `ug/sharedbuf`, `ug/pool-ledger`.
  - All 10 `ojas` branches merged: `gp/ug-alloc-error`, `gp/ug-bucket-dedup`, `gp/ug-const-dedup`, `gp/ug-adamw-scalars`, `gp/ug-adamw-msl`, `gp/ug-pool-ledger`, `gp/ug-sharedbuf`, `gp/ug-readback-clean`, `gp/ug-io-sink`, `gp/ug-tiny-step`.
- [x] Phase 4: BufferPool Child Budget, SharedBuf & Retiring dead gpu.rs (Features 8 & 12).
  - Retired 3,735 lines of duplicate `gpu.rs` in `ojas-metal`.
  - Preserved `gate_dbias_threads` in `device.rs` with bit-identical row sum ordering.
  - Wired `BudgetLedger` over child budgets for buffer pool memory tracking.
- [x] Phase 5: AdamW Scalars & Parity Verification (Features 8 & 12).
  - Unified `AdamWScalars` in `ojas-core/src/backend.rs`.
  - Shared `qwen35_adamw_math.h` with `ojas_backend.metal`.
  - Verified bit-for-bit parity in `adamw_msl_parity.rs`.
- [x] Full workspace build and test verification:
  - `tessl` checks and tests pass (`shared_runtime`, `pool_ledger`, `shared_buf`).
  - `ojas-core` tests pass (206 unit tests, `adamw_scalars`).
  - `ojas-io` tests pass (79 unit tests + 59 integration tests).
  - `ojas-metal` tests pass (`pool_ledger`, `gate_dbias`, `adamw_msl_parity`).
  - 4-Tier E2E Production Suite passes (Tier 1: 65, Tier 2: 65, Tier 3: 8, Tier 4: 4).
  - `fmt_check.sh`, `check_tracked_targets.sh`, and `sitegen -check` pass.
- [x] Handoff report completed: `.agents/teamwork/teamwork_worker_m2/handoff.md`.
