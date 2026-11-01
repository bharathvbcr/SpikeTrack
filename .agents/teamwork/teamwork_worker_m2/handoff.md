# Handoff Report: Milestone 2 — Unified Architecture & Shared Memory Pools (Features 7, 8, 9, 10, 11, 12)

## 1. Observation
- **Repositories Modified**:
  - `/Users/bharath/Code/research/ojas` (Git commits on `main`: `0e1c6bc`, `91bdd6f`, `e197e62`, `2ed3a67`, `900dd05`, `b8fafb1`, `49b2692`, `3ac4428`)
  - `/Users/bharath/Code/research/tessl` (Git commits on `main`: `81d59ce`, `1a73462`, `3c1f338`, `3a037da`)
- **Key Changes Implemented**:
  - **`ojas-core/src/hal.rs`**: Added `AllocError::InvalidAlignment`, `Timeline` methods (`current_value`, `wait_for`), and `MappedBuffer` methods (`host_ptr`, `gpu_address`, `timeline`). Verified safe reflection harness covers all variants.
  - **`ojas-core/src/tensor.rs`**: Added `Payload::Mapped` supporting zero-copy `to_host` and `host_view`. Enforced fail-closed placement checks on CPU accessors (`f32_slice`, `to_f32_vec`, `host_data`, `host_data_mut`, `write_f32`) returning `Err(OjasError::Placement { found: Some(BackendId::Metal), .. })` for device and mapped tensors.
  - **`ojas-core/src/budget.rs`**: Implemented `BudgetTrimHandler`, `trim_handler` in `Inner`, `set_trim_handler`, `trim_handler` accessors, and memory pressure retry loop in `Budget::try_reserve`.
  - **`ojas-core/src/backend.rs`**: Added `AdamWScalars` unifying the host-side f64 scalar math and f32 scalar packing across CPU, Metal, and WGPU backends.
  - **`ojas-io`**: Added `widen_in_place` in `half.rs`; added in-place zero-copy readers (`read_f32_into`, `read_bf16_bits_into`, `read_u32_into`), `INDEX_FILE_NAME`, `TensorSource` trait, and multi-shard `CheckpointView` in `safetensors.rs`; added `NpyArray`, `read_npy`, `read_npy_with_limit`, `transpose_last2`, `write_npy_f32`, and `seek_noop` in `npy.rs`.
  - **`tessl` Milestone 2 Integrations**: Merged all 6 sprint branches:
    - `ug/const-dedup`: exported layout constants (`CONV_ROWS_PER_BLOCK`, `METAL4_CONST_ARENA_BYTES`).
    - `ug/alloc-error`: `AllocFailure` taxonomy (`Limit`, `Exhausted`) and `is_allocation_failure`.
    - `ug/bucket-dedup`: deduplicated bucket rounding across runtimes.
    - `ug/tessl-shared`: `Shared`, `Timeline`, and monotone commit clock without unsafe.
    - `ug/sharedbuf`: `SharedBuf`, `SharedAlloc`, `HostFence` for zero-copy CPU/GPU sharing with timeline-fenced leasing.
    - `ug/pool-ledger`: `PoolLedger`, `PoolResident`, `reclaim`, and trim hooks.
  - **`ojas` Milestone 2 Integrations**: Merged all 10 sprint branches:
    - `gp/ug-alloc-error`, `gp/ug-bucket-dedup`, `gp/ug-const-dedup`, `gp/ug-adamw-scalars`, `gp/ug-adamw-msl`, `gp/ug-pool-ledger`, `gp/ug-sharedbuf`, `gp/ug-readback-clean`, `gp/ug-io-sink`, and `gp/ug-tiny-step`.
    - Retired 3,735 lines of legacy `ojas-metal/src/gpu.rs`, moving `gate_dbias_threads` to `device.rs`.
    - Shared tessl's `qwen35_adamw_math.h` with `ojas_backend.metal` and validated bit-for-bit parity in `ojas-metal/tests/adamw_msl_parity.rs`.

---

## 2. Logic Chain
1. **Zero-Copy & Fail-Closed Placement**: By representing UMA buffers via `Payload::Mapped` and providing `host_view()` returning `Box<dyn HostLease>`, host CPU threads can lease buffer slices with zero copies while checking that the commit timeline fence has signaled. Any direct host slice accessor (`f32_slice()`, `host_data()`) on a mapped or device tensor immediately returns `Err(OjasError::Placement)` rather than allowing un-fenced or unsynchronized data races.
2. **Authoritative Root Ledger**: Memory reservations flow through `ojas-core::Budget`. With `BudgetTrimHandler`, when capacity is exceeded, the budget invokes registered trim handlers (such as `PoolLedger::trim_free_list`) before retrying the reservation once, enabling the device buffer pool to free cached buffers under memory pressure.
3. **Shared Buffer Pool & Thread-Safe Timeline**: `HostFence` wraps Metal shared event wait logic and timeout streak counters into a `Send + Sync` primitive without unsafe code, sharing the poison latch and commit clock across multiple runtimes and host threads.
4. **Code Deduplication & Architecture Consolidation**: Porting unified readers (`CheckpointView`, `TensorSource`, `NpyArray`) to `ojas-io` eliminates duplicate file parsing code across crates. Removing the legacy `gpu.rs` (3,735 lines) eliminates duplicate kernels, unifying all attention and gating kernels under canonical `device.rs` and `tessl`.

---

## 3. Caveats
- No caveats. All 16 branches across `ojas` and `tessl` have been integrated and verified. Safe Rust (`#![forbid(unsafe_code)]`) is strictly maintained in `ojas-core` with 0 external dependencies.

---

## 4. Conclusion
Milestone 2 is complete. Code duplication has been eliminated, memory pools are unified with zero-copy UMA and timeline-fenced leases, `ojas-core::Budget` serves as the authoritative root ledger with automatic trim ladders, fail-closed placement is enforced, and all unit, integration, parity, and E2E suites pass across both repositories.

---

## 5. Verification Method
To independently verify:
```bash
# 1. Verify tessl shared runtime and buffer pool tests:
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib pool_ledger
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib shared_buf
cargo check --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --all-targets

# 2. Verify ojas core and io suites:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core --lib
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core --test adamw_scalars
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-io

# 3. Verify ojas-metal integration and parity tests:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test gate_dbias
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test adamw_msl_parity

# 4. Verify 4-tier E2E production suite:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier1_features
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier2_boundaries
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier3_combinatorial
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier4_production

# 5. Verify format, tracking, and documentation:
/Users/bharath/Code/research/ojas/scripts/fmt_check.sh
/Users/bharath/Code/research/ojas/scripts/check_tracked_targets.sh
cd /Users/bharath/Code/research/ojas && go run scripts/sitegen/main.go -check
```
