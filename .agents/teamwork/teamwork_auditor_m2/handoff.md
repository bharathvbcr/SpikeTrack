# Forensic Integrity Audit Report: Milestone 2 — Unified Architecture & Shared Memory Pools

**Work Product**: Milestone 2 Deliverables across `ojas` and `tessl`  
**Profile**: General Project (Integrity Mode: `development` per `ORIGINAL_REQUEST.md`)  
**Verdict**: **CLEAN**

---

## 1. Observation

Direct empirical observations across the audited codebases (`/Users/bharath/Code/research/ojas` and `/Users/bharath/Code/research/tessl`):

1. **Safe Rust & Invariant Enforcement**:
   - `ojas-core/src/lib.rs:9`: `#![forbid(unsafe_code)]` is strictly set.
   - Comprehensive grep confirms **zero** `unsafe` code blocks across `ojas-core`.
   - The singular `unsafe impl Send + Sync` for `SharedBuf` in `tessl/src/shared_buf.rs:172-175` is documented, audited, and strictly conforms to the invariant exception explicitly sanctioned in `ojas/AGENTS.md`.

2. **Genuine Code Deletion**:
   - `ojas-metal/src/gpu.rs` (3,760 lines) was genuinely deleted in commit `b8fafb1` (`Merge branch 'gp/ug-tiny-step' into main`).
   - The required bias gradient helper (`gate_dbias_threads`) was relocated cleanly to `ojas-metal/src/device.rs:188-212`, and validated with exact bit-for-bit parity against reference in `ojas-metal/tests/gate_dbias.rs`.

3. **No Facades, Stubs, or Hardcoded Test Mocks**:
   - Search across `ojas-core`, `tessl`, and `ojas-metal` for `assert!(true)`, `unimplemented!()`, and `todo!()` returned 0 matches.
   - Implementations of HAL traits (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`) in `ojas-core/src/hal.rs:1-495` contain complete, typed logic with exhaustive reflection tests verifying all variants.

4. **Zero-Copy Unified Memory Architecture**:
   - In `tessl/src/shared_buf.rs:178-200`, `SharedBuf` takes a `StorageModeShared` buffer and maps `ptr = mtl.contents().cast::<u8>()`.
   - Host leases (`HostRead`, `HostWrite`) yield direct slice references (`std::slice::from_raw_parts`) over unified memory with zero background memcpy.
   - In `ojas-core/src/tensor.rs:1131-1155`, `Tensor::host_view()` delegates directly to `MappedBuffer::host_read()` returning a fenced `HostLease`.
   - In `ojas-core/src/tensor.rs:1171-1179`, `Tensor::to_host()` on a `Payload::Mapped` tensor waits on the timeline fence and returns a cheap clone without memory allocation or copy.

5. **Fail-Closed Placement Enforcement**:
   - `Tensor::host_window()` (`tensor.rs:760-767`) and `Tensor::host_data()` (`tensor.rs:796-812`) explicitly check `self.device()`.
   - For `Payload::Mapped(buf)`, `self.device()` returns `Some(buf.backend())` (`Some(BackendId::Metal)`).
   - Calling CPU slice accessors (`f32_slice()`, `f32_slice_mut()`, `to_f32_vec()`, `write_f32()`, `to_ne_bytes()`) on a mapped or device tensor unconditionally returns `Err(OjasError::Placement { op, expected: None, found: Some(BackendId::Metal) })`. There is no silent host fallback.

6. **Root Budget & Freelist Trim Ladder**:
   - `ojas-core::Budget` (`budget.rs:19-280`) maintains atomic accounting (`AtomicU64::try_update`) for reservations and peaks.
   - `BudgetTrimHandler` (`budget.rs:27-29`) is invoked on `CapacityExceeded` before retrying allocation once.
   - `tessl::PoolLedger` (`tessl/src/pool_ledger.rs:99-115`) and `BudgetLedger` (`ojas-metal/src/pool_ledger.rs:53-165`) charge live and cached balances separately; under memory pressure, the pool trims the free list and retries the allocation, as verified by `runtime::pool_ledger_tests::a_refused_charge_trims_the_free_list_and_retries_once`.

7. **Empirical Test Suite Execution**:
   - `tessl` tests:
     - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime`: 4 passed, 0 failed.
     - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib pool_ledger`: 15 passed, 0 failed.
     - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib shared_buf`: 14 passed, 0 failed.
     - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test adversarial_shared_stress`: 2 passed, 0 failed.
   - `ojas-core` tests:
     - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core`: 234 passed (203 lib + 31 integration), 0 failed.
   - `ojas-io` tests:
     - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-io`: 129 passed, 0 failed.
   - `ojas-metal` tests:
     - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger --test gate_dbias`: 5 passed, 0 failed.
     - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test adamw_msl_parity`: 2 passed, 0 failed.
     - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test adversarial_pool_stress`: 2 passed, 0 failed.
   - `ojas-oracle` 4-tier E2E production suite:
     - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier1_features --test e2e_tier2_boundaries --test e2e_tier3_combinatorial --test e2e_tier4_production`: 142 passed (65 Tier 1 + 65 Tier 2 + 8 Tier 3 + 4 Tier 4), 0 failed.
   - Static documentation check:
     - `go run scripts/sitegen/main.go -check`: all 54 files up to date.

---

## 2. Logic Chain

1. **Absence of Shortcuts**: Inspection of all modified source files (`hal.rs`, `tensor.rs`, `budget.rs`, `backend.rs`, `safetensors.rs`, `half.rs`, `npy.rs`, `shared_buf.rs`, `pool_ledger.rs`) demonstrates genuine, non-trivial implementations. All error returns, type conversions, and boundary checks are fully realized. No hardcoded expected outputs, mocks, or fake returns exist.
2. **Memory Sharing Integrity**: `tessl::SharedBuf` and `ojas_core::Payload::Mapped` provide zero-copy memory access to unified memory by exposing raw host pointers via timeline-fenced leases. Data access is synchronized via commit clocks and poison latches rather than host memory duplication.
3. **Strict Placement Invariant**: Because `Payload::Mapped` reports its backend device ID (`BackendId::Metal`), attempts to access data through standard CPU slice methods (`f32_slice()`) immediately fail closed with `OjasError::Placement`. Host access is allowed exclusively via explicit `host_view()` leases or `to_host()`, ensuring memory safety without hidden computations.
4. **Authoritative Ledger & Dynamic Trim**: `Budget` acts as the root ledger across host and device allocations. The `BudgetTrimHandler` callback successfully drives cache eviction under memory pressure before retrying reservations, preventing premature out-of-memory failures while preserving strict accounting.
5. **Code Consolidation**: Duplicated SafeTensors, NPY, and JSON parsing logic was successfully consolidated into `ojas-io`, and 3,760 lines of duplicate GPU kernels in `gpu.rs` were completely excised.
6. **Empirical Verification**: All required test commands, edge-case tests, parity tests, and full 4-tier E2E integration suites execute and pass without regressions.

---

## 3. Caveats

- An untracked test file `ojas-metal/tests/adversarial_pool_stress.rs` was created in the working tree by a concurrent challenger agent. While the test itself passes (2/2 passed), because it was untracked, `./scripts/check_tracked_targets.sh` reported 1 untracked Cargo target. The committed code on `main` (`3ac4428`) is clean and has 0 untracked files in git index.
- Apple Silicon Metal tests require a local macOS environment with Metal 4 support; all tests were verified directly on the host machine.

---

## 4. Conclusion

The Milestone 2 work product satisfies all integrity, architectural, and safety requirements. There are no facade implementations, no hardcoded results, and no hidden memory copies. Safe Rust is strictly preserved in `ojas-core`. Memory pools are shared across CPU and GPU with genuine zero-copy UMA and timeline-fenced leases. All verification suites pass.

**Verdict**: **CLEAN**

---

## 5. Verification Method

To independently reproduce the forensic verification:

```bash
# 1. Verify tessl shared runtime and memory pool tests:
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib pool_ledger
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib shared_buf

# 2. Verify ojas-core and ojas-io suites:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-io

# 3. Verify ojas-metal integration and parity tests:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger --test gate_dbias
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test adamw_msl_parity

# 4. Verify 4-tier E2E production suite:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier1_features --test e2e_tier2_boundaries --test e2e_tier3_combinatorial --test e2e_tier4_production

# 5. Verify static documentation check:
cd /Users/bharath/Code/research/ojas && go run scripts/sitegen/main.go -check
```
