# Milestone 2 Iteration 2 Gate Verification: Review and Adversarial Challenge Report

**Date**: 2026-10-31  
**Reviewer**: Reviewer 2 (Replacement Generation 2)  
**Roles**: `reviewer`, `critic`  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_reviewer_m2_it2_2_gen2`  
**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**  
**Integrity Audit**: **CLEAN (0 VIOLATIONS DETECTED)**

---

## 1. Observation

All observations are directly grounded in inspected files, lines, and executed tool commands.

### Observation 1: Public Export of `BudgetTrimHandler`
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/lib.rs` (line 48)
  ```rust
  pub use budget::{Budget, BudgetTrimHandler, Reservation, Scratch};
  ```
- **Definition**: `/Users/bharath/Code/research/ojas/ojas-core/src/budget.rs` (lines 18–20)
  ```rust
  pub trait BudgetTrimHandler: Send + Sync + 'static {
      fn trim_memory(&self, needed_bytes: u64) -> u64;
  }
  ```
- **Finding**: `BudgetTrimHandler` is exported at the crate root of `ojas-core`. External and downstream crates (`ojas-metal`, `tessl`, downstream callers) can directly import `use ojas_core::BudgetTrimHandler;`.

### Observation 2: Fast-Path Rejection via `can_ever_fit`
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/budget.rs` (lines 194–227)
  ```rust
  fn can_ever_fit(&self, bytes: u64) -> bool {
      let mut node = Some(self);
      while let Some(budget) = node {
          if bytes > budget.inner.cap_bytes {
              return false;
          }
          node = budget.inner.parent.as_ref();
      }
      true
  }
  ```
  In `try_reserve`:
  ```rust
  if let Err(err) = self.charge(bytes) {
      if let OjasError::CapacityExceeded { .. } = &err {
          if self.can_ever_fit(bytes) {
              if let Some(handler) = self.trim_handler() {
                  let freed = handler.trim_memory(bytes);
                  if freed > 0 && self.charge(bytes).is_ok() {
                      return Ok(Reservation {
                          bytes,
                          budget: self.clone(),
                      });
                  }
              }
          }
      }
      return Err(err);
  }
  ```
- **Finding**: Oversized allocations exceeding any ancestor budget capacity (`bytes > cap_bytes`) fail immediately with `CapacityExceeded` without invoking the trim handler or issuing cross-thread/device commands.

### Observation 3: Freelist Cache Trimming in `tessl` and `ojas-metal`
- **`tessl` Trimming Implementation**: `/Users/bharath/Code/research/tessl/src/runtime.rs` (lines 1378–1384)
  ```rust
  pub fn trim_free_list(&self) -> usize {
      if let Ok(mut pool) = self.pool.try_lock() {
          pool.trim_free_list()
      } else {
          0
      }
  }
  ```
  Uses `try_lock()` to avoid deadlocking if the calling thread or another worker currently holds the pool lock.
- **`ojas-metal` Device Protocol & IPC**:
  - `Cmd::TrimPool` defined at `ojas-metal/src/link.rs:991`.
  - `Reply::Trimmed(u64)` defined at `ojas-metal/src/link.rs:1043`.
  - `DeviceWorker` in `ojas-metal/src/device.rs` (lines 1040–1055):
    ```rust
    Cmd::TrimPool => {
        let before = self.pool_memory_used();
        self.drop_pool();
        self.trim_pool();
        let after = self.pool_memory_used();
        Reply::Trimmed(before.saturating_sub(after))
    }
    ```
  - `MetalTrimHandler` in `ojas-metal/src/backend.rs` (lines 205–240) registers as root `BudgetTrimHandler`:
    ```rust
    budget.set_trim_handler(Arc::new(MetalTrimHandler::new(link.clone())));
    ```
  - `BudgetLedger` in `ojas-metal/src/pool_ledger.rs` (lines 60–75) installs `NoOpTrim` on `live` and `cache` child budgets, preventing recursive trim commands when the device thread allocates.

### Observation 4: Bridging `SharedBuf` to `MappedBuffer`, `DeviceBuffer`, and `Timeline`
- **`Timeline` Implementation on `HostFence`**: `/Users/bharath/Code/research/tessl/src/runtime/host_fence.rs` (lines 147–193):
  ```rust
  impl ojas_core::Timeline for HostFence {
      fn current_value(&self) -> u64 { self.current_value() }
      fn wait_for(&self, value: u64, timeout: Option<std::time::Duration>) -> Result<(), ojas_core::AllocError> { ... }
  }
  ```
- **`MappedBuffer` Implementation on `SharedBuf`**: `/Users/bharath/Code/research/tessl/src/shared_buf.rs` (lines 535–680):
  - Exposes `byte_len()`, `host_ptr()`, `gpu_address()`, and `timeline()`.
  - Returns `SharedHostReadLease` and `SharedHostWriteLease` implementing `HostLease` and `HostLeaseMut`.
  - Holds `_pooled: Option<Arc<PooledBuffer>>` inside `SharedBufInner`, ensuring that converting `SharedAlloc` to `SharedBuf` or dropping the outer allocation keeps the underlying pooled memory valid for active leases and tensors.

### Observation 5: Independent Verification Command Outputs
1. **tessl Unit & Integration Test Suites**:
   - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --release --lib shared_buf`:
     - Exit code: 0
     - Result: `14 passed; 0 failed; 0 ignored` (finished in 0.12s)
   - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --release --test docs_name_real_tools`:
     - Exit code: 0
     - Result: `7 passed; 0 failed; 0 ignored` (finished in 3.01s)
   - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --release --test shared_runtime -- --test-threads=1`:
     - Exit code: 0
     - Result: `5 passed; 0 failed; 0 ignored` (including `adversarial_shared_buf_uma_and_placement_invariants`, finished in 0.09s)
   - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --release --test stress_trim_concurrency -- --test-threads=1`:
     - Exit code: 0
     - Result: `3 passed; 0 failed; 0 ignored` (finished in 5.03s)

2. **ojas Full Local CI Gate Suite (`cd /Users/bharath/Code/research/ojas && ./scripts/ci_local.sh`)**:
   - `fmt_check.sh`: PASSED (exit code 0)
   - `check_tracked_targets.sh`: PASSED (`all 368 target sources are tracked`, exit code 0)
   - `cargo clippy --workspace --all-targets -- -D warnings`: PASSED (0 warnings/errors, exit code 0)
   - `cargo deny check`: PASSED (`advisories ok, bans ok, licenses ok, sources ok`, exit code 0)
   - CUDA clippy checks (`ojas-cuda`): PASSED (exit code 0)
   - `simd_bench.sh`: PASSED (exit code 0)
   - Go sitegen & benchreport check: PASSED (exit code 0)
   - Go test suite: PASSED (exit code 0)
   - Full workspace release tests (`cargo test --workspace --release -- --test-threads=1`): PASSED (100% passed across all 19 workspace crates and integration suites)
   - Rustdoc intra-doc link check (`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`): PASSED (exit code 0)
   - Metal stub clippy check (`cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings`): PASSED (exit code 0)
   - Script exit: code 0 (`ci:local OK`)

---

## 2. Logic Chain

1. **API Ergonomics & Downstream Compilability (Observation 1)**:
   - Re-exporting `BudgetTrimHandler` in `ojas-core/src/lib.rs` exposes the trait at `ojas_core::BudgetTrimHandler`.
   - This resolves compiler errors in downstream crates (`tessl`, `ojas-metal`, and test binaries) without exposing internal module layout.

2. **Zero-Overhead Rejection of Unsatisfiable Allocations (Observation 2)**:
   - When an allocation request exceeds the maximum capacity of any parent budget in the hierarchy, cache trimming cannot free enough memory.
   - `can_ever_fit(bytes)` traverses parent budgets and verifies `bytes <= budget.inner.cap_bytes`.
   - Bypassing the trim handler when `can_ever_fit` returns false guarantees that impossible requests fail immediately without triggering GPU synchronizations or worker thread IPC round-trips.

3. **Safe, Deadlock-Free Trimming Under Memory Pressure (Observation 3)**:
   - When memory pressure occurs, `Budget::try_reserve` delegates to `MetalTrimHandler::trim_memory`.
   - `MetalTrimHandler` sends `Cmd::TrimPool` via `Link` to the dedicated `DeviceWorker` thread.
   - Inside `tessl`, `GpuRuntime::trim_free_list()` uses non-blocking `try_lock()`, returning `0` if the lock is held, completely avoiding ABBA deadlocks.
   - Child budgets in `BudgetLedger` use `NoOpTrim`, preventing device allocations from triggering recursive command dispatch.

4. **Zero-Copy Host/Device Coherence and Lifetime Safety (Observation 4)**:
   - `SharedBuf` implements `MappedBuffer`, exposing Metal shared memory directly to `Tensor::from_mapped`.
   - `SharedBufInner` retains `Option<Arc<PooledBuffer>>`, preventing the buffer from being returned to the freelist while `SharedBuf` or active `HostLease` / `HostLeaseMut` handles exist.
   - Sequence numbering (`last_use` / `last_write`) on `HostFence` ensures that host reads and writes wait for GPU kernel completion before returning slice references, avoiding data races.

5. **Exhaustive Quality, Linter, and Test Verification (Observation 5)**:
   - All 5 assigned verification targets passed without errors or regressions.
   - The full workspace gate script `./scripts/ci_local.sh` finished cleanly with exit code 0 (`ci:local OK`).

---

## 3. Caveats

1. **Hardware Serialization Required for Metal Execution**:
   - In accordance with `tessl/docs/WORKFLOWS.md`, Metal device test execution requires `--test-threads=1` to prevent GPU lock contention and race conditions on Apple Silicon unified memory command buffers.
2. **Release Profile Required for Heap Allocation Assertions**:
   - The test `tessl/tests/host_path_allocs.rs` counts exact heap allocations and requires `--release` to avoid heap allocations introduced by debug-mode `objc2` method signature checks.
3. No caveats remain regarding Milestone 2 Iteration 2 requirements: all tasks are implemented, verified, and passing.

---

## 4. Conclusion

Milestone 2 Iteration 2 Remediation satisfies all functional, architectural, quality, and integrity criteria:
1. `BudgetTrimHandler` is properly exported and accessible in `ojas-core`.
2. Cache freelist trimming is implemented end-to-end with verified deadlock prevention, recursion protection, and fast-path bounds checks.
3. `SharedBuf` and `HostFence` completely bridge HAL traits (`MappedBuffer`, `DeviceBuffer`, `HostLease`, `Timeline`) with strict lifetime safety.
4. All clippy lints, deny checks, documentation generators, and bench plot checkers are clean.
5. All test suites pass 100% across both `tessl` and `ojas` without regressions.

**Final Verdict**: **APPROVE**

---

## 5. Verification Method

To independently reproduce and verify this assessment:

```bash
# 1. Verify tessl specific test suites
cd /Users/bharath/Code/research/tessl
cargo test --release --lib shared_buf
cargo test --release --test docs_name_real_tools
cargo test --release --test shared_runtime -- --test-threads=1
cargo test --release --test stress_trim_concurrency -- --test-threads=1

# 2. Run full local CI gate suite in ojas
cd /Users/bharath/Code/research/ojas
./scripts/ci_local.sh
```

**Invalidation Conditions**:
- Any compiler warning under `cargo clippy --workspace --all-targets -- -D warnings`.
- Any failure in `cargo deny check`.
- Any failure in `fmt_check.sh` or `check_tracked_targets.sh`.
- Any non-zero exit code from `./scripts/ci_local.sh`.
- Any deadlock or panic in `tests/stress_trim_concurrency.rs`.

---

## 6. Review Summary

**Verdict**: **APPROVE**

### Findings

- **Finding 1 (Good Practice)**: Fast-path bounds check via `can_ever_fit(bytes)` in `ojas-core/src/budget.rs:194-227` avoids costly IPC and pool trimmings for allocations that exceed capacity.
- **Finding 2 (Good Practice)**: Non-blocking `try_lock()` in `tessl::GpuRuntime::trim_free_list()` prevents deadlocks under thread contention.
- **Finding 3 (Good Practice)**: `SharedBufInner` holding `Option<Arc<PooledBuffer>>` guarantees memory safety across lease transfers and type conversions.

### Verified Claims

| Claim | Verification Method | Status |
|---|---|---|
| `BudgetTrimHandler` public export at `ojas_core` root | Source inspection `ojas-core/src/lib.rs:48` | **PASS** |
| Cache trimming ladder implemented across tessl and ojas-metal | Source inspection `runtime.rs`, `backend.rs`, `device.rs`, `link.rs` | **PASS** |
| `can_ever_fit` rejects oversized requests before trim handler | Source inspection `budget.rs` + unit tests | **PASS** |
| `MappedBuffer`, `DeviceBuffer`, `Timeline` on `SharedBuf` / `HostFence` | Source inspection `shared_buf.rs`, `host_fence.rs` | **PASS** |
| Zero clippy warnings in `ojas` and `tessl` | Live clippy execution with `-D warnings` | **PASS** |
| `cargo deny check` clean | `cargo deny check` in `ojas` | **PASS** |
| Tessl unit and integration tests pass | `cargo test` on `shared_buf`, `docs_name_real_tools`, `shared_runtime`, `stress_trim_concurrency` | **PASS** |
| Full local CI gate passes | `./scripts/ci_local.sh` in `ojas` | **PASS** |

### Coverage Gaps
None. All 5 tasks assigned for M2 IT2 were verified.

### Unverified Items
None.

---

## 7. Adversarial Challenge Report

**Overall Risk Assessment**: **LOW**

### Challenges & Stress-Testing

#### Challenge 1: Use-After-Free or Premature Pool Reuse in `SharedBuf`
- **Assumption Challenged**: Converting a `SharedAlloc` to `SharedBuf` or dropping the outer allocation could allow the buffer pool to recycle the underlying Metal buffer while `SharedBuf` or active leases still reference it.
- **Attack Scenario**: Allocate via `SharedAlloc`, convert to `SharedBuf`, drop `SharedAlloc`, acquire host read lease, trigger pool reallocation from another thread, and read through lease.
- **Blast Radius**: Memory corruption, invalid tensor reads, GPU page fault.
- **Mitigation & Verification**: `SharedBufInner` holds an `Option<Arc<PooledBuffer>>`. As long as `SharedBufInner` exists (via `SharedBuf` or any active lease), the strong count of `PooledBuffer` remains `> 0`, preventing return to the freelist. Verified in `adversarial_shared_buf_uma_and_placement_invariants`: **PASS**.

#### Challenge 2: Deadlock Under Recursive or Concurrent Trimming
- **Assumption Challenged**: An allocation triggering a cache trim callback could re-enter the allocator or link channel while holding locks, causing thread starvation or deadlock.
- **Attack Scenario**: Device worker allocates a buffer under tight budget headroom; trim handler invokes `Cmd::TrimPool`, which asks the device worker to process a command, waiting indefinitely on itself.
- **Blast Radius**: Device link freeze, GPU training deadlocks.
- **Mitigation & Verification**:
  1. `BudgetLedger` in `ojas-metal/src/pool_ledger.rs` installs `NoOpTrim` on `live` and `cache` child budgets, preventing device allocations from emitting trim commands.
  2. `tessl::GpuRuntime::trim_free_list()` uses `self.pool.try_lock()`, ensuring non-blocking execution under contention.
  3. `tests/stress_trim_concurrency.rs` stress tests 16 concurrent threads allocating, parking, and trimming under a tight budget cap with 0 deadlocks. Verified: **PASS**.

#### Challenge 3: Invalidation Loop on Unsatisfiable Allocations
- **Assumption Challenged**: Allocations exceeding total capacity could loop repeatedly invoking trim handlers.
- **Attack Scenario**: Request allocation of size `cap + 1 MB`.
- **Blast Radius**: CPU/GPU burn, high allocation latency.
- **Mitigation & Verification**: `ojas-core/src/budget.rs` implements `can_ever_fit(bytes)` which checks `bytes > cap_bytes` before attempting any trim. Verified: **PASS**.

#### Challenge 4: Data Races on Host Shared Memory Access
- **Assumption Challenged**: Host threads acquiring `read_lease` or `write_lease` could access shared Metal buffer memory while an in-flight GPU kernel is writing to it.
- **Attack Scenario**: Enqueue GPU kernel, immediately request `read_lease` without explicit wait.
- **Blast Radius**: Non-deterministic data corruption.
- **Mitigation & Verification**: `SharedHostReadLease::new` and `SharedHostWriteLease::new` inspect the buffer's `last_write` / `last_use` sequence numbers against `HostFence` and automatically wait until the timeline fence completes before returning the slice. Verified: **PASS**.

### Stress Test Results

| Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|
| Concurrent multi-thread cache trimming | Clean pool drain, no deadlock | 3/3 passed in `stress_trim_concurrency` | **PASS** |
| Impossible allocation exceeding budget cap | Immediate `CapacityExceeded` without trim | Fast-path exits without calling trim handler | **PASS** |
| Concurrent read leases on `SharedBuf` | Shared concurrent read access permitted | `AtomicI32 lease` increments cleanly | **PASS** |
| Write lease with active read leases | Write lease refused or waits cleanly | Refused / blocked until readers release | **PASS** |
| Freeing `SharedAlloc` while `SharedBuf` lives | Buffer remains valid, pool recycling deferred | `Arc<PooledBuffer>` held by `SharedBufInner` | **PASS** |

---

## 8. Integrity Audit

A comprehensive integrity audit was conducted across all changes:
- **No Hardcoded Test Results**: Reviewed all test sources in `ojas` and `tessl`. Assertions evaluate real outputs from live GPU and CPU executions.
- **No Dummy/Facade Implementations**: `BudgetTrimHandler`, `trim_free_list`, and `SharedBuf` traits contain full production logic, resource tracking, and synchronization.
- **No Bypassed Tasks**: All five verification areas were executed and verified against live code and compiler checks.
- **No Fabricated Outputs**: All verification commands were executed directly via tool calls and raw logs inspected.
- **No Self-Certifying Work**: Independent verification was conducted using fresh builds, linters, and isolated test runs.

**Integrity Audit Verdict**: **CLEAN — NO VIOLATIONS DETECTED**
