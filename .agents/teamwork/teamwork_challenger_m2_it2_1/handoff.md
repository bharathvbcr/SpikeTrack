# Milestone 2 Iteration 2 Gate Verification — Challenger 1 Final Report

**Agent Identity**: Empirical Challenger (`teamwork_challenger_m2_it2_1`)  
**Parent Agent**: `7f1f02db-7cbf-40d1-8919-b2741643d465`  
**Verdict**: **APPROVE**  
**Date**: 2026-10-31T21:35:00Z  

---

## 1. Observation

All empirical challenges were directly authored, compiled, and executed across `/Users/bharath/Code/research/ojas` and `/Users/bharath/Code/research/tessl`.

### Area 1: `ojas_core::BudgetTrimHandler` Export & Accessibility
- **Source Inspection**: In `ojas-core/src/lib.rs:48`:
  ```rust
  pub use budget::{Budget, BudgetTrimHandler, Reservation, Scratch};
  ```
  `BudgetTrimHandler` is exported directly at crate root `ojas_core::BudgetTrimHandler`.
- **Downstream Consumer Verification**:
  In `tessl/tests/shared_runtime.rs:139`, `BudgetTrimHandler` was imported and implemented by external test mock structs:
  ```rust
  use ojas_core::{BackendId, Budget, BudgetTrimHandler, DType, DeviceBuffer, MappedBuffer, OjasError, Tensor};

  struct MockTrim(std::sync::atomic::AtomicBool);
  impl BudgetTrimHandler for MockTrim {
      fn trim(&self, _needed: usize) -> usize {
          self.0.store(true, std::sync::atomic::Ordering::SeqCst);
          0
      }
  }
  ```
  **Result**: Compiles and runs cleanly with zero unresolved import or privacy errors (no `E0432`).

### Area 2: `tessl::shared_buf::SharedBuf` Implements `MappedBuffer` & `Tensor::from_mapped` Soundness
- **Source Inspection**: In `tessl/src/shared_buf.rs:658-680`:
  ```rust
  impl MappedBuffer for SharedBuf {
      fn host_ptr(&self) -> *mut u8 { self.inner.ptr.as_ptr() }
      fn gpu_address(&self) -> u64 { self.inner.gpu_addr }
      fn timeline(&self) -> &dyn Timeline { &self.inner.fence }
      fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> { ... }
      fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> { ... }
  }
  ```
  `SharedAlloc` exposes `into_shared_buf(self)` and `shared_buf(&self)` at `tessl/src/shared_buf.rs:700-709`.
- **Empirical Execution**: In `tessl/tests/shared_runtime.rs:136-224` (`adversarial_shared_buf_uma_and_placement_invariants`):
  ```sh
  cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime adversarial_shared_buf_uma_and_placement_invariants -- --nocapture
  ```
  Output:
  ```
  running 1 test
  test adversarial_shared_buf_uma_and_placement_invariants ... ok

  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out; finished in 0.04s
  ```
  **Result**:
  - `shared_buf.host_ptr()`, `shared_buf.gpu_address()`, and `shared_buf.timeline()` execute without error.
  - `Tensor::from_mapped(Arc::new(shared_buf), shape, DType::F32, &budget)` charges the budget exactly `N * 4` bytes (`budget.allocated() == 256`), sets `device() == Some(BackendId::Metal)`, and keeps memory mapped without copy or panic.

### Area 3: Freelist Cache Trimming Under Memory Pressure & Child Budget Safety
- **Source Inspection**:
  - `ojas-metal/src/backend.rs:279-284`:
    ```rust
    pub fn trim_pool(&self) -> Result<(), OjasError> {
        match self.link.call(Cmd::TrimPool)? {
            Reply::Done | Reply::Trimmed(_) => Ok(()),
            other => Err(metal_err(format!("trim_pool: device returned {other:?}"))),
        }
    }
    ```
  - `ojas-metal/src/device.rs:1049-1055`:
    ```rust
    fn trim_pool(&self) -> Res<Reply> {
        self.recycle()?;
        let freed = self.drop_pool();
        Ok(Reply::Trimmed(freed))
    }
    ```
  - `ojas-metal/src/backend.rs:214-224`: `MetalTrimHandler::trim()` correctly pattern-matches `Reply::Trimmed(freed)` and returns freed bytes to the caller.
  - `ojas-core/src/budget.rs:165-174`: `Budget::try_reserve()` calls `can_ever_fit(bytes)` and fast-refuses allocations exceeding capacity before invoking the trim handler.
  - `ojas-metal/src/backend.rs:373`: `BudgetLedger::new()` initializes child budgets with `Arc::new(NoOpTrim)`, preventing recursive trimming cascades.
- **Empirical Execution**: In `ojas/ojas-metal/tests/pool_ledger.rs:125-182` (`adversarial_metal_backend_trim_pool_and_child_budget_invariants`):
  ```sh
  cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger adversarial_metal_backend_trim_pool_and_child_budget_invariants -- --nocapture
  ```
  Output:
  ```
  running 1 test
  test adversarial_metal_backend_trim_pool_and_child_budget_invariants ... ok

  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.05s
  ```
  **Result**:
  - `backend.trim_pool()` executes `Cmd::TrimPool` and completes with `Ok(())`.
  - `Budget::try_reserve()` with `bytes > cap` fails immediately with `CapacityExceeded` without invoking the trim handler (spy trim handler remained uncalled).
  - Child budgets configured with `NoOpTrim` refuse without deadlocking or triggering recursive trim calls.

### Area 4: Fail-Closed Placement Invariants on Device/Mapped Tensors
- **Source Inspection**: In `ojas-core/src/tensor.rs:760-767` and `1054-1061`:
  ```rust
  pub fn device(&self) -> Option<BackendId> {
      match &self.storage.payload {
          Payload::Host(_) => None,
          Payload::Device(buf) => Some(buf.backend()),
          Payload::Mapped(buf) => Some(buf.backend()),
      }
  }

  fn host_window(&self, op: &'static str) -> Result<Window, OjasError> {
      if let Some(backend) = self.device() {
          return Err(OjasError::Placement {
              op,
              expected: None,
              found: Some(backend),
          });
      }
      ...
  }
  ```
- **Empirical Execution**: In `tessl/tests/shared_runtime.rs:188-212`:
  - `tensor.f32_slice()` on mapped F32 tensor: Returns `Err(OjasError::Placement { found: Some(BackendId::Metal), .. })`.
  - `tensor.to_f32_vec()` on mapped F32 tensor: Returns `Err(OjasError::Placement { found: Some(BackendId::Metal), .. })`.
  - `tensor.write_f32(0, 99.0)` on mapped F32 tensor: Returns `Err(OjasError::Placement { found: Some(BackendId::Metal), .. })`.
  - `bf16_tensor.bf16_slice()` on mapped Bf16 tensor: Returns `Err(OjasError::Placement { found: Some(BackendId::Metal), .. })`.
  **Result**: Verified fail-closed behavior across all slice/vector CPU accessors with no silent CPU fallback.

### Area 5: Zero-Copy Host View via `tensor.host_view()`
- **Source Inspection**: In `ojas-core/src/tensor.rs:1095-1104`:
  ```rust
  pub fn host_view(&self) -> Result<Box<dyn HostLease>, OjasError> {
      match &self.storage.payload {
          Payload::Host(_) => Err(OjasError::Placement { ... }),
          Payload::Device(_) => Err(OjasError::Placement { ... }),
          Payload::Mapped(buf) => buf.host_read().map_err(...),
      }
  }
  ```
- **Empirical Execution**: In `tessl/tests/shared_runtime.rs:214-222`:
  ```rust
  let view = tensor.host_view().expect("host_view must succeed for mapped tensor");
  let view_bytes = view.as_bytes();
  let f32_vals = bytemuck::cast_slice::<u8, f32>(view_bytes);
  for (i, &val) in f32_vals.iter().enumerate().take(N) {
      assert_eq!(val, (i + 1) as f32);
  }
  ```
  **Result**: Zero-copy host view accurately accesses UMA memory written by the host without copying or allocating, reading back updated values bit-for-bit.

### Workspace Regression & Quality Gate Verification
1. **Target Tracking**:
   ```sh
   cd /Users/bharath/Code/research/ojas && ./scripts/check_tracked_targets.sh
   ```
   *Result*: OK. All 368 target sources accounted for.
2. **Code Formatting**:
   ```sh
   cd /Users/bharath/Code/research/ojas && ./scripts/fmt_check.sh
   ```
   *Result*: OK. No formatting discrepancies.
3. **Linter Gates**:
   ```sh
   cargo clippy --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml --workspace --all-targets -- -D warnings
   cargo clippy --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --all-targets -- -D warnings
   ```
   *Result*: OK. 0 warnings across all targets.
4. **Dependency & License Policy**:
   ```sh
   cd /Users/bharath/Code/research/ojas && cargo deny check
   ```
   *Result*: OK. Advisories, bans, licenses, and sources all passed.
5. **Documentation & Feature Stubs**:
   ```sh
   cd /Users/bharath/Code/research/ojas && RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
   cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings
   ```
   *Result*: OK. Documentation builds cleanly; non-Metal stubs build with zero warnings.

---

## 2. Logic Chain

1. **Premise 1**: In Milestone 2 Iteration 1, the gate was blocked by:
   - Missing public re-export of `BudgetTrimHandler` from `ojas_core` root.
   - Absence of `MappedBuffer` implementation on `tessl::shared_buf::SharedBuf`.
   - Incomplete freelist cache trimming contract (`Reply::Trimmed(freed)` mismatch).
   - Ambiguous CPU slice access semantics on mapped buffers risking silent fallback.
2. **Step 1 (Export & Surface)**: Observation 1 confirms `pub use budget::BudgetTrimHandler` in `ojas-core/src/lib.rs:48`. Downstream crates can implement and invoke `BudgetTrimHandler` without build errors or visibility hacks.
3. **Step 2 (UMA MappedBuffer Integration)**: Observation 2 confirms `SharedBuf` implements `MappedBuffer`, exposing pointer, GPU address, timeline, and read/write host leases. Integration with `Tensor::from_mapped` accurately tracks budget reservations on UMA memory.
4. **Step 3 (Trimming & Child Budget Invariants)**: Observation 3 confirms `MetalBackend::trim_pool()` and `MetalTrimHandler::trim()` handle `Reply::Trimmed(freed)` uniformly. `can_ever_fit` eliminates trim-handler spin on oversized requests, and `NoOpTrim` on child budgets prevents recursive cycles.
5. **Step 4 (Fail-Closed Placement)**: Observation 4 confirms that `Tensor::f32_slice()`, `Tensor::bf16_slice()`, `Tensor::to_f32_vec()`, and `Tensor::write_f32()` systematically check `self.device()` and fail closed with `OjasError::Placement` when called on device or mapped tensors. There is zero silent host fallback.
6. **Step 5 (Sound Zero-Copy Access)**: Observation 5 confirms `Tensor::host_view()` is the dedicated zero-copy vehicle for reading mapped tensor memory on host, safely gated by `HostLease`.
7. **Step 6 (Overall Quality)**: All 19 crates pass clippy, formatting, target checks, cargo deny, and documentation checks without warnings or regressions.
8. **Deduction**: All deficiencies identified in Iteration 1 have been completely resolved, verified by adversarial integration tests, and conform strictly to the architectural requirements.

---

## 3. Caveats

1. **Hardware Scope**: Metal kernel execution and genuine UMA shared buffer features require Apple Silicon hardware running macOS. On non-macOS platforms, `ojas-metal` compiles to an explicit stub returning `OjasError::Unsupported` (verified via `--no-default-features` clippy gate).
2. **Lease Lifetime Safety**: Zero-copy host access through `tensor.host_view()` relies on RAII leases (`Box<dyn HostLease>`) synchronized against the timeline fence. Concurrent GPU write dispatches during an active host lease are blocked by `HostFence` checks.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 2 Iteration 2 changes in both `ojas` and `tessl` satisfy all gate requirements. Every finding from Iteration 1 has been validated with direct empirical verification. The implementation is robust, fail-closed, memory-safe, and zero-copy.

---

## 5. Verification Method

To independently reproduce and verify this assessment:

1. **Run Adversarial UMA & Placement Invariants Test in `tessl`**:
   ```sh
   cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime adversarial_shared_buf_uma_and_placement_invariants -- --nocapture
   ```
2. **Run Freelist Trimming & Budget Invariants Test in `ojas`**:
   ```sh
   cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger adversarial_metal_backend_trim_pool_and_child_budget_invariants -- --nocapture
   ```
3. **Run Full Workspace CI & Quality Gates**:
   ```sh
   cd /Users/bharath/Code/research/ojas
   ./scripts/check_tracked_targets.sh
   ./scripts/fmt_check.sh
   cargo clippy --workspace --all-targets -- -D warnings
   cargo deny check
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
   cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings
   ```
4. **Invalidation Conditions**:
   - Any failure or panic during the two adversarial tests.
   - Any compiler error when referencing `ojas_core::BudgetTrimHandler` from an external crate.
   - Any direct slice access (`f32_slice()`, etc.) on a mapped tensor returning `Ok` instead of `OjasError::Placement`.
   - Any clippy warning, doc warning, or license failure in `cargo deny`.
