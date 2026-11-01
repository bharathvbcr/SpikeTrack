# Handoff Report: Challenger 1 — Milestone 2 Adversarial Verification

**Milestone**: Milestone 2: Unified Architecture & Shared Memory Pools  
**Verdict**: **REQUEST_CHANGES**  
**Overall Risk Assessment**: **HIGH**

---

## 1. Observation

### Dispatched Verification Suite Execution Results
All standard test suites requested in dispatch pass without errors:
1. `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core`  
   - Result: 203 passed; 0 failed; 3 ignored.
2. `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime`  
   - Result: 4 passed; 0 failed.
3. `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib pool_ledger`  
   - Result: 15 passed; 0 failed.
4. `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier4_production`  
   - Result: 4 passed; 0 failed.
5. `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger --test gate_dbias --test adamw_msl_parity`  
   - Result: 7 passed; 0 failed.
6. `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test adversarial_shared_stress`  
   - Result: 2 passed; 0 failed.

---

### Empirical Adversarial Findings & Deficiencies

#### Observation 1: `BudgetTrimHandler` Defined but NOT Exported in `ojas-core/src/lib.rs` (CRITICAL)
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/budget.rs`, lines 27-29:
  ```rust
  pub trait BudgetTrimHandler: Send + Sync + std::fmt::Debug {
      fn trim_memory(&self, needed_bytes: u64) -> u64;
  }
  ```
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/lib.rs`:
  - Line 13: `mod budget;` (private module).
  - Line 48: `pub use budget::{Budget, Reservation, Scratch};`
- **Compiler Failure**: Attempting to use or implement `BudgetTrimHandler` in any downstream crate (`ojas-metal`, `tessl`, or external tests) fails at compile time:
  ```text
  error[E0432]: unresolved import `ojas_core::BudgetTrimHandler`
    --> ojas-metal/tests/adversarial_m2_verification.rs:24:45
     |
  24 | ...Budget, BudgetTrimHandler, DType, DeviceBuffer,
     |            ^^^^^^^^^^^^^^^^^ no `BudgetTrimHandler` in the root
  ```
- **Verbatim Impact**: Callers outside `ojas-core` cannot name `BudgetTrimHandler` or pass `Arc<dyn BudgetTrimHandler>` to `Budget::set_trim_handler`.

#### Observation 2: Zero Real Implementations of `MappedBuffer` in `tessl` or `ojas-metal` (CRITICAL)
- **File**: `ojas-core/src/hal.rs`, lines 415-452 defines `pub trait MappedBuffer: DeviceBuffer`.
- **File**: `ojas-core/src/tensor.rs`, lines 967-993 defines `Tensor::from_mapped(buffer: Arc<dyn MappedBuffer>, ...)` and `Payload::Mapped(Arc<dyn MappedBuffer>)`.
- **Grep Inspection**:
  - `grep_search Query="MappedBuffer" SearchPath="tessl/src"` → **0 results**.
  - `grep_search Query="DeviceBuffer" SearchPath="tessl/src"` → **0 results**.
  - `grep_search Query="MappedBuffer" SearchPath="ojas-metal/src"` → **0 results**.
- **Impact**: `tessl::shared_buf::SharedBuf` only implements `Send` and `Sync`. Neither `SharedBuf`, `SharedAlloc`, nor `ojas_metal::MetalBuffer` implements `MappedBuffer`. Consequently, `Tensor::from_mapped` cannot wrap any real Metal unified memory buffer; `Payload::Mapped` is only exercised via `FakeMapped` mock structs in unit tests. Zero-copy UMA is unintegrated between `ojas` and `tessl`.

#### Observation 3: `tessl::BufferPool` / `GpuRuntime` Does Not Implement `BudgetTrimHandler` (HIGH)
- **Dispatch Mandate**: Worker 2 dispatch explicitly mandated: *"In tessl/src/runtime.rs: connect BufferPool to Budget child, implement BudgetTrimHandler on BufferPool to evict idle cached buffers on pressure."*
- **Inspection**: `BudgetTrimHandler` is not referenced anywhere in `tessl`. `tessl::runtime::BufferPool` trims only internally during `charge_fresh` if its own ledger charge fails. When host-side allocations on `Budget` exceed headroom while the GPU freelist holds idle memory, `Budget::try_reserve` has no registered `BudgetTrimHandler` to evict GPU cache.

#### Observation 4: Untracked Test Target Sources Broke `./scripts/check_tracked_targets.sh` (MEDIUM)
- **Command Output**:
  ```text
  $ ./scripts/check_tracked_targets.sh
  check_tracked_targets: ojas-metal/tests/adversarial_pool_stress.rs is a Cargo target source but is not tracked
  check_tracked_targets: 1 of 369 target sources are not tracked
  ```
- **Impact**: Untracked target source left in tree breaks CI check on fresh checkouts.

---

## 2. Logic Chain

1. **Premise 1 (Interface Availability)**: Feature 8 specifies `BudgetTrimHandler` on `ojas-core::Budget` to allow memory pressure eviction of idle pool buffers.
   - *Observation*: `ojas-core/src/lib.rs` line 48 omits `BudgetTrimHandler` from `pub use budget::{...}` while `mod budget;` remains private.
   - *Deduction*: Downstream crates cannot name or implement `BudgetTrimHandler`. Any code outside `ojas-core` attempting to register a trim handler fails compilation with `E0432`.
2. **Premise 2 (Zero-Copy UMA Seam)**: Feature 9 specifies zero-copy buffer sharing between Ojas and Tessl using `Payload::Mapped(Arc<dyn MappedBuffer>)`.
   - *Observation*: Neither `tessl` nor `ojas-metal` contains any `impl MappedBuffer`. `SharedBuf` implements only `Send + Sync` with custom `read()`/`write()` methods, and `MetalBuffer` implements only `DeviceBuffer`.
   - *Deduction*: `Tensor::from_mapped` cannot accept a `SharedBuf` or `MetalBuffer`. The architectural promise that `ojas_core::Tensor` zero-copy shares Apple Silicon unified memory with `tessl` is incomplete in production code.
3. **Premise 3 (Fail-Closed Placement Integrity)**: Feature 10 specifies that CPU accessors (`f32_slice`, `bf16_slice`, `to_f32_vec`, `write_f32`) fail closed with `OjasError::Placement` without silent fallback.
   - *Observation*: Empirically verified in `ojas-core/src/tensor.rs` and on `MetalBackend` uploaded tensors that all direct host accessors fail closed with `OjasError::Placement { found: Some(BackendId::Metal), .. }`, and CPU backend compute ops refuse device and mapped tensors.
   - *Deduction*: Fail-closed placement invariance is solidly verified.

---

## 3. Caveats

- Tests requiring Apple Silicon Metal GPU execution were verified on the local Apple Silicon runtime. Linux/Windows CI does not build Metal kernels.
- `tessl::shared_buf::SharedBuf` correctly implements thread-safe concurrent reader and exclusive writer leases (`HostRead`, `HostWrite`) at the Tessl level, but lacks the trait bridge to `ojas_core::MappedBuffer`.

---

## 4. Conclusion & Verdict

**Verdict**: **REQUEST_CHANGES**

Milestone 2 cannot be approved in its current state due to two blocking architectural gaps:
1. **Unexported `BudgetTrimHandler`**: `ojas-core/src/lib.rs` must re-export `BudgetTrimHandler` so `tessl` and `ojas-metal` can implement and register cache trim hooks.
2. **Missing `MappedBuffer` Implementation**: `tessl` (or `ojas-metal`) must implement `MappedBuffer` for `SharedBuf` (or provide an adapter wrapping `SharedBuf`), connecting `host_read()` / `host_write()` to Tessl's timeline-fenced leases.
3. **Freelist Cache Trim Wiring**: A `BudgetTrimHandler` implementation must be provided for `GpuRuntime` that triggers `set_pool_cache_cap_bytes(0)` (or `trim_free_list`) when `Budget::try_reserve` experiences `CapacityExceeded`.

---

## 5. Verification Method

To reproduce and verify these findings independently:

```bash
# 1. Verify unexported BudgetTrimHandler compiler failure:
# In an external test or crate (e.g. ojas-metal), add `use ojas_core::BudgetTrimHandler;`
# Run:
cargo check --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --tests

# 2. Verify absence of MappedBuffer in tessl and ojas-metal:
rg "impl.*MappedBuffer" /Users/bharath/Code/research/tessl/src
rg "impl.*MappedBuffer" /Users/bharath/Code/research/ojas/ojas-metal/src

# 3. Verify standard Milestone 2 test suites pass:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib pool_ledger
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier4_production
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test adversarial_shared_stress
```

---

## Challenge Summary

| Challenge ID | Severity | Assumption Challenged | Attack / Failure Scenario | Status |
|---|---|---|---|---|
| **CH-M2-01** | **CRITICAL** | `BudgetTrimHandler` is ready for cross-crate integration | External crate attempts `use ojas_core::BudgetTrimHandler` → compile error `E0432`. | **CONFIRMED BUG** |
| **CH-M2-02** | **CRITICAL** | `Payload::Mapped` integrates `SharedBuf` zero-copy UMA | Cannot construct `Tensor::from_mapped` with `SharedBuf` because `SharedBuf` does not implement `MappedBuffer`. | **CONFIRMED GAP** |
| **CH-M2-03** | **HIGH** | `BufferPool` trims idle cache on root budget pressure | Host allocation on root budget fails with `CapacityExceeded` even with idle cached GPU buffers because no trim handler is registered on `BufferPool`. | **CONFIRMED GAP** |
| **CH-M2-04** | **LOW** | Direct host access on Metal/Mapped tensors fails closed | `f32_slice()`, `bf16_slice()`, `to_f32_vec()`, `write_f32()` return `Err(OjasError::Placement)` with zero silent fallback. | **ROBUST / PASS** |
| **CH-M2-05** | **LOW** | `SharedBuf` concurrent leases prevent data races | 16-thread reader/writer stress test: 0 torn reads, mutual exclusion verified. | **ROBUST / PASS** |
