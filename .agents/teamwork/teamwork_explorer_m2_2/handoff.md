# Handoff Report: Authoritative Budget Ledger, Cache Trim Ladder, and Fail-Closed Placement Enforcement

**Agent**: `teamwork_explorer_m2_2`  
**Milestone**: Milestone 2: Unified Architecture & Shared Memory Pools  
**Date**: 2026-10-31T18:31:00Z  
**Recipient**: Orchestrator (`7f1f02db-7cbf-40d1-8919-b2741643d465`) & Milestone 2 Workers  
**Type**: Hard Handoff (Investigation & Architecture Design Complete)

---

## 1. Observation

### 1.1 Root Budget Ledger & Allocations in `ojas-core`
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/budget.rs` (lines 18–34, 75–86, 170–182, 201–210)
  - `Budget` wraps `Arc<Inner>` containing `cap_bytes: u64`, `live_bytes: AtomicU64`, `peak_bytes: AtomicU64`, and `parent: Option<Budget>`.
  - `Budget::child(&self, cap_bytes: u64) -> Budget` constructs a child ledger that charges itself first (`charge_self`), then charges its parent (`charge_chain`). If the parent refuses, the child charge rolls back (`release_self`).
  - `Reservation` holds `bytes: u64` and `budget: Budget`, releasing `self.budget.release(self.bytes)` upon `Drop`.
  - `Scratch<T>` pairs a `Vec<T>` with a `Reservation`.
  - In `ojas-core/src/tensor.rs` (lines 98–115, 848–874), `Storage` holds `_reservation: Reservation`. Every tensor allocation is charged against a `Budget`.
  - **Limitation Observed**: `Budget` currently has no trim callback or pressure hook. When `live + requested > cap`, `Budget::try_reserve` immediately returns `OjasError::CapacityExceeded { requested, cap, live }` without querying external pool caches.

### 1.2 Tessl Buffer Pool & Cache Disconnect
- **File**: `/Users/bharath/Code/research/tessl/src/runtime.rs` (lines 245–360, 1192–1200, 1894–1931)
  - `BufferPool` defines:
    ```rust
    struct BufferPool {
        freelist: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
        cached_bytes: usize,
        max_cache_bytes: usize,
        max_buffer_length: usize,
    }
    ```
  - `BufferPool::alloc` checks `freelist.get_mut(&key)`: on hit, pops from freelist and decrements `self.cached_bytes`; on miss, calls `device.newBufferWithLength_options(key, MTLResourceOptions::StorageModeShared)`.
  - `BufferPool::recycle` returns buffers to `freelist` up to `max_cache_bytes` (default up to 2 GiB).
  - **Critical Defect Observed**: `BufferPool` has **no connection** to `ojas_core::Budget`. Cached buffers residing in `freelist` are untracked by `Budget`.
  - In `ojas-metal/src/backend.rs` (lines 245–250), code explicitly states:
    > "The cache is not charged to the budget (see the module docs); this hands its memory back, as wgpu's `WgpuContext::trim_pool` does."
  - This allows `pool.cache` to hold up to 2 GiB of idle buffers completely untracked, causing memory pressure or premature process kills.

### 1.3 String Scraping Allocation Classifier in `ojas-metal`
- **File**: `/Users/bharath/Code/research/ojas/ojas-metal/src/device.rs` (lines 536–551, 950–994)
  - `tessl::BufferPool::alloc` returns `Result<(Retained<MTLBuffer>, bool), String>`.
  - `ojas-metal` uses string scraping to classify allocation failures:
    ```rust
    fn classify_alloc(e: &str) -> AllocFailure {
        if e.contains("exceeds host/device allocation limit") || e.contains("exceeds device limit") {
            AllocFailure::Limit
        } else if is_new_buffer_failure(e) {
            AllocFailure::Exhausted
        } else {
            AllocFailure::Other
        }
    }
    ```
  - This string parsing is fragile and violates the unified typed interface contract.

### 1.4 Fail-Closed Placement Invariants
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/tensor.rs` (lines 701–708, 737–746, 936–941)
  - `Tensor::device(&self) -> Option<BackendId>` returns `None` for host memory and `Some(buf.backend())` for device buffers.
  - `Tensor::host_window(&self, op: &'static str)` and `host_data` check:
    ```rust
    if let Payload::Device(buf) = &self.storage.payload {
        return Err(OjasError::Placement {
            op,
            expected: None,
            found: Some(buf.backend()),
        });
    }
    ```
  - Direct CPU accessors (`f32_slice()`, `bf16_slice()`, `u32_slice()`, `f32_slice_mut()`, `to_f32_vec()`, `write_f32()`) invoke `host_window()` or `host_data()` and therefore return `Err(OjasError::Placement)`.
  - In `ojas-metal/src/backend.rs` (lines 660–672), `MetalBackend::arg` verifies:
    ```rust
    let buf = t.device_buffer().ok_or(OjasError::Placement {
        op,
        expected: Some(BackendId::Metal),
        found: None,
    })?;
    if buf.backend() != BackendId::Metal {
        return Err(OjasError::Placement {
            op,
            expected: Some(BackendId::Metal),
            found: Some(buf.backend()),
        });
    }
    ```
  - Host tensors passed to `MetalBackend` ops fail closed immediately with `OjasError::Placement`.

### 1.5 Compatibility with E2E Tier 4 Test Suite
- **File**: `/Users/bharath/Code/research/ojas/ojas-oracle/tests/e2e_tier4_production.rs`
  - Scenario C (`tier4_scenario_c_pipeline_budget_scoping_and_trim`, lines 310–331) tests `root_budget.child(16 * 1024 * 1024)` isolation and verifies live bytes return to 0 upon tensor drop.
  - Scenario D (`tier4_scenario_d_no_silent_fallback_placement_invariance`, lines 338–353) tests fail-closed error handling for malformed layouts.
  - Verified live test execution:
    ```bash
    cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-oracle --test e2e_tier4_production
    ```
    Output: `test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s`.

---

## 2. Logic Chain

1. **Root Ledger Invariant (Feature 8)**:
   - For `ojas_core::Budget` to be the sole authoritative ledger across CPU and GPU without unbudgeted drift, all GPU memory—both active resident tensors and idle cached freelist buffers—must be accounted for on the ledger.
   - When `BufferPool` caches an idle buffer in `freelist`, that buffer retains Metal physical pages in Unified Memory. Therefore, each cached buffer must hold a `Reservation` charged against a child budget (`cache_budget = root_budget.child(max_cache_bytes)`).
   - This ensures `root_budget.live_bytes()` reflects true system commitment: `active_tensors + idle_cached_buffers`.

2. **Automated Memory Pressure Trim Ladder**:
   - Because idle cached buffers hold reservations on the child budget, a large active allocation request on `root_budget` might trigger `CapacityExceeded` even though gigabytes of idle buffers could be freed.
   - Introducing `BudgetTrimHandler` on `ojas_core::Budget` closes this gap cleanly:
     - When `try_reserve(needed)` fails with `CapacityExceeded`, `Budget` queries its registered `BudgetTrimHandler`.
     - `BufferPool`'s implementation of `BudgetTrimHandler` trims the oldest/largest idle buffers from `freelist`.
     - Dropping cached buffers drops their `Reservation`s, immediately decrementing `live_bytes` in atomic fashion across the ancestor chain.
     - `Budget::try_reserve` retries charging. If sufficient headroom was recovered, the allocation succeeds with zero errors.
     - If headroom remains insufficient after a complete cache purge, `Budget` fails closed with `OjasError::CapacityExceeded`.

3. **Elimination of String Scraping**:
   - Replacing `Result<..., String>` in `BufferPool::alloc` with `Result<..., AllocError>` (where `AllocError` is the typed enum defined in `ojas-core`) enables `ojas-metal/src/device.rs` to replace `classify_alloc` with match arms over `AllocError::Exhausted`, `AllocError::DeviceLost`, and `AllocError::InvalidAlignment`.
   - `AllocError::Exhausted` maps deterministically to `OjasError::CapacityExceeded`.

4. **Fail-Closed Placement Enforcement (Feature 10)**:
   - Zero-copy UMA (Feature 9) shares underlying `StorageModeShared` buffers across CPU and GPU on Apple Silicon. However, sharing physical storage must not compromise logical placement safety.
   - Any tensor with device placement (`device() == Some(BackendId::Metal)`) strictly forbids host access (`f32_slice()`, etc.) and returns `OjasError::Placement { op, expected: None, found: Some(BackendId::Metal) }`.
   - CPU access is permitted only after explicit `to_host()`, which performs timeline fence synchronization (`Timeline::wait_for`) and produces a host-placed tensor.
   - Conversely, passing a host tensor to a `MetalBackend` op returns `OjasError::Placement { op, expected: Some(BackendId::Metal), found: None }`.
   - There is no implicit transfer or silent fallback anywhere in the pipeline.

---

## 3. Caveats

1. **Lock Ordering Safety**:
   - `BudgetTrimHandler::trim_memory` is called from `Budget::try_reserve`.
   - `Budget` uses lock-free atomics (`AtomicU64::try_update`), holding no internal mutexes during reservation.
   - `BufferPool` guards its freelist with `Mutex<BufferPool>`.
   - Calling `rt.pool.lock()` from within `trim_memory` is completely deadlock-free because no thread ever acquires a `Budget` lock while holding `pool.lock()`.
2. **OS Memory Pressure Events**:
   - In addition to budget-driven pressure trimming, `tessl::GpuRuntime` can optionally bind a Darwin GCD dispatch memory pressure source (`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`). This handles external pressure from sibling processes by purging the pool cache when macOS issues warning/critical signals.
3. **MSRV & Dependency Constraints**:
   - `ojas-core` remains `#![forbid(unsafe_code)]` with zero external dependencies (std only).
   - Dependency hierarchy `ojas-metal -> tessl -> ojas-core` is strictly acyclic.

---

## 4. Conclusion & Concrete Architectural Specifications

### Specification 1: `BudgetTrimHandler` in `ojas-core/src/budget.rs`

```rust
// In ojas-core/src/budget.rs:

/// A callback invoked by [`Budget`] when an allocation request cannot be satisfied.
///
/// Implementations free at least `needed_bytes` of idle/reclaimable memory
/// (e.g., from an internal freelist buffer cache). Returns the number of bytes actually freed.
pub trait BudgetTrimHandler: Send + Sync {
    fn trim_memory(&self, needed_bytes: u64) -> u64;
}

// In struct Inner:
struct Inner {
    cap_bytes: u64,
    live_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    parent: Option<Budget>,
    readbacks: AtomicU64,
    readback_bytes: AtomicU64,
    trim_handler: std::sync::RwLock<Option<Arc<dyn BudgetTrimHandler>>>,
}

// In impl Budget:
impl Budget {
    pub fn set_trim_handler(&self, handler: Arc<dyn BudgetTrimHandler>) {
        if let Ok(mut lock) = self.inner.trim_handler.write() {
            *lock = Some(handler);
        }
    }

    pub fn trim_handler(&self) -> Option<Arc<dyn BudgetTrimHandler>> {
        if let Ok(lock) = self.inner.trim_handler.read() {
            if let Some(h) = lock.as_ref() {
                return Some(Arc::clone(h));
            }
        }
        self.inner.parent.as_ref().and_then(|p| p.trim_handler())
    }

    pub fn try_reserve(&self, bytes: u64) -> Result<Reservation, OjasError> {
        if bytes == 0 {
            return Ok(Reservation {
                bytes: 0,
                budget: self.clone(),
            });
        }
        if let Err(err) = self.charge(bytes) {
            if let OjasError::CapacityExceeded { .. } = &err {
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
            return Err(err);
        }
        Ok(Reservation {
            bytes,
            budget: self.clone(),
        })
    }
}
```

### Specification 2: `BufferPool` Child Budget & Trim Ladder in `tessl/src/runtime.rs`

```rust
// In tessl/Cargo.toml:
// [dependencies]
// ojas-core = { path = "../ojas/ojas-core" }

// In tessl/src/runtime.rs:
use ojas_core::{AllocError, Budget, BudgetTrimHandler, Reservation};

struct CachedBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    _reservation: Reservation,
}

pub struct BufferPool {
    freelist: HashMap<usize, Vec<CachedBuffer>>,
    cached_bytes: usize,
    max_cache_bytes: usize,
    max_buffer_length: usize,
    budget: Budget, // Child budget derived from root ledger
}

impl BufferPool {
    pub fn new(root_budget: &Budget, max_cache_bytes: usize, max_buffer_length: usize) -> Self {
        let child_budget = root_budget.child(max_cache_bytes as u64);
        Self {
            freelist: HashMap::new(),
            cached_bytes: 0,
            max_cache_bytes,
            max_buffer_length,
            budget: child_budget,
        }
    }

    pub fn trim_idle(&mut self, needed_bytes: usize) -> usize {
        let mut freed = 0usize;
        let mut keys: Vec<usize> = self.freelist.keys().copied().collect();
        keys.sort_unstable_by(|a, b| b.cmp(a)); // Evict largest first
        for key in keys {
            while let Some(v) = self.freelist.get_mut(&key) {
                if let Some(cached) = v.pop() {
                    drop(cached); // Frees MTLBuffer and drops Reservation
                    self.cached_bytes = self.cached_bytes.saturating_sub(key);
                    freed += key;
                    if needed_bytes > 0 && freed >= needed_bytes {
                        return freed;
                    }
                } else {
                    break;
                }
            }
        }
        freed
    }

    pub fn alloc(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        nbytes: usize,
        kind: BufferKind,
    ) -> Result<(Retained<ProtocolObject<dyn MTLBuffer>>, bool), AllocError> {
        if nbytes > isize::MAX as usize || nbytes > self.max_buffer_length {
            return Err(AllocError::Exhausted {
                requested: nbytes as u64,
                available: self.max_buffer_length as u64,
            });
        }
        let key = Self::bucket(nbytes, kind)
            .filter(|&key| key <= self.max_buffer_length)
            .ok_or(AllocError::Exhausted {
                requested: nbytes as u64,
                available: self.max_buffer_length as u64,
            })?;
        if let Some(v) = self.freelist.get_mut(&key) {
            if let Some(cached) = v.pop() {
                self.cached_bytes = self.cached_bytes.saturating_sub(key);
                return Ok((cached.buffer, true));
            }
        }
        let b = device
            .newBufferWithLength_options(key, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| AllocError::Exhausted {
                requested: key as u64,
                available: self.cached_bytes as u64,
            })?;
        Ok((b, false))
    }

    pub fn recycle(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let key = buffer.length();
        if key > self.max_cache_bytes {
            return;
        }
        let reservation = match self.budget.try_reserve(key as u64) {
            Ok(res) => res,
            Err(_) => {
                self.trim_idle(key);
                match self.budget.try_reserve(key as u64) {
                    Ok(res) => res,
                    Err(_) => return, // Cannot fit even after trim; drop
                }
            }
        };
        self.cached_bytes += key;
        self.freelist.entry(key).or_default().push(CachedBuffer {
            buffer,
            _reservation: reservation,
        });
    }
}
```

### Specification 3: Fail-Closed Placement Enforcement in `ojas-core/src/tensor.rs`

```rust
// In ojas-core/src/tensor.rs:

impl Tensor {
    fn host_window(&self, op: &'static str) -> Result<Window, OjasError> {
        if let Some(backend) = self.device() {
            return Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(backend),
            });
        }
        if !self.is_contiguous()? {
            return Err(OjasError::Shape {
                op,
                detail: "view is not contiguous".to_string(),
            });
        }
        // ... (window bounds checks)
    }

    fn host_data(&self, op: &'static str) -> Result<&HostData, OjasError> {
        if let Some(backend) = self.device() {
            return Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(backend),
            });
        }
        match &self.storage.payload {
            Payload::Host(data) => Ok(data),
            _ => Err(OjasError::Placement {
                op,
                expected: None,
                found: self.device(),
            }),
        }
    }
}
```

### Specification 4: Retiring String Scraping in `ojas-metal/src/device.rs`

- Delete `classify_alloc` and `is_new_buffer_failure`.
- Update `Worker::retrying` to handle typed `AllocError`:
```rust
fn retrying<T>(&self, bytes: usize, mut attempt: impl FnMut() -> Result<T, AllocError>) -> Res<T> {
    let mut failed = match attempt() {
        Ok(v) => return Ok(v),
        Err(e) => e,
    };
    for step in 0..2 {
        match failed {
            AllocError::InvalidAlignment => return Err(metal_err("invalid alignment")),
            AllocError::DeviceLost => {
                return Err(OjasError::DeviceLost {
                    backend: BackendId::Metal,
                    detail: "Metal device lost".into(),
                })
            }
            AllocError::Exhausted { .. } => {}
        }
        if step == 0 {
            self.recycle()?;
        } else {
            self.drop_pool();
        }
        failed = match attempt() {
            Ok(v) => return Ok(v),
            Err(e) => e,
        };
    }
    Err(OjasError::CapacityExceeded {
        requested: bytes as u64,
        cap: self.rt.memory_info().recommended_working_set,
        live: self.rt.current_allocated_bytes(),
    })
}
```

---

## 5. Verification Method

To verify these designs once implemented by Worker:

1. **Compile & Type-Check Workspace**:
   ```bash
   cargo check --workspace --all-targets
   ```

2. **Run E2E Tier 4 Production Suite (including Scenarios C and D)**:
   ```bash
   cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
     -p ojas-oracle --test e2e_tier4_production -- --nocapture
   ```
   - Assert `tier4_scenario_c_pipeline_budget_scoping_and_trim` passes, verifying child budget isolation and zero memory leaks.
   - Assert `tier4_scenario_d_no_silent_fallback_placement_invariance` passes, confirming fail-closed error handling.

3. **Verify Fail-Closed Placement Refusal Unit Test**:
   Execute the placement invariant test in `ojas-core`:
   ```bash
   cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
     -p ojas-core --test tensor_placement
   ```
   Verify that invoking `f32_slice()`, `bf16_slice()`, `to_f32_vec()`, or `write_f32()` on a `BackendId::Metal` tensor returns `Err(OjasError::Placement { op, expected: None, found: Some(BackendId::Metal) })`.

4. **Verify Cache Trim Ladder under Simulated Budget Exhaustion**:
   - Construct root budget of 32 MiB: `let root = Budget::new(32 * 1024 * 1024)`.
   - Allocate and drop 16 MiB of tensors so they enter `BufferPool`'s freelist cache.
   - Request an allocation of 24 MiB on `root`.
   - Verify that `BufferPool` automatically trims the idle 16 MiB cached buffers via `BudgetTrimHandler`, allowing the 24 MiB allocation to succeed without returning `CapacityExceeded`.
