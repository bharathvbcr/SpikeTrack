# Handoff Report: Explorer 1 — Milestone 2 Iteration 2 (Remediation)
## Investigation: `BudgetTrimHandler` Export & Buffer Pool Cache Trim Wiring

**Target Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_1/`  
**Milestone**: Milestone 2 Iteration 2 (Remediation)  
**Topic**: `BudgetTrimHandler` public export and cross-crate freelist cache trim wiring across `ojas-core`, `tessl`, and `ojas-metal`

---

## 1. Observation

### Observation 1: `BudgetTrimHandler` Definition vs. Missing Re-export in `ojas-core`
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/budget.rs` (lines 27-29, 100-114):
  ```rust
  pub trait BudgetTrimHandler: Send + Sync + std::fmt::Debug {
      fn trim_memory(&self, needed_bytes: u64) -> u64;
  }
  ...
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
  ```
- **File**: `/Users/bharath/Code/research/ojas/ojas-core/src/lib.rs`:
  - Line 13: `mod budget;` (module is private, not `pub mod budget;`).
  - Line 48:
    ```rust
    pub use budget::{Budget, Reservation, Scratch};
    ```
- **Direct Finding**: `BudgetTrimHandler` is completely omitted from the re-exports on line 48 of `ojas-core/src/lib.rs`.
- **Verbatim Error**: Any downstream crate attempting `use ojas_core::BudgetTrimHandler;` encounters:
  ```text
  error[E0432]: unresolved import `ojas_core::BudgetTrimHandler`
    --> ojas-metal/tests/adversarial_m2_verification.rs:24:45
     |
  24 | ...Budget, BudgetTrimHandler, DType, DeviceBuffer,
     |            ^^^^^^^^^^^^^^^^^ no `BudgetTrimHandler` in the root
  ```
  Neither `use ojas_core::BudgetTrimHandler;` nor `use ojas_core::budget::BudgetTrimHandler;` can resolve because `mod budget;` is private.

---

### Observation 2: Buffer Pool Freelist Trimming in `tessl`
- **File**: `/Users/bharath/Code/research/tessl/src/runtime.rs`:
  - Lines 303-314:
    ```rust
    struct BufferPool {
        freelist: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
        cached_bytes: usize,
        max_cache_bytes: usize,
        max_buffer_length: usize,
        live_bytes: usize,
        ledger: Option<Arc<dyn PoolLedger>>,
    }
    ```
  - Lines 374-387:
    ```rust
    /// Destroy every cached buffer, releasing its ledger charge. Returns the
    /// bytes freed. Free-listed buffers are idle: they were parked after the
    /// GPU finished with them.
    fn trim_free_list(&mut self) -> usize {
        let mut freed = 0;
        for (key, buffers) in self.freelist.drain() {
            for buffer in buffers {
                drop(buffer);
                freed += key;
                if let Some(ledger) = &self.ledger {
                    ledger.release(LedgerClass::Cached, key as u64);
                }
            }
        }
        self.cached_bytes = 0;
        freed
    }
    ```
  - Lines 1363-1370:
    ```rust
    /// Cap freelist cache bytes (CLI `--pool-cache-mb`).
    pub fn set_pool_cache_cap_bytes(&self, bytes: usize) {
        if let Ok(mut info) = self.memory_info.lock() {
            info.pool_cache_cap = bytes;
        }
        if let Ok(mut pool) = self.pool.lock() {
            BufferPool::set_max_cache_bytes(&mut pool, bytes);
        }
    }
    ```
  - Lines 750-757:
    ```rust
    /// Metal encoder objects are thread-affine; do not add unsafe Send/Sync.
    /// Host mapping also excludes same-thread reentry into encode and submission.
    ///
    /// ```compile_fail,E0277
    /// use tessl::runtime::GpuRuntime;
    /// fn require_send<T: Send>() {}
    /// require_send::<GpuRuntime>();
    /// ```
    ```
- **Direct Finding**:
  - `BufferPool::trim_free_list(&mut self) -> usize` is private to `runtime.rs` and not exposed on `GpuRuntime`.
  - Calling `set_pool_cache_cap_bytes(0)` permanently sets `max_cache_bytes = 0` and does not return the reclaimed byte count.
  - `GpuRuntime` is strictly `!Send` and `!Sync` (enforced by explicit compile_fail test at line 756).

---

### Observation 3: Trimming Mechanism and Threading in `ojas-metal`
- **File**: `/Users/bharath/Code/research/ojas/ojas-metal/src/link.rs` (lines 1-13, 502-505, 1037-1043):
  - Line 3: `tessl's GpuRuntime is !Send: it holds Metal objects behind Retained pointers and a single host-access lease... One device thread creates the runtime and every buffer, and owns them for their whole lives. Handles on any thread send it a Cmd over an mpsc::Sender (which is Send + Sync) and wait for the reply.`
  - Line 505: `Cmd::TrimPool` exists in `enum Cmd`.
  - Line 1042: `Reply::Done` is returned for `Cmd::TrimPool`. No byte count is currently returned.
- **File**: `/Users/bharath/Code/research/ojas/ojas-metal/src/device.rs` (lines 1040-1051):
  ```rust
  fn drop_pool(&self) {
      let cap = self.rt.memory_info().pool_cache_cap;
      self.rt.set_pool_cache_cap_bytes(0);
      self.rt.set_pool_cache_cap_bytes(cap);
  }

  fn trim_pool(&self) -> Res<Reply> {
      self.recycle()?;
      self.drop_pool();
      Ok(Reply::Done)
  }
  ```
- **File**: `/Users/bharath/Code/research/ojas/ojas-metal/src/backend.rs` (lines 205-212):
  ```rust
  pub fn new(budget: Budget) -> Result<Self, OjasError> {
      let waits = Arc::new(Waits::default());
      let (tx, device) = crate::device::spawn(Arc::clone(&waits), budget.cap_bytes())?;
      Ok(Self {
          link: Arc::new(Link::new(tx, device, waits, budget.cap_bytes())),
          budget,
      })
  }
  ```
- **Direct Finding**: `MetalBackend::new` receives `budget: Budget`, but does not register a `BudgetTrimHandler`. `Cmd::TrimPool` drops the pool by toggling cache cap, but neither computes nor returns the reclaimed byte count.

---

### Observation 4: Child Budgets in `BudgetLedger` & Deadlock Hazard
- **File**: `/Users/bharath/Code/research/ojas/ojas-metal/src/pool_ledger.rs` (lines 53-69, 140-165):
  ```rust
  pub struct BudgetLedger {
      live: Budget,
      cache: Budget,
      held: Mutex<Held>,
  }

  impl BudgetLedger {
      pub fn new(root: &Budget, cache_cap_bytes: u64) -> Arc<Self> {
          Arc::new(Self {
              live: root.child(root.cap_bytes()),
              cache: root.child(cache_cap_bytes),
              held: Mutex::new(Held::default()),
          })
      }
  ```
  And in `try_charge`:
  ```rust
  fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
      let reservation = self
          .budget(class)
          .try_reserve(bytes)
          .map_err(|e| self.refused(class, bytes, e))?;
  ```
- **Direct Finding**:
  - `self.live` is created via `root.child(...)`.
  - When `live.try_reserve(bytes)` fails because `root` is full: `Budget::try_reserve` calls `self.trim_handler()`, which walks up to `self.inner.parent` (i.e. `root`) if no handler is installed on `live`.
  - If `root` has a trim handler that sends `Cmd::TrimPool` via `Link` to the device thread, and this allocation was initiated on the device thread inside `GpuRuntime::alloc_buffer_kind` -> `BufferPool::alloc` -> `charge_fresh`:
    The device thread sends a message to itself and blocks waiting for its own reply -> **RECURSIVE SELF-DEADLOCK**!
  - In `tessl/src/runtime.rs` line 360, `BufferPool::charge_fresh` already contains its own ladder:
    ```rust
    if self.trim_free_list() == 0 {
        return Err(first.to_string());
    }
    match &self.ledger {
        Some(ledger) => ledger.try_charge(LedgerClass::Live, key as u64)...
    ```
    Therefore, `self.live.try_reserve` inside `ledger.try_charge` must fail fast without walking up to the device thread's trim handler!

---

## 2. Logic Chain

1. **Step 1 (Interface Accessibility)**:
   - *Premise*: Crate consumers and sibling crates (`tessl`, `ojas-metal`, tests) must implement or reference `BudgetTrimHandler`.
   - *Observation*: `ojas-core/src/lib.rs` line 48 exports only `{Budget, Reservation, Scratch}`. `mod budget;` is private.
   - *Deduction*: Adding `BudgetTrimHandler` to `pub use budget::{Budget, BudgetTrimHandler, Reservation, Scratch};` in `ojas-core/src/lib.rs` immediately exposes `ojas_core::BudgetTrimHandler` across all crates without breaking `#![forbid(unsafe_code)]` or adding dependencies.

2. **Step 2 (Memory Reclamation & Accounting)**:
   - *Premise*: In `Budget::try_reserve`:
     ```rust
     let freed = handler.trim_memory(bytes);
     if freed > 0 && self.charge(bytes).is_ok() {
         return Ok(Reservation { bytes, budget: self.clone() });
     }
     ```
     `Budget::try_reserve` retries `self.charge(bytes)` **only if** `freed > 0`.
   - *Observation*: `Cmd::TrimPool` currently returns `Reply::Done` (ignoring byte counts). `set_pool_cache_cap_bytes(0)` returns `()`.
   - *Deduction*: Trimming must compute `before.saturating_sub(after)` (or call `BufferPool::trim_free_list` directly) and return `freed as u64`. `Cmd::TrimPool` must reply with `Reply::Trimmed(u64)`.

3. **Step 3 (Safe Cross-Thread Wiring in `ojas-metal`)**:
   - *Premise*: `Budget` is shared across multiple CPU threads. `BudgetTrimHandler` requires `Send + Sync + std::fmt::Debug`. However, `GpuRuntime` is `!Send + !Sync` and must only be accessed from the single device worker thread.
   - *Observation*: `ojas_metal::link::Link` contains `tx: mpsc::Sender<Msg>`, `DeviceInfo`, `waits: Arc<Waits>`, and is pure safe Rust (`Send + Sync`).
   - *Deduction*: Wrapping `Arc<Link>` inside `MetalTrimHandler` allows any host thread calling `Budget::try_reserve` to safely send `Cmd::TrimPool` to the device thread and await `Reply::Trimmed(freed)`.

4. **Step 4 (Deadlock & Lock Cycle Elimination)**:
   - *Premise*: Invoking `trim_memory` must not deadlock, violate lock order, or cause recursive self-wait.
   - *Observations*:
     - `Budget::try_reserve` holds no locks when invoking `handler.trim_memory` (the `RwLockReadGuard` on `self.inner.trim_handler` is dropped before the invocation).
     - In `BudgetLedger::release`, `self.held.lock()` is popped and dropped before dropping the reservation. Lock order is strictly `BufferPool.pool` -> `BudgetLedger.held`.
     - When `GpuRuntime::alloc` runs on the device thread, `BufferPool::charge_fresh` calls `BudgetLedger::try_charge`, which calls `live.try_reserve`. If `live` delegates to `root`'s trim handler, the device thread calls `link.call(Cmd::TrimPool)` to itself, deadlocking.
   - *Deduction*:
     - `BudgetLedger::new` must install a `NoOpTrimHandler` on its child budgets (`live` and `cache`). Because `live.trim_handler()` returns `Some(NoOpTrimHandler)`, it stops recursion and never delegates to `root`. `BufferPool::charge_fresh` will receive `Err(LedgerRefused)` and execute its own internal `self.trim_free_list()`.
     - `GpuRuntime::trim_free_list(&self)` must use `self.pool.try_lock()` rather than `lock()`, guaranteeing zero blocking if called while `pool` is busy.

---

## 3. Caveats

- `GpuRuntime` remains single-threaded (`!Send`, `!Sync`) by design. Do not attempt to add `unsafe impl Send + Sync` for `GpuRuntime` as `objc2-metal` objects (`MTLDevice`, `MTLLibrary`, `MTLCommandQueue`) are thread-affine. Cross-thread trimming is canonically mediated through `ojas-metal::Link`.
- In standalone `tessl` unit tests without `ojas-metal`, `GpuRuntime::trim_free_list(&self)` operates directly on the test thread.
- Memory reclaimed by `trim_free_list()` consists of idle buffers parked in the freelist (`LedgerClass::Cached`). Active in-flight GPU buffers (`LedgerClass::Live`) are never trimmed until completed and recycled.

---

## 4. Conclusion & Concrete Recommendations for Worker 2

### Remediation Roadmap for Worker 2

#### Action 1: Re-export `BudgetTrimHandler` in `ojas-core/src/lib.rs`
In `/Users/bharath/Code/research/ojas/ojas-core/src/lib.rs`, line 48:
```rust
// Replace:
pub use budget::{Budget, Reservation, Scratch};

// With:
pub use budget::{Budget, BudgetTrimHandler, Reservation, Scratch};
```

#### Action 2: Expose `GpuRuntime::trim_free_list` in `tessl/src/runtime.rs`
In `/Users/bharath/Code/research/tessl/src/runtime.rs`, add at line 1371 (in `impl GpuRuntime`):
```rust
    /// Destroy every cached buffer in the pool free list, releasing its ledger
    /// charge. Returns the bytes freed. Free-listed buffers are idle: they
    /// were parked after the GPU finished with them.
    ///
    /// Uses `try_lock` to avoid recursive deadlock if called while the pool
    /// lock is already held on the calling thread.
    pub fn trim_free_list(&self) -> usize {
        if let Ok(mut pool) = self.pool.try_lock() {
            pool.trim_free_list()
        } else {
            0
        }
    }
```

#### Action 3: Add `Reply::Trimmed(u64)` and Byte Tracking in `ojas-metal/src/link.rs` & `device.rs`
In `/Users/bharath/Code/research/ojas/ojas-metal/src/link.rs`:
```rust
#[derive(Debug)]
pub(crate) enum Reply {
    Bufs(Vec<NewBuf>),
    Bytes(Vec<u8>),
    Norm(f32),
    Memory(crate::MetalMemory),
    Done,
    Trimmed(u64),
}
```

In `/Users/bharath/Code/research/ojas/ojas-metal/src/device.rs`:
```rust
    fn drop_pool(&self) -> u64 {
        let before = self.rt.pool_resident().cached_bytes;
        let cap = self.rt.memory_info().pool_cache_cap;
        self.rt.set_pool_cache_cap_bytes(0);
        self.rt.set_pool_cache_cap_bytes(cap);
        let after = self.rt.pool_resident().cached_bytes;
        before.saturating_sub(after)
    }

    /// [`Cmd::TrimPool`]: recycle what was freed since the last waited
    /// commit, then release the whole cache and report bytes freed.
    fn trim_pool(&self) -> Res<Reply> {
        self.recycle()?;
        let freed = self.drop_pool();
        Ok(Reply::Trimmed(freed))
    }
```

#### Action 4: Register `MetalTrimHandler` on Root Budget in `ojas-metal/src/backend.rs`
In `/Users/bharath/Code/research/ojas/ojas-metal/src/backend.rs`:
```rust
use ojas_core::BudgetTrimHandler;

#[derive(Debug)]
pub(crate) struct MetalTrimHandler {
    link: Arc<crate::link::Link>,
}

impl BudgetTrimHandler for MetalTrimHandler {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        match self.link.call(crate::link::Cmd::TrimPool) {
            Ok(crate::link::Reply::Trimmed(freed)) => freed,
            _ => 0,
        }
    }
}

impl MetalBackend {
    pub fn new(budget: Budget) -> Result<Self, OjasError> {
        let waits = Arc::new(Waits::default());
        let (tx, device) = crate::device::spawn(Arc::clone(&waits), budget.cap_bytes())?;
        let link = Arc::new(Link::new(tx, device, waits, budget.cap_bytes()));

        // Wire automatic cache trim hook: when host or backend reservations hit
        // capacity, evict idle pool freelist buffers to reclaim headroom.
        let trim_handler = Arc::new(MetalTrimHandler {
            link: Arc::clone(&link),
        });
        budget.set_trim_handler(trim_handler);

        Ok(Self { link, budget })
    }

    pub fn trim_pool(&self) -> Result<(), OjasError> {
        match self.link.call(Cmd::TrimPool)? {
            Reply::Done | Reply::Trimmed(_) => Ok(()),
            other => Err(metal_err(format!("trim_pool: device returned {other:?}"))),
        }
    }
```

#### Action 5: Isolate Child Budgets in `BudgetLedger::new` (`ojas-metal/src/pool_ledger.rs`)
In `/Users/bharath/Code/research/ojas/ojas-metal/src/pool_ledger.rs`:
```rust
use ojas_core::BudgetTrimHandler;

#[derive(Debug)]
struct NoOpTrim;

impl BudgetTrimHandler for NoOpTrim {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        0
    }
}

impl BudgetLedger {
    pub fn new(root: &Budget, cache_cap_bytes: u64) -> Arc<Self> {
        let live = root.child(root.cap_bytes());
        let cache = root.child(cache_cap_bytes);

        // Child budgets must not delegate to root's trim handler.
        // BufferPool::charge_fresh already drives its own ladder,
        // so try_charge must fail fast without recursing back to the device thread.
        let noop = Arc::new(NoOpTrim);
        live.set_trim_handler(Arc::clone(&noop));
        cache.set_trim_handler(noop);

        Arc::new(Self {
            live,
            cache,
            held: Mutex::new(Held::default()),
        })
    }
```

---

## 5. Verification Method

To independently verify these recommendations once implemented:

```bash
# 1. Verify BudgetTrimHandler export from ojas-core:
cargo check --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core --test adamw_scalars

# 2. Verify compilation in ojas-metal and tessl using BudgetTrimHandler:
cargo check --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --all-targets
cargo check --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --tests

# 3. Verify tessl pool ledger tests pass without regressions:
cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib pool_ledger

# 4. Verify ojas-metal pool ledger tests pass without deadlocks:
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-metal --test pool_ledger

# 5. Verify format and tracking:
/Users/bharath/Code/research/ojas/scripts/fmt_check.sh
/Users/bharath/Code/research/ojas/scripts/check_tracked_targets.sh
```

**Invalidation Conditions**:
- If `live.try_reserve` inside `BudgetLedger::try_charge` recurses to `MetalTrimHandler`, the device thread will hang on `Cmd::TrimPool` channel receive.
- If `GpuRuntime::trim_free_list` uses `lock()` rather than `try_lock()`, re-entrant calls within `alloc` will deadlock.
- If `MetalTrimHandler::trim_memory` returns `0` despite freeing buffers, `Budget::try_reserve` will abort without retrying the charge.
