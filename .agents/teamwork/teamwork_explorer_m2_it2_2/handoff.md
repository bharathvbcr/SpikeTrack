# Handoff Report: Explorer 2 — Bridging `tessl::shared_buf::SharedBuf` to `ojas_core::hal::MappedBuffer`

**Milestone**: Milestone 2 Iteration 2 (Remediation)  
**Task**: Architectural Analysis and Concrete Implementation Design for Bridging `tessl::shared_buf::SharedBuf` to `ojas_core::hal::MappedBuffer`  
**Author**: Explorer 2 (Investigation & Synthesis)  

---

## 1. Observation

### 1.1 The Missing Seam Identified by Challenger 1
- **File**: `ojas-core/src/hal.rs`, lines 415–452 defines `pub trait MappedBuffer: DeviceBuffer`.
- **File**: `ojas-core/src/tensor.rs`, lines 967–993 defines `Tensor::from_mapped(buffer: Arc<dyn MappedBuffer>, ...)` and `Payload::Mapped(Arc<dyn MappedBuffer>)`.
- **Empirical Search**:
  - `grep_search Query="MappedBuffer" SearchPath="tessl/src"` returned **0 results**.
  - `grep_search Query="MappedBuffer" SearchPath="ojas-metal/src"` returned **0 results**.
  - Neither `tessl::shared_buf::SharedBuf`, `tessl::shared_buf::SharedAlloc`, nor `ojas_metal::MetalBuffer` implements `MappedBuffer`.
  - In unit tests, `Payload::Mapped` is exercised exclusively with a test mock `FakeMapped` in `ojas-core/src/tensor.rs`. No real Metal unified memory buffer could be wrapped into `Tensor::from_mapped`.

### 1.2 Trait Contracts in `ojas-core`
1. **`DeviceBuffer`** (`ojas-core/src/tensor.rs:20–34`):
   ```rust
   pub trait DeviceBuffer: Any + Send + Sync + fmt::Debug {
       fn backend(&self) -> BackendId;
       fn byte_len(&self) -> usize;
       fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError>;
       fn as_any(&self) -> &dyn Any;
   }
   ```
2. **`MappedBuffer`** (`ojas-core/src/hal.rs:415–452`):
   ```rust
   pub trait MappedBuffer: DeviceBuffer {
       fn host_ptr(&self) -> *mut u8 { std::ptr::null_mut() }
       fn gpu_address(&self) -> u64 { 0 }
       fn timeline(&self) -> &dyn Timeline { ... }
       fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> { ... }
       fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> { ... }
   }
   ```
3. **`Timeline`** (`ojas-core/src/hal.rs:288–344`):
   ```rust
   pub trait Timeline: Send + Sync {
       fn submitted(&self) -> Result<u64, FenceError>;
       fn completed(&self) -> Result<u64, FenceError>;
       fn current_value(&self) -> u64 { self.completed().unwrap_or(0) }
       fn wait_for(&self, value: u64, timeout: Option<std::time::Duration>) -> Result<(), AllocError>;
       fn wait(&self, value: u64) -> Result<(), FenceError>;
   }
   ```
4. **`HostLease` & `HostLeaseMut`** (`ojas-core/src/hal.rs:353–402`):
   ```rust
   pub trait HostLease: Send + Sync {
       fn as_bytes(&self) -> &[u8];
       fn as_f32(&self) -> Result<&[f32], OjasError>;
       fn as_u32(&self) -> Result<&[u32], OjasError>;
   }

   pub trait HostLeaseMut: HostLease {
       fn as_bytes_mut(&mut self) -> &mut [u8];
       fn as_f32_mut(&mut self) -> Result<&mut [f32], OjasError>;
       fn as_u32_mut(&mut self) -> Result<&mut [u32], OjasError>;
   }
   ```

### 1.3 State of `tessl::shared_buf::SharedBuf` and `HostFence`
- **File**: `tessl/src/shared_buf.rs:128–144`:
  ```rust
  pub struct SharedBuf {
      ptr: NonNull<u8>,
      len: usize,
      _mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
      last_use: AtomicU64,
      last_write: AtomicU64,
      lease: AtomicI32,
      fence: HostFence,
  }
  ```
  - Carries the only audited `unsafe impl Send + Sync` (`tessl/src/shared_buf.rs:172–175`).
  - Implements thread-safe leases via `pub fn read(&self) -> Result<HostRead<'_>, HostAccessError>` and `pub fn write(&self) -> Result<HostWrite<'_>, HostAccessError>`.
  - `HostRead<'a>` and `HostWrite<'a>` borrow `&'a SharedBuf`.
- **File**: `tessl/src/runtime/host_fence.rs:35–45`:
  ```rust
  pub(crate) struct HostFence {
      lifetime_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
      submitted: Arc<AtomicU64>,
      wait_ms: Arc<AtomicU64>,
      timeouts: Arc<AtomicU32>,
      poison: Arc<RuntimePoison>,
  }
  ```
  - Holds `MTLSharedEvent` (which is `Send + Sync` in `objc2-metal`) and atomic counters.
  - Implements `submitted(&self) -> u64`, `checked_value(&self) -> Result<u64, String>`, and `wait_stamp(&self, stamp: u64) -> Result<(), HostAccessError>`.
- **File**: `tessl/Cargo.toml:112`:
  - `ojas-core = { path = "../ojas/ojas-core" }` is already an explicit dependency of `tessl`.

### 1.4 Architectural and Safety Constraints
- `PROJECT.md:4` & `PROJECT.md:68`:
  - `MappedBuffer: Trait implemented by Tessl's SharedBuf`.
- `ojas/AGENTS.md` and crate roots:
  - `ojas-core/src/lib.rs:9`: `#![forbid(unsafe_code)]`.
  - `ojas-metal/src/lib.rs:10`: `#![forbid(unsafe_code)]`.
  - "The one sanctioned exception outside these crates is in the sibling tessl checkout, not in this workspace: tessl's `SharedBuf` (`src/shared_buf.rs`) carries a single audited `unsafe impl Send + Sync`... ojas code reaches shared memory only through tessl's safe `SharedBuf` API; `ojas-core` and `ojas-metal` stay `#![forbid(unsafe_code)]` and must not add their own `unsafe impl Send` or `Sync` for a device buffer."

---

## 2. Logic Chain

1. **Location of the Implementation (Rust Orphan Rules)**:
   - `MappedBuffer` is defined in `ojas-core`.
   - `SharedBuf` is defined in `tessl`.
   - In `ojas-metal`, both `MappedBuffer` and `SharedBuf` are foreign types. Under Rust's orphan rule (`E0117`), `ojas-metal` **cannot** implement `MappedBuffer` for `SharedBuf` without wrapping it in a newtype.
   - In `tessl`, `SharedBuf` is a local type and `tessl` depends on `ojas-core`. Therefore, `tessl` **can and must** implement `MappedBuffer` for `SharedBuf` directly, aligning with `PROJECT.md:68`.

2. **The Lifetime & Boxing Paradox**:
   - `MappedBuffer::host_read(&self)` returns `Result<Box<dyn HostLease>, FenceError>`.
   - In Rust, `Box<dyn Trait>` defaults to `Box<dyn Trait + 'static>`.
   - The current `HostRead<'a>` in `shared_buf.rs` borrows `&'a SharedBuf`. An instance of `HostRead<'a>` has lifetime `'a` and cannot be cast into `Box<dyn HostLease + 'static>` unless `'a == 'static`.
   - Furthermore, if a caller calls `let lease = tensor.host_view()?; drop(tensor);`, the `Tensor` and its underlying `Arc<dyn MappedBuffer>` drop. If the lease does not retain the underlying buffer, accessing memory via the lease becomes a use-after-free.
   - **Deduction**: `HostLease` must own an `Arc` reference to the shared buffer's inner state.
   - **Resolution**: Refactor `SharedBuf` to wrap an `Arc<SharedBufInner>`:
     ```rust
     #[derive(Clone)]
     pub struct SharedBuf {
         inner: Arc<SharedBufInner>,
     }
     ```
     `SharedHostReadLease` and `SharedHostWriteLease` will hold `Arc<SharedBufInner>`. This provides `'static` lifetime, guarantees memory validity, prevents data races, and requires **zero new `unsafe` blocks**.

3. **Pool Allocation Lifecycle Safety**:
   - In `tessl/src/tensor.rs:82–99`, `PooledBuffer::drop` returns cold buffers to `GpuRuntime`'s freelist (`schedule_cold_recycle`).
   - If `SharedBuf` held only `Retained<ProtocolObject<dyn MTLBuffer>>` without retaining `PooledBuffer`, dropping `SharedAlloc` would recycle the buffer into the pool freelist while `SharedBuf` or `Tensor::from_mapped` is still using it!
   - **Resolution**: `SharedBufInner` must hold `_pooled: Option<Arc<PooledBuffer>>`. As long as `SharedBuf`, `Tensor`, or any active `HostLease` is alive, the `PooledBuffer` strong count remains $\ge 1$, completely preventing premature pool recycling.

4. **Timeline Synchronization Bridge**:
   - `HostFence` in `tessl/src/runtime/host_fence.rs` already wraps the private `MTLSharedEvent`, tracks `submitted`, checks poison, and fences via `wait_stamp(stamp)`.
   - Implementing `ojas_core::hal::Timeline for HostFence` directly maps:
     - `submitted()` $\rightarrow$ `self.submitted()`
     - `completed()` $\rightarrow$ `self.checked_value()`
     - `wait(value)` $\rightarrow$ `self.wait_stamp(value)`
   - `SharedBuf::timeline(&self) -> &dyn Timeline` returns `&self.inner.fence`.

5. **Safety Invariance Preservation**:
   - `SharedBufInner` continues to rely on the existing, single audited `unsafe impl Send for SharedBufInner {}` and `unsafe impl Sync for SharedBufInner {}`.
   - All trait implementations (`Timeline`, `HostLease`, `HostLeaseMut`, `DeviceBuffer`, `MappedBuffer`) contain **zero `unsafe` code**.
   - `ojas-core` and `ojas-metal` remain strictly `#![forbid(unsafe_code)]`.

---

## 3. Caveats

1. **`MetalBackend::upload` Integration**:
   - Reviewer 1 Finding 2 noted that `MetalBackend::upload()` in `ojas-metal/src/backend.rs` does not yet intercept `Payload::Mapped`.
   - Once `SharedBuf` implements `MappedBuffer`, `MetalBackend::upload` should inspect `tensor.mapped_buffer()`. If present and `mbuf.backend() == BackendId::Metal`, `upload` can adopt or clone the tensor zero-copy instead of allocating a fresh `MetalBuffer`.
2. **`Tensor::to_host` Behavior on Mapped Tensors**:
   - Reviewer 1 Finding 1 noted that `Tensor::to_host()` returns `Ok(self.clone())` which retains `Payload::Mapped`. Because `self.device()` is `Some(BackendId::Metal)`, host accessors like `f32_slice()` fail with `OjasError::Placement`.
   - For zero-copy in-place access, `tensor.host_view()` is the intended API. When an explicit decoupled host copy is required, `to_host()` should copy through `read_bytes` or use `to_f32_vec()`/`host_view()`.

---

## 4. Conclusion & Concrete Design for Worker 2

The bridge between `tessl::shared_buf::SharedBuf` and `ojas_core::hal::MappedBuffer` should be implemented directly in `tessl`.

### Implementation Specification

#### File 1: `tessl/src/runtime/host_fence.rs`
Add `Timeline` implementation for `HostFence`:

```rust
use ojas_core::backend::BackendId;
use ojas_core::hal::{FenceError, Timeline};
use ojas_core::OjasError;

impl Timeline for HostFence {
    fn submitted(&self) -> Result<u64, FenceError> {
        if self.ensure_not_poisoned().is_err() {
            return Err(FenceError::Poisoned);
        }
        Ok(self.submitted())
    }

    fn completed(&self) -> Result<u64, FenceError> {
        if self.ensure_not_poisoned().is_err() {
            return Err(FenceError::Poisoned);
        }
        self.checked_value().map_err(|_| FenceError::Poisoned)
    }

    fn current_value(&self) -> u64 {
        self.checked_value().unwrap_or(0)
    }

    fn wait(&self, value: u64) -> Result<(), FenceError> {
        match self.wait_stamp(value) {
            Ok(()) => Ok(()),
            Err(HostAccessError::NotSubmitted { needed, submitted }) => {
                Err(FenceError::NotSubmitted {
                    value: needed,
                    submitted,
                })
            }
            Err(HostAccessError::Busy(why)) => Err(FenceError::Busy { detail: why }),
            Err(HostAccessError::Runtime(why)) => {
                if why.contains("poisoned") || why.contains("DeviceLost") {
                    Err(FenceError::Poisoned)
                } else if why.contains("timeout") || why.contains("exceeded") {
                    Err(FenceError::TimedOut {
                        value,
                        completed: self.current_value(),
                    })
                } else {
                    Err(FenceError::Refused(OjasError::Backend {
                        id: BackendId::Metal,
                        detail: why,
                    }))
                }
            }
        }
    }
}
```

#### File 2: `tessl/src/shared_buf.rs`
Refactor `SharedBuf` to wrap `Arc<SharedBufInner>` and implement `DeviceBuffer`, `MappedBuffer`, `HostLease`, and `HostLeaseMut`:

```rust
use ojas_core::backend::BackendId;
use ojas_core::hal::{FenceError, HostLease, HostLeaseMut, MappedBuffer, Timeline};
use ojas_core::tensor::DeviceBuffer;
use ojas_core::OjasError;
use std::any::Any;
use std::sync::Arc;

/// The Send + Sync core of a shared buffer, wrapping Arc<SharedBufInner>.
#[derive(Clone)]
pub struct SharedBuf {
    pub(crate) inner: Arc<SharedBufInner>,
}

pub(crate) struct SharedBufInner {
    ptr: NonNull<u8>,
    len: usize,
    gpu_addr: u64,
    _mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
    _pooled: Option<Arc<crate::tensor::PooledBuffer>>,
    last_use: AtomicU64,
    last_write: AtomicU64,
    lease: AtomicI32,
    fence: HostFence,
}

unsafe impl Send for SharedBufInner {}
unsafe impl Sync for SharedBufInner {}

impl SharedBuf {
    pub(crate) fn new(gpu: &GpuBuffer, len: usize, fence: HostFence) -> Result<Self, String> {
        let mtl = &gpu.inner.buffer;
        let mode = mtl.storageMode();
        if mode != MTLStorageMode::Shared {
            return Err(format!("shared buffer needs StorageModeShared storage, got {mode:?}"));
        }
        if len == 0 || len > gpu.nbytes() {
            return Err(format!(
                "shared buffer length {len} is outside the allocation ({} bytes)",
                gpu.nbytes()
            ));
        }
        let inner = SharedBufInner {
            ptr: mtl.contents().cast::<u8>(),
            len,
            gpu_addr: gpu.inner.gpu_addr,
            _mtl: mtl.clone(),
            _pooled: Some(Arc::clone(&gpu.inner)),
            last_use: AtomicU64::new(0),
            last_write: AtomicU64::new(0),
            lease: AtomicI32::new(0),
            fence,
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn last_use(&self) -> u64 {
        self.inner.last_use.load(Ordering::SeqCst)
    }

    pub fn last_write(&self) -> u64 {
        self.inner.last_write.load(Ordering::SeqCst)
    }

    pub fn host_addr(&self) -> usize {
        self.inner.ptr.as_ptr() as usize
    }

    pub fn check_gpu_access(&self, role: GpuRole) -> Result<(), HostAccessError> {
        Self::gpu_access(self.inner.lease.load(Ordering::SeqCst), role)
    }

    pub(crate) fn stamp_gpu(&self, commit: u64, role: GpuRole) -> Result<(), HostAccessError> {
        self.inner.last_use.fetch_max(commit, Ordering::SeqCst);
        if role == GpuRole::Write {
            self.inner.last_write.fetch_max(commit, Ordering::SeqCst);
        }
        Self::gpu_access(self.inner.lease.load(Ordering::SeqCst), role)
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    pub(crate) fn bytes_mut(&self) -> &mut [u8] {
        self.inner.bytes_mut()
    }

    fn acquire_read(&self) -> Result<SharedHostReadLease, FenceError> {
        self.inner.fence.ensure_not_poisoned().map_err(|_| FenceError::Poisoned)?;
        let mut cur = self.inner.lease.load(Ordering::SeqCst);
        loop {
            if cur < 0 {
                return Err(FenceError::Busy {
                    detail: "a host write lease is live".into(),
                });
            }
            if cur == i32::MAX {
                return Err(FenceError::Busy {
                    detail: "too many host read leases".into(),
                });
            }
            match self.inner.lease.compare_exchange_weak(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        let stamp = self.inner.last_write.load(Ordering::SeqCst);
        if let Err(e) = self.inner.fence.wait_stamp(stamp) {
            self.inner.lease.fetch_sub(1, Ordering::SeqCst);
            return Err(map_host_access_error(e, stamp, &self.inner.fence));
        }
        Ok(SharedHostReadLease {
            inner: Arc::clone(&self.inner),
        })
    }

    fn acquire_write(&self) -> Result<SharedHostWriteLease, FenceError> {
        self.inner.fence.ensure_not_poisoned().map_err(|_| FenceError::Poisoned)?;
        if self.inner.lease.compare_exchange(0, -1, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err(FenceError::Busy {
                detail: "a host lease is live".into(),
            });
        }
        let stamp = self.inner.last_use.load(Ordering::SeqCst);
        if let Err(e) = self.inner.fence.wait_stamp(stamp) {
            self.inner.lease.store(0, Ordering::SeqCst);
            return Err(map_host_access_error(e, stamp, &self.inner.fence));
        }
        Ok(SharedHostWriteLease {
            inner: Arc::clone(&self.inner),
        })
    }

    pub fn read(&self) -> Result<HostRead<'_>, HostAccessError> {
        self.inner.fence.ensure_not_poisoned().map_err(HostAccessError::Runtime)?;
        let mut cur = self.inner.lease.load(Ordering::SeqCst);
        loop {
            if cur < 0 {
                return Err(HostAccessError::Busy("a host write lease is live".into()));
            }
            if cur == i32::MAX {
                return Err(HostAccessError::Busy("too many host read leases".into()));
            }
            match self.inner.lease.compare_exchange_weak(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        let lease = HostRead { buf: self };
        self.inner.fence.wait_stamp(self.inner.last_write.load(Ordering::SeqCst))?;
        Ok(lease)
    }

    pub fn write(&self) -> Result<HostWrite<'_>, HostAccessError> {
        self.inner.fence.ensure_not_poisoned().map_err(HostAccessError::Runtime)?;
        if self.inner.lease.compare_exchange(0, -1, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err(HostAccessError::Busy("a host lease is live".into()));
        }
        let lease = HostWrite { buf: self };
        self.inner.fence.wait_stamp(self.inner.last_use.load(Ordering::SeqCst))?;
        Ok(lease)
    }
}

impl SharedBufInner {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[allow(clippy::mut_from_ref)]
    fn bytes_mut(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

fn map_host_access_error(e: HostAccessError, stamp: u64, fence: &HostFence) -> FenceError {
    match e {
        HostAccessError::NotSubmitted { needed, submitted } => {
            FenceError::NotSubmitted { value: needed, submitted }
        }
        HostAccessError::Busy(detail) => FenceError::Busy { detail },
        HostAccessError::Runtime(why) => {
            if why.contains("poisoned") || why.contains("DeviceLost") {
                FenceError::Poisoned
            } else if why.contains("timeout") || why.contains("exceeded") {
                FenceError::TimedOut {
                    value: stamp,
                    completed: fence.current_value(),
                }
            } else {
                FenceError::Refused(OjasError::Backend {
                    id: BackendId::Metal,
                    detail: why,
                })
            }
        }
    }
}

pub struct SharedHostReadLease {
    inner: Arc<SharedBufInner>,
}

impl HostLease for SharedHostReadLease {
    fn as_bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_f32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }

    fn as_u32(&self) -> Result<&[u32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_u32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }
}

impl Drop for SharedHostReadLease {
    fn drop(&mut self) {
        self.inner.lease.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct SharedHostWriteLease {
    inner: Arc<SharedBufInner>,
}

impl HostLease for SharedHostWriteLease {
    fn as_bytes(&self) -> &[u8] {
        self.inner.bytes()
    }

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_f32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }

    fn as_u32(&self) -> Result<&[u32], OjasError> {
        words(self.inner.bytes()).ok_or_else(|| OjasError::Shape {
            op: "HostLease::as_u32",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }
}

impl HostLeaseMut for SharedHostWriteLease {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.bytes_mut()
    }

    fn as_f32_mut(&mut self) -> Result<&mut [f32], OjasError> {
        words_mut(self.inner.bytes_mut()).ok_or_else(|| OjasError::Shape {
            op: "HostLeaseMut::as_f32_mut",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }

    fn as_u32_mut(&mut self) -> Result<&mut [u32], OjasError> {
        words_mut(self.inner.bytes_mut()).ok_or_else(|| OjasError::Shape {
            op: "HostLeaseMut::as_u32_mut",
            detail: "buffer length or address is not 4-aligned".to_string(),
        })
    }
}

impl Drop for SharedHostWriteLease {
    fn drop(&mut self) {
        self.inner.lease.store(0, Ordering::SeqCst);
    }
}

impl fmt::Debug for SharedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedBuf")
            .field("len", &self.inner.len)
            .field("gpu_addr", &self.inner.gpu_addr)
            .field("last_use", &self.last_use())
            .field("last_write", &self.last_write())
            .field("lease", &self.inner.lease.load(Ordering::SeqCst))
            .finish()
    }
}

impl DeviceBuffer for SharedBuf {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.inner.len
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        const OP: &str = "SharedBuf::read_bytes";
        let end = offset.checked_add(len).ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "offset + len overflows".into(),
        })?;
        if end > self.inner.len {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("range {offset}..{end} exceeds buffer length {}", self.inner.len),
            });
        }
        let lease = self.acquire_read().map_err(|e| e.into_ojas(BackendId::Metal))?;
        Ok(lease.as_bytes()[offset..end].to_vec())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MappedBuffer for SharedBuf {
    fn host_ptr(&self) -> *mut u8 {
        self.inner.ptr.as_ptr()
    }

    fn gpu_address(&self) -> u64 {
        self.inner.gpu_addr
    }

    fn timeline(&self) -> &dyn Timeline {
        &self.inner.fence
    }

    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        let lease = self.acquire_read()?;
        Ok(Box::new(lease))
    }

    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        let lease = self.acquire_write()?;
        Ok(Box::new(lease))
    }
}

impl SharedAlloc {
    /// Convert this allocation into an owning `SharedBuf` (implements `MappedBuffer`),
    /// keeping the pool allocation alive as long as the `SharedBuf` is alive.
    pub fn into_shared_buf(self) -> SharedBuf {
        self.core
    }

    /// Access the underlying `SharedBuf` (implements `MappedBuffer`).
    pub fn shared_buf(&self) -> SharedBuf {
        self.core.clone()
    }
}
```

---

## 5. Verification Method

To independently verify this bridge once implemented by Worker 2:

1. **Verify Trait Conformance in `tessl`**:
   ```bash
   cargo check --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --all-targets
   ```
2. **Verify End-to-End Metal UMA `Tensor::from_mapped`**:
   Write an integration test in `tessl/tests/mapped_buffer_uma.rs` or `ojas-metal/tests/mapped_buffer_uma.rs`:
   ```rust
   let rt = GpuRuntime::new().unwrap();
   let alloc = SharedAlloc::new(&rt, 1024 * 4).unwrap();
   let shared_buf = alloc.into_shared_buf();
   let mapped: Arc<dyn MappedBuffer> = Arc::new(shared_buf);
   
   let budget = Budget::new(64 * 1024 * 1024);
   let tensor = Tensor::from_mapped(mapped, &[1024], DType::F32, &budget).unwrap();
   
   // Zero-copy host read lease:
   let view = tensor.host_view().unwrap();
   let slice = view.as_f32().unwrap();
   assert_eq!(slice.len(), 1024);
   
   // Direct host access fails closed without silent fallback:
   assert!(tensor.f32_slice().is_err());
   ```
3. **Verify Existing Suites Pass Without Regressions**:
   ```bash
   cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --lib shared_buf
   cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --test shared_runtime
   cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core
   ```
