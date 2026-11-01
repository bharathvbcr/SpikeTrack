# Handoff Report: Milestone 2 — HAL Traits & Zero-Copy Unified Buffer Architecture (Features 7 & 9)

**Author**: Explorer 1 (Milestone 2)  
**Date**: 2026-10-31  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_1/`  
**Target Scope**: Feature 7 (HAL Traits in `ojas-core`), Feature 9 (Zero-Copy UMA Buffer Sharing on `Tensor`)  

---

## 1. Observation

### 1.1 Existing Codebase Assets in `ojas-core`
- **Pure Rust and `#![forbid(unsafe_code)]`**:
  `ojas-core/src/lib.rs:9` enforces `#![forbid(unsafe_code)]`.
  `ojas-core/Cargo.toml:10` has an empty `[dependencies]` table, confirming zero external dependencies.
- **Existing HAL Foundation (`ojas-core/src/hal.rs`)**:
  `ojas-core/src/hal.rs` already contains foundational definitions for:
  - `AllocError` (lines 61-76): `OverBudget`, `Exhausted`, `TooLarge`, `DeviceLost`.
  - `FenceError` (lines 169-186): `NotSubmitted`, `Busy`, `Poisoned`, `TimedOut`, `Refused`.
  - `Timeline` (lines 269-297): `submitted()`, `completed()`, `wait()`.
  - `HostLease` / `HostLeaseMut` (lines 306-355): `as_bytes()`, `as_f32()`, `as_u32()`, `as_bytes_mut()`, `as_f32_mut()`, `as_u32_mut()`.
  - `MappedBuffer: DeviceBuffer` (lines 368-387): `host_read()`, `host_write()`.
  - `Provenance` (lines 397-430): `Exact`, `Fast`.
  - **Critical Test Invariant**: Line 572 `the_hal_list_names_every_trait_method` uses compile-time reflection (`include_str!("hal.rs")`) to verify that every method declared on `Timeline`, `HostLease`, `HostLeaseMut`, and `MappedBuffer` is explicitly tested in `default_calls()`. All 14 tests in `hal.rs` currently pass (`cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core --lib hal::tests`).
- **Current Tensor Storage (`ojas-core/src/tensor.rs`)**:
  - `Payload` (lines 152-156) only defines:
    ```rust
    #[derive(Debug)]
    enum Payload {
        Host(HostData),
        Device(Arc<dyn DeviceBuffer>),
    }
    ```
  - `DeviceBuffer` (lines 19-33):
    ```rust
    pub trait DeviceBuffer: Any + Send + Sync + fmt::Debug {
        fn backend(&self) -> BackendId;
        fn byte_len(&self) -> usize;
        fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError>;
        fn as_any(&self) -> &dyn Any;
    }
    ```
  - `to_host` (lines 1005-1080): Currently forces an expensive chunked readback via `buf.read_bytes(offset + done, n)` into a freshly allocated `HostData::try_zeroed` buffer, charging `Budget::device_readbacks` and performing multiple memcpys.
- **Current Metal Device Readback Overhead (`ojas-metal/src/device.rs`)**:
  - In `ojas-metal/src/device.rs:1823-1854`, `read` waits on the GPU via `self.settle(Scan::All, Wait::Read)?`, reads `buf.try_contents_u8()`, copies bytes with `map[off..end].to_vec()`, passes them over an IPC channel via `Reply::Bytes`, and `Tensor::to_host` copies them again into `HostData`. This is a double memcpy on unified memory where bytes are already host-resident.

### 1.2 Memory & Runtime Architecture in `tessl`
- **StorageModeShared by Default**:
  `tessl/src/runtime.rs:299`:
  ```rust
  let b = device
      .newBufferWithLength_options(key, MTLResourceOptions::StorageModeShared)
      .ok_or_else(|| format!("newBuffer({key}) failed"))?;
  ```
  Every buffer allocated in `BufferPool` is `MTLResourceOptions::StorageModeShared`.
- **Apple Silicon Hardware Coherence**:
  On Apple Silicon, `StorageModeShared` provides hardware cache coherence between CPU and GPU. The host virtual address `buffer.contents()` is fixed and valid for the entire lifetime of the buffer.
- **GPU Timeline Synchronization**:
  `tessl/src/runtime.rs:391`: `Metal4EncodePackage.lifetime_event` is a retained `MTLSharedEvent`.
  `tessl/src/runtime.rs:1466`: `m4.lifetime_event.waitUntilSignaledValue_timeoutMS(value, budget)` blocks on the completion of GPU work. `lifetime_event.signaledValue()` returns the current monotonically increasing completed timestamp.
- **Spike Benchmark Proof (`ojas-ug-spike-uma/src/main.rs`)**:
  Proves zero-copy pointer sharing across CPU and GPU on `MTLBuffer` using `contents()` and `MTLSharedEvent` synchronization without data corruption or memory copies.

---

## 2. Logic Chain

1. **Pure Rust HAL Requirements**:
   `ojas-core` is the root dependency (`ojas-metal -> tessl -> ojas-core`). Because `ojas-core` enforces `#![forbid(unsafe_code)]` and zero external dependencies, its HAL traits cannot directly invoke Apple Metal APIs or use `unsafe` dereferences.
   *Inference*: The HAL traits in `ojas-core` must define pure safe interfaces (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`) with safe accessor types (`*mut u8` pointer passing is valid safe syntax in Rust, but slice conversions `as_bytes()` must be provided by implementations or through `HostLease`).

2. **Unified Memory Coherence on Apple Silicon**:
   Because `tessl` allocates all buffers with `MTLResourceOptions::StorageModeShared`, both CPU and GPU access the identical physical DRAM pages.
   *Inference*: Memory copies during `to_host()` and `upload()` are entirely redundant for shared buffers. The only required operation when transitioning buffer ownership between CPU and GPU is an execution fence on the timeline (`lifetime_event`).

3. **Placement Invariant & No Silent Fallback**:
   `PROJECT.md` dictates that placement remains strictly explicit (`BackendId::Cpu` vs `BackendId::Metal`). CPU kernels cannot silently execute on GPU buffers without explicit leasing.
   *Inference*: `Tensor::host_view()` and `Tensor::host_view_mut()` must provide short-scoped RAII `HostLease` guards that explicitly synchronize with the timeline and enforce reader/writer exclusion, while standard CPU tensor slice accessors (`f32_slice`) continue to refuse device tensors with `OjasError::Placement`.

4. **Zero-Copy Tensor Representation**:
   Adding `Payload::Mapped(Arc<dyn MappedBuffer>)` to `Tensor` allows `Tensor` to hold a reference to unified memory.
   *Inference*:
   - `to_host()` on a `Payload::Mapped` tensor waits on the timeline fence and returns a zero-copy representation sharing the underlying buffer without invoking `read_bytes` or allocating a new `HostData` buffer.
   - `upload()` on `MetalBackend` inspects the payload: if it is already a `MappedBuffer` allocated in tessl's pool, it verifies timeline synchronization and returns a device tensor without copying bytes.

5. **Authoritative Budget Accounting**:
   `ojas_core::Budget` tracks live allocations and readback counts.
   *Inference*: `tessl::BufferPool` must reserve buffer allocations from a child `Budget` (`pool.cache`). Allocation failures must produce typed `AllocError::Exhausted { requested, available }`, eliminating string scraping.

---

## 3. Caveats

1. **Objective-C / Metal Thread-Safety (`Send + Sync`)**:
   `objc2-metal` does not automatically mark `MTLBuffer` as `Send` or `Sync`. `SharedBuf` in `tessl` wraps `Retained<ProtocolObject<dyn MTLBuffer>>` and requires:
   ```rust
   unsafe impl Send for SharedBuf {}
   unsafe impl Sync for SharedBuf {}
   ```
   *Safety Justification (Decision D6)*: On Apple Silicon, `buffer.contents()` is fixed and stable. The wrapper serializes concurrent access via atomic reader/writer counters and timeline event waits.
2. **Discrete GPU and Non-Metal Backends**:
   Backends like CUDA, HIP, and portable `wgpu` do not have unified memory on typical host systems. For those backends, `MappedBuffer` defaults refuse with `FenceError::Refused(OjasError::Unsupported)`, preserving explicit copy semantics without silent fallbacks.
3. **Numerics & Provenance**:
   Metal GPU computations operate under `Numerics::Fast`. Bytes produced on Metal carry `Provenance::Fast`. When leased to CPU, operations under `Numerics::Exact` must account for fast-tier provenance according to project policy.

---

## 4. Conclusion & Detailed Technical Specifications

### 4.1 Feature 7: HAL Traits in `ojas-core`

#### File: `ojas-core/src/hal.rs`

```rust
use crate::backend::{BackendId, Numerics};
use crate::tensor::DeviceBuffer;
use crate::OjasError;
use std::any::Any;
use std::fmt;
use std::time::Duration;

/// Typed allocation failure from a unified or device buffer pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocError {
    /// Allocation requested more than the budget allows.
    OverBudget { requested: u64, cap: u64, live: u64 },
    /// Memory pool exhausted; available memory is insufficient.
    Exhausted { requested: u64, available: u64 },
    /// Request exceeds physical device limit.
    TooLarge { requested: u64, cap: u64, live: u64 },
    /// Requested byte size or address violates hardware alignment constraints.
    InvalidAlignment { requested: usize, alignment: usize },
    /// Hardware device loss or fatal timeline poisoning.
    DeviceLost { backend: BackendId, detail: String },
}

impl AllocError {
    pub fn requested(&self) -> Option<u64> {
        match self {
            AllocError::OverBudget { requested, .. }
            | AllocError::Exhausted { requested, .. }
            | AllocError::TooLarge { requested, .. } => Some(*requested),
            AllocError::InvalidAlignment { requested, .. } => Some(*requested as u64),
            AllocError::DeviceLost { .. } => None,
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(self, AllocError::OverBudget { .. } | AllocError::Exhausted { .. })
    }
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OverBudget { requested, cap, live } => {
                write!(f, "allocation over budget: requested {requested} bytes, cap {cap}, live {live}")
            }
            AllocError::Exhausted { requested, available } => {
                write!(f, "device memory exhausted: requested {requested} bytes, available {available}")
            }
            AllocError::TooLarge { requested, cap, live } => {
                write!(f, "allocation exceeds device limit: requested {requested} bytes, limit {cap}, live {live}")
            }
            AllocError::InvalidAlignment { requested, alignment } => {
                write!(f, "invalid buffer alignment: {requested} bytes not aligned to {alignment}")
            }
            AllocError::DeviceLost { backend, detail } => {
                write!(f, "backend {backend:?}: device lost: {detail}")
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<AllocError> for OjasError {
    fn from(e: AllocError) -> OjasError {
        match e {
            AllocError::OverBudget { requested, cap, live } => {
                OjasError::CapacityExceeded { requested, cap, live }
            }
            AllocError::Exhausted { requested, available } => {
                OjasError::CapacityExceeded { requested, cap: available, live: 0 }
            }
            AllocError::TooLarge { requested, cap, live } => {
                OjasError::CapacityExceeded { requested, cap, live }
            }
            AllocError::InvalidAlignment { requested, alignment } => {
                OjasError::Shape {
                    op: "alloc",
                    detail: format!("alignment violation: requested {requested} with alignment {alignment}"),
                }
            }
            AllocError::DeviceLost { backend, detail } => OjasError::DeviceLost { backend, detail },
        }
    }
}

/// Monotonically increasing execution completion fence.
pub trait Timeline: Send + Sync {
    /// Highest committed submission stamp.
    fn submitted(&self) -> Result<u64, FenceError> {
        Err(FenceError::unsupported("Timeline::submitted", "timeline"))
    }

    /// Highest completed GPU stamp.
    fn completed(&self) -> Result<u64, FenceError> {
        Err(FenceError::unsupported("Timeline::completed", "timeline"))
    }

    /// Current completed timeline value (convenience accessor).
    fn current_value(&self) -> u64 {
        self.completed().unwrap_or(0)
    }

    /// Block until `value` is reached or timeout expires.
    fn wait_for(&self, value: u64, timeout: Option<Duration>) -> Result<(), AllocError> {
        match self.wait(value) {
            Ok(()) => Ok(()),
            Err(FenceError::TimedOut { value, completed }) => Err(AllocError::DeviceLost {
                backend: BackendId::Metal,
                detail: format!("timeline wait timed out at {completed} waiting for {value}"),
            }),
            Err(FenceError::Poisoned) => Err(AllocError::DeviceLost {
                backend: BackendId::Metal,
                detail: "timeline poisoned".to_string(),
            }),
            Err(e) => Err(AllocError::DeviceLost {
                backend: BackendId::Metal,
                detail: e.to_string(),
            }),
        }
    }

    /// Block indefinitely until `value` is reached.
    fn wait(&self, _value: u64) -> Result<(), FenceError> {
        Err(FenceError::unsupported("Timeline::wait", "timeline"))
    }
}

/// RAII shared read-only lease over mapped host memory.
pub trait HostLease: Send + Sync {
    fn as_bytes(&self) -> &[u8];

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLease::as_f32",
            detail: "lease does not view as f32".to_string(),
        })
    }

    fn as_u32(&self) -> Result<&[u32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLease::as_u32",
            detail: "lease does not view as u32".to_string(),
        })
    }
}

/// RAII exclusive mutable lease over mapped host memory.
pub trait HostLeaseMut: HostLease {
    fn as_bytes_mut(&mut self) -> &mut [u8];

    fn as_f32_mut(&mut self) -> Result<&mut [f32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLeaseMut::as_f32_mut",
            detail: "lease does not view as f32".to_string(),
        })
    }

    fn as_u32_mut(&mut self) -> Result<&mut [u32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLeaseMut::as_u32_mut",
            detail: "lease does not view as u32".to_string(),
        })
    }
}

/// Unified buffer accessible to both CPU and GPU without memcpy.
pub trait MappedBuffer: DeviceBuffer {
    /// Host virtual address of the shared mapping.
    fn host_ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    /// GPU virtual address of the buffer.
    fn gpu_address(&self) -> u64 {
        0
    }

    /// Associated timeline synchronization fence.
    fn timeline(&self) -> &dyn Timeline {
        struct NoTimeline;
        impl Timeline for NoTimeline {}
        static NO_TIMELINE: NoTimeline = NoTimeline;
        &NO_TIMELINE
    }

    /// Acquire a shared read lease, waiting for GPU writers to complete.
    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        Err(FenceError::unsupported("MappedBuffer::host_read", "mapped buffer"))
    }

    /// Acquire an exclusive write lease, waiting for all GPU users to complete.
    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        Err(FenceError::unsupported("MappedBuffer::host_write", "mapped buffer"))
    }
}
```

---

### 4.2 Feature 9: Zero-Copy Memory Representation on `Tensor`

#### File: `ojas-core/src/tensor.rs`

1. **Payload Variant Addition**:
   ```rust
   #[derive(Debug)]
   enum Payload {
       Host(HostData),
       Device(Arc<dyn DeviceBuffer>),
       Mapped(Arc<dyn MappedBuffer>),
   }
   ```

2. **Storage Length & Window Checking**:
   ```rust
   impl Storage {
       fn len(&self) -> usize {
           match &self.payload {
               Payload::Host(data) => data.byte_len(),
               Payload::Device(buf) => buf.byte_len(),
               Payload::Mapped(buf) => buf.byte_len(),
           }
       }
   }
   ```

3. **Explicit Placement Enforcement in Accessors**:
   ```rust
   fn host_window(&self, op: &'static str) -> Result<Window, OjasError> {
       match &self.storage.payload {
           Payload::Device(buf) => Err(OjasError::Placement {
               op,
               expected: None,
               found: Some(buf.backend()),
           }),
           Payload::Mapped(buf) => Err(OjasError::Placement {
               op,
               expected: None,
               found: Some(buf.backend()),
           }),
           Payload::Host(_) => {
               // Contiguity and window checks...
           }
       }
   }
   ```

4. **Constructors for Mapped Tensors**:
   ```rust
   impl Tensor {
       /// Wrap a unified memory buffer with reservation charged against budget.
       pub fn from_mapped(
           buffer: Arc<dyn MappedBuffer>,
           shape: &[usize],
           dtype: DType,
           budget: &Budget,
       ) -> Result<Self, OjasError> {
           let nbytes = u64_len(buffer.byte_len(), "Tensor::from_mapped")?;
           let reservation = budget.try_reserve(nbytes)?;
           Self::from_mapped_reserved(buffer, shape, dtype, reservation)
       }

       /// Wrap a unified memory buffer with pre-reserved budget.
       pub fn from_mapped_reserved(
           buffer: Arc<dyn MappedBuffer>,
           shape: &[usize],
           dtype: DType,
           reservation: Reservation,
       ) -> Result<Self, OjasError> {
           let nbytes = u64_len(buffer.byte_len(), "Tensor::from_mapped_reserved")?;
           if reservation.bytes() != nbytes {
               return Err(OjasError::OutOfRange {
                   op: "Tensor::from_mapped_reserved",
                   detail: format!("reservation {} != buffer length {}", reservation.bytes(), nbytes),
               });
           }
           let strides = contiguous_strides(shape)?;
           let tensor = Self {
               storage: Arc::new(Storage::new(Payload::Mapped(buffer), reservation)),
               shape: shape.to_vec().into_boxed_slice(),
               strides: strides.into_boxed_slice(),
               dtype,
               byte_offset: 0,
           };
           tensor.check_window()?;
           Ok(tensor)
       }

       /// Access mapped buffer if payload is mapped.
       pub fn mapped_buffer(&self) -> Option<&Arc<dyn MappedBuffer>> {
           match &self.storage.payload {
               Payload::Mapped(buf) => Some(buf),
               _ => None,
           }
       }
   }
   ```

5. **Timeline-Fenced Zero-Copy `to_host` and `upload`**:
   - In `Tensor::to_host`:
     ```rust
     pub fn to_host(&self, budget: &Budget) -> Result<Self, OjasError> {
         match &self.storage.payload {
             Payload::Host(_) => Ok(self.clone()),
             Payload::Mapped(mbuf) => {
                 // Zero-Copy UMA Fence: Wait for pending GPU writes on timeline.
                 let timeline = mbuf.timeline();
                 timeline.wait_for(timeline.current_value(), None)
                     .map_err(OjasError::from)?;
                 // Return shared clone without memcpy or secondary allocation.
                 Ok(self.clone())
             }
             Payload::Device(buf) => {
                 // Traditional chunked readback for discrete devices...
             }
         }
     }
     ```
   - In `Tensor::host_view`:
     ```rust
     pub fn host_view(&self) -> Result<Box<dyn HostLease>, OjasError> {
         match &self.storage.payload {
             Payload::Mapped(mbuf) => mbuf.host_read().map_err(|e| e.into_ojas(mbuf.backend())),
             Payload::Host(_) => {
                 // Return safe host slice wrapper lease
                 Ok(Box::new(HostSliceLease::new(self)?))
             }
             Payload::Device(buf) => Err(OjasError::Placement {
                 op: "Tensor::host_view",
                 expected: None,
                 found: Some(buf.backend()),
             }),
         }
     }
     ```
   - In `MetalBackend::upload` (`ojas-metal/src/backend.rs`):
     ```rust
     fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
         if let Some(mbuf) = tensor.mapped_buffer() {
             if mbuf.backend() == BackendId::Metal {
                 // Verify timeline synchronization: no pending host writes
                 let timeline = mbuf.timeline();
                 timeline.wait_for(timeline.current_value(), None)
                     .map_err(OjasError::from)?;
                 return Ok(tensor.clone());
             }
         }
         // Fallback to copy path for pure host tensors...
     }
     ```

---

### 4.3 Feature 9: Implementation of `MappedBuffer` by `tessl::SharedBuf`

#### In `tessl/src/tensor.rs` (or `tessl/src/shared_buf.rs`):

```rust
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use ojas_core::{AllocError, BackendId, DeviceBuffer, FenceError, HostLease, HostLeaseMut, MappedBuffer, OjasError, Timeline};
use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

pub struct SharedBuf {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) host_ptr: *mut u8,
    pub(crate) gpu_addr: u64,
    pub(crate) byte_len: usize,
    pub(crate) timeline: Arc<dyn Timeline>,
    pub(crate) last_write: AtomicU64,
    pub(crate) last_use: AtomicU64,
    pub(crate) readers: AtomicUsize,
    pub(crate) writer: AtomicBool,
}

// SAFETY: StorageModeShared guarantees stable virtual address mapping for buffer lifetime.
// Read/write concurrency is strictly guarded by readers/writer atomics and timeline fences.
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

impl DeviceBuffer for SharedBuf {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        let end = offset.checked_add(len).ok_or_else(|| OjasError::OutOfRange {
            op: "SharedBuf::read_bytes",
            detail: "overflow".to_string(),
        })?;
        if end > self.byte_len {
            return Err(OjasError::OutOfRange {
                op: "SharedBuf::read_bytes",
                detail: format!("end {end} > len {}", self.byte_len),
            });
        }
        self.timeline.wait(self.last_write.load(Ordering::Acquire))
            .map_err(|e| e.into_ojas(BackendId::Metal))?;
        let mut out = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(self.host_ptr.add(offset), out.as_mut_ptr(), len);
        }
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MappedBuffer for SharedBuf {
    fn host_ptr(&self) -> *mut u8 {
        self.host_ptr
    }

    fn gpu_address(&self) -> u64 {
        self.gpu_addr
    }

    fn timeline(&self) -> &dyn Timeline {
        &*self.timeline
    }

    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        let last_write = self.last_write.load(Ordering::Acquire);
        self.timeline.wait(last_write)?;
        if self.writer.load(Ordering::Acquire) {
            return Err(FenceError::Busy {
                detail: "buffer is locked for writing".to_string(),
            });
        }
        self.readers.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(SharedBufLease {
            ptr: self.host_ptr,
            len: self.byte_len,
            readers: &self.readers,
        }))
    }

    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        let last_use = self.last_use.load(Ordering::Acquire);
        self.timeline.wait(last_use)?;
        if self.readers.load(Ordering::Acquire) > 0 || self.writer.swap(true, Ordering::SeqCst) {
            return Err(FenceError::Busy {
                detail: "buffer is leased for reading or writing".to_string(),
            });
        }
        Ok(Box::new(SharedBufLeaseMut {
            ptr: self.host_ptr,
            len: self.byte_len,
            writer: &self.writer,
        }))
    }
}

struct SharedBufLease {
    ptr: *mut u8,
    len: usize,
    readers: *const AtomicUsize,
}
unsafe impl Send for SharedBufLease {}
unsafe impl Sync for SharedBufLease {}

impl HostLease for SharedBufLease {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl Drop for SharedBufLease {
    fn drop(&mut self) {
        unsafe { (*self.readers).fetch_sub(1, Ordering::SeqCst); }
    }
}

struct SharedBufLeaseMut {
    ptr: *mut u8,
    len: usize,
    writer: *const AtomicBool,
}
unsafe impl Send for SharedBufLeaseMut {}
unsafe impl Sync for SharedBufLeaseMut {}

impl HostLease for SharedBufLeaseMut {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl HostLeaseMut for SharedBufLeaseMut {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for SharedBufLeaseMut {
    fn drop(&mut self) {
        unsafe { (*self.writer).store(false, Ordering::SeqCst); }
    }
}
```

---

## 5. Step-by-Step Implementation Plan for Worker

### Phase 1: `ojas-core` HAL Trait Extensions
1. Edit `/Users/bharath/Code/research/ojas/ojas-core/src/hal.rs`:
   - Extend `AllocError` with `InvalidAlignment { requested: usize, alignment: usize }`.
   - Update `AllocError::Exhausted { requested: u64, available: u64 }` and its mapping in `From<AllocError> for OjasError`.
   - Add `current_value(&self) -> u64` and `wait_for(&self, value: u64, timeout: Option<Duration>) -> Result<(), AllocError>` to `Timeline`.
   - Add default methods `host_ptr(&self) -> *mut u8`, `gpu_address(&self) -> u64`, and `timeline(&self) -> &dyn Timeline` to `MappedBuffer`.
   - Update `default_calls()` in `hal.rs::tests` to include the new methods so `the_hal_list_names_every_trait_method` passes.
2. Run `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core --lib hal::tests`.

### Phase 2: `ojas-core` Zero-Copy Tensor Representation
1. Edit `/Users/bharath/Code/research/ojas/ojas-core/src/tensor.rs`:
   - Add `Payload::Mapped(Arc<dyn MappedBuffer>)`.
   - Update `len(&self)` on `Storage` to handle `Payload::Mapped`.
   - Implement `Tensor::from_mapped` and `Tensor::from_mapped_reserved`.
   - Add `Tensor::mapped_buffer(&self) -> Option<&Arc<dyn MappedBuffer>>`.
   - Update `Tensor::to_host(&self, budget: &Budget)`: on `Payload::Mapped`, wait on timeline fence and return zero-copy clone.
   - Implement `Tensor::host_view(&self)` and `Tensor::host_view_mut(&mut self)`.
   - Update `device()` to return `Some(mbuf.backend())` for mapped tensors.
   - Enforce placement refusals on `f32_slice()` / `host_window()` for `Payload::Mapped`.
2. Run `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core`.

### Phase 3: `tessl` SharedBuf & Budget Root Ledger Integration
1. Implement `SharedBuf` in `tessl/src/tensor.rs` implementing `DeviceBuffer` and `MappedBuffer`.
2. Connect `SharedBuf.timeline` to `tessl::MetalTimeline` wrapping `lifetime_event`.
3. In `tessl/src/runtime.rs`, update `BufferPool::alloc` to accept `Budget`, charge bucket requests against `budget.child()`, and trigger the freelist trim ladder on pressure.

### Phase 4: `ojas-metal` Zero-Copy Dispatch Wiring
1. In `ojas-metal/src/backend.rs`, wrap kernel outputs using `Tensor::from_mapped_reserved` over `SharedBuf`.
2. In `MetalBackend::upload`, short-circuit when `tensor.mapped_buffer()` is already on Metal, synchronizing the timeline and wrapping without memcpy.
3. Verify with end-to-end readback and upload tests: assert `Budget::device_readbacks()` remains 0 for mapped tensors.

---

## 6. Verification Method

1. **Pure Trait Invariant Verification**:
   ```sh
   cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core --lib hal::tests
   ```
   *Expected Result*: All 14 tests pass, including `the_hal_list_names_every_trait_method`.

2. **Core Tensor Test Suite**:
   ```sh
   cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-core
   ```
   *Expected Result*: All 177 unit tests in `ojas-core` pass without regressions.

3. **Zero-Copy Readback Parity Test**:
   - Construct a `Tensor` with `Payload::Mapped` over a mock `MappedBuffer`.
   - Call `tensor.to_host(&budget)`.
   - Verify `budget.device_readbacks()` equals `(0, 0)`.
   - Call `tensor.host_view()`.
   - Verify leased bytes match input bytes exactly.
   - Attempt concurrent `host_write()` while lease is held; verify it fails with `FenceError::Busy`.

4. **Invalidation Conditions**:
   - Any introduction of `unsafe` blocks into `ojas-core` violates `#![forbid(unsafe_code)]`.
   - Any fallback to host `memcpy` when `to_host()` is invoked on a mapped buffer violates the zero-copy UMA contract.
   - Any CPU slice accessor (`f32_slice`) succeeding directly on a device-mapped tensor violates the fail-closed placement contract.
