# Architecture & Memory Exploration: Ojas & Tessl Unified Architecture and Shared Memory Pools

**Author:** Explorer 1 (`teamwork_explorer`)  
**Target Codebases:** `/Users/bharath/Code/research/ojas` and `/Users/bharath/Code/research/tessl`  
**Working Directory:** `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_1/`  
**Date:** 2026-10-31  

---

## 1. Observation

### 1.1 Repository Layout and Dependencies
- **Ojas Workspace Layout** [verified: `ojas/Cargo.toml:1-24`]:
  Contains 19 workspace members: `ojas-core`, `ojas-cpu`, `ojas-autograd`, `ojas-oracle`, `ojas-data`, `ojas-io`, `ojas-infer`, `ojas-metal`, `ojas-device`, `ojas-kernels`, `ojas-wgpu`, `ojas-cuda`, `ojas-hip`, `ojas-capi`, `ojas-gusset-engine`, `ojas-simd`, `ojas-qwen35`, `ojas-model`, `ojas-train`.
  Workspace `rust-version = "1.97"` [verified: `ojas/Cargo.toml:32`].
- **Sibling Path Dependencies** [verified: `ojas/AGENTS.md:31-34`]:
  `ojas` has path dependencies to sibling checkouts: `../tessl` (`/Users/bharath/Code/research/tessl`) and `../../devtools/gusset`.
  In `ojas-metal/Cargo.toml:22`: `tessl = { path = "../../tessl", optional = true }`.
  In `ojas-qwen35/Cargo.toml:21`: `tessl = { path = "../../tessl" }`.
- **Tessl Package Layout** [verified: `tessl/Cargo.toml:1-17`]:
  `name = "tessl"`, `rust-version = "1.82"`. Tessl is a single root crate with an unpublished path dev-dependency `numcheck = { path = "numcheck" }`. It is not a Cargo workspace.
  `tessl/Cargo.toml:17` declares `links = "tessl"` and `tessl/build.rs:57` emits `cargo:kernels={kernels_dir}` to export `DEP_TESSL_KERNELS` to dependent build scripts.
  `tessl` is strictly macOS / Apple Silicon only: unconditionally depends on `objc2-metal = "0.3"` [verified: `tessl/Cargo.toml:66-105`] and compiles Metal 4 shaders during `build.rs`.

### 1.2 DevMap and GitPulse Code Intelligence
- **Ojas DevMap Status** [verified: `devmap_status`]:
  Database at `/Users/bharath/Code/research/ojas/.devcouncil/codeintel/devmap.sqlite`. Generation 7156, total files 1,735, total symbols 18,807, total edges 95,402. Status: `available: true`, `is_fresh: false` (source edits present across active worktrees).
- **Tessl DevMap Status** [verified: `devmap_status`]:
  Database at `/Users/bharath/Code/research/tessl/.devmap/codeintel/devmap.sqlite`. Generation 1471, total files 820, total symbols 11,874, total edges 56,571. Status: `available: true`, `is_fresh: true`.
- **GitPulse Worktrees & Tasks** [verified: `gitpulse_insights`, `gitpulse_list_tasks`]:
  Ojas has 28 worktrees, including 15 active `gp/ug-*` unification lanes (`gp/ug-core`, `gp/ug-sharedbuf`, `gp/ug-tessl-shared`, `gp/ug-pool-ledger`, `gp/ug-one-runtime`, `gp/ug-alloc-error`, `gp/ug-bucket-dedup`, `gp/ug-const-dedup`, `gp/ug-plan-doc`, `gp/ug-metal-attn-bwd`, `gp/ug-metal-encoder-bwd`, `gp/ug-metal-maxpool`, `gp/ug-metal-spatial-bwd`).
  Tessl has 10 worktrees, including 9 paired `ug/*` worktrees (`ug/sharedbuf`, `ug/tessl-shared`, `ug/pool-ledger`, `ug/retire-train-prep`, `ug/alloc-error`, `ug/bucket-dedup`, `ug/const-dedup`, `ug/adamw-msl`, `ug/docs`).
  The governing unification plan document was located at `/Users/bharath/Code/research/ojas-ug-plan-doc/docs/unification-plan.md` [verified].

### 1.3 Memory Representations in Ojas and Tessl
- **Ojas Core Memory Hierarchy** [verified: `ojas-core/src/tensor.rs:89-165`]:
  ```rust
  pub struct Tensor {
      storage: Arc<Storage>,
      shape: Box<[usize]>,
      strides: Box<[usize]>,
      dtype: DType,
      byte_offset: usize,
  }
  struct Storage {
      payload: Payload,
      finite: AtomicU8,
      compute: AtomicU8,
      _reservation: Reservation,
  }
  enum Payload {
      Host(HostData),
      Device(Arc<dyn DeviceBuffer>),
  }
  enum HostData {
      F32(Vec<f32>),
      U32(Vec<u32>),
      Half(Vec<u16>),
  }
  ```
  `DeviceBuffer` trait [verified: `ojas-core/src/tensor.rs:19-33`]:
  ```rust
  pub trait DeviceBuffer: Any + Send + Sync + fmt::Debug {
      fn backend(&self) -> BackendId;
      fn byte_len(&self) -> usize;
      fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError>;
      fn as_any(&self) -> &dyn Any;
  }
  ```
- **Ojas Metal Backend Memory Indirection** [verified: `ojas-metal/src/backend.rs:124-182`]:
  ```rust
  pub struct MetalBuffer {
      id: u64,
      len: usize,
      link: Arc<Link>,
      ids: Option<Ids>,
  }
  impl DeviceBuffer for MetalBuffer { ... }
  ```
  `MetalBuffer` does not hold an `MTLBuffer` or memory pointer; it holds a 64-bit integer ID communicating with an actor thread:
  In `ojas-metal/src/device.rs:553-556`:
  ```rust
  struct Worker {
      rt: Arc<GpuRuntime>,
      bufs: HashMap<u64, GpuBuffer>,
      next_id: u64, ...
  }
  ```
- **Tessl Buffer and Memory Pool Hierarchy** [verified: `tessl/src/tensor.rs:67-100`, `tessl/src/runtime.rs:245-305`]:
  `tessl` allocates all buffers as Metal 4 shared buffers:
  ```rust
  device.newBufferWithLength_options(key, MTLResourceOptions::StorageModeShared)
  ```
  `StorageModeShared` on Apple Silicon is unified memory (coherent between CPU and GPU).
  `GpuBuffer` owns `Arc<PooledBuffer>` which holds `buffer: Retained<ProtocolObject<dyn MTLBuffer>>` and `gpu_addr: u64`.
  Host mapping [verified: `tessl/src/tensor.rs:113-224`]:
  `tessl::GpuBuffer::map_host<T>(&self) -> Result<HostMapping<'_, T>, String>` grants direct host access via `self.metal().contents().as_ptr()` guarded by an exclusive `RuntimeAccess` lease.
- **Tessl BufferPool** [verified: `tessl/src/runtime.rs:245-338`]:
  Maintains a freelist `HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>` with power-of-two buckets up to 1 MiB (`POW2_BUCKET_MAX_BYTES`) and 16 KiB multiples (`ALLOC_PAGE_BYTES`) above it. Default cache cap is 2 GiB (`DEFAULT_POOL_CACHE_BYTES = 2 * 1024 * 1024 * 1024`).

### 1.4 Code Duplication and Seams Between Ojas and Tessl
- **SafeTensors, JSON, and NPY Parsers** [verified: `tessl/src/safetensors.rs:1-951`, `ojas/ojas-io/src/safetensors.rs:1-1768`, `tessl/src/json.rs:1-517`, `ojas/ojas-io/src/json.rs:1-1236`]:
  Both repositories implement custom, zero-dependency RFC 8259 JSON parsers and strict SafeTensors file readers independently (~2,028 lines in Tessl duplicating `ojas-io`).
- **Qwen3.5 Whole-Step Training Orchestration** [verified: `tessl/src/qwen35_train.rs:1-4725`, `tessl/src/qwen35_adamw.rs:1-1518`, `tessl/src/qwen35_bwd.rs:1-988`, `ojas-qwen35/src/step.rs:1-3062`, `ojas-metal/src/gpu.rs:1-3735`]:
  Tessl has 7,231 lines of training step, AdamW scalar math, and gradient bank orchestration.
  `ojas-qwen35` wraps this Tessl step (3,062 lines).
  `ojas-metal/src/gpu.rs` contains a legacy, unwired 3,735-line training step (`Session`, `TinyState`).
  Meanwhile, `ojas-model/src/qwen35/step_provider.rs` already runs Qwen3.5 training via `ojas-autograd::Tape` and `ojas-train` (`TapeStepProvider`).
- **`StepProvider` Trait Duplication** [verified: `ojas-train/src/lappi/step.rs:267`, `ojas-cuda/src/step.rs:285`]:
  Defined twice independently within the `ojas` workspace.
- **AdamW Scalar Computation** [verified: `tessl/src/qwen35_adamw.rs:507`, `ojas-metal/src/backend.rs:1403`, `ojas-train/src/lappi/adamw.rs:166`]:
  Duplicated across three locations; MSL math in `tessl/kernels/qwen35_adamw_math.h` duplicated in `ojas-metal/kernels/ojas_backend.metal:1528-1541`.
- **Memory Accounting & String Scraping** [verified: `ojas-metal/src/device.rs:515-551, 5062`]:
  `ojas-metal` contains an `include_str!` pin of `tessl/src/runtime.rs` to match exact string formats of `newBuffer(...) failed` to classify OOMs.

---

## 2. Logic Chain

### 2.1 The Inefficiencies of Current Memory Management
1. **Redundant Copies on Unified Memory Architecture (UMA)**:
   - On Apple Silicon, DRAM is physically shared between the CPU and GPU.
   - Tessl's allocations in `BufferPool` use `MTLResourceOptions::StorageModeShared`, which maps memory into both CPU address space (`MTLBuffer.contents()`) and GPU virtual address space (`MTLBuffer.gpuAddress()`).
   - However, Ojas represents CPU memory as `HostData::F32(Vec<f32>)` (allocated via the system allocator) and Metal memory as a remote `id: u64` in `Worker.bufs`.
   - Transferring data via `upload()` or `to_host()` causes an explicit buffer allocation and a full memcpy across the boundary (`ojas-metal/src/device.rs:1823-1825` and `:1848-1854`).
   - For a large tensor (e.g. SAM 3.1 trunk weights ~1.5 GB or Qwen 2B embedding table ~2 GB), this wastes GBs of physical RAM and burns memory bandwidth.

2. **Divergent Accounting Between Budget and BufferPool**:
   - `ojas_core::Budget` enforces strict reservation accounting (`Budget::try_reserve`).
   - Tessl's `BufferPool` holds up to 2 GiB of freed Metal buffers in its freelist cache (`DEFAULT_POOL_CACHE_BYTES`).
   - Because `BufferPool`'s cache is uncharged against Ojas's `Budget`, the process can trigger macOS jetsam memory kills under memory pressure while Ojas reports its budget is not exceeded.
   - To fix this, `BufferPool` must charge every allocated bucket to an `ojas_core::Budget` child ledger, and must expose a trim-on-pressure ladder.

3. **Guaranteed "No Silent Fallback" Under Unified Memory**:
   - Core Ojas Invariant: *"No silent fallback. A GPU backend that cannot run an op returns OjasError::Unsupported. It never computes on the host behind the caller's back."* [verified: `ojas/AGENTS.md:37-39`].
   - If unified memory buffers were accessible transparently without checks, CPU ops might execute on GPU-resident tensors without caller awareness.
   - Therefore, zero-copy sharing must NOT mean implicit execution. Tensor placement (`BackendId::Cpu` vs `BackendId::Metal`) must remain strictly explicit.
   - A tensor on `BackendId::Metal` MUST continue to refuse CPU access (`tensor.f32_slice()` -> `OjasError::Placement`), and CPU backend ops MUST continue to refuse device tensors.
   - Zero-copy is achieved when the caller *explicitly* requests transfer (`tensor.to_host()` or `backend.upload()`): instead of allocating and copying bytes, the runtime synchronizes pending GPU commands and wraps the exact same underlying unified memory buffer in a host lease or device buffer.

### 2.2 Architectural Seam & Dependency Arrow
1. **The Cargo Dependency Direction**:
   - The owner's binding decision: *"The Cargo arrow points from the extension to ojas's core, and the Metal backend is tessl's. ojas is the universal framework."*
   - Dependency order: `ojas-metal -> tessl -> ojas-core`.
   - `ojas-core` has `#![forbid(unsafe_code)]` and zero external dependencies [verified: `ojas-core/Cargo.toml:10`].
   - By introducing abstract Hardware Abstraction Layer (HAL) traits in `ojas-core` (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`), `ojas-core` remains pure, deterministic, and universal (compiling on Linux, macOS, Windows).
   - `tessl` depends on `ojas-core` and implements these HAL traits using Metal 4 primitives (`MTLBuffer`, `MTLSharedEvent`).
   - `ojas-metal` depends on `tessl` to dispatch Metal compute kernels and manage the device link.

2. **Division of Responsibilities**:
   - **Tessl**:
     - Metal 4 GPU runtime, residency sets, command encoding, argument tables.
     - BufferPool with `StorageModeShared` unified memory allocations.
     - Heavy Mac inference optimizations: ICB (Indirect Command Buffers), quantized GEMM/GEMV (Q4 MLX, Q8, Int8), KV caches, decode graphs.
     - Multimodal inference engines: EmbeddingGemma 2 (text encoder, vision tower, audio, multimodal projection), BERT, and native media decoders (`media/`: ImageIO, AVFoundation, AudioToolbox).
     - Canonical Metal kernel sources and headers (exported via `DEP_TESSL_KERNELS`).
     - Metal-native training kernel entry points (e.g. `gdn_train.metal`, `qwen35_bwd.metal`, `qwen35_adamw.metal`).
   - **Ojas**:
     - Universal training engine and framework: `ojas-core` (tensors, shapes, limits, dtype, budget, backend trait).
     - Bit-identical CPU reference backend (`ojas-cpu` with `Numerics::Exact`).
     - SIMD micro-kernels (`ojas-simd`).
     - Reverse-mode automatic differentiation (`ojas-autograd::Tape`).
     - High-level model towers, optimizers (AdamW, Muon NS5), trainer loop, and `TapeStepProvider` (`ojas-model`, `ojas-train`).
     - IO and checkpointing (`ojas-io`, `ojas-data`).
     - Backend implementations: `ojas-cpu`, `ojas-metal` (via Tessl), `ojas-wgpu` (cross-platform WGSL), `ojas-cuda` (NVIDIA), `ojas-hip` (AMD).

---

## 3. Caveats
- **MSRV Probe Gate (Decision D2)**:
  `ojas-core` inherits workspace `rust-version = "1.97"`, whereas `tessl` currently declares `rust-version = "1.82"`. Before `tessl` can directly depend on `ojas-core`, either `cargo +1.82 check -p ojas-core` must succeed (proving `ojas-core` syntax and dependencies build at 1.82), or `tessl`'s MSRV must be raised to 1.97.
- **Unification Plan Open Decisions**:
  Eleven owner decisions (D1 through D11) in `unification-plan.md` remain formally marked as OPEN. The recommendations synthesized here reflect the consensus verdict (Proposal B + HAL traits in `ojas-core`), but require owner sign-off on D5 (retiring the Tessl C-ABI training symbols vs keeping them for `tessl_torch`).
- **ICB Replay Re-stamping**:
  When indirect command buffers (ICBs) replay commands, timestamp fences on replayed buffers must be updated so CPU fences don't read stale completion values.
- **`Send + Sync` on Metal Buffers (Decision D6)**:
  `objc2-metal` marks `MTLDevice` and `MTLCommandQueue` as thread-safe, but `MTLBuffer` is not marked `Send` or `Sync` by default. Tessl's `SharedBuf` wrapper requires an explicit, audited `unsafe impl Send for SharedBuf` and `unsafe impl Sync for SharedBuf` (as prototyped in `ojas-ug-spike-uma/src/main.rs:64-65`).

---

## 4. Conclusion

1. **Target Crate Architecture**:
   ```
                            ojas workspace (universal)
    ojas-core:      Tensor, DType, OjasError, Budget (sole ledger), Backend trait,
                    HAL traits: AllocError, MappedBuffer, HostLease, Timeline, Provenance
    ojas-autograd:  Tape, autograd graph, gradcheck
    ojas-cpu/simd:  CPU reference backend (Numerics::Exact bit-identical contract)
    ojas-model/train: Towers, TapeStepProvider, AdamW, Muon, trainer loops
    ojas-io/data:   SafeTensors, JSON, NPY, BPE tokenizer (sole canonical IO)
    ojas-metal:     Metal op router (impl Backend for MetalBackend), Link/Cmd device thread
        │
        ▼ (path dependency)
    tessl (sibling repository):
        Depends on: ojas-core (after MSRV probe)
        Owns: Metal 4 runtime, BufferPool (StorageModeShared), SharedBuf + Timeline (impl HAL),
              canonical kernel sources (DEP_TESSL_KERNELS), inference engines (Qwen3.5, Gemma, BERT),
              native media decode (ImageIO, AVFoundation).
   ```

2. **Shared Memory Pools & Zero-Copy Protocol (No Silent Fallback)**:
   - **One Ledger**: `ojas_core::Budget` is the single root ledger. Tessl's `BufferPool` reserves against an `ojas` budget child before calling `newBufferWithLength_options(key, StorageModeShared)`. The 2 GiB freelist cache is tracked under a `pool.cache` child with an automatic trim ladder on memory pressure.
   - **Zero-Copy Host Access**: `ojas_core::Tensor` gains `Payload::Mapped(Arc<dyn MappedBuffer>)`. On Apple Silicon, `to_host()` waits for `Timeline::completed >= last_write` and returns an exclusive `HostLease` over the exact same unified memory without memcpy.
   - **Zero-Copy Device Transfer**: `upload()` verifies the buffer is already in unified memory, waits on the timeline, registers GPU residency, and returns a `Tensor` placed on `BackendId::Metal` without memcpy.
   - **Fail-Closed Placement Enforcement**: Calling `tensor.f32_slice()` on a `BackendId::Metal` tensor continues to return `OjasError::Placement`. CPU ops refuse device tensors. Placement transitions must be explicit. Non-unified backends (wgpu, CUDA, HIP) return `OjasError::Unsupported` for `MappedBuffer` and retain explicit copies.

3. **Code Duplication Elimination Plan**:
   - **SafeTensors / JSON / NPY**: Delete `tessl/src/safetensors.rs` (950 lines), `tessl/src/json.rs` (516 lines), `tessl/src/npy.rs` (417 lines), and `tessl/src/plain.rs` (145 lines). Tessl uses `ojas-io` with a slab sink (`pread` directly into pooled unified memory).
   - **Training Orchestration**: Retire Tessl's whole-step training layer (`qwen35_train.rs`, `qwen35_adamw.rs`, `qwen35_bwd.rs`), delete legacy `ojas-metal/src/gpu.rs` (3,735 lines), and retire `ojas-qwen35` step shell. Qwen3.5 training is unified under `ojas-model::TapeStepProvider`.
   - **Optimizers**: Move AdamW scalar generation into `ojas-core` (`adamw_scalars_for_step -> [f32; 8]`); `ojas-metal` entry points `#include` Tessl's `qwen35_adamw_math.h` via `DEP_TESSL_KERNELS`.
   - **Memory Accounting**: Delete string-scraping `classify_alloc` and the `include_str!` pin in `ojas-metal/src/device.rs:5062`, replaced by typed `AllocError -> OjasError::CapacityExceeded`.

---

## 5. Verification Method

### 5.1 Code and Contract Inspections
1. **Verify Crate Boundaries & Forbidden Code**:
   ```sh
   # Verify ojas-core has empty dependencies and forbids unsafe code
   rg -n 'forbid\(unsafe_code\)' /Users/bharath/Code/research/ojas/ojas-core/src/lib.rs
   # Verify tessl links declaration and kernel export
   rg -n 'links = "tessl"' /Users/bharath/Code/research/tessl/Cargo.toml
   rg -n 'cargo:kernels=' /Users/bharath/Code/research/tessl/build.rs
   ```
2. **Verify Memory Allocation Modes**:
   ```sh
   # Verify StorageModeShared in tessl
   rg -n 'StorageModeShared' /Users/bharath/Code/research/tessl/src/runtime.rs
   # Verify Ojas device readback copy path
   rg -n 'read_bytes' /Users/bharath/Code/research/ojas/ojas-core/src/tensor.rs
   ```
3. **Verify DevMap & GitPulse Indices**:
   ```sh
   # In ojas:
   devmap status
   gitpulse insights
   # In tessl:
   devmap status
   gitpulse insights
   ```

### 5.2 Test Execution (Read-Only Verification Baseline)
- Check tracked targets in Ojas:
  ```sh
  cd /Users/bharath/Code/research/ojas && ./scripts/check_tracked_targets.sh
  ```
- Run formatting check in Ojas:
  ```sh
  cd /Users/bharath/Code/research/ojas && ./scripts/fmt_check.sh
  ```
- Run unit and integration tests across Ojas workspace:
  ```sh
  cd /Users/bharath/Code/research/ojas && cargo test --workspace --release -- --test-threads=1
  ```
- Run unit and integration tests in Tessl:
  ```sh
  cd /Users/bharath/Code/research/tessl && cargo test --release -- --test-threads=1
  ```

### 5.3 Invalidation Conditions
- If Apple Silicon Metal driver requires GPU-private storage (`MTLStorageMode::Private`) for optimal MPP TensorOps matrix multiplication, zero-copy sharing for weight matrices would require a blit copy, invalidating zero-copy for weight storage (though activations remain shared).
- If `cargo +1.82 check -p ojas-core` fails due to 2024-edition or 1.97-specific syntax in `ojas-core`, `tessl` cannot directly depend on `ojas-core` without raising `tessl`'s MSRV or introducing an intermediate C-ABI/shim crate.
