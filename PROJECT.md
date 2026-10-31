# Project: Ojas & Tessl Unified Architecture, Shared Memory Pools, and SAM/Gemma Kernel Parity

## Architecture
- **Dependency Flow**: `ojas-metal -> tessl -> ojas-core`.
  - `ojas-core`: Pure Rust, `#![forbid(unsafe_code)]`, zero-dependency. Houses `Tensor`, `DType`, `OjasError`, `Budget` (root ledger), `Backend` trait, and abstract HAL traits (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`).
  - `tessl`: Metal 4 GPU runtime, `BufferPool` with `StorageModeShared` unified memory, Apple Silicon Mac inference optimizations (ICB, quantization, KV caches, EmbeddingGemma 2 multimodal towers), and canonical Metal kernel sources (`DEP_TESSL_KERNELS`).
  - `ojas-cpu`: CPU reference backend with bit-identical `Numerics::Exact` contract.
  - `ojas-autograd`: Reverse-mode `Tape`, gradcheck helpers, and autograd records.
  - `ojas-model` / `ojas-train`: High-level model towers, optimizers (AdamW, Muon), and `TapeStepProvider`.
  - `ojas-metal`: Implements `Backend` for `MetalBackend`, routing ops to Tessl runtime and Metal compute kernels.
- **Shared Memory Pools (Zero-Copy UMA)**:
  - `ojas_core::Budget` serves as the sole authoritative ledger for host allocations and Tessl buffer pool cache (`pool.cache` child with automatic pressure trim).
  - `ojas_core::Tensor` gains `Payload::Mapped(Arc<dyn MappedBuffer>)`.
  - On Apple Silicon, `to_host()` returns timeline-fenced `HostLease` over the same `StorageModeShared` buffer without memcpy.
  - `upload()` verifies the buffer is in unified memory, waits on timeline, and wraps into device tensor without memcpy.
  - **No Silent Fallback**: Placement remains strictly explicit (`BackendId::Cpu` vs `BackendId::Metal`). CPU methods refuse device tensors with `OjasError::Placement`.
- **Code Duplication Elimination**:
  - SafeTensors / JSON / NPY readers in `tessl` deleted in favor of `ojas-io`.
  - Tessl's whole-step training layers (`qwen35_train.rs`, `qwen35_adamw.rs`, `qwen35_bwd.rs`) and legacy `ojas-metal/src/gpu.rs` retired in favor of `ojas-model::TapeStepProvider`.
  - AdamW scalar generation consolidated into `ojas-core`.
  - String-scraping memory allocation classifier replaced by typed `AllocError`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | GitPulse Task Categorization & Priority Audit | Audit, categorize (P0/P1/P2), and status-update 243 tasks across ojas, tessl, and SpikeTrack boards | M1 | Survey (Explorer 3) |
| 2 | Fast Inner Dev Loop Workflow | Sub-15s compile, clippy, and unit test script (`dev_fast_check.sh`) | M1 | Survey (Explorer 3) |
| 3 | Tiered Testing Pipeline | Structured 5-tier testing pipeline from pure validators to full workspace | M1 | Survey (Explorer 3) |
| 4 | Isolated PyTorch Parity Verification Workflow | Test harness runner against `ojas-oracle` reference fixtures with exact tolerances | M1 | Survey (Explorer 3) |
| 5 | Metal Test Execution & GPU Locking Workflow | Exclusive `mac_heavy.sh` GPU lock, budget isolation, and thread serialization | M1 | Survey (Explorer 3) |
| 6 | Cross-Workspace Sync & Collision Safety | Sibling path validation and GitPulse worktree collision checks | M1 | Survey (Explorer 3) |
| 7 | HAL Traits in `ojas-core` | `MappedBuffer`, `HostLease`, `Timeline`, `AllocError` definitions in pure `ojas-core` | M2 | Survey (Explorer 1) |
| 8 | Shared Budget Ledger & Cache Trim Ladder | `ojas_core::Budget` as root ledger; Tessl `BufferPool` reserves against child budget | M2 | Survey (Explorer 1) |
| 9 | Zero-Copy UMA Buffer Sharing | `Payload::Mapped` on `Tensor`; timeline-fenced zero-copy `to_host` and `upload` | M2 | Survey (Explorer 1) |
| 10 | Fail-Closed Explicit Placement Enforcement | Refusal of CPU ops on Metal tensors without silent fallback; explicit placement checks | M2 | Survey (Explorer 1) |
| 11 | SafeTensors & JSON Reader Dedup | Retire `tessl/src/safetensors.rs`, `json.rs`, `npy.rs`; route through `ojas-io` | M2 | Survey (Explorer 1) |
| 12 | Training Orchestration & Optimizer Dedup | Retire `tessl/src/qwen35_train.rs` & `ojas-metal/src/gpu.rs`; unify AdamW scalars in `ojas-core` | M2 | Survey (Explorer 1) |
| 13 | Metal `max_pool2d_forward` Kernel | 2D max pooling forward on Metal GPU with exact tie-breaking | M3 | Survey (Explorer 2) |
| 14 | Generalized Metal `conv_transpose2d_forward` | Generalized transposed conv2d via col2im gathering on Metal | M3 | Survey (Explorer 2) |
| 15 | High-Dimensional Tiled Cross-Attention ($D=256$) | Decoupled $T_k \ge T_q$ tiled cross-attention on Metal, solving 74% tracker bottleneck | M3 | Survey (Explorer 2) |
| 16 | Biased Attention Forward (`boxRPB`) | Additive logit bias support in fast tiled attention for DETR decoder | M3 | Survey (Explorer 2) |
| 17 | Fused Vision QKV + 2D RoPE + Window Partition | Fused Metal kernel for SAM ViT trunk attention dispatch optimization | M3 | Survey (Explorer 2) |
| 18 | BF16 ViT Activation Tier & BF16 LayerNorm | End-to-end BF16 activation persistence and fused GEMM `GeluErf` epilogue | M3 | Survey (Explorer 2) |
| 19 | Autograd Tape Vision Ops Wiring | Tape records and backward dispatch for Relu, GroupNorm, Bilinear, RoPE, Concat, Slice | M3 | Survey (Explorer 2) |
| 20 | SAM 3.1 PyTorch Parity Verification | Parity verification against `ojas-oracle` fixtures under strict tolerances | M3 | Survey (Explorer 2) |
| 21 | `gelu_tanh` / GeGLU Forward & Backward | Exact PyTorch tanh approximation GELU on CPU and Metal for Gemma MLP | M4 | Survey (Explorer 2) |
| 22 | Symmetric Non-Causal Sliding Window Attention | Symmetric inclusive sliding window ($|i - j| \le W$) on CPU and Metal | M4 | Survey (Explorer 2) |
| 23 | Masked Sequence Mean Pooling (`segment_mean_rows`) | Token-masked sequence mean pooling on CPU and Metal for Gemma embedding | M4 | Survey (Explorer 2) |
| 24 | L2 Normalization with Matryoshka Slicing | `l2_normalize_rows` with prefix truncation on CPU and Metal | M4 | Survey (Explorer 2) |
| 25 | Per-Head V RMSNorm (`rms_qkv_norm_forward`) | Weightless RMSNorm on V per head in Gemma attention | M4 | Survey (Explorer 2) |
| 26 | Gemma Embedding PyTorch Parity Verification | Parity verification against HuggingFace Gemma embedding reference | M4 | Survey (Explorer 2) |
| 27 | Full Workspace E2E Regression Pass | Full `cargo test --workspace` across ojas and tessl passing without regressions | M5 | R1/R2/R3 Acceptance Criteria |
| 28 | Adversarial Coverage Hardening | Edge cases, OOM recovery, zero-length tensors, multi-threaded contention | M5 | Project Pattern Phase 2 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | GitPulse Tasks & Accelerated Workflows | Features 1, 2, 3, 4, 5, 6 | none | IN_PROGRESS |
| M2 | Unified Architecture & Shared Memory Pools | Features 7, 8, 9, 10, 11, 12 | M1 | PLANNED |
| M3 | SAM 3.1 & Vision Kernels with PyTorch Parity | Features 13, 14, 15, 16, 17, 18, 19, 20 | M2 | PLANNED |
| M4 | Gemma Embedding Kernels with PyTorch Parity | Features 21, 22, 23, 24, 25, 26 | M2 | PLANNED |
| M5 | E2E Parity Verification & Coverage Hardening | Features 27, 28 (Pass 100% tests & adversarial hardening) | M3, M4 | PLANNED |

## Interface Contracts

### `ojas-core` ↔ `tessl` (HAL Traits)
- `AllocError`: Typed enumeration (`Exhausted { requested, available }`, `DeviceLost`, `InvalidAlignment`) mapped to `OjasError::CapacityExceeded`.
- `MappedBuffer`: Trait implemented by Tessl's `SharedBuf`.
  ```rust
  pub trait MappedBuffer: Any + Send + Sync + fmt::Debug {
      fn byte_len(&self) -> usize;
      fn host_ptr(&self) -> *mut u8;
      fn gpu_address(&self) -> u64;
      fn timeline(&self) -> &dyn Timeline;
      fn as_any(&self) -> &dyn Any;
  }
  ```
- `Timeline`: Monotonically increasing GPU execution fence.
  ```rust
  pub trait Timeline: Send + Sync {
      fn current_value(&self) -> u64;
      fn wait_for(&self, value: u64, timeout: Option<Duration>) -> Result<(), AllocError>;
  }
  ```
- `Budget`: Sole root ledger. Tessl `BufferPool::new(device, budget: Budget)` reserves all pool memory from `budget`.

### `ojas-core` ↔ `ojas-metal` (Backend Trait Extensions)
- Added methods on `Backend`:
  - `max_pool2d_forward(&self, input: &Tensor, kernel: [usize; 2], stride: [usize; 2]) -> Result<Tensor, OjasError>`
  - `conv_transpose2d_forward(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, params: &ConvTranspose2dParams) -> Result<Tensor, OjasError>`
  - `gelu_tanh_forward(&self, input: &Tensor) -> Result<Tensor, OjasError>`
  - `gelu_tanh_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError>`
  - `segment_mean_rows(&self, input: &Tensor, segments: &Tensor) -> Result<Tensor, OjasError>`
  - `l2_normalize_rows(&self, input: &Tensor, truncate_dim: Option<usize>, eps: f32) -> Result<Tensor, OjasError>`
  - `rms_qkv_norm_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, q_w: Option<&Tensor>, k_w: Option<&Tensor>, eps: f32) -> Result<(Tensor, Tensor, Tensor), OjasError>`

## Code Layout
- `/Users/bharath/Code/research/ojas`: Universal training system, CPU reference, autograd, IO, models, and metal backend extension.
- `/Users/bharath/Code/research/tessl`: Metal 4 inference engine, shared memory pools, optimized matrix kernels, EmbeddingGemma 2.
- `/Users/bharath/Code/research/SpikeTrack`: Application consumer, SAM 3.1 video tracker multiplex, and fixtures.
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/`: Orchestration and agent metadata only.
