# Project: Ojas & Tessl Unified Architecture, Shared Memory Pools, and SAM/Gemma Kernel Parity

## Architecture

### 3-Tier Modular Compute Ecosystem
- **`ojas` (Central Universal Framework)**:
  - `ojas-core`: Pure Rust, `#![forbid(unsafe_code)]`, zero-dependency. Houses `Tensor`, `DType`, `OjasError`, `Budget` (root ledger), `Backend` trait, and abstract HAL traits (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`).
  - `ojas-cpu`: CPU reference backend with bit-identical `Numerics::Exact` contract.
  - `ojas-autograd`: Reverse-mode `Tape`, gradcheck helpers, and autograd records.
  - `ojas-model` / `ojas-train`: High-level model towers, optimizers (AdamW, Muon), and `TapeStepProvider`.
  - `ojas-metal`: Implements `Backend` for `MetalBackend`, routing ops to Tessl runtime and Metal compute kernels.
  - `ojas-wgpu` / `ojas-cuda` / `ojas-hip`: Portable cross-platform compute and accelerator probes.
- **`tessl` (Dedicated Apple Silicon / Metal 4 Extension & Mac Inference)**:
  - Metal 4 GPU runtime, `BufferPool` with `StorageModeShared` unified memory, Apple Silicon Mac inference optimizations (ICB, quantization, KV caches, EmbeddingGemma 2 multimodal towers), and canonical Metal kernel sources (`DEP_TESSL_KERNELS`).
- **`sparsl` (Dedicated Sparse Computing Extension)**:
  - Deterministic sparse matrix representations (CSR, CSC), sparse linear algebra (SpMV), chunked prefix scans over affine maps, and event-driven simulation.
  - No dependency edge to `ojas`, `tessl` or `SpikeTrack` (`sparsl/Cargo.toml` `[dependencies]` names none). `SpikeTrack`'s `st-gate` builds its own change maps and bitsets with no dependencies (`crates/st-gate/Cargo.toml:10-13`) and names `sparsl` as a later replacement for its bitset (`crates/st-gate/src/bits.rs:5`); it does not use it today.

### Dependency Edges (as built)
- `SpikeTrack → ojas`: `ojas-core`, `ojas-cpu`, `ojas-io`, `ojas-metal` by path (`Cargo.toml:15-18`; used by `st-model` and `spiketrack`).
- `ojas-metal → tessl`: optional, behind the `metal` feature (`ojas/ojas-metal/Cargo.toml:14,22`).
- `tessl → ojas-core`, `ojas-io`: path dependencies (`tessl/Cargo.toml:112-113` at `origin/main` `a00521f`). Between the two checkouts this is a cycle (`ojas-metal → tessl → ojas-core`); between crates it is not, because `ojas-core` and `ojas-io` depend on neither `tessl` nor `ojas-metal`. It is recorded as an architecture issue for the owner.
- `tessl` declares `ojas-io` but no `ojas_io` path appears anywhere in `tessl` (`rg -uu`); only `ojas_core` is used.

### Modularity Rationale Across Checkouts
1. **Decoupled Compile Cycles:** Xcode 26 Metal toolchain compilation (`xcrun metal`, `metallib`) and intensive sparse property testing (`proptest`, mutation suites) remain isolated from universal training CI and cross-platform builds.
2. **Isolated Hardware Requirements:** Apple Silicon M-series specifics live in `tessl`; generic CPU and optional Metal sparse kernels live in `sparsl`; portable multi-platform training lives in `ojas`.
3. **Independent Profiling & Micro-Benchmarking:** GEMM tile sweeps and register tuning (tessl) vs sparse crossover thresholds (sparsl) vs end-to-end training step throughput (ojas) iterate on distinct benchmark suites.
4. **Independent Release Cadence & Separate Checkouts:** Downstream consumers take only what they need without code or runtime bloat.

### Shared Memory Pools (Zero-Copy UMA)
- `ojas_core::Budget` serves as the sole authoritative ledger for host allocations and Tessl buffer pool cache (`pool.cache` child with automatic pressure trim).
- `ojas_core::Tensor` gains `Payload::Mapped(Arc<dyn MappedBuffer>)`.
- On Apple Silicon, `to_host()` returns timeline-fenced `HostLease` over the same `StorageModeShared` buffer without memcpy.
- `upload()` verifies the buffer is in unified memory, waits on timeline, and wraps into device tensor without memcpy.
- **No Silent Fallback**: Placement remains strictly explicit (`BackendId::Cpu` vs `BackendId::Metal`). CPU methods refuse device tensors with `OjasError::Placement`.

### Code Duplication Elimination
- SafeTensors / JSON / NPY readers: **not deduplicated.** `tessl/src/safetensors.rs`, `json.rs`, `npy.rs` and `plain.rs` all still exist (950 + 516 + 417 + 145 = 2,028 lines at `origin/main`). `ojas-qwen35` still imports `tessl::safetensors` (`ojas-qwen35/src/names.rs:13`) and `tessl::npy` (`ojas-qwen35/tests/common/mod.rs:10`), and `tessl/Cargo.toml:113` declares `ojas-io` while tessl uses none of it. What exists is `ojas-io`'s own readers (`ojas-io/src/safetensors.rs`, `json.rs`, `npy.rs`), which seven ojas crates depend on (`ojas-data`, `-oracle`, `-capi`, `-model`, `-train`, `-qwen35`, `-cuda`).
- Tessl's whole-step training layer: `qwen35_train.rs` (tessl `102b89d`, whose message says "not built or run" and "DO-NOT-MERGE-BEFORE-AB") and the legacy `ojas-metal/src/gpu.rs` (ojas `b7b26c6`) are retired in favor of `ojas-model::TapeStepProvider`. `qwen35_adamw.rs` (218 lines, the free f32 `adamw_step` launcher) and `qwen35_bwd.rs` (1,031 lines, per-op backward launchers) were kept on purpose and are still `pub mod` (`tessl/src/lib.rs:273-274`). No caller of `tessl::qwen35_adamw` was found in ojas source by `rg`.
- AdamW scalar formation: two owners remain, down from the three the plan counted (the earlier count is not re-verified). `ojas-core` owns `AdamWScalars` (`ojas-core/src/backend.rs:277-314`); `ojas-cpu`, `ojas-metal` and `ojas-wgpu` call `AdamWScalars::new`, and `ojas-cuda` (`k11_host.rs:182`) and `ojas-train` (`lappi/adamw.rs:167`) call `from_bias_corrections` with libm bias corrections. `tessl/src/qwen35_adamw.rs:86` (`adamw_scalars`) forms its own, independently.
- String-scraping memory allocation classifier replaced by typed `AllocError`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | GitPulse Task Categorization & Priority Audit | Audit and categorize (P0/P1/P2) the GitPulse boards; the audit read the ojas and tessl boards and changed neither (stated by the audit, board history not checked). On 2026-11-01 the boards hold 134 (ojas), 114 (tessl) and 41 (SpikeTrack) cards (`gitpulse_list_tasks`); 19 of SpikeTrack's came from the roadmap import. The earlier 243-card total is not reproducible from the boards (ojas and tessl alone hold 248 now) | M1 | Survey (Explorer 3) |
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
| 15 | Tiled Cross-Attention for SAM Memory Attention ($D=32$, 8 heads) | Decoupled $T_k \ge T_q$ cross-attention on Metal at head dim 32, $H=8$ (`memory_attention.rs:20-22`): split-K tiled kernel (`attention_tiled.rs`), pad-q route (`ENCODER_ATTN_MAX_PAD_RATIO`, `ojas-metal/src/backend.rs:1688`) and TensorOps flash route (`flash_cross_route`, `backend.rs:1904`), the last asserted to take the flash kernel at $T_q=5184$, $T_k=15584$, $D=32$ by `attention_flash.rs:466` (on a TensorOps device). Head dim 256 flash is compiled (`device.rs:229`, `FLASH_CROSS_HEAD_DIMS` `backend.rs:1861`) but no test observes it. Memory attention is still 74% of each tracker state, profiled at ojas `ea3cf39` with the flash route in the build (`bench/results/profile-tracker/README.md:41`, `versions.txt:2`); no before/after measurement of this work exists | M3 | Survey (Explorer 2) |
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
| M1 | GitPulse Tasks & Accelerated Workflows | Features 1, 2, 3, 4, 5, 6 | none | DONE |
| M2 | Unified Architecture & Shared Memory Pools | Features 7, 8, 9, 10, 11, 12 | M1 | PARTIAL |
| M3 | SAM 3.1 & Vision Kernels with PyTorch Parity | Features 13, 14, 15, 16, 17, 18, 19, 20 | M2 | PARTIAL |
| M4 | Gemma Embedding Kernels with PyTorch Parity | Features 21, 22, 23, 24, 25, 26 | M2 | DONE |
| M5 | E2E Parity Verification & Coverage Hardening | Features 27, 28 (Pass 100% tests & adversarial hardening) | M3, M4 | DONE |

Status notes (2026-11-01; code read, nothing run). M1, M4 and M5 keep their earlier status and were not re-examined here.
- **M2 PARTIAL.** 7: the HAL traits are in `ojas-core`, and tessl implements them (`tessl/src/shared_buf.rs:680`, `tessl/src/runtime/host_fence.rs:147`). 8 and 10: not re-verified here. 9: the surface exists (`Payload::Mapped`, `Tensor::from_mapped`, `ojas-core/src/tensor.rs:955-985`) but no code outside tests calls it in ojas or tessl (`rg`), so zero-copy sharing is not in use. 11: not landed (see Code Duplication Elimination). 12: partial; orchestration retired, scalar formation has two owners.
- **M3 PARTIAL.** 14: Metal runs only `stride == kernel` with no padding or output padding (`ojas-metal/src/backend.rs:1606-1619`), not the generalized op. 15: see the row. 13 and 16-20: not re-verified here.

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
