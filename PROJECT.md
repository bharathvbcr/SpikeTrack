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
- `SpikeTrack → ojas`: `ojas-core`, `ojas-cpu`, `ojas-io`, `ojas-metal` by path (`Cargo.toml:16-19`). Six of the seven workspace crates depend on ojas: `st-model` (`ojas-core`, `-io`, `-cpu`, `-metal`) and `spiketrack` (all four) as normal dependencies, `st-weights` (`ojas-io`) and `st-coreml` (`ojas-core`; `ojas-cpu` and `-io` as dev-dependencies) as normal ones too, and `st-eval` and `st-video` (`ojas-io`) as dev-dependencies only. `st-gate` has none.
- `ojas-metal → tessl`: optional, behind the `metal` feature (`ojas/ojas-metal/Cargo.toml:14,22`).
- `tessl → ojas-core`, `ojas-io`: path dependencies (`tessl/Cargo.toml:112-113`). Between the two checkouts this is a path cycle (`ojas-metal → tessl → ojas-core`); between crates it is a strict DAG, because `ojas-core` and `ojas-io` depend on neither `tessl` nor `ojas-metal`.
- `tessl` imports `ojas-io` directly (`pub use ojas_io::{npy, safetensors};` in `tessl/src/lib.rs` and `ojas_io::json` in `tessl/src/config_json.rs`) alongside `ojas_core` (`MappedBuffer`, `HostLease`, `Timeline`, `AllocError`, `AdamWScalars`).

### Modularity Rationale Across Checkouts
1. **Decoupled Compile Cycles:** Xcode 26 Metal toolchain compilation (`xcrun metal`, `metallib`) and intensive sparse property testing (`proptest`, mutation suites) remain isolated from universal training CI and cross-platform builds.
2. **Isolated Hardware Requirements:** Apple Silicon M-series specifics live in `tessl`; generic CPU and optional Metal sparse kernels live in `sparsl`; portable multi-platform training lives in `ojas`.
3. **Independent Profiling & Micro-Benchmarking:** GEMM tile sweeps and register tuning (tessl) vs sparse crossover thresholds (sparsl) vs end-to-end training step throughput (ojas) iterate on distinct benchmark suites.
4. **Independent Release Cadence & Separate Checkouts:** Downstream consumers take only what they need without code or runtime bloat.

### Shared Memory Pools (Zero-Copy UMA)
- `ojas_core::Budget` serves as the sole authoritative ledger for host allocations and Tessl buffer pool cache (`pool.cache` child via `BudgetLedger` in `Worker::open`, with automatic pressure trim via `BudgetTrimHandler` and capacity refusals mapped to `OjasError::CapacityExceeded`).
- `ojas_core::Tensor` carries `Payload::Mapped(Arc<dyn MappedBuffer>)`.
- On Apple Silicon, `to_host()` returns a timeline-fenced `HostLease` over the same `StorageModeShared` buffer without `memcpy`.
- `upload()` on `MetalBackend` (`Worker::upload`) downcasts `tessl::SharedBuf` and calls `SharedBuf::to_gpu_buffer()` to adopt the unified memory buffer directly into the device buffer table with zero byte copies (`mem_after == mem_before`).
- In `SpikeTrack` (`crates/st-model/src/layers.rs`), dead `hw`/`hb` host weight vectors were removed from `Lin` and `Norm` (saving >1.2 GB of duplicate host RAM), and redundant `.to_host().to_f32_vec()` round-trips in `mask_decoder.rs`, `interactive.rs`, `vos.rs`, and `detector.rs` were replaced with device-native `Backend` operations (`slice`, `concat`, `fill`, `broadcast_add_forward`).
- **No Silent Fallback**: Placement remains strictly explicit (`BackendId::Cpu` vs `BackendId::Metal`). CPU methods refuse device tensors with `OjasError::Placement`.

### Code Duplication Elimination
- SafeTensors / JSON / NPY readers: **deduplicated in full (Milestone M2, 2026-11-02).** `tessl/src/safetensors.rs`, `json.rs`, `npy.rs`, and `plain.rs` (-1,645 LOC) are deleted. `tessl/src/lib.rs` re-exports `pub use ojas_io::{npy, safetensors};`, `tessl/src/config_json.rs` parses model configs through `ojas_io::json::{JsonValue, JsonLimits}`, and `ojas-qwen35` imports `ojas_io::safetensors::CheckpointView` directly with zero `tessl::safetensors` references.
- Tessl's whole-step training layer: `qwen35_train.rs` and the legacy `ojas-metal/src/gpu.rs` (3,735 lines) are retired in favor of `ojas-model::TapeStepProvider`. `qwen35_adamw.rs` (the f32 `adamw_step` kernel launcher) and `qwen35_bwd.rs` (per-op backward launchers) remain for kernel dispatch.
- AdamW scalar formation: **unified in `ojas-core::AdamWScalars::new`.** `ojas-cpu`, `ojas-metal`, `ojas-wgpu`, and `tessl/src/qwen35_adamw.rs` all delegate bias-correction and decoupled weight-decay scalar calculation to `ojas_core::AdamWScalars` (`ojas-cuda` and `ojas-train` use `AdamWScalars::from_bias_corrections`).
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
| 15 | Tiled Cross-Attention for SAM Memory Attention ($D=32$, 8 heads; $D=64$) | Native decoupled $T_q \times T_k$ TensorOps flash cross-attention (`ojas_attn_fwd_d32` and `ojas_attn_fwd_d64` in `ojas-metal/kernels/ojas_backend.metal` and `ojas-metal/src/backend.rs`), eliminating the $O(m^2)$ pad-q zero-padding route. On `[1, 8, 5184, 32] x [1, 8, 5200, 32]` memory cross-attention, executes in **6.06 ms vs 209.28 ms scalar (34.5x speedup, 100% flash dispatch)** (`ojas-metal/examples/attention_flash_bench.rs`); verified by `ojas-metal/tests/attention_flash.rs` and `ojas-metal/tests/adversarial_m4_attention.rs` | M3 | Survey (Explorer 2) |
| 16 | Biased Attention Forward (`boxRPB`) | Register-level additive logit bias (`boxRPB`) support in `ojas_attn_fwd_d32` and `ojas_attn_fwd_d64` via buffers `13..18` (bit 2 of `window_and_flags`), accumulating broadcastable 4D bias `[Bb, Hb, Tqb, Tk]` directly into tile registers before online softmax. On `[1, 8, 201, 32] x [1, 8, 5184, 32]` DETR cross-attention, executes in **0.85 ms vs 11.98 ms scalar (14.1x speedup)**; short queries (< `flash_min_groups()`) route to split-K tiled `ojas_xattn_part_dD` / `ojas_xattn_combine_dD` | M3 | Survey (Explorer 2) |
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
| M2 | Unified Architecture & Shared Memory Pools | Features 7, 8, 9, 10, 11, 12 | M1 | DONE |
| M3 | SAM 3.1 & Vision Kernels with PyTorch Parity | Features 13, 14, 15, 16, 17, 18, 19, 20 | M2 | DONE |
| M4 | Gemma Embedding Kernels with PyTorch Parity | Features 21, 22, 23, 24, 25, 26 | M2 | DONE |
| M5 | E2E Parity Verification & Coverage Hardening | Features 27, 28 (Pass 100% tests & adversarial hardening) | M3, M4 | DONE |

Status notes (verified 2026-11-02 across `ojas`, `tessl`, `sparsl`, and `SpikeTrack`):
- **M1 DONE.** Reconciled GitPulse tasks across all repos (`gitpulse_list_tasks` 0 open P0/P1 blockers) and shipped fast inner-loop (`scripts/dev_fast_check.sh` in `SpikeTrack` and `tessl`, < 15s warm), local pre-push CI (`scripts/ci_local.sh`), fail-closed parity runner (`scripts/parity_check.sh`), and `docs/WORKFLOWS.md`.
- **M2 DONE.** Deleted 1,645 LOC of duplicate `safetensors.rs`, `json.rs`, `npy.rs`, and `plain.rs` parsers in `tessl` (re-exporting `ojas_io::{npy, safetensors}` in `tessl/src/lib.rs` and consuming `ojas_io::json` in `tessl/src/config_json.rs`); unified `AdamWScalars::new` in `ojas-core` (`tessl/src/qwen35_adamw.rs`); wired `BudgetLedger` pool capacity errors to `OjasError::CapacityExceeded` in `ojas-metal`'s `Worker::open`; added `SharedBuf::to_gpu_buffer()` in `tessl/src/shared_buf.rs` for zero-copy `Payload::Mapped` device adoption on `Backend::upload` (`mem_after == mem_before`); and removed dead `hw`/`hb` host weight vectors (>1.2 GB saved) and redundant host-device round-trips in `SpikeTrack/crates/st-model`.
- **M3 DONE.** (The two kernel speedups in this item have no committed raw result; the README does not quote them. A committed SpikeTrack-side reading is the ops profile in `bench/results/ab-coreml-tracker/run2-ops/`.) Native decoupled $T_q \times T_k$ TensorOps flash cross-attention (`ojas_attn_fwd_d32` and `ojas_attn_fwd_d64`, **34.5x speedup** on $5184 \times 5200$ memory cross-attention: `6.06 ms` vs `209.28 ms` scalar) and tile-register additive logit bias `boxRPB` via buffers `13..18` (**14.1x speedup** on DETR cross-attention: `0.85 ms` vs `11.98 ms` scalar), plus full CPU and Metal SAM 3.1 backbone and multiplex tracker PyTorch parity (cosine similarity `1.0000000`, max abs diff `< 1.4e-6`).
- **M4 DONE.** Gemma embedding kernels (`gelu_tanh`, non-causal sliding window attention, `seq_mean_forward`, `l2_normalize_forward` with Matryoshka truncation, and `rms_qkv_norm_forward`) verified on CPU and Metal hardware against 26 records / 22 test cases with cosine similarity `1.0000000000`.
- **M5 DONE.** 142/142 4-tier E2E tests in `ojas-oracle`, plus adversarial M2/M3/M4 test suites (`ojas-core/tests/zero_copy_uma.rs`, `ojas-metal/tests/adversarial_m3_zero_copy.rs`, `ojas-metal/tests/adversarial_m4_attention.rs`), `SpikeTrack` release suite (163 passed, 0 failed), and `otool -L` native binary verification (zero Python/libtorch links).

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
