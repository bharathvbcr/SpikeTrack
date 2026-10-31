# Handoff Report: GitPulse Tasks & Accelerated Workflows Survey

**Author**: Explorer 3 (Workflow & Tasks Explorer)  
**Date**: 2026-10-31T17:25:00Z  
**Target Codebases**: `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/tessl`, `/Users/bharath/Code/research/SpikeTrack`  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_3`  

---

## 1. Observation

### 1.1 GitPulse Task Boards and Worktree Inventory
Direct observation via GitPulse MCP tools (`gitpulse_list_tasks`, `gitpulse_insights`, `gitpulse_get_task`) and local file inspection:
- **`ojas` Task Board**: 128 active tasks returned (`repository.id: 95adcf33-d1b5-4599-9a36-09cb299223cf`).
- **`tessl` Task Board**: 105 active tasks returned (`repository.id: 9d836940-ed83-46b6-a042-7374157df380`).
- **`SpikeTrack` Task Board**: 10 active tasks returned (`repository.id: 20c324e3-e92c-4997-b76f-c483b75a8830`).
- **Worktree State (`ojas`)**: 28 worktrees detected via `gitpulse_insights`. 3 worktrees dirty (main has 15 uncommitted files, `ojas-ug-cpu-gemma` has 13 dirty files, `ojas-ug-metal-gemma` has 11 dirty files).
- **Task Brief Files in Repositories**:
  - `ojas/tasks/`: 66 markdown task briefs (e.g. `gp-spatial-ops.md`, `gp-vision-norm-act-bias.md`, `gp-bidirectional-attention.md`, `gp-structural-dedup.md`).
  - `tessl/tasks/`: 17 markdown task briefs (e.g. `gp-decode-icb-31b-and-range-batch-default.md`, `gp-metal4-fault-escalation-and-reader-bounds.md`).
  - `SpikeTrack/tasks/`: 19 markdown task briefs (e.g. `gp-vision-backbone.md`, `gp-reference-oracle.md`, `gp-detector-and-mask-head.md`).
  - `ojas/backups/branch-consolidation/classification-draft.md`: Records active state of unification waves (wave 2, kernel workflow, sprint workflow, wave 3).

### 1.2 Architecture Unification & Shared Memory State
Direct inspection of `ft-9802688f79d7b37b3b2a23525a90cb96` ("Unify ojas and tessl: ojas owns training, tessl owns Metal and Mac inference; one memory pool, one runtime and one ledger, no duplicated code") revealed:
- **Ownership Split**: 
  - `ojas` owns universal training in Rust: `Tensor`, `DType`, `OjasError`, `Budget`, the `Backend` trait, shape validators, autocast, autograd `Tape`, optimizers, data loaders, checkpoints, CPU reference backend. Light inference (sampling/eval) only.
  - `tessl` owns Metal + heavily optimized inference on Apple Silicon: prefill/decode, ICB decode, quantization, `EmbeddingGemma 2`, BERT, KV caches, fused kernels. Acts as `ojas`'s Metal extension (kernels, runtime, allocator).
- **Duplication & Friction Points Cited**:
  - **Memory Pools**: Every `tessl` buffer is `StorageModeShared` (`tessl/src/runtime.rs:299`), allowing unified CPU/GPU memory, but `ojas::Tensor` holds a host `Vec` plus an opaque `DeviceBuffer` with copy-only `read_bytes` (`ojas-core/src/tensor.rs:19-33`).
  - **Duplicate Runtimes**: Four distinct `GpuRuntime::new` sites in a single process: `ojas-metal/src/device.rs:576`, `ojas-metal/src/gpu.rs:330`, `ojas-qwen35/src/step.rs:399`, and `gemma-metal/src/kernels.rs:816`. Each has its own command queue, pool, pipeline cache, and residency set.
  - **Host Mapping Lockout**: `tessl`'s `GpuRuntime` is `!Send/!Sync` (`tessl/src/runtime.rs:1023-1030`). `tessl` `map_host` acquires an exclusive lease blocking GPU encoding (`tessl/src/tensor.rs:110-113`).
  - **Error Classification**: `tessl` returns `String` errors. `ojas` classifies OOM by matching string fragments (`classify_alloc`, `ojas-metal/src/device.rs:515-551`) and contains an `include_str!` test of `tessl/src/runtime.rs` (`ojas-metal/src/device.rs:5062`).
  - **Duplicate Kernel Families & AdamW**: 7 AdamW sites; duplicated RMSNorm, softmax, argmax, reduce, residual add, embedding gather, KV write, and attention across `ojas` and `tessl`.

### 1.3 SAM 3.1 & Gemma Embedding Kernel Status
From `ojas/tasks/gp-spatial-ops.md`, `gp-vision-norm-act-bias.md`, `gp-bidirectional-attention.md`, and `ojas-oracle/src/sam_fixtures.rs`:
- **SAM 3.1 Gaps in Ojas `Backend`**:
  - Attention: `Backend` had only causal attention (`causal_sdpa_*`). SAM 3.1 requires non-causal attention across 32 ViT layers (28 windowed 24x24 in 3x3 grids, 4 global 5,184 tokens), cross-attention (Tq != Tk), additive relative position bias (`boxRPB`), and key-length masks.
  - Normalization & Activation: `LayerNorm` (weight, bias, eps), `LayerNorm2d` (over C of NCHW), exact `GELU` (erf variant; tanh variant breaks parity), `GroupNorm` (for pixel decoder).
  - Spatial & Convolutions: `Conv2d` (via im2col + GEMM for patch embed 14x14 stride 14), 2-D axial RoPE (without norm; theta 10,000 over 72x72 grid), transposed conv2d stride 2 (GEMM + pixel shuffle), max pool 2x2, bilinear interpolation (`align_corners`).
  - Elementwise & Linear: `linear_bias` (linear forward with optional bias / fused epilogue), broadcast add/mul, standalone last-axis softmax.
- **Parity Harness & Fixtures in `ojas-oracle`**:
  - `ojas-oracle/src/sam_fixtures.rs` already defines exact isolated op parity harnesses: `axial_tables`, `rope_interleaved`, `attention`, `gelu`, `group_norm`, `layer_norm_affine_recovered`, `layer_norm_weighted`, `linear_bias`, `qkv_linear`, `conv2d`, `conv_transpose2d`.
  - Tolerances rigorously defined:
    - `TABLE_TOL`: `max_abs: 1e-6`, `max_rel: 1e-3`, `min_cosine: 1.0 - 1e-9`
    - `ROPE_TOL`: `max_abs: 1e-5`, `max_rel: 1e-3`, `min_cosine: 1.0 - 1e-9`
    - `GELU_TOL`: `max_abs: 1e-5`, `max_rel: 1e-3`, `min_cosine: 1.0 - 1e-9`
    - `LAYER_NORM_TOL`: `max_abs: 1e-4`, `max_rel: 1e-3`, `min_cosine: 1.0 - 1e-9`
    - `GROUP_NORM_TOL`: `max_abs: 1e-4`, `max_rel: 1e-3`, `min_cosine: 1.0 - 1e-9`
    - `ATTENTION_TOL`: `max_abs: 1e-4`, `max_rel: 1e-2`, `min_cosine: 1.0 - 1e-8`
  - Metal fixture test runner in `ojas-metal/tests/sam_fixtures_metal.rs` reads `OJAS_SAM_FIXTURES` (`fixtures/backbone/visem_22_f0.safetensors`).
- **Gemma Embedding Status**:
  - `tessl`: `embedgemma2::encoder_attn(_bf16)` provides bidirectional attention over right-padded batches with per-sequence lengths (tiled for D=64/256/512). Tasks: `ft-4efcd42fe048496d20b61cfaada11f1c` (speed & calibration), `ft-3aa3f7384dfebfc842cefeac9be61fb9` (evidence runs, native tokenizers), `ft-663f5b350361b943591df19fd3286c5c` (Gemma 4 forward for Lappi v2 with head dims 256/512).
  - `ojas`: `ft-3a9bc23c078680191aa866be4148bb9a` (Gemma 4 training path for Lappi v2), `ft-a95d2145479571d79e85440dc6e63396` (gemma-metal build.rs fix & crate dedup).

### 1.4 Existing Build, Test, and CI Workflows
- **`ojas/scripts/ci_local.sh`**:
  - Executes 12 stages: `fmt_check.sh`, `check_tracked_targets.sh`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo deny check`, `cargo clippy -p ojas-cuda --features cuda`, `simd_bench.sh`, `sitegen -check`, `benchreport plots -check`, `cargo build -p ojas-gusset-engine && go test`, `cargo test --workspace --release -- --test-threads=1`, `cargo doc --workspace`, and `clippy ojas-metal without metal`.
  - Bottleneck: Full workspace release testing serialized with `--test-threads=1` takes upwards of 15–25 minutes. Runs go tests, documentation checks, and site generators on every invocation.
- **`tessl/scripts/ci_local.sh`**:
  - Executes 15+ stages: `cargo fmt`, `cargo build --release`, `cargo clippy`, `cargo doc`, `hosted_tests.sh` (device hidden under sandbox denying IOKit), `dialect_lint`, `check_kernel_ir.sh`, emulator host contract (`check_qwen35.py`), emulator kernel cases (`check_qwen35_model.py`), comparator lanes, python binding tests, Go unit & device tests, `cargo test --release --no-fail-fast -- --test-threads=1`, `mtl_tensor` tests, and examples.
  - Requires `MAC_HEAVY_LABEL` via `tools/mac_heavy.sh` for exclusive GPU access to avoid test failures and noise.
- **`ojas/.github/workflows/test.yml`**:
  - Runs Linux (Ubuntu 24.04 with software Vulkan `lavapipe`), macOS (runs `cargo test --workspace --release --exclude ojas-metal` and only builds `ojas-metal` tests because hosted runners lack Metal 4 hardware), Go, GPU-compile, MSRV (1.97), and cargo-deny.
- **`SpikeTrack/Cargo.toml`**:
  - Path dependencies currently point to `../ojas-st/` (worktree detached at `ea3cf39`) to resolve relative sibling paths (`../../tessl`).

---

## 2. Logic Chain

1. **Premise 1: Unification Seam Ownership**: The authoritative direction in `ft-9802688f79d7b37b3b2a23525a90cb96` and `ORIGINAL_REQUEST.md` establishes that `ojas` is the universal framework centered on training, while `tessl` is the canonical Metal inference engine and `ojas`'s Metal backend provider.
2. **Premise 2: Memory Inefficiency Root Cause**: Hardware Apple Silicon unified memory uses `StorageModeShared` buffers. However, `ojas::Tensor` maintains a host `Vec` alongside an opaque `DeviceBuffer`, requiring explicit host-device copies (`read_bytes` / `upload`). Concurrently, `tessl`'s exclusive host lease (`map_host`) blocks asynchronous GPU encoding. Resolving this requires a thread-safe `Send + Sync` shared buffer abstraction with an explicit fence contract.
3. **Premise 3: Runtime Fragmentation**: Four separate `GpuRuntime::new` instantiations in one process fragment memory budgets and pipeline caches. Consolidating into a single process-wide runtime with an agreed allocation ledger eliminates memory over-commit and OOM misclassifications.
4. **Premise 4: Kernel Parity Requirements**: SAM 3.1 inference in `SpikeTrack` and training in `ojas` require non-causal attention, 2D RoPE, exact GELU, LayerNorm (and LayerNorm2d), Conv2d (im2col + GEMM), transposed conv2d, and GroupNorm. `tessl` already contains optimized Metal implementations for several of these (`bert::layer_norm`, `embedgemma2::encoder_attn`, GEMM). The architecture requires unifying op dispatch so `ojas-metal` delegates to `tessl` rather than maintaining duplicate kernel implementations.
5. **Premise 5: Workflow Bottleneck**: Development velocity is currently paralyzed because developers and agents run monolithic `ci_local.sh` scripts that serialize entire release test suites (`--test-threads=1`). A multi-tier workflow decoupling fast local feedback (<15s) from heavy parity verification and gated CI runs is mandatory.

---

## 3. Actionable Task Categorization (P0 / P1 / P2)

### Priority P0 — Critical Path / Blockers

| Task ID | Repo | Type | Status | Title & Scope |
|---|---|---|---|---|
| `ft-b4f9ce4e` | `ojas` | Bug | Review | Main CI red on push: Linux fmt failures, sitegen check, unrun Linux tests. Must be green to unblock PR merges. |
| `ft-53d639de` | `ojas` | Feature | Review | Bidirectional and cross attention with additive bias and key-length masks (SAM 3.1 critical requirement). |
| `ft-96baeabc` | `ojas` | Feature | In Progress | LayerNorm, exact GELU, linear bias and broadcast elementwise ops (SAM 3.1 ViT & neck blocker). |
| `ft-a050deca` | `ojas` | Bug | Review | Metal GDN backward refuses every call since tessl chunked (WY) merge (workspace size mismatch). |
| `ft-cab0fdf1` | `ojas` | Bug | Review | Device OOM misclassified on GPU backends (Metal retry never fires on real tessl error). |
| `ft-7f859b95` | `tessl` | Bug | Review | Main CI red since emulator host contract check failure. Blocks clean baseline. |
| `gp-vision-backbone` | `SpikeTrack`| Feature | In Progress | Vision backbone forward on Metal (im2col + tessl GEMM, 95 stages parity vs PyTorch). |
| `gp-reference-oracle`| `SpikeTrack`| Research| Ready | Offline PyTorch reference fixtures generation and validation baseline. |

### Priority P1 — High Priority (Architecture Unification, Shared Memory & Core Kernels)

| Task ID | Repo | Type | Status | Title & Scope |
|---|---|---|---|---|
| `ft-9802688f` | `ojas` | Refactor | Backlog | **Unify ojas and tessl**: one memory pool, one runtime, one ledger, eliminate duplicate kernels. Umbrella task for R2. |
| `ft-ee2a3753` | `ojas` | Bug | Review | Eliminate safetensors double-copy host transients, enforce child-first `Budget::check_room`, wire zero-copy UMA. |
| `ft-e55afcb5` | `ojas` | Refactor | Backlog | Share platform-neutral Qwen3.5 provider half between Metal and CUDA (config, names, groups, validators). |
| `ft-92c5e9c6` | `ojas` | Feature | In Progress | Conv2d, depthwise and transposed conv, pooling, interpolation, and 2-D RoPE for SAM 3.1. |
| `ft-f3f95dbe` | `ojas` | Perf | In Progress | Metal small/medium f32 GEMM throughput in SpikeTrack (conv2d im2col + linear_bias). |
| `ft-301ddd4c` | `ojas` | Chore | In Progress | Unify CI automation, test parity, multi-threaded readback accounting, and repo hygiene. |
| `ft-ab2f1daf` | `ojas` | Test | Review | Test coverage for new ops: torch goldens for GQA, sliding window, sampler, GPU trainers. |
| `ft-68d0f581` | `ojas` | Bug | Review | Compile and run unverified commits (AdamW 100-step oracle, LoRA memory preflight). |
| `ft-a3b7e231` | `tessl` | Bug | Review | Async encode without waiting sync exhausts 16 MiB const arena and poisons runtime. |
| `ft-6b9d840b` | `tessl` | Bug | Review | Runtime frees/recycles GPU memory on public SharedEvent (lifetime safety hazard). |
| `ft-c125d853` | `tessl` | Bug | Review | Safe code can bind or copy past GPU allocations (Binder & DecodeIcb soundness). |
| `ft-4efcd42f` | `tessl` | Perf | Review | EmbedGemma2 speed: calibrate FORWARD_COST_ROWS, overlap forwards, parallel segment_mean. |
| `ft-c4777436` | `SpikeTrack`| Feature | Backlog | Re-run memory-encode mask-upload measurement after ojas fresh_from upload lands. |
| `ft-4a8f33de` | `SpikeTrack`| Chore | Backlog | Zero-copy Backend::read_with / upload_with and tessl mapping a never-committed buffer. |

### Priority P2 — Medium Priority (Optimization, Follow-ups & Backlog)

| Task ID | Repo | Type | Status | Title & Scope |
|---|---|---|---|---|
| `ft-8d3aa337` | `ojas` | Feature | Review | SAM 3.1 op follow-ups: group_norm, pad/concat/slice, tiled Metal attention for bias/cross cases. |
| `ft-503c08c9` | `ojas` | Feature | Review | CPU backward for conv2d, conv_transpose2d, max_pool2d with Tape ops. |
| `ft-f6dcf4c5` | `ojas` | Perf | Backlog | Zero-copy `Backend::read_with` / `upload_with` for SpikeTrack frame I/O (95 MB readback, 170 MB mask). |
| `ft-3a9bc23c` | `ojas` | Feature | Backlog | Gemma 4 training path for Lappi v2 (step provider, LoRA on producer layers, softcap-aware 17-row slice). |
| `ft-6964cdbd` | `ojas` | Refactor | Backlog | One owner per behaviour: shape_product, training policy, head-dim constant, conformance suite. |
| `ft-a95d2145` | `ojas` | Chore | Ready | Fix gemma-metal build.rs panic, add CI/DevCouncil config, converge duplicate crates. |
| `ft-b90b30ae` | `ojas` | Chore | Backlog | Decide imported research trees arch02/ and gemma-metal/ (18-20 MB outside workspace). |
| `ft-6326debf` | `ojas` | Perf | Ready | Lift remaining red rows against PyTorch on Metal (grad accum, CE backward, hd64 attention). |
| `ft-4b1313de` | `tessl` | Spike | Review | Runtime lease exclusive: second read-only HostMapping fails "runtime busy". |
| `ft-3aa3f738` | `tessl` | Bench | Review | EmbeddingGemma 2 quiet evidence runs for text fusion/vision tower, native tokenizers. |
| `ft-663f5b35` | `tessl` | Feature | Backlog | Gemma 4 forward for Lappi v2: PLE, KV-share, sliding/global attention at head dims 256/512. |
| `ft-c8e7726b` | `tessl` | Refactor | Review | Share config.json parsing, batch validation, and upload_segments across BERT, EmbedGemma2, Qwen3.5. |
| `ft-01dd5d8a` | `tessl` | Refactor | Review | Benchmark harness convergence: bench_paired, src/bin/common, tests/common. |
| `ft-b37609de` | `SpikeTrack`| Feature | Backlog | E22: split and speed up detector decoder host prep (52 ms/frame on VISEM 82). |

---

## 4. Accelerated Workflows Design

To eliminate the massive latency of executing monolithic verification scripts on every iteration, we design and document five executable, layered workflows.

```
+---------------------------------------------------------------------------------------+
|                                ACCELERATED WORKFLOW SUITE                             |
+---------------------------------------------------------------------------------------+
|  [WF1: Inner Dev Loop]       -> Fast edit/compile check (< 15s)                       |
|  [WF2: Tiered Testing]       -> Unit -> CPU Exact -> Autograd -> Workspace            |
|  [WF3: Parity Verification]  -> Isolated op check against PyTorch fixtures (Tolerances)|
|  [WF4: Metal Execution]      -> Heavy GPU lock, serialized execution, leak isolation  |
|  [WF5: Worktree & Sync]      -> Sibling paths, worktree cleanup, collision safety     |
+---------------------------------------------------------------------------------------+
```

### Workflow 1: Fast Inner Development Loop (< 15s)
**Goal**: Rapid iteration during op implementation without triggering full workspace builds, docs generation, or serialized test suites.

```sh
#!/usr/bin/env bash
# scripts/dev_fast_check.sh <crate-name>
# Example: ./scripts/dev_fast_check.sh ojas-core
set -euo pipefail

CRATE="${1:-ojas-core}"

echo "==> [Fast Check] Syntax & Typecheck ($CRATE)"
cargo check -p "$CRATE"

echo "==> [Fast Check] Surgical Clippy ($CRATE)"
cargo clippy -p "$CRATE" -- -D warnings

echo "==> [Fast Check] In-crate Unit Tests (Debug, Parallel)"
cargo test -p "$CRATE" --lib -- --nocapture
```

### Workflow 2: Tiered Testing Pipeline
**Goal**: Structured gate execution from cheapest to most expensive, failing fast.

| Tier | Scope | Target | Timeout | Command |
|---|---|---|---|---|
| **Tier 1** | Pure shape/dims validators & core traits | `ojas-core` | < 10s | `cargo test -p ojas-core --lib` |
| **Tier 2** | CPU reference backend & bit-exact numerics | `ojas-cpu` | < 30s | `cargo test -p ojas-cpu --lib --test exact_golden` |
| **Tier 3** | Autograd reverse-mode Tape & gradchecks | `ojas-autograd` | < 45s | `cargo test -p ojas-autograd --test tape_parity` |
| **Tier 4** | Metal backend unit tests (Metal hardware) | `ojas-metal` | < 60s | `cargo test -p ojas-metal --lib -- --test-threads=1` |
| **Tier 5** | Release Workspace Smoke (Pre-commit only) | Workspace | < 3m | `cargo test --workspace --release --exclude ojas-metal -- --test-threads=4` |

### Workflow 3: PyTorch Parity Verification Loop
**Goal**: Verify bitwise numerical correctness of isolated ops against PyTorch reference fixtures without spinning up end-to-end models.

- **Fixture Location**: `SpikeTrack/fixtures/backbone/visem_22_f0.safetensors`
- **CPU Op Parity Command**:
  ```sh
  OJAS_SAM_FIXTURES="$HOME/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors" \
    cargo test -p ojas-oracle --test sam_fixtures_cpu -- <op_name> --ignored --nocapture
  ```
- **Metal Op Parity Command**:
  ```sh
  OJAS_SAM_FIXTURES="$HOME/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors" \
    cargo test -p ojas-metal --test sam_fixtures_metal -- <op_name> --ignored --nocapture --test-threads=1
  ```
- **Pass Criteria Check**: Output must satisfy all three metrics evaluated in `ojas_oracle::sam_fixtures::compare`:
  1. $\max |y_{\text{got}} - y_{\text{want}}| \le \text{tol}_{\text{abs}}$
  2. $\frac{\max |y_{\text{got}} - y_{\text{want}}|}{\max(|y_{\text{want}}|, 10^{-3} \cdot \max |y_{\text{want}}|)} \le \text{tol}_{\text{rel}}$
  3. $\frac{y_{\text{got}} \cdot y_{\text{want}}}{\|y_{\text{got}}\| \|y_{\text{want}}\|} \ge \text{tol}_{\text{cosine}}$

### Workflow 4: Metal Test Execution Workflow (Deterministic & Non-Contended)
**Goal**: Prevent GPU resource contention, flaky allocations, and test failures caused by concurrent processes sharing the Metal device.

- **Problem Addressed**:
  - `ojas-metal` tests reading the process-wide device allocation gauge fail under parallel test threads (`ft-bd5d290e104a55f3e7e715388c342e0e`).
  - Benchmarks produce distorted timings if background rendering or another agent runs GPU jobs simultaneously.
- **Workflow Protocol**:
  1. **Acquire Exclusive GPU Lock**:
     ```sh
     # Wrap test runs using mac_heavy lock
     /Users/bharath/Code/research/tessl/tools/mac_heavy.sh cargo test -p ojas-metal --test <test_name> -- --test-threads=1 --nocapture
     ```
  2. **Device-Free Fallback Policy**:
     On machines without Metal hardware (e.g. Linux CI or remote nodes), export `OJAS_ALLOW_NO_GPU=1` to allow clean skips; on local Apple Silicon, ensure `OJAS_ALLOW_NO_GPU=0` (or unset) to enforce fail-closed execution.
  3. **Pool Reset & Budget Isolation**:
     Tests instantiate `MetalBackend::new(Budget::new(bytes))` with a dedicated budget; test assertions must verify peak charged bytes vs pool allocations (`allocated_bytes_for`).

### Workflow 5: Cross-Workspace Synchronization & Worktree Management
**Goal**: Safely coordinate edits across sibling checkouts (`ojas`, `tessl`, `SpikeTrack`) without breaking path dependencies or causing branch collisions.

- **Dependency Graph**:
  ```
  SpikeTrack  ──>  ojas (ojas-core, ojas-cpu, ojas-metal, ojas-io)  ──>  tessl
       │                                                                  ▲
       └──────────────────────────────────────────────────────────────────┘
  ```
- **Sync Protocol**:
  1. **Check Collision Risk First**:
     Run `gitpulse_collision_risk` on each repository before editing shared interfaces (`ojas-core/src/backend.rs`, `tessl/src/runtime.rs`).
  2. **Sibling Path Hygiene**:
     Ensure `SpikeTrack/Cargo.toml` points to clean local checkouts (`../ojas/` rather than detached `../ojas-st/`).
  3. **Staged Worktree Pruning**:
     Consolidate or archive inactive `ojas-ug-*` worktrees to avoid GitPulse deadline timeouts and dirty ledger states.

---

## 5. Caveats

1. **Uncommitted Worktrees in Ojas**: 28 worktrees exist on the host machine. In particular, `ojas-ug-cpu-gemma` (13 dirty files) and `ojas-ug-metal-gemma` (11 dirty files) hold uncommitted edits from prior sprint sessions. These must be reviewed or stashed before touching those modules.
2. **GPU Lock Prerequisite**: `tools/mac_heavy.sh` must be present and executable for running heavy Metal device tests without race conditions.
3. **Fixture Disk Footprint**: Full SAM 3.1 activation fixtures require ~3.4 GB of disk space. Tests correctly fail closed if `OJAS_SAM_FIXTURES` is unset.
4. **No Implementation Undertaken**: In strict accordance with explorer identity and team protocol, this investigation was completely read-only. No source files outside the explorer working directory were modified.

---

## 6. Conclusion

1. **Architecture Unification Contract Established**: Card `ft-9802688f79d7b37b3b2a23525a90cb96` provides the complete specification for unifying `ojas` and `tessl`: `ojas` owns the universal Rust training surface, `tessl` owns Metal inference and runtime kernels, and a shared Send+Sync buffer handle eliminates host-device copies on Apple Silicon unified memory.
2. **Missing Kernel Inventory Complete**: SAM 3.1 requirements (bidirectional attention with additive bias/key-length masks, exact GELU, LayerNorm/LayerNorm2d, Conv2d via im2col+GEMM, transposed conv2d, 2-D axial RoPE) and Gemma embedding requirements are mapped to existing P0/P1 tasks with established numerical tolerances in `ojas-oracle`.
3. **Workflows Solved**: The 5 proposed accelerated workflows replace monolithic, slow scripts with sub-15s compile checks, tiered test execution, isolated PyTorch parity evaluation, and lock-protected Metal test execution.

---

## 7. Verification Method

To independently verify the observations, tasks, and workflows documented in this report:

1. **Verify GitPulse Tasks & Unification Brief**:
   ```sh
   # Inspect ft-9802688f on ojas board
   # Via MCP: gitpulse_get_task(repo_path: "/Users/bharath/Code/research/ojas", task_id: "ft-9802688f79d7b37b3b2a23525a90cb96")
   ```
2. **Verify Fast Inner Loop Syntax Check**:
   ```sh
   cd /Users/bharath/Code/research/ojas
   cargo check -p ojas-core -p ojas-cpu
   ```
3. **Verify Tier 1 & Tier 2 Tests**:
   ```sh
   cd /Users/bharath/Code/research/ojas
   cargo test -p ojas-core --lib
   cargo test -p ojas-cpu --lib
   ```
4. **Verify SAM Parity Fixture Harness Structure**:
   ```sh
   cd /Users/bharath/Code/research/ojas
   # Inspect sam_fixtures.rs tolerances and functions
   cat ojas-oracle/src/sam_fixtures.rs | grep -E "pub const.*TOL" -A 4
   ```
5. **Verify Tessl Hosted Tests Isolation**:
   ```sh
   cd /Users/bharath/Code/research/tessl
   ./scripts/hosted_tests.sh
   ```

*Invalidation Conditions*:
- If `ft-9802688f79d7b37b3b2a23525a90cb96` is deleted or overridden by an owner directive changing the ownership split.
- If `SpikeTrack` changes its model trunk from SAM 3.1 to an architecture not requiring Conv2d, 2D RoPE, or bidirectional attention.
- If Apple Silicon deprecates unified `StorageModeShared` memory buffers.
