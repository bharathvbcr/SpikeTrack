# SpikeTrack Developer Workflows

This document defines the standard developer workflows for **SpikeTrack** and its acceleration extension **tessl**.
These workflows ensure rapid inner-loop iteration, rigorous pre-push validation, exact numerical parity against PyTorch goldens, and contention-free Metal GPU execution on Apple Silicon.

---

## Workflow Quick Reference

| Workflow | Primary Command | Target Duration | When to Use |
|---|---|---|---|
| **WF1: Fast Inner Loop** | `./scripts/dev_fast_check.sh [crate]` | **< 15 seconds** | Every code change during active development (compile/lint/unit test). |
| **WF1 (tessl): Fast Loop** | `../tessl/scripts/dev_fast_check.sh [filter]` | **< 15 seconds** | When modifying tessl Metal 4 kernels or runtime. |
| **WF2: Local Pre-Push CI** | `./scripts/ci_local.sh` | **~ 30–60 seconds** | Before pushing commits or opening pull requests. Validates formatting, no-Python invariant, clippy, and release tests. |
| **WF3: Parity Verification** | `./scripts/parity_check.sh [options]` | **~ 10–30 seconds** | After changing backbone or tracker modeling code; checks numerical agreement against PyTorch oracle fixtures. |
| **WF4: Serialized Metal Run** | `mac_heavy.sh ./scripts/parity_check.sh --backend metal` | **Variable** | Running GPU-heavy workloads to avoid Metal command buffer contention and memory thrashing. |

---

## WF1: Fast Inner Loop (`dev_fast_check.sh`)

### Purpose
The fast inner development loop provides sub-15-second feedback for developers working on individual crates or components. It avoids the overhead of compiling release binaries or loading multi-gigabyte neural network weights.

### Architecture & Stages
`scripts/dev_fast_check.sh` executes three focused stages:

1. **Stage 1: Syntax & Typecheck (`cargo check`)**:
   - Performs rapid compiler checks for syntax, borrow checking, and type resolution.
2. **Stage 2: Surgical Clippy (`cargo clippy -D warnings`)**:
   - Catches compiler warnings, idiom violations, and anti-patterns strictly within the target crate. Omitted by default for sub-15s inner loop iteration; pass `--clippy` to enable.
3. **Stage 3: In-crate Parallel Unit Tests (`cargo test --lib -- --nocapture`)**:
   - Executes non-ignored, self-contained unit tests in parallel.

### Usage & Examples

```bash
# Default check (targets st-model with Stage 1 check + Stage 3 unit tests, clippy omitted)
./scripts/dev_fast_check.sh

# Run clippy stage explicitly (opt-in for inner loop)
./scripts/dev_fast_check.sh --clippy

# Target a specific crate
./scripts/dev_fast_check.sh st-gate
./scripts/dev_fast_check.sh st-weights
./scripts/dev_fast_check.sh st-video
./scripts/dev_fast_check.sh st-coreml
./scripts/dev_fast_check.sh spiketrack

# Target multiple crates simultaneously
./scripts/dev_fast_check.sh st-model st-weights

# Fast syntax-only check (skips clippy and tests)
./scripts/dev_fast_check.sh --check-only

# Skip clippy explicitly (default behavior)
./scripts/dev_fast_check.sh --skip-clippy st-model

# Workspace-wide fast check
./scripts/dev_fast_check.sh --all
```

### Measured Performance
On Apple Silicon M-series hardware with warm caches:
- `st-model` (Stage 1 check + Stage 3 unit tests with 49 tests): **~11 seconds** (well within the < 15s budget). Note: initial cold cache compilation may take ~40 seconds before settling into incremental sub-15s iteration.
- `--check-only`: **~2–4 seconds** (warm cache).

### Sister Fast Loop in `tessl`
When working on Metal 4 compute kernels or runtime buffers in the sibling checkout `../tessl`, use:

```bash
cd /Users/bharath/Code/research/tessl

# Default fast check (runs Stage 1 check + Stage 3 runtime::tests, < 15s warm)
./scripts/dev_fast_check.sh

# Run clippy stage explicitly (skipped by default for sub-15s inner loop)
./scripts/dev_fast_check.sh --clippy

# Fast check on specific modules
./scripts/dev_fast_check.sh gemm::tests
./scripts/dev_fast_check.sh tensor

# Syntax only
./scripts/dev_fast_check.sh --check-only   # ~2–4 seconds warm
```
Note: `tessl` tests are serialized with `--test-threads=1` to prevent Metal device allocation race conditions. The script enforces fail-closed validation: if a filter matches 0 unit tests, it exits immediately with code 1.

---

## WF2: Local Pre-Push CI (`ci_local.sh`)

### Purpose
`scripts/ci_local.sh` mirrors the GitHub Actions CI pipeline locally. It prevents broken builds or formatting regressions from reaching upstream.

### Architecture & Gates

1. **Gate 0: Sibling Dependency Verification**:
   - Verifies the required sibling repositories exist:
     - `../ojas` (Universal autograd and core traits HAL)
     - `../tessl` (Apple Silicon Metal 4 acceleration)
     - `../sparsl` (Sparse computing extension)
2. **Gate 1: Code Formatting Verification (`cargo fmt`)**:
   - Discovers workspace packages via `cargo metadata` and validates formatting strictly on SpikeTrack packages (`-p st-model -p st-gate ...`), preventing accidental modification or checks on external sibling path dependencies.
3. **Gate 2: House Language Policy Enforcement (`scripts/no_python_check.sh`)**:
   - Ensures the release binary (`target/release/spiketrack`) is 100% Rust-native.
   - Inspects Mach-O dynamic library bindings (`otool -L`) to verify zero non-system / non-Apple libraries and zero Python runtime bindings.
4. **Gate 3: Strict Workspace Clippy (`cargo clippy --workspace --all-targets -- -D warnings`)**:
   - Enforces zero compiler warnings across all workspace members and test targets.
5. **Gate 4: Release Mode Lib Test Suite (`cargo test --workspace --release --lib`)**:
   - Runs full workspace library tests compiled with release optimizations, serialized via `--test-threads=1`.

### Usage & Examples

```bash
# Full local CI verification
./scripts/ci_local.sh

# Automatically apply formatting fixes
./scripts/ci_local.sh --fix-fmt

# Skip specific gates during focused debugging
./scripts/ci_local.sh --skip-fmt
./scripts/ci_local.sh --skip-python
./scripts/ci_local.sh --skip-clippy
./scripts/ci_local.sh --skip-tests
```

### Measured Performance
- Full release library tests across all 6 workspace crates (140 tests): **~1.39 seconds** execution time once compiled.

---

## WF3: PyTorch Golden Parity Verification (`parity_check.sh`)

### Purpose
SpikeTrack guarantees bit-level and numerical equivalence to PyTorch upstream models. `scripts/parity_check.sh` automates the execution of parity tests against precomputed PyTorch oracle fixtures (`visem_22_f0.safetensors`, `judo.safetensors`).

### Reference Fixtures
The script automatically discovers reference fixtures at standard locations:
- **SAM 3.1 Multiplex Weights**: `models/sam3.1/sam3.1_multiplex.safetensors` (~3.4 GB)
- **Vision Backbone Fixtures**: `fixtures/backbone/visem_22_f0.safetensors` (~3.4 GB)
- **Tracker Fixtures**: `fixtures/tracker/judo.safetensors` (~2.6 GB)

Paths can be overridden via command-line arguments (`--weights`, `--fixtures`, `--tracker-fixtures`) or environment variables (`SPIKETRACK_WEIGHTS`, `SPIKETRACK_FIXTURES`, `SPIKETRACK_TRACKER_FIXTURES`).

### Execution Matrix

```bash
# Run CPU backbone parity (default) — executes backbone_matches_pytorch_fp32_on_five_images
./scripts/parity_check.sh

# Run CPU tracker parity — executes multiplex_mask_decoder_matches_upstream
./scripts/parity_check.sh --component tracker --backend cpu

# Run Metal GPU backbone parity — executes backbone_matches_pytorch_fp32_on_five_images_on_metal under GPU lock
./scripts/parity_check.sh --component backbone --backend metal

# Run Metal GPU tracker parity — executes multiplex_mask_decoder_matches_upstream_on_metal under GPU lock
./scripts/parity_check.sh --component tracker --backend metal

# Run all components across both CPU and Metal
./scripts/parity_check.sh --component all --backend all

# Run specific individual test with exact filter name
./scripts/parity_check.sh --component backbone --test vit_layer_norm_eps_matches_upstream
./scripts/parity_check.sh --component tracker --backend cpu --test multiplex_mask_decoder_matches_upstream
./scripts/parity_check.sh --component tracker --backend metal --test multiplex_mask_decoder_matches_upstream_on_metal
```

### Exact Matching & Backend Isolation Guard
`parity_check.sh` appends `--exact` to the Cargo test invocation (`cargo test ... -- --exact --ignored --nocapture --test-threads=1`). This guarantees that:
1. **Zero Substring Bleed**: When running CPU suites (e.g. `multiplex_mask_decoder_matches_upstream`), Cargo does not substring-match or execute the Metal equivalent (`multiplex_mask_decoder_matches_upstream_on_metal`).
2. **GPU Lock Protection**: Metal tests are only ever executed when the target backend is `metal`, and are wrapped under the `mac_heavy.sh` GPU lock to avoid Apple Silicon UMA contention.
3. **Fail-Closed CPU Guard**: If a user requests `--backend cpu` with a filter naming a Metal test, the script immediately refuses execution with exit code 1.

### Integrity & Fail-Closed Guard
`parity_check.sh` enforces a fail-closed check: if the test command reports `0 tests run` or `0 passed; 0 failed` (e.g. if a test filter was misspelled), the script exits with an error code rather than falsely reporting success.

### Parity Criteria
- **Tracker Multiplex Mask Decoder**: Cosine similarity **1.0000000** across all 7 video frames.
- **Vision Backbone Normalization & Attention**: Max absolute difference **< 1.4e-6** (exact f32 precision).

---

## WF4: Metal GPU Serialized Execution & GPU Lock (`mac_heavy.sh`)

### Contention Risk on Apple Silicon
Apple Silicon uses Unified Memory Architecture (UMA) shared between the CPU and GPU. When multiple GPU-heavy processes (such as Metal compute kernels, CoreML models, or tensor allocations) execute concurrently:
1. Command buffer completions can exceed timeout thresholds, triggering Metal assertion failures (`MTLCommandBufferStatusError`).
2. High memory residency spikes can force macOS into aggressive swap compression or out-of-memory terminations.

### GPU Lock Integration
`parity_check.sh` integrates with `mac_heavy.sh` (discovered automatically in sibling research tooling, e.g. `../Lappi-decision/tools/mac_heavy.sh`):

1. **Lock Detection**:
   - If already running under an active `mac_heavy` lock context (`MAC_HEAVY_LABEL` is set), `parity_check.sh` inherits the lock.
   - If not in a lock context, `parity_check.sh` automatically wraps Metal test commands with `mac_heavy.sh spiketrack-parity`.
   - If `mac_heavy.sh` is absent, the script falls back gracefully to serializing runs with `--test-threads=1`.
2. **Manual Override**:
   - To bypass GPU lock checks during isolated interactive runs, pass the `--no-gpu-lock` flag:
     ```bash
     ./scripts/parity_check.sh --backend metal --no-gpu-lock
     ```
3. **Manual Wrapping**:
   - You can also explicitly run any script under `mac_heavy.sh`:
     ```bash
     /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh my-run ./scripts/parity_check.sh --backend metal
     ```

---

## Maintenance & House Policies

1. **Executable Permissions**:
   All scripts in `scripts/` must retain executable permissions (`chmod +x`).
2. **No Python Dependency**:
   SpikeTrack and tessl are native Rust engines. No Python scripts, runtimes, or bindings may be introduced into the build, test, or deployment path.
3. **Isolated Formatting**:
   Do not run `cargo fmt --all` from the workspace root if it formats out-of-tree path dependencies (`../tessl`, `../ojas`, `../sparsl`). Always use `scripts/ci_local.sh --fix-fmt` to format SpikeTrack packages exclusively.
