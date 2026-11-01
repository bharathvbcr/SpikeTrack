# Forensic Audit Report: Milestone 4 (Features 21–26)

**Work Product**: Milestone 4 Implementation (Features 21–26: Gemma Embedding Kernels with PyTorch Parity) in `/Users/bharath/Code/research/ojas` and `/Users/bharath/Code/research/tessl`  
**Profile**: General Project (Forensic Integrity)  
**Integrity Mode**: `development` (per `ORIGINAL_REQUEST.md`)  
**Verdict**: **`CLEAN`**

---

### Phase Results
- **Hardcoded Output Detection**: **PASS** — Verified `gelu_tanh`, `seq_mean_forward`, `l2_normalize_forward`, `rms_qkv_norm_forward`, and symmetric sliding window attention compute genuine floating-point math across dynamic inputs without lookup tables or hardcoded values.
- **Facade Detection**: **PASS** — No dummy/stub implementations. Real numerical kernels executed across CPU and Metal GPU pipelines.
- **Pre-populated Artifact Detection**: **PASS** — `fixtures/gemma_ops/golden.txt` is an authentic reference dataset generated from deterministic PyTorch formulas in `gemma_ops_fixture.py`, dynamically asserted at runtime by `ojas-oracle/src/gemma_golden.rs`.
- **Placement Invariants & Zero Silent Fallback**: **PASS** — CPU host accessors (`host_window`, `host_data`, `host_data_mut` in `ojas-core/src/tensor.rs:761-845`) fail closed with `OjasError::Placement` on device buffers; Metal backend (`ojas-metal/src/backend.rs:708-715`) refuses host buffers; no silent CPU fallback occurs.
- **Un-Stubbed Tier 4 Pipeline**: **PASS** — `tier4_scenario_b_gemma_embedding_production_inference_pipeline` in `ojas-oracle/tests/e2e_tier4_production.rs:255-379` executes genuine backend methods (`rms_qkv_norm_forward`, `attention_forward`, `gelu_tanh_forward`, `seq_mean_forward`, `l2_normalize_forward`) and validates unit sphere norm conservation $\left|\|y\|_2 - 1.0\right| < 10^{-6}$.
- **Compiler & Repository Hygiene**: **PASS** — `#![forbid(unsafe_code)]` enforced across `ojas-core`, `ojas-cpu`, `ojas-metal`, `ojas-oracle`, `ojas-autograd`, `ojas-capi`; zero clippy warnings with `-D warnings`; `cargo deny check` passes; git status clean with all 382 target sources tracked.

---

## 1. Observation

### 1.1 Direct Source Code Observations
1. **Feature 21: `gelu_tanh` / GeGLU forward & backward**:
   - `ojas-core/src/gelu_tanh.rs:47-69`: Implements exact closed-form `f64` formulation `gelu_tanh_exact(x: f32) -> f32` and `gelu_tanh_backward_exact(x: f32, dy: f32) -> f32` using stable sigmoid parameterization $s = \text{sigmoid}(2\sqrt{2/\pi}(x + 0.044715 x^3))$ to prevent catastrophic cancellation in negative tails ($x \le -8.0$).
   - `ojas-cpu/src/gemma_ops.rs:23-60`: Multi-threaded CPU dispatch for `gelu_tanh_forward` and `gelu_tanh_backward` invoking `gelu_tanh_exact` and `gelu_tanh_backward_exact` through `fill_rows`.
   - `ojas-autograd/src/tape.rs`: Tape record `Rec::GeluTanh` handles reverse-mode differentiation and gradient chaining.
   - `ojas-metal/kernels/gemma_ops.metal:35-79`: Metal compute kernels `ojas_gelu_tanh_fwd` and `ojas_gelu_tanh_bwd` perform element-wise floating-point math with status flags for non-finite detection.
2. **Feature 22: Symmetric non-causal sliding window attention**:
   - `ojas-cpu/src/attn/general.rs:150-157`: Attention keys range computed as `(i.saturating_sub(r), i.saturating_add(r).saturating_add(1).min(len))` with dead query padding row masking (`if self.q_len == self.kv_len && i >= len { return 0..0; }`).
   - `ojas-metal/kernels/ojas_backend.metal` & `ojas-metal/src/backend.rs`: Sliding window radius flag bit `8u` and query padding row masking flag bit `16u`.
3. **Feature 23: Masked sequence mean pooling (`seq_mean_forward`)**:
   - `ojas-cpu/src/gemma_ops.rs:69-104`: Computes row mean per sequence using column-blocked `f64` accumulation over active tokens `rows = min(lens[b], seq)`, leaving zeros if `rows == 0`.
   - `ojas-metal/kernels/gemma_ops.metal:83-94`: `ojas_segments_from_lens` GPU kernel generates range segments directly on device, routed to Tessl's `segment_mean_rows_f32`.
4. **Feature 24: Matryoshka prefix L2 normalization (`l2_normalize_forward`)**:
   - `ojas-cpu/src/gemma_ops.rs:111-143`: Row-wise L2 norm in `f64` with $\epsilon$ clamping: `let denom = sum_sq.sqrt().max(eps); *slot = (f64::from(v) / denom) as f32;`.
   - `ojas-metal/src/gemma_ops.rs:19-22`: Dispatches to Tessl's GPU `l2_normalize_rows_f32` with fixed $\epsilon = 10^{-12}$.
5. **Feature 25: Per-head V RMSNorm (`rms_qkv_norm_forward`)**:
   - `ojas-core/src/shapes.rs:724-760`: `rms_qkv_norm_forward_dims` validates rank $\ge 1$, matching head dimensions, and optional weight shapes.
   - `ojas-cpu/src/backend.rs:706-733`: Normalizes Q and K using learned weights (or ones if weightless), and normalizes V weightlessly using a unit vector.
   - `ojas-metal/src/backend.rs:2532-2562`: Metal GPU implementation uploading unit vectors for weightless normalization.
6. **Feature 26: Golden Fixture & Parity Harness**:
   - `ojas-oracle/fixtures/gemma_ops/golden.txt`: Authentic 160-line fixture file containing 21 test cases (25 op evaluations) generated by PyTorch with seed `20261101`.
   - `ojas-oracle/src/gemma_golden.rs:250-465`: Real test runner executing ops dynamically on backends, asserting max absolute error $\le 1.0 \times 10^{-4}$ and cosine similarity $\ge 1.0 - 1.0 \times 10^{-8}$.
7. **Placement Invariance & Memory Safety**:
   - `ojas-core/src/tensor.rs:760-845`: `host_window`, `host_data`, and `host_data_mut` inspect `self.device()` and fail closed with `OjasError::Placement` if called on device or mapped tensors.
   - `#![forbid(unsafe_code)]` confirmed present in `ojas-core/src/lib.rs:9`, `ojas-cpu/src/lib.rs:63`, `ojas-metal/src/lib.rs:10`, `ojas-oracle/src/lib.rs:34`, `ojas-autograd/src/lib.rs:8`, `ojas-capi/src/lib.rs:13`.

### 1.2 Independent Empirical Verification Results

1. **CPU Gemma Golden Tests**:
   - Command: `cargo test -p ojas-oracle --test gemma_golden_cpu -- --nocapture`
   - Result: `test gemma_ops_match_torch_goldens_on_cpu ... ok` (25 ops evaluated across 21 cases, max absolute error $\le 8.94 \times 10^{-7}$, cosine = 1.0000000000).
2. **CPU Gemma Ops Unit Tests**:
   - Command: `cargo test -p ojas-cpu --test gemma_ops -- --nocapture`
   - Result: 15 passed, 0 failed, 0 ignored.
3. **Autograd Tape Verification**:
   - Command: `cargo test -p ojas-autograd --test gelu_tanh_tape -- --nocapture`
   - Result: 4 passed, 0 failed, 0 ignored.
4. **Metal GPU Gemma Golden Tests** (under `mac_heavy.sh` lock):
   - Command: `bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_parity bash -c "cd /Users/bharath/Code/research/ojas && cargo test -p ojas-metal --test gemma_golden_metal -- --nocapture"`
   - Result: `test gemma_ops_match_torch_goldens_on_metal ... ok` (25 ops evaluated across 21 cases, max absolute error $\le 1.79 \times 10^{-6}$, cosine = 1.0000000000).
5. **Metal GPU Gemma Ops Tests** (under `mac_heavy.sh` lock):
   - Command: `bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_ops bash -c "cd /Users/bharath/Code/research/ojas && cargo test -p ojas-metal --test gemma_ops -- --nocapture"`
   - Result: 4 passed, 0 failed, 0 ignored.
6. **Tier 4 Production Pipeline Scenarios**:
   - Command: `bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_tier4 bash -c "cd /Users/bharath/Code/research/ojas && cargo test --release -p ojas-oracle --test e2e_tier4_production -- --nocapture"`
   - Result: 4 passed, 0 failed, 0 ignored (`tier4_scenario_a_sam31_video_tracker_multiplex_pipeline`, `tier4_scenario_b_gemma_embedding_production_inference_pipeline`, `tier4_scenario_c_pipeline_budget_scoping_and_trim`, `tier4_scenario_d_no_silent_fallback_placement_invariance`).
7. **Full Local CI Gate Execution**:
   - Command: `bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ci_local bash -c "cd /Users/bharath/Code/research/ojas && ./scripts/ci_local.sh"`
   - Result: `ci:local OK` (fmt ok, 382 target sources tracked, clippy 0 warnings with `-D warnings`, cargo deny ok, bench charts up to date, all workspace release tests passed, rustdoc intra-doc links ok, metal stub build ok).
8. **Git Status & Working Tree Hygiene**:
   - Command: `git -C /Users/bharath/Code/research/ojas status`
   - Result: All changes cleanly staged, 0 untracked files, 0 dirty modifications.

---

## 2. Logic Chain

1. **Absence of Hardcoded Facades**:
   - Inspection of `ojas-core/src/gelu_tanh.rs`, `ojas-cpu/src/gemma_ops.rs`, `ojas-cpu/src/attn/general.rs`, and `ojas-metal/kernels/gemma_ops.metal` demonstrated that every mathematical operation processes dynamic float buffers through analytical and iterative mathematical algorithms.
   - No fixed output arrays, pre-computed lookup tables, or bypass branches match incoming test parameters.
2. **Fixture Authenticity**:
   - Inspection of `ojas-oracle/python/gemma_ops_fixture.py` verified that references were computed from genuine PyTorch operations in `float64` under random seeds.
   - The test evaluator `ojas-oracle/src/gemma_golden.rs` parses these tensors and computes actual L2 norms, maximum absolute differences, and cosine similarities between backend outputs and fixture data. Every case asserts within mathematical tolerance bounds ($\le 1.79 \times 10^{-6}$ max abs error).
3. **Fail-Closed Placement & Zero Fallback**:
   - `Tensor::host_window`, `Tensor::host_data`, and `Tensor::host_data_mut` explicitly enforce that if `self.device()` is not `None`, an `OjasError::Placement` is immediately returned.
   - Metal backend methods inspect tensor buffers and refuse host tensors with `OjasError::Placement`.
   - `e2e_tier4_production.rs::tier4_scenario_d_no_silent_fallback_placement_invariance` validates that invalid tensor configurations fail closed with typed errors.
4. **End-to-End Pipeline Integrity**:
   - `tier4_scenario_b_gemma_embedding_production_inference_pipeline` exercises the entire EmbeddingGemma 2 pipeline end-to-end (projections, per-head RMSNorm, sliding window attention, GeGLU feedforward, sequence mean pooling, Matryoshka slicing, and L2 normalization).
   - Invariants (unit norm $\|y\|_2 = 1.0 \pm 10^{-6}$, cosine alignment with unnormalized direction) are validated across dimensions [64, 32, 16].
5. **Code Quality and Standards**:
   - The workspace maintains `#![forbid(unsafe_code)]` in all target crates.
   - Clippy reports zero warnings under `-D warnings`.
   - All 382 targets are tracked in git.
   - Full CI passes cleanly.

---

## 3. Caveats

- **Test Execution Lock**: Metal GPU tests require the `mac_heavy.sh` execution lock to prevent Metal command buffer serialization stalls when run concurrently with other GPU processes.
- **External Checkpoint Weight Download**: Evaluation against full 271M parameter HuggingFace weights requires offline weight downloading; the test suite verifies PyTorch operator parity through self-contained golden fixtures.

---

## 4. Conclusion

The forensic integrity audit of Milestone 4 (Features 21–26) is complete.
All 6 features have been implemented with genuine, un-stubbed numerical algorithms on both CPU and Metal GPU backends.
No hardcoded lookups, facades, or synthetic shortcuts exist.
PyTorch golden parity is confirmed to within $1.79 \times 10^{-6}$ max abs error.
Placement invariants and zero silent fallback are strictly enforced.
Compiler and repository hygiene meet all standards (`ci:local OK`).

**Verdict: `CLEAN`**. Work product is approved.

---

## 5. Verification Method

To independently reproduce this forensic audit:

1. **Verify Formatting & Tracked Targets**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/fmt_check.sh
   ./scripts/check_tracked_targets.sh
   ```
2. **Verify Clippy & Cargo Deny**:
   ```bash
   cargo clippy --workspace --all-targets -- -D warnings
   cargo deny check
   ```
3. **Run CPU Gemma Golden & Unit Tests**:
   ```bash
   cargo test -p ojas-oracle --test gemma_golden_cpu -- --nocapture
   cargo test -p ojas-cpu --test gemma_ops -- --nocapture
   cargo test -p ojas-autograd --test gelu_tanh_tape -- --nocapture
   ```
4. **Run Metal GPU Tests under `mac_heavy.sh` lock**:
   ```bash
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_parity \
       bash -c "cd /Users/bharath/Code/research/ojas && cargo test -p ojas-metal --test gemma_golden_metal -- --nocapture"
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_ops \
       bash -c "cd /Users/bharath/Code/research/ojas && cargo test -p ojas-metal --test gemma_ops -- --nocapture"
   ```
5. **Run Tier 4 Production Pipelines**:
   ```bash
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_tier4 \
       bash -c "cd /Users/bharath/Code/research/ojas && cargo test --release -p ojas-oracle --test e2e_tier4_production -- --nocapture"
   ```
6. **Run Full CI Local Script**:
   ```bash
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ci_local \
       bash -c "cd /Users/bharath/Code/research/ojas && ./scripts/ci_local.sh"
   ```
   Confirm output ends with `ci:local OK`.
