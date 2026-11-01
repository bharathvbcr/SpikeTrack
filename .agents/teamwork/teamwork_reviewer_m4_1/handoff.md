# Milestone 4 Code Review & Adversarial Audit Report: Gemma Embedding Kernels with PyTorch Parity (Features 21–26)

**Reviewer**: `teamwork_reviewer_m4_1`  
**Roles**: Reviewer, Adversarial Critic  
**Date**: 2026-11-01  
**Milestone**: Milestone 4 (Features 21–26)  
**Target Repository**: `/Users/bharath/Code/research/ojas`  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_reviewer_m4_1`  
**Final Verdict**: **`APPROVE`**

---

## 1. Observation

### 1.1 Scope & Code Changes Examined
Milestone 4 introduces 6 core features (Features 21–26) establishing full PyTorch parity and Metal GPU acceleration for Gemma Embedding models across `ojas-core`, `ojas-cpu`, `ojas-autograd`, `ojas-metal`, `ojas-capi`, and `ojas-oracle`:

1. **Feature 21: `gelu_tanh` / GeGLU Forward & Backward**:
   - `ojas-core/src/gelu_tanh.rs`: Formulated $x \cdot \text{sigmoid}(u)$ with $u = 2\sqrt{2/\pi}(x + 0.044715 x^3)$ using closed-form `f64` arithmetic over `exp_exact::exp_neg(u.abs())`. The derivative $\frac{dy}{dx} = s + x \cdot s(1 - s) \cdot u'$ computes $s(1 - s) = \frac{e^{-|u|}}{(1 + e^{-|u|})^2}$ avoiding catastrophic cancellation in both tails.
   - `ojas-cpu/src/gemma_ops.rs`: Multi-threaded `gelu_tanh_forward` and `gelu_tanh_backward` row execution with `fill_rows`, single `f32` rounding, and overflow detection (`OjasError::NonFinite`).
   - `ojas-autograd/src/tape.rs`: Added `Rec::GeluTanh` record, tape method `gelu_tanh`, and backward chaining accumulating gradients into leaf variables.
   - `ojas-metal/kernels/gemma_ops.metal`: Compute kernels `ojas_gelu_tanh_fwd` (calling tessl's canonical `tessl_gelu_pytorch_tanh` from `act.h`) and `ojas_gelu_tanh_bwd` (evaluating derivative with clamped inner arguments $|x| \le 20, |u| \le 10$).
   - `ojas-metal/src/backend/gemma_host.rs` & `ojas-metal/src/device/gemma_dev.rs`: Metal host and device dispatch with `ST_IN` / `ST_OUT` non-finite status word tracking.

2. **Feature 22: Symmetric Non-Causal Sliding Window Attention ($|i - j| \le W$) with Query Padding Row Masking**:
   - `ojas-core/src/shapes.rs`: Boundary validation enforcing $R \ge 1$, rejecting causal + radius, requiring $T_q == T_k$, and normalizing $R \ge T_k - 1$ to `None`.
   - `ojas-cpu/src/attn/general.rs`: `Geom::keys` determines visible keys $j \in [\max(0, i - R), \min(\text{len}_b, i + R + 1))$. If $T_q == T_k$ and $i \ge \text{len}_b$, query row is marked as padding ($0..0$ keys), writing zeroes to output and $0.0$ to `lse`.
   - `ojas-metal/kernels/ojas_backend.metal`: Added flag bit 3 (`8u` for symmetric band) and flag bit 4 (`16u` for dead query rows $i \ge \text{len}_b$).
   - `ojas-metal/src/backend.rs`: Routes sliding window calls to either tiled `encoder_attn` ($D \in \{32, 64, 256, 512\}$) or scalar general attention ($D \le 256$).

3. **Feature 23: Masked Sequence Mean Pooling (`seq_mean_forward`)**:
   - `ojas-core/src/shapes.rs`: `seq_mean_forward_dims` strictly enforces $x$ rank 3 $[B, T, D]$ `F32` and `key_lens` rank 1 $[B]$ `U32`.
   - `ojas-cpu/src/gemma_ops.rs`: Multi-threaded CPU dispatch over active tokens $\min(\text{len}_b, T)$, two-pass accumulation in `f64` in chunks of 16 columns (`MEAN_COLUMNS = 16`). Never reads padded rows for reduction, while validating entire tensor for finiteness.
   - `ojas-metal/kernels/gemma_ops.metal`: Dedicated `ojas_segments_from_lens` kernel turns device `lens` buffer into $[B, 2]$ ranges $(b \cdot T, b \cdot T + \min(\text{len}_b, T))$ directly on GPU without host readback.
   - `ojas-metal/src/device/gemma_dev.rs`: Dispatches `ojas_segments_from_lens` followed by tessl's optimized `segment_mean_rows_f32`.

4. **Feature 24: Matryoshka Prefix L2 Normalization (`l2_normalize_forward`)**:
   - `ojas-core/src/shapes.rs`: `l2_normalize_forward_dims` validates rank 2 $[N, D]$, finite $\epsilon > 0$.
   - `ojas-cpu/src/gemma_ops.rs`: Multi-threaded row-wise L2 norm with `max(norm, eps)` denominator. Zero rows output $+0.0$ without NaN.
   - `ojas-metal/src/backend/gemma_host.rs`: Enforces $\epsilon = 10^{-12}$ (failing closed with `OjasError::Unsupported` for any other $\epsilon$).
   - `ojas-metal/src/device/gemma_dev.rs`: Routes to tessl's `l2_normalize_rows_f32`.

5. **Feature 25: Per-Head V RMSNorm (`rms_qkv_norm_forward`)**:
   - `ojas-core/src/backend.rs`: Extended `Backend` trait with `rms_qkv_norm_forward`, updated `with_backend_methods!`, `every_call_reaches`, and `autocast.rs`.
   - `ojas-core/src/shapes.rs`: Pure validator `rms_qkv_norm_forward_dims` supporting learned weights on $Q$ and $K$, and weightless unit-scaling on $V$.
   - `ojas-cpu/src/backend.rs` & `ojas-metal/src/backend.rs`: Full CPU and Metal implementations.
   - `ojas-capi/src/gate_w11_adversarial.rs`: Pinned ops count updated from 98 to 99, refused ops from 94 to 95.

6. **Feature 26: Gemma Embedding PyTorch Parity & Golden Fixtures**:
   - `ojas-oracle/python/gemma_ops_fixture.py`: Offline reference oracle producing `fixtures/gemma_ops/golden.txt` (21 cases, 25 op evaluations, format `ojas-gemma-ops-golden-v1`).
   - `ojas-oracle/src/gemma_golden.rs`: Parser and evaluator for `golden.txt`.
   - `ojas-oracle/tests/gemma_golden_cpu.rs` & `ojas-metal/tests/gemma_golden_metal.rs`: CPU and Metal golden parity harnesses.
   - `ojas-oracle/tests/e2e_tier4_production.rs`: Scenario B un-stubbed with genuine backend ops.

---

### 1.2 Verbatim Independent Execution Results

#### 1. Repository Invariants & `#![forbid(unsafe_code)]` Compliance
Verified all crates in `ojas` have `#![forbid(unsafe_code)]` (excluding sanctioned low-level crates `ojas-simd` and C ABI `ojas-gusset-engine`).
Grepped for `unsafe` across all newly created files (`gelu_tanh.rs`, `gemma_ops.rs`, `gemma_host.rs`, `gemma_dev.rs`, `gemma_ops.metal`, `gemma_golden.rs`):
**Result**: 0 occurrences. Safe Rust strictly maintained.

#### 2. Compiler Hygiene
- **Formatting**:
  ```bash
  cd /Users/bharath/Code/research/ojas && ./scripts/fmt_check.sh
  ```
  Result: Exit 0 (all files properly formatted).
- **Tracked Targets**:
  ```bash
  cd /Users/bharath/Code/research/ojas && ./scripts/check_tracked_targets.sh
  ```
  Result: `check_tracked_targets: all 384 target sources are tracked` (Exit 0).
- **Clippy**:
  ```bash
  cargo clippy --workspace --all-targets -- -D warnings
  ```
  Result: Exit 0, 0 warnings.
- **Cargo Deny**:
  ```bash
  cargo deny check
  ```
  Result: `advisories ok, bans ok, licenses ok, sources ok` (Exit 0).

#### 3. CPU Golden Parity Verification
```bash
cargo test -p ojas-oracle --test gemma_golden_cpu -- --nocapture
```
Output:
```
cpu: Gemma golden tests: 25 ops evaluated across 21 cases
  gelu_tanh_fwd  standard           max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_bwd  standard           max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_fwd  tails              max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_bwd  tails              max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_fwd  near_zero          max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_bwd  near_zero          max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_fwd  batch_matrix       max_abs: 0.00e0  cosine: 1.0000000000
  gelu_tanh_bwd  batch_matrix       max_abs: 0.00e0  cosine: 1.0000000000
  seq_mean       ragged             max_abs: 0.00e0  cosine: 1.0000000000
  seq_mean       full               max_abs: 0.00e0  cosine: 1.0000000000
  seq_mean       zero_len           max_abs: 0.00e0  cosine: 1.0000000000
  seq_mean       cancellation       max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   embed_768          max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   matryoshka_512     max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   matryoshka_256     max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   matryoshka_128     max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   small_dim          max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   zero_row           max_abs: 0.00e0  cosine: 1.0000000000
  rms_qkv_norm   mha                max_abs: 3.58e-7  cosine: 1.0000000000
  rms_qkv_norm   gqa                max_abs: 2.38e-7  cosine: 1.0000000000
  rms_qkv_norm   vit_eps            max_abs: 4.77e-7  cosine: 1.0000000000
  swa            basic              max_abs: 4.77e-7  cosine: 1.0000000000
  swa            gqa                max_abs: 5.96e-7  cosine: 1.0000000000
  swa            wide_window        max_abs: 8.94e-7  cosine: 1.0000000000
  swa            w1_nearest         max_abs: 2.38e-7  cosine: 1.0000000000
test gemma_ops_match_torch_goldens_on_cpu ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; finished in 0.02s
```

#### 4. Metal GPU Golden Parity Verification (under `mac_heavy.sh` lock)
```bash
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_parity cargo test -p ojas-metal --test gemma_golden_metal -- --nocapture
```
Output:
```
metal: Gemma golden tests: 25 ops evaluated across 21 cases
  gelu_tanh_fwd  standard           max_abs: 1.19e-7  cosine: 1.0000000000
  gelu_tanh_bwd  standard           max_abs: 1.19e-7  cosine: 1.0000000000
  gelu_tanh_fwd  tails              max_abs: 2.29e-7  cosine: 1.0000000000
  gelu_tanh_bwd  tails              max_abs: 1.79e-6  cosine: 1.0000000000
  gelu_tanh_fwd  near_zero          max_abs: 4.66e-10  cosine: 1.0000000000
  gelu_tanh_bwd  near_zero          max_abs: 1.49e-8  cosine: 1.0000000000
  gelu_tanh_fwd  batch_matrix       max_abs: 2.38e-7  cosine: 1.0000000000
  gelu_tanh_bwd  batch_matrix       max_abs: 5.18e-7  cosine: 1.0000000000
  seq_mean       ragged             max_abs: 1.19e-7  cosine: 1.0000000000
  seq_mean       full               max_abs: 5.96e-8  cosine: 1.0000000000
  seq_mean       zero_len           max_abs: 5.96e-8  cosine: 1.0000000000
  seq_mean       cancellation       max_abs: 0.00e0  cosine: 1.0000000000
  l2_normalize   embed_768          max_abs: 1.49e-8  cosine: 1.0000000000
  l2_normalize   matryoshka_512     max_abs: 1.49e-8  cosine: 1.0000000000
  l2_normalize   matryoshka_256     max_abs: 1.49e-8  cosine: 1.0000000000
  l2_normalize   matryoshka_128     max_abs: 1.49e-8  cosine: 1.0000000000
  l2_normalize   small_dim          max_abs: 5.96e-8  cosine: 1.0000000000
  l2_normalize   zero_row           max_abs: 5.96e-8  cosine: 1.0000000000
  rms_qkv_norm   mha                max_abs: 4.77e-7  cosine: 1.0000000000
  rms_qkv_norm   gqa                max_abs: 4.77e-7  cosine: 1.0000000000
  rms_qkv_norm   vit_eps            max_abs: 2.38e-7  cosine: 1.0000000000
  swa            basic              max_abs: 2.38e-7  cosine: 1.0000000000
  swa            gqa                max_abs: 2.38e-7  cosine: 1.0000000000
  swa            wide_window        max_abs: 1.13e-6  cosine: 1.0000000000
  swa            w1_nearest         max_abs: 2.38e-7  cosine: 1.0000000000
test gemma_ops_match_torch_goldens_on_metal ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; finished in 0.70s
```

#### 5. Metal Gemma Ops Unit Tests (under `mac_heavy.sh` lock)
```bash
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_ops cargo test -p ojas-metal --test gemma_ops -- --nocapture
```
Output:
```
MAXERR gelu_tanh_forward: 2.3841858e-7
MAXERR l2_normalize_forward: 2.9802322e-8
MAXERR seq_mean_forward: 2.9802322e-8
MAXERR rms_qkv_norm_forward q: 3.5762787e-7
MAXERR rms_qkv_norm_forward k: 4.7683716e-7
MAXERR l2_normalize_forward zero row: 0e0
MAXERR gelu_tanh_backward: 5.9604645e-7
test l2_normalize_forward_matches_cpu ... ok
test gelu_tanh_forward_and_backward_match_cpu ... ok
test seq_mean_forward_matches_cpu ... ok
MAXERR rms_qkv_norm_forward v: 3.5762787e-7
MAXERR rms_qkv_norm_forward weightless q: 4.7683716e-7
MAXERR rms_qkv_norm_forward weightless k: 5.9604645e-7
MAXERR rms_qkv_norm_forward weightless v: 3.5762787e-7
test rms_qkv_norm_forward_matches_cpu ... ok
test result: ok. 4 passed; 0 failed; 0 ignored; finished in 0.71s
```

#### 6. Metal General Attention Unit Tests (under `mac_heavy.sh` lock)
```bash
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh attn_metal cargo test -p ojas-metal --test attention_general -- --nocapture
```
Output:
```
test malformed_calls_are_refused_as_the_cpu_refuses_them ... ok
test head_dims_follow_the_core_limit ... ok
test offset_views_are_read_in_place ... ok
test a_non_finite_operand_is_reported_at_the_next_sync ... ok
test a_sharp_softmax_rescales_instead_of_overflowing ... ok
test causal_window_and_prefill_alignment_match_cpu ... ok
test key_lengths_match_cpu_including_empty_and_oversized ... ok
test every_bias_layout_matches_cpu ... ok
test bidirectional_and_cross_attention_match_cpu ... ok
test result: ok. 9 passed; 0 failed; 0 ignored; finished in 0.79s
```

#### 7. Full 4-Tier E2E Test Suite Execution
- **Tier 1 (Feature Isolation, 65 tests)**:
  `cargo test -p ojas-oracle --test e2e_tier1_features`: **65 passed, 0 failed** in 0.03s.
- **Tier 2 (Boundary & Corner, 65 tests)**:
  `cargo test -p ojas-oracle --test e2e_tier2_boundaries`: **65 passed, 0 failed** in 0.00s.
- **Tier 3 (Pairwise Combinatorial, 8 tests)**:
  `cargo test -p ojas-oracle --test e2e_tier3_combinatorial`: **8 passed, 0 failed** in 0.00s.
- **Tier 4 (Real-World Production Pipelines, 4 tests)**:
  `cargo test -p ojas-oracle --test e2e_tier4_production`: **4 passed, 0 failed** in 0.07s.
  Specifically, `tier4_scenario_b_gemma_embedding_production_inference_pipeline` verified unit norm condition $\|y\|_2 = 1.0 \pm 10^{-6}$ and cosine similarity preservation $1.0 \pm 10^{-6}$ across Matryoshka dimensions (64, 32, 16) with all mock loops replaced with real operations.

#### 8. Adversarial Stress Test Execution
- **Autograd Adversarial Suite**:
  `cargo test -p ojas-autograd --test adversarial_challenge_m4`: **6 passed, 0 failed** in 0.12s.
  Stressed extreme ranges $x \in [-1000, 1000]$, $x \le -20$ zero saturation, $x \ge 20$ linear saturation, HDR cancellation resistance in `seq_mean`, and bit-identity across thread counts.
- **Metal Adversarial Suite (under `mac_heavy.sh` lock)**:
  `cargo test -p ojas-metal --test adversarial_challenger_m4 -- --test-threads=1 --nocapture`: **7 passed, 0 failed** in 0.76s.
  Stressed dual routing across head dimensions $D \in \{32, 64, 256, 512\}$, fail-closed epsilon validation on Metal, zero-row normalization without NaN, multi-batch ragged segment generation without readback, and multi-threaded concurrency fences.

#### 9. Full Local CI Gate Execution
```bash
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ci_local ./scripts/ci_local.sh
```
Output:
```
==> fmt
==> cargo target sources are tracked
check_tracked_targets: all 384 target sources are tracked
==> clippy
==> cargo-deny (deny.toml)
advisories ok, bans ok, licenses ok, sources ok
==> cuda check
==> nightly simd (not a Cargo target)
portable_simd f32x8 add matches the scalar lane
==> go (stable)
==> site (reference pages and docs mirror must match the markdown)
sitegen: generated files are up to date (54 files)
==> bench charts (site/assets/plots/bench and docs/bench-plots.md must match bench/results)
ok  	command-line-arguments	0.163s
plots: 83 charts and docs/bench-plots.md up to date
==> test (release, serialized)
...
==> rustdoc (intra-doc links resolve; no link to a private item)
==> ojas-metal without the metal feature (stub build, every target)
ci:local OK
mac_heavy: 2026-11-01T02:48:47Z 'ci_local' finished with exit 0
```

---

## 2. Logic Chain

1. **Integrity Violations Audit**:
   - Actively inspected source code in `ojas-core/src/gelu_tanh.rs`, `ojas-cpu/src/gemma_ops.rs`, `ojas-metal/src/backend/gemma_host.rs`, `ojas-metal/src/device/gemma_dev.rs`, `ojas-metal/kernels/gemma_ops.metal`, and `ojas-oracle/src/gemma_golden.rs`.
   - Found **zero hardcoded outputs**, **zero facade or dummy implementations**, and **zero shortcuts**.
   - Arithmetic in `gelu_tanh_exact` and `gelu_tanh_backward_exact` implements the exact mathematical closed forms with tail stabilization.
   - `seq_mean_forward` executes genuine two-pass `f64` accumulation over active tokens.
   - `l2_normalize_forward` executes genuine Euclidean norm division with $\epsilon$ clamping.
   - `rms_qkv_norm_forward` normalizes $V$ per-head with unit weights.
   - `e2e_tier4_production.rs` Scenario B exercises real backend calls end-to-end.
   - Verification logs and test outputs were produced by independent live execution.

2. **Numerical Stability & Zero Cancellation in Tails**:
   - For `gelu_tanh`, the sigmoid-based reformulations $x \cdot \text{sigmoid}(2u)$ and $\frac{e^{-|u|}}{(1 + e^{-|u|})^2}$ completely avoid catastrophic cancellation in the deep negative tail ($x \le -8$) where standard floating point $1 + \tanh(u)$ vanishes to 0.
   - On CPU, single `f32` rounding of intermediate `f64` values ensures bit-identical results across thread counts (1, 2, 5) and numerics tiers (`Exact` vs `Fast`).
   - On Metal GPU, clamp boundaries $|x| \le 20$ and $|u| \le 10$ ensure no float overflow occurs in the cubic term while matching torch behavior to $< 1.8 \times 10^{-6}$ max abs error.

3. **Query Padding Row Masking & Non-Causal SWA**:
   - Query padding row masking enforces that when $T_q == T_k$, queries $i \ge \text{len}_b$ are treated as padding, outputting zeros and $0.0$ for `lse`.
   - In CPU `Geom::keys`, this returns $0..0$. In Metal `ojas_backend.metal`, flag bit 4 (`16u`) sets $hi = 0$.
   - Both backends guarantee that padded queries produce zeros and do not contaminate key gradients or attention scores.

4. **Fail-Closed Placement & Parameter Validation**:
   - Boundary checks in `ojas-core/src/shapes.rs` (`rms_qkv_norm_forward_dims`, `seq_mean_forward_dims`, `l2_normalize_forward_dims`) validate operand ranks, dtypes, and finite parameter ranges prior to any memory allocation or kernel launch.
   - `l2_normalize_forward` rejects non-finite $\epsilon$ with `OjasError::NonFinite` and $\epsilon \le 0$ with `OjasError::OutOfRange`.
   - Metal GPU rejects $\epsilon \ne 10^{-12}$ with `OjasError::Unsupported`, preventing silent misuse of tessl's fixed-epsilon kernel.
   - Metal backend enforces explicit placement: calling CPU methods on GPU buffers or vice-versa fails closed with `OjasError::Placement` with zero silent host fallbacks.

5. **PyTorch Parity & Zero Python at Test Time**:
   - PyTorch was utilized strictly offline to generate `fixtures/gemma_ops/golden.txt`.
   - In-repo Rust harnesses in `ojas-oracle` parse the committed fixture and evaluate CPU and Metal backends without any Python runtime dependency.
   - Measured max absolute error $\le 8.94 \times 10^{-7}$ on CPU and $\le 1.79 \times 10^{-6}$ on Metal, with cosine similarity $\ge 1.0 - 10^{-9}$ across all 25 op evaluations, exceeding the acceptance criteria.

---

## 3. Caveats

1. **Full Model Checkpoint Weights**: The op-level golden fixtures and E2E production pipelines validate all architectural primitives and numerical invariants of HuggingFace EmbeddingGemma 2. End-to-end inference on the complete 271M parameter model checkpoint requires downloading the full weights from HuggingFace, which is out of scope for self-contained CI testing.
2. **Metal GPU Exclusive Lock**: Execution of Metal GPU parity and integration tests requires serial execution under the `mac_heavy.sh` lock to prevent queue contention and timeout on shared Apple Silicon hardware.

---

## 4. Conclusion

1. **Milestone 4 (Features 21–26: Gemma Embedding Kernels with PyTorch Parity) is APPROVED**.
2. All required deliverables are completely implemented, mathematically robust, and rigorously tested on both CPU and Metal GPU backends.
3. Integrity audit confirmed zero violations: no hardcoded outputs, no facades, no task bypasses, and no self-certification.
4. `#![forbid(unsafe_code)]` is strictly preserved with zero unsafe blocks added.
5. All compiler hygiene checks (`fmt_check.sh`, `check_tracked_targets.sh`, `cargo clippy -D warnings`, and `cargo deny check`) pass cleanly with 0 warnings.
6. Target unit tests, golden parity suites, 4-tier E2E suites (142 tests), and adversarial stress suites pass with 100% success rate.
7. Local workspace CI (`./scripts/ci_local.sh`) passes with `ci:local OK`.

**Verdict**: **`APPROVE`**

---

## 5. Verification Method

To independently reproduce the complete verification:

1. **Check Hygiene & Linters**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/fmt_check.sh
   ./scripts/check_tracked_targets.sh
   cargo clippy --workspace --all-targets -- -D warnings
   cargo deny check
   ```

2. **Verify CPU & Metal Gemma Golden Parity**:
   ```bash
   cargo test -p ojas-oracle --test gemma_golden_cpu -- --nocapture
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh gemma_metal_parity \
       cargo test -p ojas-metal --test gemma_golden_metal -- --nocapture
   ```

3. **Verify E2E Production Workloads**:
   ```bash
   cargo test -p ojas-oracle --test e2e_tier4_production -- --nocapture
   ```

4. **Verify Adversarial Stress Suites**:
   ```bash
   cargo test -p ojas-autograd --test adversarial_challenge_m4
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh adv_metal_all \
       cargo test -p ojas-metal --test adversarial_challenger_m4 -- --test-threads=1 --nocapture
   ```

5. **Run Full Workspace CI Script**:
   ```bash
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ci_local ./scripts/ci_local.sh
   ```
   Expected termination line: `ci:local OK`.
