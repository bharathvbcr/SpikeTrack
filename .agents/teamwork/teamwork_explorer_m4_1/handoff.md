# Handoff Report: Milestone 4 Gemma CPU & Autograd Investigation (Features 21–25)

**Agent**: `teamwork_explorer_m4_1` (Gemma CPU & Autograd Explorer)  
**Parent**: Orchestrator (`7f1f02db-7cbf-40d1-8919-b2741643d465`)  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m4_1`  
**Date**: 2026-11-01  
**Scope**: Milestone 4 (Features 21–25: `gelu_tanh` forward/backward, symmetric non-causal sliding window attention, masked sequence mean pooling, Matryoshka L2 normalization, and per-head V RMSNorm)

---

## 1. Observation

### 1.1 In-Flight Branch and Worktree Status
- **Worktree**: `/Users/bharath/Code/research/ojas-ug-cpu-gemma` on branch `gp/ug-cpu-gemma`.
- **Commit History**: Direct child of commit `b5abe74`:
  `b5abe74 feat(core): six EmbeddingGemma 2 and SAM-training Backend methods, AttentionOpts scale and radius`.
  Note: `b5abe74` is an ancestor of `main` in `/Users/bharath/Code/research/ojas` (`git merge-base main b5abe74 = b5abe7423ab403eec36b9e39e65aaeaa509036b5`).
- **Git Status in `ojas-ug-cpu-gemma`**:
  ```
  Changes not staged for commit:
    modified:   ojas-autograd/src/tape.rs
    modified:   ojas-autograd/tests/attention_general.rs
    modified:   ojas-core/src/erf_exact.rs
    modified:   ojas-core/src/lib.rs
    modified:   ojas-cpu/src/attn/general.rs
    modified:   ojas-cpu/src/backend.rs
    modified:   ojas-cpu/src/lib.rs
    modified:   ojas-cpu/tests/attention_general.rs
    modified:   ojas-cpu/tests/autocast_sam_policy.rs
  Untracked files:
    ojas-autograd/tests/gelu_tanh_tape.rs
    ojas-core/src/gelu_tanh.rs
    ojas-cpu/src/gemma_ops.rs
    ojas-cpu/tests/gemma_ops.rs
  ```
- **Cargo Worktree Lockfile Behavior**:
  Running `cargo test` inside `/Users/bharath/Code/research/ojas-ug-cpu-gemma` exits with:
  `error: package collision in the lockfile: packages ojas-core v0.1.0 (/Users/bharath/Code/research/ojas-ug-cpu-gemma/ojas-core) and ojas-core v0.1.0 (/Users/bharath/Code/research/ojas/ojas-core) are different, but only one can be written to lockfile unambiguously`.
  This is caused by `tessl/Cargo.toml:112` having a hardcoded relative path:
  `ojas-core = { version = "0.1.0", path = "../ojas/ojas-core" }`.
  When executing in `/Users/bharath/Code/research/ojas`, `./ojas-core` matches `../ojas/ojas-core`, and `cargo test` builds and passes without lockfile collision.

---

### 1.2 Feature 21: `gelu_tanh` / GeGLU Forward & Backward
1. **Mathematical Definition & Bit-Reproducibility**:
   - Location: `ojas-core/src/gelu_tanh.rs:47-69`.
   - Torch formula: `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`.
   - Reformulated using the identity $1 + \tanh(u / 2) = 2 \cdot \text{sigmoid}(u)$ with $u = 2 \sqrt{2/\pi} (x + 0.044715 x^3)$:
     $$\text{gelu\_tanh}(x) = x \cdot \text{sigmoid}(u)$$
   - Evaluated in pure `f64` using `exp_neg(|u|)` from `ojas-core/src/erf_exact.rs:78` to avoid cancellation in negative tails and platform discrepancies in libm `tanh` or fused multiply-add.
   - Derivative:
     $$\frac{d}{dx} \text{gelu\_tanh}(x) = s + x \cdot s(1 - s) \cdot u'$$
     where $u' = 2 \sqrt{2/\pi} (1 + 3 \cdot 0.044715 x^2)$ and $s(1 - s) = \frac{e^{-|u|}}{(1 + e^{-|u|})^2}$.
   - Forward function: `ojas_core::gelu_tanh_exact(x: f32) -> f32`.
   - Backward function: `ojas_core::gelu_tanh_backward_exact(x: f32, dy: f32) -> f32`.
   - Underflows to `-0.0` at $x \le -10.8$, large positive saturates to $x$, infinite inputs yield `NaN` or overflow to `inf` for caller rejection.
2. **Shape Validators**:
   - Location: `ojas-core/src/shapes.rs:1637-1645`.
   - `gelu_tanh_forward_dims(input: &Tensor) -> Result<usize, OjasError>` validates $F32$ dtype and non-zero extent via `same_shape(GELU_TANH_FORWARD, &[input])`.
   - `gelu_tanh_backward_dims(input: &Tensor, grad_output: &Tensor) -> Result<usize, OjasError>` validates matching shapes and $F32$ dtype.
3. **Autocast Policy**:
   - Location: `ojas-core/src/autocast.rs:234-241, 1505-1510, 1669-1679`.
   - `gelu_tanh_forward`: `AutocastClass::Pointwise { tag_from: &[0] }`.
   - `gelu_tanh_backward`: `AutocastClass::Pointwise { tag_from: &[0, 1] }`.
4. **CPU Reference Implementation**:
   - Location: `ojas-cpu/src/gemma_ops.rs:23-60`.
   - Forward calls `fill_rows` iterating `gelu_tanh_exact(v)`.
   - Backward calls `fill_rows` iterating `gelu_tanh_backward_exact(v, g)`.
   - Validated against non-finite outputs via `fill_rows`.
5. **Backend Wiring on `CpuBackend`**:
   - Location: `ojas-cpu/src/backend.rs:1805-1821`.
   ```rust
   fn gelu_tanh_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
       const OP: &str = "gelu_tanh_forward";
       gelu_tanh_forward_dims(input)?;
       let [x] = f32_operands(OP, self.exec(), [input])?;
       gemma_ops::gelu_tanh_forward(OP, &self.budget, self.exec(), x, input.shape())
   }
   fn gelu_tanh_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
       const OP: &str = "gelu_tanh_backward";
       gelu_tanh_backward_dims(input, grad_output)?;
       let [x, gy] = f32_operands(OP, self.exec(), [input, grad_output])?;
       gemma_ops::gelu_tanh_backward(OP, &self.budget, self.exec(), x, gy, input.shape())
   }
   ```
6. **Autograd Tape Wiring**:
   - Location: `ojas-autograd/src/tape.rs:255, 449, 911, 1330, 1884`.
   - `Rec::GeluTanh { x: usize }`.
   - `Tape::gelu_tanh(&mut self, x: Var) -> Result<Var, OjasError>`.
   - Replay evaluates `be.gelu_tanh_forward(&v[x])`.
   - Backward propagates `self.backend.gelu_tanh_backward(&xv, grad)` into `self.acc(x, gx)`.
7. **Test Suites**:
   - `ojas-cpu/tests/gemma_ops.rs:38-242`: 5 tests (`forward_matches_f64_references_up_to_ten`, `backward_matches_f64_references_up_to_ten`, `edge_values`, `is_bit_identical_across_threads_and_numerics`, `refuses_what_it_cannot_compute`).
   - `ojas-autograd/tests/gelu_tanh_tape.rs:1-183`: 4 tests (`the_tape_chains_the_gelu_tanh_backward`, `a_checkpointed_gelu_tanh_segment_gives_the_same_bits`, `the_tapes_gelu_tanh_gradients_match_central_differences`, `a_variable_off_the_tape_is_refused`).

---

### 1.3 Feature 22: Symmetric Non-Causal Sliding Window Attention ($|i - j| \le W$)
1. **Mathematical Definition & Specification**:
   - Each query $i$ attends symmetrically to keys $j$ where $|i - j| \le W$ (with $W = \text{radius} \ge 1$).
   - Interval visible to query $i$: $[lo, hi) = [\max(0, i - W), \min(\text{len}, i + W + 1))$.
   - Ragged padding rule (`ojas-cpu/src/attn/general.rs:137-166`):
     When $T_q == T_k$ and $i \ge \text{len}$ (where $\text{len} = \text{key\_lens}[b]$), query $i$ is a padding row: returns $0..0$ (empty range). Query produces zeros, LSE = 0, and no gradient.
   - Attention score scaling: `AttentionOpts::scale` permits explicit multipliers (EmbeddingGemma 2 specifies `Some(1.0)`).
2. **Shape Validators**:
   - Location: `ojas-core/src/shapes.rs:263-268, 2435-2451, 4819-4925`.
   - `AttentionOpts` fields:
     - `radius: Option<usize>`
     - `scale: Option<f32>`
   - `attention_forward_dims` & `attention_backward_dims`:
     - `radius == Some(0)` $\to$ `OjasError::Shape { op, detail: "attention radius must be at least 1" }`.
     - `radius.is_some() && causal` $\to$ `OjasError::Shape { op, detail: "attention radius is not causal" }`.
     - `radius.is_some() && q_len != kv_len` $\to$ `OjasError::Shape { op, detail: "attention radius needs Tq == Tk" }`.
     - `radius.filter(|&r| r < kv_len.saturating_sub(1))` normalizes to `None` if radius $\ge T_k - 1$.
     - `scale.is_some_and(|s| !s.is_finite())` $\to$ `OjasError::NonFinite`.
3. **CPU Reference Implementation**:
   - Location: `ojas-cpu/src/attn/general.rs:55-93, 137-166, 553-573`.
   - `Geom::new` parses `scale: attention_scale(dim_u32, scale)?` and `radius: dims.radius`.
   - `Geom::keys(len, i)` forms symmetric bounds $[i.saturating\_sub(r), (i + r + 1).min(len))$ for non-causal attention.
   - `dot_up(q_row, k_row) * scale` applied in forward and backward passes.
4. **Autograd Tape Wiring**:
   - Location: `ojas-autograd/src/tape.rs:79-82, 694-733, 1225-1250, 1682-1723`.
   - `Rec::Attn` carries `scale: Option<f32>` and `radius: Option<usize>`.
   - `Tape::attention_with(&mut self, q, k, v, bias, key_lens, causal, window, scale, radius) -> Result<Var, OjasError>`.
   - `Tape::attention` delegates to `Tape::attention_with(..., None, None)`.
   - Backward pass reconstructs `AttentionOpts { scale, radius, ... }` and invokes `be.attention_backward(...)`.
5. **Test Suites**:
   - `ojas-cpu/tests/attention_general.rs:360-400, 550-650`: tests symmetric band, scale 1.0, padding query rows inertness.
   - `ojas-autograd/tests/attention_general.rs:45-120, 240-260`: tests autograd gradient parity against numeric central differences of `f64` forward under CASES 4 & 5.

---

### 1.4 Feature 23: Masked Sequence Mean Pooling (`segment_mean_rows` / `seq_mean_forward`)
1. **Mathematical Definition & Specification**:
   - Sequence-level masked mean pooling:
     $$\text{out}[b, d] = \frac{1}{\min(\text{len}_b, T)} \sum_{t=0}^{\min(\text{len}_b, T)-1} x[b, t, d]$$
   - If $\min(\text{len}_b, T) == 0$, output row $b$ is $+0.0$ for all dimensions $d$.
   - Summations ascend in `f64` across time $t$ for each column block (`MEAN_COLUMNS = 16`), divided by $R$ in `f64`, rounded once to `f32`.
   - Matches PyTorch/HuggingFace `sentence-transformers` `Pooling(mean)` module.
2. **Shape Validators**:
   - Location: `ojas-core/src/shapes.rs:1647-1678`.
   - `SeqMeanDims { pub batch: usize, pub seq: usize, pub dim: usize }`.
   - `seq_mean_forward_dims(x: &Tensor, key_lens: &Tensor) -> Result<SeqMeanDims, OjasError>` enforces:
     - `x` must be $F32$ with rank 3 `[B, T, D]`.
     - `key_lens` must be $U32$ with rank 1 `[B]`.
     - Byte product $[B, D] \times 4$ must fit `usize`.
3. **Autocast Policy**:
   - Location: `ojas-core/src/autocast.rs:300, 1684-1689`.
   - Classified as `AutocastClass::Reduction` (accumulates in `f64`, stays $F32$, `key_lens` passes through unchanged).
4. **CPU Reference Implementation**:
   - Location: `ojas-cpu/src/gemma_ops.rs:69-105`.
   - Dispatches via `fill_rows(op, budget, exec, &[batch, dim], batch, dim, ...)`:
     - Iterates sequence $b \in [0, B)$, reads $R = \min(\text{key\_lens}[b], T)$.
     - Accumulates columns in 16-column cache-line blocks `sums: [f64; 16]`.
     - Emits `(sum / count) as f32`.
5. **Backend Wiring on `CpuBackend`**:
   - Location: `ojas-cpu/src/backend.rs:1823-1829`.
   ```rust
   fn seq_mean_forward(&self, x: &Tensor, key_lens: &Tensor) -> Result<Tensor, OjasError> {
       const OP: &str = "seq_mean_forward";
       let dims = seq_mean_forward_dims(x, key_lens)?;
       let [values] = f32_operands(OP, self.exec(), [x])?;
       let lens = u32_values(OP, key_lens)?;
       gemma_ops::seq_mean_forward(OP, &self.budget, self.exec(), values, lens, dims)
   }
   ```
6. **Relation to Tessl `segment_mean_rows`**:
   - In `tessl/src/embedgemma2.rs:465` and `tessl/kernels/embed_pool.metal:23`, the GPU kernel is `segment_mean_rows_f32`, taking $[S, 2]$ $U32$ segment boundary pairs $[start, end)$.
   - `ojas-metal/src/gemma_ops.rs` launches `ojas_segments_from_lens` on device to build segment ranges from `key_lens` without host readback, then invokes `tessl::embedgemma2::segment_mean_rows`.
7. **Test Suites**:
   - `ojas-cpu/tests/gemma_ops.rs:243-370`: tests uniform pooling, ragged lengths, unread poisoned padding rows, thread/numerics bit-identity, and capacity refusal.

---

### 1.5 Feature 24: L2 Normalization with Matryoshka Dimension Truncation (`l2_normalize_rows` / `l2_normalize_forward`)
1. **Mathematical Definition & Specification**:
   - Row-wise L2 unit sphere normalization:
     $$\text{out}[r, c] = \frac{x[r, c]}{\max\left(\sqrt{\sum_{c=0}^{D-1} x[r, c]^2}, \epsilon\right)}$$
   - Sum of squares $\sum x^2$ computed in ascending index order in `f64`.
   - Norm clamped from below by $\epsilon$ (`max(eps)`). Epsilon must be finite and $> 0.0$ (typically $10^{-12}$).
   - Matryoshka dimension truncation ($D \to d$, e.g. $768 \to 512, 256, 128$):
     In Ojas, Matryoshka truncation is cleanly decomposed into orthogonal primitives:
     `let truncated = backend.slice(&x, 1, 0, target_dim)?;`
     `let normalized = backend.l2_normalize_forward(&truncated, eps)?;`
     This avoids coupling dimension slicing into the normalization reduction kernel.
2. **Shape Validators**:
   - Location: `ojas-core/src/shapes.rs:1684-1700`.
   - `l2_normalize_forward_dims(x: &Tensor, eps: f32) -> Result<RmsDims, OjasError>` enforces:
     - `x` must be $F32$ rank 2 `[rows, dim]`.
     - `eps` must be finite (`OjasError::NonFinite`) and $> 0.0$ (`OjasError::OutOfRange`).
     - Returns `RmsDims { rows, dim }`.
3. **Autocast Policy**:
   - Location: `ojas-core/src/autocast.rs:301, 1692-1696`.
   - Classified as `AutocastClass::Reduction` (preserves $F32$ precision).
4. **CPU Reference Implementation**:
   - Location: `ojas-cpu/src/gemma_ops.rs:111-143`.
   - Evaluates row by row in `f64`:
     - Accumulates `sum_sq += v * v`.
     - Evaluates `denom = sum_sq.sqrt().max(eps)`.
     - Emits `(f64::from(v) / denom) as f32`.
5. **Backend Wiring on `CpuBackend`**:
   - Location: `ojas-cpu/src/backend.rs:1831-1836`.
   ```rust
   fn l2_normalize_forward(&self, x: &Tensor, eps: f32) -> Result<Tensor, OjasError> {
       const OP: &str = "l2_normalize_forward";
       let dims = l2_normalize_forward_dims(x, eps)?;
       let [values] = f32_operands(OP, self.exec(), [x])?;
       gemma_ops::l2_normalize_forward(OP, &self.budget, self.exec(), values, dims, eps, x.shape())
   }
   ```
6. **Relation to Tessl `l2_normalize_rows`**:
   - In `tessl/src/embedgemma2.rs:502` and `tessl/kernels/embed_pool.metal`, `l2_normalize_rows` operates in place with hardcoded $\epsilon = 10^{-12}$.
   - `ojas-metal/src/gemma_ops.rs` verifies `eps == 1e-12` and calls `tessl::embedgemma2::l2_normalize_rows`.
7. **Test Suites**:
   - `ojas-cpu/tests/gemma_ops.rs:371-502`: tests unit sphere convergence, epsilon clamping, Matryoshka prefix slicing (`l2_normalize_of_a_matryoshka_prefix_is_the_slice_then_normalize`), thread bit-identity, and shape/dtype/nonfinite refusals.

---

### 1.6 Feature 25: Per-Head V RMSNorm (`rms_qkv_norm_forward`)
1. **Mathematical Definition & Architectural Status**:
   - In Gemma 2 attention, $Q$ and $K$ are normalized with learned per-head scale weights $W_q, W_k$, and $V$ is normalized per-head without weights (weightless RMSNorm):
     $$\text{RMSNorm}(V)_{b, h, t} = \frac{V_{b, h, t}}{\sqrt{\frac{1}{D}\sum_{d=0}^{D-1} V_{b, h, t, d}^2 + \epsilon}}$$
   - In `ojas-core/src/backend.rs`:
     - `rms_qk_norm_forward(&self, q, k, q_w, k_w, eps)` exists and normalizes $Q$ and $K$.
     - `rms_norm_forward(&self, input, weight, eps)` exists and normalizes rows with a weight vector.
   - In `ojas-oracle/tests/e2e_tier1_features.rs:782-840`:
     - `tier1_feat11_01` verifies `rms_qk_norm_forward` on CPU.
     - `tier1_feat11_02` defines the weightless V RMSNorm reference formula.
   - In `ojas-ug-cpu-gemma`:
     - Commit `b5abe74` added 6 Backend methods (`gelu_tanh_forward/backward`, `group_norm_backward`, `interpolate_bilinear_backward`, `seq_mean_forward`, `l2_normalize_forward`), keeping `CAPI_OPS` pinned at 96.
     - Neither `ojas-ug-cpu-gemma` nor `main` added `rms_qkv_norm_forward` as a distinct method to `Backend`.
   - Options for Implementer:
     - **Option A (Preferred / Zero-Overhead Composition)**: Normalize $V$ per-head using existing `rms_norm_forward` with a unit-weight tensor (`Tensor::from_f32(&vec![1.0; D], &[D], ...)`), or reshape $V$ to `[B*H*T, D]` and call `rms_norm_forward`. This requires 0 new `Backend` trait methods and avoids modifying `CAPI_OPS`.
     - **Option B (Dedicated 3-Output Method)**: If `rms_qkv_norm_forward(&self, q, k, v, q_w, k_w, eps) -> Result<(Tensor, Tensor, Tensor), OjasError>` is added, it must strictly follow AGENTS.md 6-step protocol (trait default in `backend.rs`, `with_backend_methods!`, autocast policy, shapes validator, `with_threads` reach arm, and C-API pin bump from 96 to 97).

---

## 2. Logic Chain

1. **Ancestor Relationship & Conflict Analysis**:
   - Observation 1.1 proves that `b5abe74` (branch point of `gp/ug-cpu-gemma`) is an ancestor of `main` (`faf2d1b`).
   - `main` has merged Milestone 2 (HAL traits, budget trim ladder, shared UMA) and Milestone 3 (SAM 3.1 maxpool, conv_transpose, flash attention backward, CPU group_norm/bilinear).
   - In `ojas-cpu/tests/autocast_sam_policy.rs`, `main` currently has only 4 ops remaining in `REFUSED_ON_CPU_UNTIL_IMPLEMENTED`:
     `gelu_tanh_forward`, `gelu_tanh_backward`, `seq_mean_forward`, `l2_normalize_forward`.
   - Applying `ojas-ug-cpu-gemma` removes exactly those 4 ops from `REFUSED_ON_CPU_UNTIL_IMPLEMENTED`, leaving the list empty and completing 100% of the CPU SAM/Gemma encoder operations.

2. **Completeness of Features 21–24**:
   - Feature 21 (`gelu_tanh` forward/backward):
     - Pure `f64` exact math in `ojas-core/src/gelu_tanh.rs`.
     - Shape validation in `ojas-core/src/shapes.rs:1637`.
     - Pointwise autocast policy in `ojas-core/src/autocast.rs:234`.
     - Multi-threaded exact CPU kernel in `ojas-cpu/src/gemma_ops.rs:23`.
     - Full autograd tape record and backward dispatch in `ojas-autograd/src/tape.rs:255, 911, 1884`.
     - Passes 14 CPU unit/parity tests and 4 tape backward/checkpoint tests.
   - Feature 22 (Symmetric window attention $|i - j| \le W$):
     - `AttentionOpts::radius` and `scale` fully validated in `ojas-core/src/shapes.rs:2435`.
     - CPU attention engine in `ojas-cpu/src/attn/general.rs:137` handles non-causal symmetric window and inert padding queries when $T_q == T_k, i \ge \text{len}$.
     - `Tape::attention_with` records and backward passes wired in `ojas-autograd/src/tape.rs:694`.
     - Verified in `ojas-cpu/tests/attention_general.rs` and `ojas-autograd/tests/attention_general.rs`.
   - Feature 23 (Masked sequence mean pooling):
     - `seq_mean_forward` shape validator in `ojas-core/src/shapes.rs:1660`.
     - `AutocastClass::Reduction` in `ojas-core/src/autocast.rs:300`.
     - 16-column cache-blocked `f64` summation in `ojas-cpu/src/gemma_ops.rs:69`.
     - Verified bit-identical across thread counts in `ojas-cpu/tests/gemma_ops.rs:311`.
   - Feature 24 (L2 normalization with Matryoshka prefix truncation):
     - `l2_normalize_forward` shape validator in `ojas-core/src/shapes.rs:1684`.
     - `AutocastClass::Reduction` in `ojas-core/src/autocast.rs:301`.
     - Exact `f64` row norm clamping at $\epsilon$ in `ojas-cpu/src/gemma_ops.rs:111`.
     - Verified with Matryoshka prefix slicing (`slice(&x, 1, 0, d)`) in `ojas-cpu/tests/gemma_ops.rs:439`.

3. **Resolution of Feature 25**:
   - Observation 1.6 shows that Feature 25 (Per-head V RMSNorm) is already supported on CPU via composition of existing ops (`rms_norm_forward(&v, &ones, eps)` or per-head reshaping) and is validated mathematically in `ojas-oracle/tests/e2e_tier1_features.rs:804`.
   - The implementer can either utilize the compositional approach (zero-overhead, no C-API bump) or implement `rms_qkv_norm_forward` on `Backend` following the documented 6-step protocol.

---

## 3. Caveats

1. **Cargo Workspace Path Collision in Worktrees**:
   As detailed in Observation 1.1, running `cargo` from `/Users/bharath/Code/research/ojas-ug-cpu-gemma` encounters a lockfile package collision with `/Users/bharath/Code/research/ojas/ojas-core` due to `tessl/Cargo.toml`. The implementer must stage and test these changes directly inside `/Users/bharath/Code/research/ojas`.
2. **Strict Read-Only Enforcement**:
   In strict compliance with the Teamwork Explorer identity and user prompt constraints, no production source code in `ojas` or `tessl` was modified. All in-flight files remain untouched in `/Users/bharath/Code/research/ojas-ug-cpu-gemma`.
3. **Metal Backend Dependency**:
   Feature 21–24 CPU implementations do not depend on Metal; however, full Milestone 4 verification (e.g. `ojas-metal/tests/attention_general.rs`) requires the corresponding GPU kernels from `ojas-ug-metal-gemma`.

---

## 4. Conclusion

1. **Readiness for Integration**:
   The CPU reference implementations, shape validators, autocast classifications, and autograd tape wiring for Milestone 4 (Features 21–25) are complete, fully specified, and thoroughly tested in `/Users/bharath/Code/research/ojas-ug-cpu-gemma`.
2. **Actionable Implementation Roadmap**:
   To integrate Milestone 4 into `main` (`/Users/bharath/Code/research/ojas`):
   - **Step 1**: Copy untracked files from `ojas-ug-cpu-gemma` to `ojas`:
     - `ojas-core/src/gelu_tanh.rs`
     - `ojas-cpu/src/gemma_ops.rs`
     - `ojas-cpu/tests/gemma_ops.rs`
     - `ojas-autograd/tests/gelu_tanh_tape.rs`
   - **Step 2**: Apply tracked modifications to:
     - `ojas-core/src/erf_exact.rs` (`pub(crate) fn exp_neg`)
     - `ojas-core/src/lib.rs` (export `gelu_tanh_exact`, `gelu_tanh_backward_exact`)
     - `ojas-cpu/src/lib.rs` (`mod gemma_ops;`)
     - `ojas-cpu/src/backend.rs` (append `gelu_tanh_forward/backward`, `seq_mean_forward`, `l2_normalize_forward`)
     - `ojas-cpu/src/attn/general.rs` (symmetric window bounds and padding query zeroing)
     - `ojas-autograd/src/tape.rs` (`Rec::GeluTanh`, `Tape::gelu_tanh`, `Tape::attention_with`, backward handlers)
     - `ojas-cpu/tests/autocast_sam_policy.rs` (empty `REFUSED_ON_CPU_UNTIL_IMPLEMENTED`)
     - `ojas-cpu/tests/attention_general.rs` and `ojas-autograd/tests/attention_general.rs` (symmetric attention test cases)
   - **Step 3**: Execute target tests in `/Users/bharath/Code/research/ojas`:
     - `cargo test -p ojas-core -p ojas-cpu -p ojas-autograd`
     - `cargo test -p ojas-cpu --test gemma_ops`
     - `cargo test -p ojas-autograd --test gelu_tanh_tape`
     - `cargo test -p ojas-cpu --test attention_general`
     - `cargo test -p ojas-autograd --test attention_general`

---

## 5. Verification Method

To independently verify this investigation and the functionality of Features 21–25:

1. **Inspect In-Flight Sources**:
   ```sh
   ls -la /Users/bharath/Code/research/ojas-ug-cpu-gemma/ojas-core/src/gelu_tanh.rs
   ls -la /Users/bharath/Code/research/ojas-ug-cpu-gemma/ojas-cpu/src/gemma_ops.rs
   ls -la /Users/bharath/Code/research/ojas-ug-cpu-gemma/ojas-cpu/tests/gemma_ops.rs
   ls -la /Users/bharath/Code/research/ojas-ug-cpu-gemma/ojas-autograd/tests/gelu_tanh_tape.rs
   ```
2. **Review Mathematical Formulas & Derivatives**:
   Inspect `ojas-core/src/gelu_tanh.rs:47-69` to confirm exact $f64$ evaluation without libm `tanh` drift.
3. **Integration and Test Execution on `main`**:
   Once applied to `/Users/bharath/Code/research/ojas`, execute:
   ```sh
   cd /Users/bharath/Code/research/ojas
   cargo test -p ojas-cpu --test gemma_ops -- --nocapture
   cargo test -p ojas-autograd --test gelu_tanh_tape -- --nocapture
   cargo test -p ojas-cpu --test attention_general -- --nocapture
   cargo test -p ojas-autograd --test attention_general -- --nocapture
   cargo test -p ojas-cpu --test autocast_sam_policy -- --nocapture
   ```
4. **Invalidation Conditions**:
   - Any divergence between `Numerics::Exact` and `Numerics::Fast` (violating `AGENTS.md` bit-identity invariant).
   - Any non-finite float leakage without raising `OjasError::NonFinite`.
   - Any test failure in `gemma_ops`, `gelu_tanh_tape`, or `autocast_sam_policy`.
