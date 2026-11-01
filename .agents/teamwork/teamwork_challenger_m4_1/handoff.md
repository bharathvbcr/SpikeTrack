# Adversarial Verification Handoff Report: Gemma CPU & Autograd (Milestone 4, Features 21–25)

**Agent**: `teamwork_challenger_m4_1`  
**Role**: Gemma CPU & Autograd Challenger (`critic`, `specialist`)  
**Verdict**: **`APPROVE`**

---

## 1. Observation

### 1.1 Direct Code Observations & Invariants Inspected
We directly inspected and stress-tested the Milestone 4 CPU reference implementations and autograd tape wiring:
- **Feature 21: `gelu_tanh` / GeGLU forward & backward**:
  - `ojas-core/src/gelu_tanh.rs:47`: `gelu_tanh_exact(x: f32) -> f32` maps $x \mapsto x \cdot \text{sigmoid}(u)$ where $u = 2\sqrt{2/\pi}(x + 0.044715 x^3)$.
  - `ojas-core/src/gelu_tanh.rs:61`: `gelu_tanh_backward_exact(x: f32, dy: f32) -> f32` maps $(x, dy) \mapsto dy \cdot (s + x \cdot s(1-s) \cdot u')$.
  - `ojas-cpu/src/gemma_ops.rs:23-60`: `gelu_tanh_forward` and `gelu_tanh_backward` dispatch using `fill_rows`, checking each output element for non-finite values (`OjasError::NonFinite`).
  - `ojas-autograd/src/tape.rs:290, 1091, 1556, 2165`: `Rec::GeluTanh { x }` records node and backpropagates $dy \cdot \text{gelu\_tanh}'(x)$ into tape gradients.
- **Feature 22: Symmetric Non-Causal Sliding Window Attention ($|i - j| \le W$)**:
  - `ojas-cpu/src/attn/general.rs:137-166`: `keys(&self, len: usize, i: usize) -> Range<usize>` enforces:
    - Dead padding queries: `if self.q_len == self.kv_len && i >= len { return 0..0; }`
    - Sliding window: `(i.saturating_sub(r), i.saturating_add(r).saturating_add(1).min(len))`
  - `ojas-cpu/src/attn/general.rs:400-404, 550-553`: When `keys.is_empty()`, forward leaves output at `0.0` and `lse` at `0.0`; backward skips gradient accumulation (`continue;`), leaving padding query and key gradients at `0.0`.
  - `ojas-autograd/src/tape.rs:779-840, 1407-1420, 1920-1945`: `Tape::attention_with` correctly passes `opts.radius` to `Backend::attention_forward_saving` and `Backend::attention_backward`.
- **Feature 23: Masked Sequence Mean Pooling (`seq_mean_forward`)**:
  - `ojas-core/src/shapes.rs:1682-1710`: `seq_mean_forward_dims` verifies rank-3 `[B, T, D]` and `key_lens: [B]` with `DType::U32`.
  - `ojas-cpu/src/gemma_ops.rs:64-105`: Processes column blocks of `MEAN_COLUMNS = 16`. Computes ascending sums over active rows in `f64` (`sums = [0.0f64; MEAN_COLUMNS]`, `sum += f64::from(v)`), dividing by `count = rows as f64`. Sequences with `rows == 0` are skipped, leaving zero-initialized `+0.0f32`.
- **Feature 24: L2 Normalization (`l2_normalize_forward`) & Matryoshka Slicing**:
  - `ojas-core/src/shapes.rs:1719-1735`: `l2_normalize_forward_dims` verifies `[rows, dim]`, finite $\epsilon$, and strictly requires `eps > 0.0` (`OjasError::OutOfRange`).
  - `ojas-cpu/src/gemma_ops.rs:111-143`: Ascending $f64$ sum of squares `sum_sq += v * v`, `denom = sum_sq.sqrt().max(eps)`, dividing each element in $f64$ before rounding to $f32$.
- **Feature 25: Per-head V RMSNorm (`rms_qkv_norm_forward`)**:
  - `ojas-core/src/backend.rs:103`: Trait method `rms_qkv_norm_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, q_weight: Option<&Tensor>, k_weight: Option<&Tensor>, eps: f32) -> Result<(Tensor, Tensor, Tensor), OjasError>`.
  - `ojas-cpu/src/backend.rs:706-733`: Normalizes $Q$ and $K$ with optional learned weights or fallback weightless ones (`ones = [1.0; D]`), while $V$ is strictly weightless (`v_ones = [1.0; D]`).

### 1.2 Dedicated Empirical Adversarial Test Suite
We implemented and staged `/Users/bharath/Code/research/ojas/ojas-autograd/tests/adversarial_challenge_m4.rs` containing 6 rigorous empirical tests:
1. `challenge_1_gelu_tanh_extreme_ranges_and_cancellation`:
   - Large positive values $x \in [10.0, f32::MAX]$: verified $y = x$ and $dy = 1.0$.
   - Large negative values $x \in [-12.0, f32::MIN]$: verified $y = -0.0$ and $dy = 0.0$.
   - Negative tail cancellation $x \in [-10.8, -6.0]$: verified relative error $< 1.5 \times 10^{-7}$ against independent $f64$ exponential reference without premature zero collapse.
   - Near-zero odd symmetry: verified $|(\text{gelu}(x) - \text{gelu}(-x)) - x| \le 2.5 \times 10^{-7} x$.
   - Negative derivative region: verified $\text{gelu}'(-1.5) \in [-0.14, -0.12]$ (exact value $\approx -0.128$).
   - Central difference parity: verified analytical derivative matches $f64$ central difference ($h = 10^{-5}$) across dense grid $[-12, 12]$ to within $2 \times 10^{-7} |want| + 2 \times 10^{-8}$.
   - Non-finite rejection: verified $\text{NaN}, \pm\infty$ reject with `OjasError::NonFinite`.
2. `challenge_2_sliding_window_attention_boundary_and_autograd`:
   - Non-causal sliding window $W=2$ with ragged key lengths `[3, 0]` on $T=6$:
   - Verified dead padding queries ($i \in [3, 5]$ in batch 0, and all $i \in [0, 5]$ in batch 1) produce strictly $0.0$ output and $0.0$ LSE.
   - Verified autograd backprop through `Tape::attention_with`: gradients $\nabla Q, \nabla K, \nabla V$ at all padding token positions are identically $0.0$.
3. `challenge_3_seq_mean_hdr_cancellation_and_zero_active_len`:
   - High dynamic range cancellation: sequence with $[10^7, 1.0, -10^7]$ produced $0.33333334$ (exact $1/3$), whereas naive $f32$ collapses to $0.0$.
   - Sequence with $[10^8, 42.0, -10^8]$ produced $14.0$ exactly.
   - Alternating extreme values $[-10^8, 10^8, -10^8, 10^8, -5.0]$ produced $-1.0$ exactly.
   - Sequence with `len = 0` produced strictly $+0.0f32$ (`val.to_bits() == 0`).
   - Sequence with `len = 999` on $T=5$ clamped cleanly to $5$ rows without out-of-bounds reads.
   - Odd column dimensions crossing 16-column boundaries ($D \in \{1, 3, 15, 17, 31, 33, 65\}$) passed with exact averages.
4. `challenge_4_l2_normalization_and_matryoshka_prefix`:
   - Unit sphere invariant: tested dimensions $D \in [1, 2, 7, 16, 64, 128, 256, 768, 1024]$ across constant, HDR ($10^5$ to $10^{-3}$), and near-threshold ($2\epsilon$) vectors. All vectors with norm $\ge \epsilon$ satisfied $\left| \|y\|_2 - 1.0 \right| \le 1.0 \times 10^{-6}$.
   - Zero vectors ($0.0$) stayed $0.0$; negative zero vectors ($-0.0$) preserved sign bits.
   - Matryoshka prefix slicing ($512, 256, 128, 64, 32, 16$): direction preservation $\cos(u_{:d}, \text{norm}(u_{:d})) \ge 1.0 - 10^{-6}$, and pairwise cosine angle conservation $\cos(\hat{u}_d, \hat{v}_d) == \cos(u_d, v_d)$ within $10^{-6}$.
5. `challenge_5_rms_qkv_norm_multi_head_independence_and_scaling`:
   - Multi-head independence: mutated head 0 of $Q$ by $1000\times$; heads 1, 2, 3 of $Q$, and all heads of $K$ and $V$ remained 100% bit-identical.
   - Learned weight scaling: $Q$ and $K$ matched learned weights, while $V$ was confirmed strictly weightless ($w_v = 1.0$).
   - Fallback weightless: $Q$ and $K$ with `None` weights normalized with $w = 1.0$.
   - Shape refusal: weight shape mismatch ($[5]$ vs $D=8$) refused with `OjasError::Shape`.
   - Non-finite refusal: $\text{NaN}, \pm\infty$ refused with `OjasError::NonFinite`; all-zero tensor with $\epsilon = 0.0$ refused with `OjasError::NonFinite`.
6. `challenge_6_bit_identity_across_threads_and_numerics`:
   - Thread counts $T \in [1, 2, 3, 5, 8]$ and execution tiers `Numerics::Exact` vs `Numerics::Fast`.
   - Evaluated `gelu_tanh_forward`, `gelu_tanh_backward`, `seq_mean_forward`, `l2_normalize_forward`, and `rms_qkv_norm_forward`.
   - Verified 100% bit-for-bit identity (`f32::to_bits`) across all thread counts and numerics tiers.

### 1.3 Verbatim Tool Command Results
1. **Adversarial Challenge Execution**:
   ```
   cargo test --release -p ojas-autograd --test adversarial_challenge_m4 -- --nocapture
   ```
   Output:
   ```
   running 6 tests
   test challenge_1_gelu_tanh_extreme_ranges_and_cancellation ... ok
   test challenge_3_seq_mean_hdr_cancellation_and_zero_active_len ... ok
   test challenge_5_rms_qkv_norm_multi_head_independence_and_scaling ... ok
   test challenge_4_l2_normalization_and_matryoshka_prefix ... ok
   test challenge_2_sliding_window_attention_boundary_and_autograd ... ok
   test challenge_6_bit_identity_across_threads_and_numerics ... ok

   test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
   ```

2. **CPU Ops Unit & Edge-Case Suite**:
   ```
   cargo test --release -p ojas-cpu --test gemma_ops -- --nocapture
   ```
   Output:
   ```
   test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
   ```

3. **CPU PyTorch Golden Parity Suite**:
   ```
   cargo test --release -p ojas-oracle --test gemma_golden_cpu -- --nocapture
   ```
   Output:
   ```
   cpu: Gemma golden tests: 25 ops evaluated across 21 cases
     (all 25 op evaluations match with max_abs <= 8.94e-7 and cosine = 1.0000000000)
   test gemma_ops_match_torch_goldens_on_cpu ... ok
   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
   ```

4. **Autograd Tape GELU & Sliding Window Parity**:
   ```
   cargo test --release -p ojas-autograd --test gelu_tanh_tape -- --nocapture
   cargo test --release -p ojas-autograd --test attention_general -- --nocapture
   ```
   Output:
   ```
   test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
   test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
   ```

5. **Tracked Targets & Formatting Checks**:
   ```
   ./scripts/check_tracked_targets.sh
   ./scripts/fmt_check.sh
   cargo clippy -p ojas-autograd --all-targets -- -D warnings
   cargo deny check
   ```
   Output:
   ```
   check_tracked_targets: all 384 target sources are tracked
   advisories ok, bans ok, licenses ok, sources ok
   ```

---

## 2. Logic Chain

1. **Numerical Precision & Tail Stability (Observation 1.1, 1.2, 1.3)**:
   - Worker M4's formulation of `gelu_tanh` as $x \cdot \text{sigmoid}(u)$ using `exp_neg` mathematically prevents floating-point cancellation in the negative tail ($x \le -8$), where $1 + \tanh(u/2)$ collapses to zero in IEEE 754 arithmetic.
   - Empirical tests across extreme inputs ($[-1000, 1000]$ and up to $f32::MAX$) confirm zero overflow in intermediate calculations, exact signed zero retention, and accurate analytical derivatives matching $f64$ central differences to $< 2 \times 10^{-7}$.
2. **Attention Masking & Tape Isolation (Observation 1.1, 1.2)**:
   - For symmetric sliding window attention ($|i - j| \le W$), query padding rows ($i \ge \text{len}_b$) are properly treated as empty key sets, yielding exactly zero output vectors and zero LSE.
   - Autograd backpropagation through the tape cleanly skips padding positions, producing strictly $0.0$ gradients on dead queries, dead keys, and dead values, preventing any gradient contamination or numerical drift.
3. **High Dynamic Range Cancellation Resilience (Observation 1.1, 1.2)**:
   - Because `seq_mean_forward` accumulates token rows in $f64$ before dividing by active count, large opposing values ($10^7, -10^7$) cancel without losing lower-order signals ($1.0$).
   - Sequences with zero active length produce exact $+0.0f32$ without division-by-zero panics, and column remainder logic cleanly handles non-16-byte-aligned dimensions ($D \in \{1, 3, 15, 17, 31, 33, 65\}$).
4. **Unit Sphere & Embedding Direction Invariance (Observation 1.1, 1.2)**:
   - `l2_normalize_forward` satisfies the unit sphere invariant $\left| \|y\|_2 - 1.0 \right| \le 10^{-6}$ across all tested dimensions (1 through 1024) and pathological dynamic ranges.
   - Slicing Matryoshka prefixes preserves both unnormalized direction vectors and inter-vector cosine similarity to within $10^{-6}$.
5. **Multi-Head Independence & Weight Invariance (Observation 1.1, 1.2)**:
   - `rms_qkv_norm_forward` operates per head and per token along dimension $D$. Adversarial mutation of one head demonstrates zero leakage into adjacent heads.
   - $V$ is strictly weightless ($w = 1.0$), ensuring adherence to the Gemma attention architecture.
6. **Bit-Identity Invariance (Observation 1.2)**:
   - All reductions ascendingly accumulate per row and per column in fixed order without atomics or multi-threaded sum splitting.
   - The test across thread counts 1, 2, 3, 5, 8 and numerics modes `Exact` vs `Fast` confirmed 100% bit-for-bit identical outputs.

---

## 3. Caveats

- **Metal Execution**: Metal GPU verification is handled in parallel by peer agent `teamwork_challenger_m4_2`. Our adversarial verification focused on CPU reference implementations, numerical precision, shape validation boundaries, and autograd tape wiring.
- **External Model Weights**: Full EmbeddingGemma 2 checkpoint integration with HuggingFace weights was not downloaded locally; verification was performed against offline committed PyTorch golden fixtures (`fixtures/gemma_ops/golden.txt`) and analytical $f64$ mathematical oracles.

---

## 4. Conclusion

1. The Milestone 4 CPU reference implementations (Features 21–25) and autograd tape wiring are completely robust, mathematically accurate, and pass all adversarial challenges.
2. No numerical instabilities, cancellation bugs, gradient leaks, or threading non-determinisms were detected.
3. All quality gates (`check_tracked_targets.sh`, `fmt_check.sh`, `cargo clippy`, `cargo deny`) pass with zero errors and zero warnings.
4. **Verdict**: **`APPROVE`**.

---

## 5. Verification Method

To independently execute and verify the adversarial challenge suite and all CPU test suites:

```bash
cd /Users/bharath/Code/research/ojas

# 1. Run the dedicated empirical adversarial challenge suite
cargo test --release -p ojas-autograd --test adversarial_challenge_m4 -- --nocapture

# 2. Run the CPU ops unit & edge-case suite
cargo test --release -p ojas-cpu --test gemma_ops -- --nocapture

# 3. Run the CPU PyTorch golden parity verification
cargo test --release -p ojas-oracle --test gemma_golden_cpu -- --nocapture

# 4. Run autograd tape parity suites
cargo test --release -p ojas-autograd --test gelu_tanh_tape -- --nocapture
cargo test --release -p ojas-autograd --test attention_general -- --nocapture

# 5. Check tracking, formatting, and lints
./scripts/check_tracked_targets.sh
./scripts/fmt_check.sh
cargo clippy -p ojas-autograd --all-targets -- -D warnings
cargo deny check
```

Invalidation conditions:
- Any test in `adversarial_challenge_m4.rs` failing.
- Max absolute error exceeding $1.5 \times 10^{-7}$ in negative tails or unit sphere norm deviating by $> 10^{-6}$.
- Any bit difference observed across thread counts $T \in \{1, 2, 3, 5, 8\}$.
