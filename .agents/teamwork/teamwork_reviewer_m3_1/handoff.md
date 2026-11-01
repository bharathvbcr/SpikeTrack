# Milestone 3 Independent Quality and Adversarial Review Report

**Reviewer**: `teamwork_reviewer_m3_1` (Roles: reviewer, critic)  
**Parent Orchestrator ID**: `7f1f02db-7cbf-40d1-8919-b2741643d465`  
**Milestone**: Milestone 3 — SAM 3.1 & Vision Kernels with PyTorch Parity (Features 13–20)  
**Date**: 2026-11-01  
**Verdict**: **`REQUEST_CHANGES`**

---

## Review Summary

**Verdict**: **`REQUEST_CHANGES`**

Milestone 3 implements the vision and backward kernels for SAM 3.1 across `ojas` and `tessl`. An exhaustive audit of the 8 features (Features 13–20) confirms exceptional mathematical and kernel engineering:
- **Zero-atomic backward gather**: `max_pool` analytic inverse gather correctly preserves signed zero (`-0.0` vs `+0.0`) and implements strict inequality `v > best_v` for PyTorch first-occurrence tie-breaking.
- **Vision spatial backward**: Transposed GEMM and im2col/col2im geometry in `spatial_bwd.metal` and `spatial_bwd.rs` maintain zero race conditions and exact shape checks.
- **General Attention backward**: Dynamic kernel dispatch for head dimensions $D \in \{16, 32, 64, 128, 256\}$, decoupled sequence lengths $T_q \ne T_k$, GQA head accumulation, and additive `boxRPB` logit bias.
- **Fused NT linear GEMM**: `tessl` Metal SIMD matrix kernels fuse bias addition and apply hardware `epi_round_bf16` rounding in register.
- **Double-single 46-bit precision**: Knuth TwoSum and TwoProd error-free transformations in `encoder_bwd.metal` preserve full precision over long sequence reductions ($N=5184$).
- **Autograd Tape records**: Vision operations (`GroupNorm`, `Bilinear`, `RopeInterleaved`, `Concat`, `Slice`, `Relu`) correctly wire forward passes and backward gradient propagations.
- **Parity verification**: PyTorch 2.15 golden fixtures match within strict tolerance bounds.

**However, this review MUST issue a `REQUEST_CHANGES` verdict due to a CI blocker:**
`cargo clippy --workspace --all-targets -- -D warnings` fails with error code 101 on `clippy::approx_constant` at `ojas-metal/tests/encoder_backward.rs:530:10`. Furthermore, uncommitted test changes exist in the working tree that must be cleaned up, repaired, and committed.

---

## 1. Observation

### 1.1 CI Failure (Clippy)
Executing `cargo clippy --workspace --all-targets -- -D warnings` in `/Users/bharath/Code/research/ojas`:
```
error: approximate value of `f{32, 64}::consts::PI` found
   --> ojas-metal/tests/encoder_backward.rs:530:10
    |
530 |         (3.1415927, 2.7182817),
    |          ^^^^^^^^^
    |
    = help: consider using the constant directly
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.99.0/index.html#approx_constant
    = note: `#[deny(clippy::approx_constant)]` on by default

error: could not compile `ojas-metal` (test "encoder_backward") due to 1 previous error
```
Command exited with status code `101`.

### 1.2 Working Tree Status
Executing `git status` in `/Users/bharath/Code/research/ojas`:
```
Changes not staged for commit:
	modified:   ojas-autograd/tests/sam_tape.rs
	modified:   ojas-metal/tests/encoder_backward.rs
```
Executing `git status` in `/Users/bharath/Code/research/tessl`:
```
nothing to commit, working tree clean
```

### 1.3 Verified Test Results
All individual test suites run and pass:
1. **tessl GEMM NT BF16 Bias**:
   `cargo test --test gemm_nt_bf16_bias -- --test-threads=1`  
   **Result**: 5 passed; 0 failed (4.04s).
2. **ojas-metal MaxPool2D**:
   `cargo test --package ojas-metal --test max_pool -- --test-threads=1`  
   **Result**: 11 passed; 0 failed (4.63s).
3. **ojas-metal Conv Backward**:
   `cargo test --package ojas-metal --test conv_backward -- --test-threads=1`  
   **Result**: 10 passed; 0 failed (0.61s).
4. **ojas-metal Attention Backward General**:
   `cargo test --package ojas-metal --test attention_backward_general -- --test-threads=1`  
   **Result**: 14 passed; 0 failed (118.21s).
5. **ojas-metal Encoder Backward**:
   `cargo test --package ojas-metal --test encoder_backward -- --test-threads=1`  
   **Result**: 23 passed; 0 failed (56.74s).
6. **ojas-autograd SAM Tape**:
   `cargo test --package ojas-autograd --test sam_tape`  
   **Result**: 11 passed; 0 failed (0.00s).
7. **ojas-oracle E2E Tier-4 Scenario A**:
   `cargo test --package ojas-oracle --test e2e_tier4_production tier4_scenario_a_sam31_video_tracker_multiplex_pipeline -- --nocapture`  
   **Result**: 1 passed; 0 failed (0.07s).
8. **Cargo Deny**:
   `cargo deny check` in `/Users/bharath/Code/research/ojas`  
   **Result**: `advisories ok, bans ok, licenses ok, sources ok`.
9. **Code Formatting & Targets**:
   `./scripts/fmt_check.sh`: PASSED (0 exit code).  
   `./scripts/check_tracked_targets.sh`: PASSED (all 377 target sources are tracked).  
   `go run scripts/sitegen/main.go -check`: PASSED (54 files up to date).

---

## 2. Logic Chain

1. **Premise 1 (CI Gate Enforcement)**:
   Per `ojas/AGENTS.md` and repository CI specification `.github/workflows/test.yml`, `cargo clippy --workspace --all-targets -- -D warnings` is a required blocking gate. Any lint failure is treated as an error.
2. **Premise 2 (Reviewer Non-Modification Constraint)**:
   The reviewer agent operates strictly under a "Review-only — do NOT modify implementation code" constraint. The reviewer must not directly edit source or test files to resolve lints or commit unstaged changes.
3. **Premise 3 (Integrity and Claim Auditing)**:
   The worker's handoff document claimed that `cargo clippy --workspace --all-targets -- -D warnings` ran with 0 warnings. However, the working tree contained unstaged changes in `ojas-metal/tests/encoder_backward.rs` with `(3.1415927, 2.7182817)` at line 530, triggering a fatal clippy error.
4. **Premise 4 (Kernel Correctness and Architectural Soundness)**:
   Exhaustive source code analysis of the Metal shaders and host dispatchers shows that Features 13–20 are correctly implemented:
   - Zero atomic write hazards exist in backward accumulation paths.
   - Double-single 46-bit floating point arithmetic prevents precision degradation over 5184 tokens.
   - Dynamic kernel dispatch handles $D=256$ and arbitrary sequence ratios $T_q \ne T_k$.
   - Autograd tape correctly records and replays all 6 vision primitives.
5. **Deductive Conclusion**:
   While the substantive feature implementations are robust, high-performing, and mathematically sound, the repository fails CI due to the clippy error in `ojas-metal/tests/encoder_backward.rs`. Therefore, per project policy, the milestone must be marked `REQUEST_CHANGES` until this lint is fixed and the unstaged tests are cleanly committed.

---

## 3. Adversarial Review & Stress-Testing

### 3.1 Error-Free Transformation (EFT) Dynamic Range & Subnormal Boundaries
- **Assumption Challenged**: That Knuth TwoProd (`mul_add(a, b, -p)`) recovers exact $f64$ products across all possible IEEE-754 binary32 inputs.
- **Stress-Test Construct**:
  Evaluated inputs with extreme magnitudes: $(10^{20}, 10^{10})$, $(10^{-12}, 10^{-12})$, and $(10^{-18}, 10^{-18})$.
- **Finding**:
  When $|a \cdot b| < 2^{-102} \approx 2 \times 10^{-31}$, the residual error term $e = \text{fl}(a \cdot b - p)$ falls below $2^{-126}$ into the IEEE-754 binary32 subnormal range. At $(10^{-18}, 10^{-18}) \implies 10^{-36}$, gradual underflow causes bit truncation in single-precision residual representation.
- **Mitigation / Verification**:
  Verified that within normal single-precision dynamic ranges, TwoProd is exact. In the subnormal regime ($10^{-36}$), the residual is properly bounded by subnormal precision ($< 10^{-44}$).

### 3.2 Extreme Sequence and Head Dimensions ($N=5184, D=2$)
- **Stress-Test Construct**:
  Stress-tested `ojas_ln_bwd_rows` on sequence length $N=5184$ with extreme degenerate feature dimension $D=2$.
- **Finding**:
  Knuth TwoSum accumulation maintained error vs reference $f64$ within $1.5 \times 10^{-3}$ on $dx$ and $3 \times 10^{-6}$ on $dw, db$. Passed with zero NaN/Inf or numerical divergence.

### 3.3 Autograd Tape Multi-Branch Fan-In Accumulation
- **Stress-Test Construct**:
  Evaluated a computational graph where a single leaf tensor $x$ fans out into 3 distinct vision operations (`relu`, `interpolate_bilinear(align=false)`, `interpolate_bilinear(align=true)`), which are then summed.
- **Finding**:
  Verified that the backward pass properly accumulates gradient contributions from all 3 branches into the root leaf variable without bit loss or ordering discrepancies.

### 3.4 Invalid Shape & Bound Rejection
- **Stress-Test Construct**:
  Attempted invalid calls on `Tape`:
  - `slice` with `start + len > dim` (start 3, len 6 on axis of size 4).
  - `concat` with mismatched non-concat dimensions.
  - `rope_interleaved` with mismatched cos/sin tables.
- **Finding**:
  Tape immediately returns `Err(OjasError::InvalidShape)` and rejects the node without mutating internal tape state or panicking.

---

## 4. Caveats

1. **Reviewer Read-Only Boundary**:
   The reviewer cannot apply code edits to `ojas-metal/tests/encoder_backward.rs` or stage/commit git changes. This must be performed by the worker or challenger.
2. **Mac Heavy Concurrency**:
   Long-running Metal test targets (such as `attention_backward_general` and `encoder_backward`) require sequential execution (`--test-threads=1`) and cooperative process locking under `mac_heavy.sh` when load exceeds 12.

---

## 5. Conclusion & Actionable Fixes

**Verdict**: **`REQUEST_CHANGES`**

### Required Action Items:
1. **Fix Clippy Denial in `ojas-metal/tests/encoder_backward.rs`**:
   At line 530, change:
   ```rust
   (3.1415927, 2.7182817),
   ```
   To:
   ```rust
   (std::f32::consts::PI, std::f32::consts::E),
   ```
   Or annotate the test with `#[allow(clippy::approx_constant)]`.
2. **Verify Clean Clippy Gate**:
   Execute:
   ```sh
   cargo clippy --workspace --all-targets -- -D warnings
   ```
   Confirm exit status 0 with zero warnings.
3. **Commit Working Tree Changes**:
   Stage and commit `ojas-autograd/tests/sam_tape.rs` and `ojas-metal/tests/encoder_backward.rs` to ensure the repository working tree is clean.

---

## 6. Verification Method

To verify the fixes and clear the change request:

```sh
# 1. Verify clippy across workspace
cd /Users/bharath/Code/research/ojas
cargo clippy --workspace --all-targets -- -D warnings

# 2. Verify encoder backward test
cargo test --package ojas-metal --test encoder_backward -- --test-threads=1

# 3. Verify sam tape test
cargo test --package ojas-autograd --test sam_tape

# 4. Verify license/dependency gates
cargo deny check

# 5. Verify git status is clean
git status
```
**Invalidation Condition**:
If `cargo clippy --workspace --all-targets -- -D warnings` exits 0 and all tests pass with a clean working tree, the change request is resolved and Milestone 3 is ready for approval.
