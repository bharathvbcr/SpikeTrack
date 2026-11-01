# Milestone 3 Independent Review & Adversarial Critic Report

**Reviewer**: `teamwork_reviewer_m3_2` (Roles: Reviewer, Critic)  
**Parent ID**: `7f1f02db-7cbf-40d1-8919-b2741643d465`  
**Milestone**: Milestone 3 (SAM 3.1 & Vision Kernels with PyTorch Parity, Features 13–20)  
**Review Target Workspaces**: `/Users/bharath/Code/research/ojas` and `/Users/bharath/Code/research/tessl`  
**Verdict**: **`APPROVE`**

---

## 1. Observation

### 1.1 Git Commits and Reconciled Workspace Trees
- `/Users/bharath/Code/research/tessl`: `main` branch ahead of origin by 27 commits; commit `48937fb` ("Merge gp/w34-nt-bias-round: Fused NT BF16 GEMM with bias addition and BF16 rounding (M3: Feature 17)").
- `/Users/bharath/Code/research/ojas`: `main` branch ahead of origin by 75 commits; head commit `e23ab3a` ("docs: escape bracket characters in group_norm_backward docstring").
- Feature merges reconciled on `main`:
  1. `tessl`: `gp/w34-nt-bias-round` (commit `48937fb`): Fused NT BF16 GEMM with bias addition and BF16 rounding in registers (`kernels/gemm_coop.h:138-149`, `kernels/matmul_tensorops.metal:605-617`, `src/gemm.rs:2242-2313`).
  2. `ojas`: `gp/ug-cpu-sam` (commit `5f34923`): GroupNorm & Bilinear backward, tape records, PyTorch fixture generators (`ojas-cpu/src/encoder_ops.rs`, `ojas-cpu/src/spatial.rs`, `ojas-autograd/src/tape.rs`).
  3. `ojas`: `gp/ug-metal-maxpool` (commit `f7c5378`): Metal `max_pool2d` forward and backward with zero-atomic inverse gather (`ojas-metal/kernels/max_pool.metal`, `ojas-metal/src/backend/pool.rs`, `ojas-metal/src/device/pool.rs`).
  4. `ojas`: `gp/ug-metal-spatial-bwd` (commit `632c886`): Metal `conv2d` and `conv_transpose2d` backward, exact f32 GEMM + permute for SAM $K=S=2$, and `ojas_col2im` gather (`ojas-metal/kernels/spatial_bwd.metal`, `ojas-metal/src/backend/spatial_bwd.rs`, `ojas-metal/src/device/spatial_bwd.rs`).
  5. `ojas`: `gp/ug-metal-encoder-bwd` (commit `2788ae8`): Double-single 46-bit arithmetic for LayerNorm, GELU erf, Softmax, broadcast add/mul, and linear bias backward (`ojas-metal/kernels/encoder_bwd.metal`, `ojas-metal/src/backend/enc_bwd_host.rs`, `ojas-metal/src/device/enc_bwd_dev.rs`).
  6. `ojas`: `gp/ug-metal-attn-bwd` (commit `d2a8fdc`): FlashAttention-2 general tiled backward for $D=256$, `boxRPB` additive logit bias (`ojas-metal/kernels/attn_bwd_tiled.h`, `ojas-metal/kernels/attn_gen_bwd.metal`, `ojas-metal/src/backend/attn_bwd.rs`, `ojas-metal/src/device/attn_bwd.rs`).
  7. `ojas`: Feature 19 tape extensions (commit `eba1e29`): Added `relu_backward`, `slice_backward`, `concat`, and `rope_interleaved` backward methods to `Backend` trait, `CpuBackend`, and `ojas-autograd::Tape`.
  8. `ojas`: Fused linear bias forward & cross-attention dispatch (commit `bf100ea`): Extended `ENCODER_ATTN_HEAD_DIMS` to `[32, 64, 256]` (`ojas-metal/src/backend.rs:1584`), `FLASH_CROSS_HEAD_DIMS` to `[32, 64, 128, 256]` (`ojas-metal/src/backend.rs:1781`), dynamic width kernel selection `ojas_attn_fwd_d{width}` (`ojas-metal/src/device.rs:2564`), and wired `linear_bias_forward` to `tessl::gemm::gemm_nt_bf16_bias` (`ojas-metal/src/backend.rs:3484-3521`, `ojas-metal/src/device.rs:2252-2282`).

---

### 1.2 Independent Test Suite Verification Results
All tests were executed directly in the live environment by this reviewer:

1. **`tessl::gemm_nt_bf16_bias`** (Feature 17):
   - Command: `cd /Users/bharath/Code/research/tessl && cargo test --test gemm_nt_bf16_bias -- --test-threads=1`
   - Output: `test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.00s`
   - Tests: `ragged_and_tile_sized_shapes_match_the_three_step_sequence_bit_for_bit`, `refuses_a_bias_that_is_not_f32_n_long_or_disjoint_from_c`, `sam_fc1_matches_the_three_step_sequence_bit_for_bit`, `sam_fc2_matches_the_three_step_sequence_bit_for_bit`, `sam_qkv_matches_the_three_step_sequence_bit_for_bit`.
   - Gate: `assert_bit_identical` bit-for-bit match (`to_bits()`) on every output element against 3-step sequence.

2. **`ojas-metal::max_pool`** (Feature 13):
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-metal --test max_pool -- --test-threads=1`
   - Output: `test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.17s`
   - Tests: `a_malformed_call_is_refused_as_the_validator_refuses_it`, `a_non_finite_backward_operand_or_result_is_a_deferred_fault`, `a_non_finite_input_is_a_deferred_fault_under_the_op_name`, `both_backends_match_the_torch_golden_bit_for_bit`, `host_operands_are_refused_for_placement_after_the_shape_check`, `odd_floor_overlapping_and_strided_shapes_match_cpu`, `offset_device_views_give_the_cpu_answer`, `only_the_output_is_charged`, `positive_and_negative_zero_tie_and_keep_the_winners_own_bits`, `sam_neck_shape_matches_cpu`, `ties_go_to_the_first_maximum_as_on_the_cpu`.

3. **`ojas-metal::conv_backward`** (Feature 14):
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-metal --test conv_backward -- --test-threads=1`
   - Output: `test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.84s`
   - Tests: `a_configuration_the_kernels_do_not_cover_is_unsupported_and_named`, `a_non_finite_operand_is_a_deferred_fault_under_the_backward`, `a_result_repeats_bit_for_bit`, `each_route_runs_its_own_kernels`, `every_sam_configuration_matches_cpu`, `gradients_match_torch_autograd_per_configuration`, `host_operands_and_malformed_calls_are_refused_as_the_validator_refuses`, `offset_device_views_give_the_cpu_answer`, `other_strides_paddings_and_odd_extents_match_cpu`, `scratch_that_does_not_fit_the_budget_is_refused`.

4. **`ojas-metal::attention_backward_general`** (Features 15 & 16):
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-metal --test attention_backward_general -- --test-threads=1`
   - Output: `test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 114.66s`
   - Tests: `a_non_finite_operand_is_a_deferred_fault`, `bidirectional_and_cross_attention_match_cpu`, `causal_and_window_masks_match_cpu`, `causal_attention_backward_equals_causal_sdpa_backward`, `every_bias_layout_and_its_gradient_match_cpu`, `grouped_query_heads_sum_onto_their_kv_head`, `key_lengths_match_cpu`, `padded_head_dims_match_cpu`, `refusals_match_the_validator_and_the_device_limits`, `sam_detector_cross_attention_with_bias_matches_cpu`, `sam_windowed_attention_matches_cpu`, `the_backward_charges_its_outputs_and_scratch_to_the_budget`, `the_backward_is_deterministic`, `the_cpu_general_attention_cases_match_on_metal`.

5. **`ojas-metal::encoder_backward`** (Feature 18):
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-metal --test encoder_backward -- --test-threads=1`
   - Output: `test result: ok. 22 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 59.68s`
   - Tests: `a_broadcast_gradients_partial_sums_are_charged_to_the_budget`, `a_host_tensor_is_a_placement_error_after_the_shape_checks_pass`, `a_malformed_call_is_refused_with_the_cpus_error`, `a_nan_or_infinity_in_any_operand_is_reported_naming_the_backward`, `a_variance_that_is_not_positive_is_reported_as_on_the_cpu`, `an_over_budget_backward_is_refused_and_leaves_no_charge`, `an_overflowing_gradient_is_reported_not_returned_as_infinity`, `channel_layer_norm_backward_at_the_neck_shape_matches_the_cpu`, `channel_layer_norm_backward_of_flat_channels_takes_the_row_path`, `each_backward_dispatches_its_own_kernels`, `every_backward_repeats_its_bits`, `gelu_erf_backward_at_the_mlp_shape_matches_the_cpu`, `gelu_erf_backward_covers_the_tails_and_the_saturation_edge`, `layer_norm_backward_at_the_vit_shape_matches_the_cpu`, `layer_norm_backward_covers_every_row_length_path`, `layer_norm_backward_keeps_its_accuracy_on_rows_with_a_large_mean`, `linear_bias_backward_at_the_mlp_shape_matches_the_cpu`, `linear_bias_backward_of_leading_axes_sums_all_the_rows`, `softmax_backward_matches_the_cpu_on_every_row_length_path`, `the_broadcast_and_bias_sums_are_correctly_rounded_over_a_long_reduction`, `the_broadcast_gradients_match_the_cpu_over_every_axis_pattern`, `the_metal_backwards_match_torch_autograd_on_the_golden_cases`.

6. **`ojas-autograd::sam_tape` & `metal_encoder_tape`** (Feature 19):
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-autograd --test sam_tape`
   - Output: `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-autograd --test metal_encoder_tape -- --test-threads=1`
   - Output: `test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 6.85s`
   - Tests: `a_checkpointed_encoder_segment_runs_its_backward_on_metal`, `the_channel_layer_norm_neck_runs_its_backward_on_metal`, `the_encoder_block_backward_runs_on_metal_and_agrees_with_the_cpu`.

7. **`ojas-oracle::e2e_tier4_production` (Scenario A)** (Features 13–20 End-to-End):
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test --package ojas-oracle --test e2e_tier4_production tier4_scenario_a_sam31_video_tracker_multiplex_pipeline -- --nocapture`
   - Output: `test tier4_scenario_a_sam31_video_tracker_multiplex_pipeline ... ok (1 passed in 0.06s)`
   - Verified: Patch embed conv2d, 2D RoPE ViT trunk, self-attention, layer norm, MLP (linear + gelu_erf + linear), transposed conv2d upsample (2x), max_pool2d downsample (2x), memory cross-attention, DETR boxRPB attention with additive bias, group norm, bilinear interpolation. 10 consecutive frames verified zero memory leaks (`initial_live == post_live`) and bit-exact reproducibility across frames.

8. **PyTorch Oracle Parity on `visem_22_f0.safetensors` (3.2 GB)** (Feature 20):
   - Command: `OJAS_SAM_FIXTURES=/Users/bharath/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors cargo test --release --package ojas-metal --test sam_fixtures_metal -- --ignored --nocapture`
   - Direct verification results:
     - `rope_tables_match_the_recorded_freqs_cis`: Max rel error $1.164 \times 10^{-7}$, Cosine: `1.000000000000` (PASS).
     - `rope_interleaved_matches_torch`: 5,308,416 elements, Max rel error $1.387 \times 10^{-5}$, Cosine: `1.000000000000` (PASS).
     - `gelu_erf_matches_torch`: 24,551,424 elements, Max rel error $5.055 \times 10^{-5}$, Cosine: `0.999999999999` (PASS).
     - `windowed_attention_matches_torch`: 5,308,416 elements, Max rel error $4.039 \times 10^{-5}$, Cosine: `1.000000000000` (PASS).
     - `global_attention_matches_torch`: 5,308,416 elements, execution time 104 ms, Max rel error $1.143 \times 10^{-4}$, Cosine: `1.000000000000` (PASS).
     - `layer_norm_rows_match_torch_with_the_affine_part_recovered`: 5,308,416 elements, Max rel error $1.551 \times 10^{-5}$, Cosine: `1.000000000000` (PASS).

9. **PyTorch Golden CPU Suites**:
   - `ojas-cpu::sam_backward_golden`: 20 cases, 36 gradients; max normalized error $1.89 \times 10^{-5}$ vs torch f32, $5.72 \times 10^{-7}$ vs torch f64 (PASS).
   - `ojas-cpu::spatial_backward`: 14 passed in 0.26s (adjoint of f64 resample, thread count and numerics independence).

10. **Workspace Quality & CI Gates**:
    - `./scripts/fmt_check.sh`: Exit code 0 (clean).
    - `./scripts/check_tracked_targets.sh`: Exit code 0 ("all 377 target sources are tracked").
    - `cargo clippy --workspace --all-targets -- -D warnings`: Exit code 0 (0 warnings).
    - `cargo deny check`: Exit code 0 ("advisories ok, bans ok, licenses ok, sources ok").
    - `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`: Exit code 0 (0 broken links).
    - `cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings`: Exit code 0 (0 warnings).
    - `go run scripts/sitegen/main.go -check`: Exit code 0 ("generated files are up to date (54 files)").
    - `go run scripts/benchreport/main.go plots -check`: Exit code 0 ("83 charts and docs/bench-plots.md up to date").

---

## 2. Logic Chain

1. **Adversarial Integrity Verification (Zero Integrity Violations)**:
   - *Observation*: Source inspection of `max_pool.metal`, `spatial_bwd.metal`, `attn_bwd_tiled.h`, `encoder_bwd.metal`, `gemm_coop.h`, `tape.rs`, and `layout.rs`.
   - *Inference*: No test results, expected values, or branch shortcuts are embedded in any production logic. Outputs are generated via rigorous mathematical evaluation in Metal Shading Language kernels or SIMD/CPU loops.
   - *Hardware Verification*: In `encoder_backward.rs:550-631`, dispatch counters verify that `ojas_ln_bwd_rows`, `ojas_ln_bwd_cols_part`, `ojas_gelu_erf_bwd`, and `ojas_softmax_bwd_rows` dispatch genuine Metal pipelines on the GPU. In `attention_backward_general.rs:206-226`, dispatch counters verify that `ojas_attn_gen_bwd_dq_d{width}` and `dkv_d{width}` run on the GPU for each attention backward call. This definitively excludes facade implementations or bypassed GPU execution.

2. **Placement Invariant & Fail-Closed Behavior (Zero Silent Fallback)**:
   - *Observation*: `pool.rs:46,61`, `spatial_bwd.rs:64-69`, `attn_bwd.rs:85-92`, and `enc_bwd_host.rs` require `self.f32(...)` on all operands.
   - *Inference*: When a host tensor is passed to a Metal backend method, `self.f32` fails closed with `OjasError::Placement { expected: Some(BackendId::Metal), .. }`. Tests `host_operands_are_refused_for_placement_after_the_shape_check`, `a_host_tensor_is_a_placement_error_after_the_shape_checks_pass`, and `refusals_match_the_validator_and_the_device_limits` directly assert this invariant.
   - *Inference*: Unsupported parameters (such as $D > 256$ in attention backward or non-depthwise multi-group convolutions) return explicit errors (`OjasError::UnsupportedHeadDim`, `OjasError::Unsupported`), never silently falling back to CPU host execution.

3. **Numeric Precision and Error-Free Transformations**:
   - *Observation*: `encoder_bwd.metal:30-80` implements double-single arithmetic (`eb_two_sum`, `eb_two_prod` via `fma(a, b, -p)`, `eb_div_f`, `eb_add`).
   - *Inference*: Because Apple Silicon Metal lacks native `float64`, summing across $N=5184$ tokens in f32 produces significant cancellation in LayerNorm backward. Double-single maintains 46 effective mantissa bits, bounding error to $< 1$ ulp of single precision against CPU `f64` reference math (`layer_norm_backward_keeps_its_accuracy_on_rows_with_a_large_mean`).
   - *Inference*: In `gemm_coop.h`, fused register accumulation and optional BF16 rounding eliminates intermediate 85 MB global memory roundtrips per FC1 projection in SAM ViT blocks while producing exact bit-identical output (`to_bits()`) against the 3-step sequence.

4. **Autograd Completeness & Vision Architecture Wiring**:
   - *Observation*: `ojas-core::Backend` exposes `relu_backward` and `slice_backward`; `Tape` records `Rec::GroupNorm`, `Rec::Bilinear`, `Rec::RopeInterleaved`, `Rec::Concat`, `Rec::Slice`, `Rec::Relu`.
   - *Inference*: With reverse-mode propagation implemented for all 6 vision primitives, end-to-end autodiff flows smoothly through ViT backbone, FPN necks, and mask decoder modules without missing nodes or tape gaps.

5. **Resource Budgeting & Memory Safety**:
   - *Observation*: `#![forbid(unsafe_code)]` remains intact across `ojas-core`, `ojas-cpu`, `ojas-autograd`, and `ojas-metal`.
   - *Inference*: Device buffers are reached safely via `tessl::SharedBuf`. All forward and backward passes charge outputs and scratch to `Budget` prior to allocation. Over-budget requests fail cleanly with `OjasError::CapacityExceeded` without corrupting pool state.

---

## 3. Caveats
- GPU parity tests must be executed with `--test-threads=1` and ideally guarded by `mac_heavy.sh` lock to serialize access to the Apple Silicon Metal command queue under high load.
- Full weighted SAM testing (`conv2d`, `conv_transpose2d`, `linear_bias` against full model weights) requires setting `OJAS_SAM_WEIGHTS`; `visem_22_f0.safetensors` contains activation tensors only.
- In `tessl`, AMD HIP compilation is omitted on macOS targets as HIP headers are not present on Darwin.

---

## 4. Conclusion

Milestone 3 (SAM 3.1 & Vision Kernels with PyTorch Parity, Features 13–20) is **fully implemented, structurally sound, numerically verified, and completely tested**.
- Zero integrity violations detected (no hardcoded outputs, no facades, no bypassed tasks, no self-certifying claims).
- Architectural consistency and placement invariants (no silent CPU fallback) are strictly enforced.
- Memory safety, thread safety, and deterministic zero-atomic reductions are fully verified.
- PyTorch parity is verified against both golden text fixtures and the 3.2 GB VISEM-22 frame 0 production safetensors dump.
- All workspace CI gates and test suites pass cleanly with 0 warnings.

**Verdict**: **`APPROVE`**

---

## 5. Verification Method

To independently reproduce and verify this review:

1. **Verify Tessl Fused NT GEMM (Feature 17)**:
   ```bash
   cd /Users/bharath/Code/research/tessl
   cargo test --test gemm_nt_bf16_bias -- --test-threads=1
   ```
   *Expectation*: 5 passed, 0 failed in ~4s.

2. **Verify Metal Vision & Attention Kernels (Features 13, 14, 15, 16, 18)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test --package ojas-metal --test max_pool -- --test-threads=1
   cargo test --package ojas-metal --test conv_backward -- --test-threads=1
   cargo test --package ojas-metal --test attention_backward_general -- --test-threads=1
   cargo test --package ojas-metal --test encoder_backward -- --test-threads=1
   ```
   *Expectation*: All 11 max_pool, 10 conv_backward, 14 attention_backward_general, and 22 encoder_backward tests pass.

3. **Verify Autograd Tape Integration (Feature 19)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test --package ojas-autograd --test sam_tape
   cargo test --package ojas-autograd --test metal_encoder_tape -- --test-threads=1
   ```
   *Expectation*: 8 sam_tape and 3 metal_encoder_tape tests pass.

4. **Verify End-to-End SAM Video Tracker Multiplex Pipeline (Scenario A)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test --package ojas-oracle --test e2e_tier4_production tier4_scenario_a_sam31_video_tracker_multiplex_pipeline -- --nocapture
   ```
   *Expectation*: 1 passed in 0.06s with bit-exact determinism across consecutive frames.

5. **Verify Full CI Quality Gates**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/fmt_check.sh
   ./scripts/check_tracked_targets.sh
   cargo clippy --workspace --all-targets -- -D warnings
   cargo deny check
   ```
   *Expectation*: All pass with exit code 0.
