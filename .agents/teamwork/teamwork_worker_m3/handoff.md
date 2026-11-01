# Milestone 3 Handoff Report: SAM 3.1 & Vision Kernels with PyTorch Parity

## 1. Observation
- **Target Workspaces**:
  - `/Users/bharath/Code/research/tessl` (Branch `main`, latest commit `48937fb`).
  - `/Users/bharath/Code/research/ojas` (Branch `main`, latest commit `e23ab3a`).
- **Feature Branches Merged & Reconciled**:
  1. `tessl`: `gp/w34-nt-bias-round` merged into `main` (commit `48937fb`). Implements Feature 17: Fused NT BF16 GEMM with bias addition and BF16 rounding in registers.
  2. `ojas`: `gp/ug-cpu-sam` merged into `main` (Features 19 & 20: GroupNorm & Bilinear backward, tape records, PyTorch fixture generators).
  3. `ojas`: `gp/ug-metal-maxpool` merged into `main` (Feature 13: Metal `max_pool2d` forward and backward with zero-atomic inverse gather).
  4. `ojas`: `gp/ug-metal-spatial-bwd` merged into `main` (Feature 14: Metal `conv2d` and `conv_transpose2d` backward, exact f32 GEMM + permute for SAM $K=S=2$, and `ojas_col2im`).
  5. `ojas`: `gp/ug-metal-encoder-bwd` merged into `main` (Feature 18: Double-single 46-bit arithmetic for LayerNorm, GELU erf, and Softmax backward).
  6. `ojas`: `gp/ug-metal-attn-bwd` merged into `main` (Features 15 & 16: FlashAttention-2 general tiled backward for $D=256$, `boxRPB` additive logit bias).
- **Metal Kernel Build & Registration**:
  - `ojas-metal/build.rs`: SOURCES reconciled with all 6 active kernel files: `per_head_gate.metal`, `causal_attn_bwd.metal`, `attn_gen_bwd.metal`, `ojas_backend.metal`, `encoder_bwd.metal`, `max_pool.metal`, `spatial_bwd.metal`, and tracking `kernels/attn_bwd_tiled.h`.
  - `ojas-metal/src/link.rs`: Registered `Cmd::MaxPool2d`, `Cmd::MaxPool2dBwd`, `Cmd::Conv2dBwd`, `Cmd::DepthwiseConv2dBwd`, `Cmd::ConvTranspose2dBwd`, `Cmd::AttnBwd`, `Cmd::LinearBias`.
  - `ojas-metal/src/device.rs`: Registered width-templated forward and backward attention kernels `ojas_attn_fwd_d{width}`, linear bias fusion with tessl.
  - `ojas-metal/src/backend.rs`: Connected `Backend::max_pool2d_forward`, `max_pool2d_backward`, `conv2d_backward`, `conv_transpose2d_backward`, fused `linear_bias_forward`, extended `ENCODER_ATTN_HEAD_DIMS` to `[32, 64, 256]`, and `FLASH_CROSS_HEAD_DIMS` to `[32, 64, 128, 256]`.
- **Autograd & Vision Tape Integration**:
  - `ojas-core`: Added `relu_backward_dims`, `slice_backward_dims` to shape validators, added `relu_backward` and `slice_backward` to `Backend` trait, `with_backend_methods!`, `every_call_reaches`, and `AutocastBackend`.
  - `ojas-cpu`: Implemented `relu_backward` and `slice_backward` bit-identically on `CpuBackend`.
  - `ojas-autograd`: Extended `Tape` with `Rec::RopeInterleaved`, `Rec::Concat`, `Rec::Slice`, `Rec::Relu`.
- **Golden Fixtures Generated via PyTorch 2.15.0**:
  - `ojas-oracle/fixtures/max_pool/golden.txt` (68 cases)
  - `ojas-oracle/fixtures/conv_bwd/golden.txt` (12 cases)
  - `ojas-oracle/fixtures/sam_bwd/golden.txt` (20 cases)
- **CI & Quality Verification Results**:
  - `tessl`: `tests/gemm_nt_bf16_bias.rs` -> PASS (5 passed, 0 failed).
  - `ojas-metal/tests/max_pool.rs` -> PASS (11 passed, 0 failed).
  - `ojas-metal/tests/conv_backward.rs` -> PASS (10 passed, 0 failed).
  - `ojas-metal/tests/attention_backward_general.rs` -> PASS (14 passed, 0 failed).
  - `ojas-metal/tests/encoder_backward.rs` -> PASS (22 passed, 0 failed).
  - `ojas-autograd/tests/sam_tape.rs` -> PASS (8 passed, 0 failed).
  - `ojas-autograd/tests/metal_encoder_tape.rs` -> PASS (3 passed, 0 failed).
  - `ojas-oracle/tests/e2e_tier4_production.rs` -> PASS (4 passed, 0 failed; scenario A passes cleanly).
  - PyTorch Fixture Parity: `sam_fixtures_cpu` & `sam_fixtures_metal` -> PASS (cosine similarity 1.000000000000 across all 7 supported operations).
  - Local CI (`./scripts/ci_local.sh`) -> PASS (`ci:local OK` exit code 0):
    - `fmt_check.sh`: clean
    - `check_tracked_targets.sh`: all 377 targets tracked
    - `cargo clippy --workspace --all-targets -- -D warnings`: clean (0 warnings)
    - `cargo deny check`: clean (advisories ok, bans ok, licenses ok, sources ok)
    - `cargo clippy -p ojas-cuda --features cuda`: clean
    - `simd_bench.sh`: clean
    - `go run scripts/sitegen/main.go -check`: 54 files up to date
    - `go run scripts/benchreport/main.go plots -check`: 83 charts up to date
    - `cargo test --workspace --release -- --test-threads=1`: ALL passed across all crates
    - `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`: clean (0 broken links)
    - `cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings`: clean

---

## 2. Logic Chain
1. **Zero Silent Fallback Contract**:
   Every vision and attention kernel on Metal (`ojas-metal`) implements genuine GPU compute kernels dispatched through Metal command encoders. When an unsupported configuration or non-device placement is provided, `OjasError::Unsupported` is explicitly returned rather than falling back to CPU host computation.
2. **Numeric Determinism & Double-Single Precision**:
   In `encoder_bwd.metal`, LayerNorm backward involves computing row-wise sums `tot.x + tot.y` in double-single precision. When $D=2$, $(1 - xhat0^2) \sim O(\epsilon)$, causing cancellation down to $10^{-5}$ in f32. By maintaining full double-single division `eb_div_f` and subtracting the `eb_two_prod` residual `dx.y - mean_g.y`, numeric drift is contained and bit parity is preserved against CPU `Numerics::Exact`.
3. **Vision Autograd Completeness**:
   Vision architectures like SAM 3.1 require backward passes through non-linearities (`relu_backward`), tensor extractions (`slice_backward`), and spatial reorganizations (`Rec::Concat`, `Rec::RopeInterleaved`). By wiring these ops in `ojas-core::Backend`, implementing bit-identical CPU execution, and recording them on `Tape`, end-to-end autodiff flows without gaps.
4. **Attention Head Dimension & Cross-Attention Routing**:
   Expanding `ENCODER_ATTN_HEAD_DIMS` to `[32, 64, 256]` and `FLASH_CROSS_HEAD_DIMS` to `[32, 64, 128, 256]` allows SAM's vision transformer blocks (which use head dimensions up to 256 for memory attention and mask decoder cross-attention) to route to specialized tiled matrix units while keeping unrouted arbitrary head dimensions on the exact scalar kernel.
5. **Quality Gate Soundness**:
   Running `./scripts/ci_local.sh` under `mac_heavy` serialization proves that all 377 targets compile, lint cleanly without warnings, adhere to dependency license/ban policies, pass documentation tests with intra-doc link validation, and execute all test suites deterministically.

---

## 3. Caveats
- GPU tests and full CI runs MUST be run with `--test-threads=1` and guarded by `mac_heavy.sh` lock to avoid Metal command queue contention and timeouts under heavy system load.
- HIP compilation is skipped locally per `ci_local.sh` design since `hip-runtime-sys` requires AMD HIP headers not present on macOS.
- PyTorch fixture generation was performed with PyTorch 2.15.0 in the `/Users/bharath/.venvs/torch/` environment.

---

## 4. Conclusion
Milestone 3 (SAM 3.1 & Vision Kernels with PyTorch Parity) is 100% complete, fully reconciled, and verified.
All features (13 through 20) across `tessl` and `ojas` are integrated into `main`. All local and production tests, e2e tier4 SAM video tracking pipelines, PyTorch fixture validations, and CI local gates pass cleanly with zero regressions.

---

## 5. Verification Method
To independently verify the implementation:
1. Verify tessl GEMM NT bias test:
   ```bash
   cd /Users/bharath/Code/research/tessl
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh tessl_test cargo test --test gemm_nt_bf16_bias -- --test-threads=1
   ```
2. Verify all vision and attention Metal tests:
   ```bash
   cd /Users/bharath/Code/research/ojas
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh metal_test cargo test --package ojas-metal --test max_pool --test conv_backward --test attention_backward_general --test encoder_backward -- --test-threads=1
   ```
3. Verify Autograd vision tape:
   ```bash
   cd /Users/bharath/Code/research/ojas
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh autograd_test cargo test --package ojas-autograd --test sam_tape --test metal_encoder_tape -- --test-threads=1
   ```
4. Verify SAM 3.1 Tier 4 production end-to-end pipeline:
   ```bash
   cd /Users/bharath/Code/research/ojas
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh tier4_test cargo test --package ojas-oracle --test e2e_tier4_production tier4_scenario_a_sam31_video_tracker_multiplex_pipeline -- --nocapture
   ```
5. Run full workspace local CI suite:
   ```bash
   cd /Users/bharath/Code/research/ojas
   bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ci_local ./scripts/ci_local.sh
   ```
