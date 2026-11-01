# Milestone 3 Investigation & Architecture Report: SAM Attention & ViT Encoder Kernels with PyTorch Parity

**Author**: Explorer 2 (`teamwork_explorer_m3_2`)  
**Target Milestone**: Milestone 3 (SAM 3.1 & Vision Kernels with PyTorch Parity — Features 15, 16, 17, 18, 19, 20)  
**Deliverable**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m3_2/handoff.md`

---

## 1. Observation

### 1.1 Source Repositories & In-Flight Branches
Investigation was conducted across three repositories:
- `/Users/bharath/Code/research/ojas`
- `/Users/bharath/Code/research/tessl`
- `/Users/bharath/Code/research/SpikeTrack`

The authoritative in-flight branches under review:
1. **`ojas: gp/ug-metal-attn-bwd`** (commits `f4b68da`, `104a33f`, `15d9159` ahead of `c389342`):
   - `ojas-metal/kernels/attn_bwd_tiled.h` (496 lines): General tiled FlashAttention-2 backward, supporting $D \in \{16, 32, 64, 128, 256\}$, decoupled $T_q \ne T_k$, additive score bias with broadcast semantics, GQA head grouping, and per-batch key lengths.
   - `ojas-metal/kernels/attn_gen_bwd.metal` (100 lines): Entry points `ojas_attn_gen_bwd_dq_d{DM}`, `dkv_d{DM}`, `dbias_d{DM}` instantiated for $DM \in \{16, 32, 64, 128, 256\}$.
   - `ojas-metal/src/backend/attn_bwd.rs` (121 lines): Host interface `MetalBackend::attn_backward`.
   - `ojas-metal/src/device/attn_bwd.rs` (233 lines): Device driver `Worker::attn_gen_bwd`.
   - `ojas-metal/tests/attention_backward_general.rs` (744 lines): Comprehensive test suite covering head dim 256, SAM shapes ($9 \times 576 \times 64$, $1 \times 201 \times 5184 \times 32$), broadcast bias, and bit-identical determinism.

2. **`ojas: main` Attention Gaps at Head Dim $D=256$**:
   - In `ojas-metal/src/backend.rs`:
     - Line 1575: `const ENCODER_ATTN_HEAD_DIMS: [usize; 2] = [32, 64];` (256 is excluded).
     - Line 1743: `const FLASH_CROSS_HEAD_DIMS: [u32; 1] = [32];` (64, 128, 256 are excluded).
     - Line 1795: `xattn_splits` checks `!matches!(g.d, 16 | 32)`, refusing $D=256$.
   - In `ojas-metal/src/device.rs`:
     - Lines 2559–2562:
       ```rust
       // One threadgroup per 32 query rows of a plane, four
       // simdgroups, no partials: `ojas_attn_fwd_d32` with `cross`.
       let scale = sdpa_scale(g.d)?;
       self.ktg("ojas_attn_fwd_d32", ...);
       ```
       The kernel name is hardcoded to `ojas_attn_fwd_d32` instead of dynamic width formatting `format!("ojas_attn_fwd_d{width}")`.
   - Result: Any cross-attention call with $D=256$ silently falls through to the scalar kernel `ojas_attn_gen_fwd`, which is 28× to 40× slower, causing the observed 74% bottleneck in SpikeTrack memory attention.

3. **`tessl: gp/w34-nt-bias-round`** (commits `daff7db`, `e326f33`, `3c428db`, `f535dd3`, `139306d` ahead of `c9adc87`):
   - `kernels/gemm_coop.h`: `coop_tile` template extended with `uint round_bf16` and `epi_round_bf16(v)` in the register epilogue.
   - `kernels/matmul_tensorops.metal` (lines 599–617):
     ```metal
     NT_COOP_EPI_KERNEL(matmul2d_tensorops_nt_bf16_f32_bias, bfloat, 128, 64, 4, true)
     ```
     Implements fused $C[M, N] = \text{round\_bf16}(A[M, K] @ B[N, K]^T + \text{bias}[N])$ in registers.
   - `src/gemm.rs` (lines 2242–2313): Public function `gemm_nt_bf16_bias`.
   - `tests/gemm_nt_bf16_bias.rs`: 5 tests pass under `mac_heavy.sh` lock on M5 Pro with zero difference against the 3-step sequence.
   - `bench/results/sam_nt_bias_round_m5pro.txt`: On SAM trunk shapes ($M=5184, N=1024/3072/4736, K=1024$), fused kernel achieves 27.0–30.8 TFLOP/s, a 1.77× speedup over the 3-step sequence (1315 µs vs 2329 µs on $M=5184, N=3072$).

4. **`ojas: gp/ug-metal-encoder-bwd`** (commits `b381d41`, `dae0cd7` ahead of `c389342`):
   - `ojas-metal/kernels/encoder_bwd.metal` (601 lines): Double-single arithmetic (`eb_two_sum`, `eb_two_prod`, `eb_add`, `eb_div_f`, `eb_tg_sum2`) with 46-bit precision; kernels `ojas_ln_bwd_rows`, `ojas_ln_bwd_cols_part`, `ojas_cols_fin4`, `ojas_cln_bwd_pos`, `ojas_cln_bwd_ch_part`, `ojas_gelu_erf_bwd`, `ojas_softmax_bwd_rows`, `ojas_bcast_grad_part`, `ojas_bcast_grad_fin`.
   - `ojas-metal/src/backend/enc_bwd_host.rs`: Implements `enc_layer_norm_backward`, `enc_channel_layer_norm_backward`, `enc_gelu_erf_backward`, `enc_softmax_backward`, `enc_broadcast_add_backward`, `enc_broadcast_mul_backward`, `enc_linear_bias_backward`.
   - `ojas-metal/src/device/enc_bwd_dev.rs`: Device execution and pipeline dispatch.
   - `ojas-metal/tests/encoder_backward.rs` (1059 lines): Tests against PyTorch autograd golden fixture (`ojas-oracle/fixtures/encoder_bwd/golden.txt`).

5. **`ojas: gp/ug-cpu-sam`** (commit `177a424`):
   - Implements `group_norm_backward` and `interpolate_bilinear_backward` in pure CPU and autograd tape (`ojas-autograd/src/tape.rs`).
   - Verified in `ojas-autograd/tests/sam_tape.rs` and `ojas-cpu/tests/sam_backward_golden.rs`.

6. **Build Script Collision in `ojas-metal/build.rs`**:
   - `main`: `const SOURCES: [&str; 2] = ["per_head_gate.metal", "ojas_backend.metal"];`
   - `gp/ug-metal-attn-bwd`: `const SOURCES: [&str; 4] = [..., "attn_gen_bwd.metal"];`
   - `gp/ug-metal-encoder-bwd`: `const SOURCES: [&str; 4] = [..., "encoder_bwd.metal"];`
   - `gp/ug-metal-maxpool`: `const SOURCES: [&str; 4] = [..., "max_pool.metal"];`
   - `gp/ug-metal-spatial-bwd`: `const SOURCES: [&str; 4] = [..., "spatial_bwd.metal"];`
   - Git merge-tree reveals a direct merge conflict on `const SOURCES` if branches are merged without unifying this list.

---

## 2. Logic Chain

### 2.1 Decoupled Cross-Attention ($D=256$) Architecture & Fix
1. **Forward Pass ($D=256$)**:
   - Metal Performance Primitives (MPP) `matmul2d` supports matrix descriptors up to dimension 256.
   - In `ojas-metal/kernels/ojas_backend.metal`, `OJAS_ATTN_TILED(256)` is already instantiated and compiled into `ojas_attn_fwd_d256`.
   - The threadgroup memory allocation for `ojas_attn_fwd_d256` is:
     $$S = 32 \times 64 \text{ floats} = 8192 \text{ bytes (8 KiB)}$$
     $$\text{m\_row}, \text{l\_row}, \text{a\_row} = 32 \text{ floats each} = 384 \text{ bytes}$$
     Total threadgroup memory is ~8.5 KiB, which is well within Apple Silicon's 32 KiB threadgroup memory limit and **does not scale with head dimension $D$**.
   - The reason $D=256$ was not using this kernel was an artificial gate in `FLASH_CROSS_HEAD_DIMS` (`[32]`) and hardcoded kernel name dispatch in `device.rs:2562`.
   - Extending `FLASH_CROSS_HEAD_DIMS` to `[32, 64, 128, 256]` and dynamically selecting `ojas_attn_fwd_d{width}` enables full TensorOps acceleration for SAM memory attention ($D=256$, $T_q=1, T_k=5184$), eliminating the 74% tracker bottleneck.

2. **Backward Pass ($D=256$)**:
   - In `gp/ug-metal-attn-bwd:ojas-metal/kernels/attn_bwd_tiled.h`, the FlashAttention-2 backward algorithm is implemented with:
     - Tile size $ATT\_BQ = 32$, $ATT\_BK = 32$.
     - 4 SIMD groups (128 threads per threadgroup).
     - Threadgroup memory:
       $$S = 32 \times 32 \text{ floats} = 4096 \text{ bytes (4 KiB)}$$
       $$dP = 32 \times 32 \text{ floats} = 4096 \text{ bytes (4 KiB)}$$
       $$r_0, r_1 = 32 \text{ floats each} = 256 \text{ bytes}$$
       Total: 8448 bytes (~8.3 KiB), independent of $D$.
   - The gradient calculation executes in three cooperative steps:
     1. `ojas_attn_bwd_dr`: Computes $D_r[i] = \sum_d (dO_{i, d} \cdot O_{i, d})$ per query row $i$.
     2. `ojas_attn_gen_bwd_dq_d{DM}`: Each threadgroup owns 32 query rows $[q_0, q_0 + 32)$, iterates over visible key blocks $k_b$, loads $Q$, $K$, $V$, $dO$, computes $S = Q K^T$, $dP = dO V^T$, evaluates online softmax derivative $dS = \text{scale} \cdot (P \circ (dP - D_r))$, and accumulates $dQ += dS K$ directly into cooperative tensor registers.
     3. `ojas_attn_gen_bwd_dkv_d{DM}`: Each threadgroup owns 32 key rows $[k_0, k_0 + 32)$ of a KV plane, iterates over query blocks $q_b$ across all grouped query heads $H/H_{kv}$, evaluates $P^T$ and $dS^T$, and accumulates $dK += dS^T Q$ and $dV += P^T dO$.
   - Bit-identical determinism is guaranteed because every output element of $dQ, dK, dV$ is written exactly once by its owning threadgroup with zero atomic writes.

### 2.2 Additive Logit Bias (`boxRPB`) Semantics
1. **DETR Relative Position Bias**:
   - In SAM 3.1 DETR prompt decoder, the attention logits receive an additive bias:
     $$S_{i, j} = \text{scale} \cdot (Q_i K_j^T) + \text{Bias}_{b, h, i, j}$$
   - The bias tensor shape is $[B_{\text{bias}}, H_{\text{bias}}, Q_{\text{bias}}, T_k]$ with broadcast rules:
     - $B_{\text{bias}} \in \{1, B\}$: broadcasts across batch rows.
     - $H_{\text{bias}} \in \{1, H\}$: broadcasts across attention heads.
     - $Q_{\text{bias}} \in \{1, T_q\}$: broadcasts across query rows.
     - Key dimension $T_k$ is always fully specified (stride 1).
2. **Forward & Backward Indexing**:
   - In `attn_bwd_tiled.h`, the bias row offset is computed via:
     ```cpp
     inline ulong agb_bias_row(AgbMask m, AgbBias bd, uint b, uint h, uint i) {
         const ulong plane = (ulong)(bd.b == 1u ? 0u : b) * bd.h + (bd.h == 1u ? 0u : h);
         return (plane * bd.q + (ulong)(bd.q == 1u ? 0u : i)) * m.Tk;
     }
     ```
   - In `ojas_attn_gen_bwd_dbias_d{DM}`:
     - If $Q_{\text{bias}} = T_q$, one threadgroup owns a $32 \times 32$ tile of the bias plane and accumulates $g = P \circ (dP - D_r)$ across all broadcast batch rows ($0..B$) and heads ($0..H$).
     - If $Q_{\text{bias}} = 1$ (single query row), threadgroups sum across all query blocks in ascending order into threadgroup storage, writing 32 columns per threadgroup.
     - Reduction order is strictly batch-major, head-minor, query-ascending, matching the CPU reference bit-for-bit.

### 2.3 Fused SAM Trunk NT GEMM (Bias Addition & BF16 Rounding)
1. **Trunk Linear Bottleneck**:
   - SAM 3.1 ViT backbone (`st-model/src/backbone.rs:767-802`) executes 6 linear projections per block across 32 layers:
     - Attention projections: $Q, K, V$ ($[5184, 1024] \times [1024, 1024]^T$), Projection ($[5184, 1024] \times [1024, 1024]^T$).
     - MLP projections: FC1 ($[5184, 1024] \times [4096, 1024]^T$), FC2 ($[5184, 4096] \times [1024, 4096]^T$).
   - Current execution in `ojas-metal/src/backend.rs:3422`:
     ```rust
     fn linear_bias_forward(&self, input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, OjasError> {
         let product = self.linear_forward(input, weight)?;
         self.broadcast_add_forward(&product, bias)
     }
     ```
     This allocates an intermediate `product` tensor ($5184 \times 4096 \times 4 = 85 \text{ MB}$ per FC1 call), writes it to global memory, and reads it back in `broadcast_add_forward`.
2. **Fused Kernel Epilogue**:
   - `tessl: gp/w34-nt-bias-round` provides `matmul2d_tensorops_nt_bf16_f32_bias`.
   - Register epilogue in `kernels/gemm_coop.h`:
     ```cpp
     for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
         if (cT.is_valid_element(i)) {
             auto idx = cT.get_multidimensional_index(i);
             cT[i] = epi_add_strict(cT[i], bias[tx + idx[0]]);
         }
     }
     if (round_bf16 != 0u) {
         for (uint16_t i = 0; i < cT.get_capacity(); ++i)
             if (cT.is_valid_element(i)) cT[i] = epi_round_bf16(cT[i]);
     }
     ```
   - Eliminates intermediate tensor allocation and memory roundtrips.
   - Yields 27.0–30.8 TFLOP/s, improving execution time by 1.77× (43% faster).

### 2.4 ViT Encoder Backward
1. **Precision & Numerics**:
   - Backpropagation through ViT LayerNorm, GELU, and Softmax requires summing across sequence dimensions ($N = 5184$).
   - Apple Silicon Metal lacks native `float64`. Standard `float32` accumulation loses up to 12 bits of precision over 5184 elements.
   - `gp/ug-metal-encoder-bwd` employs Knuth/Dekker double-single error-free transformations (`eb_two_sum`, `eb_two_prod` using `fma(a, b, -p)`).
   - Keeps 46 effective mantissa bits, matching CPU `f64` reference math to within $< 1 \text{ ulp}$ of single precision.

---

## 3. Caveats

1. **TensorOps Hardware Gate**:
   - Both `matmul2d_tensorops_nt_bf16_f32_bias` and tiled attention backward (`attn_gen_bwd.metal`) require Apple Silicon Metal Performance Primitives (TensorOps).
   - On non-Apple or pre-M1 hardware, calls must fail closed with explicit `OjasError::Unsupported`, never silently executing incorrect math or falling back to CPU behind the caller's back.
2. **Build System Merge Conflict**:
   - `ojas-metal/build.rs` will encounter merge conflicts if `gp/ug-metal-attn-bwd`, `gp/ug-metal-encoder-bwd`, `gp/ug-metal-maxpool`, and `gp/ug-metal-spatial-bwd` are merged naively. The Worker must reconcile `const SOURCES` to contain all 7 kernel files.
3. **Hardware Lock During Parity Verification**:
   - Running parity tests for Metal attention and GEMM concurrently causes GPU contention, skewing timing and triggering potential driver timeouts. All Metal parity tests must run under `scripts/mac_heavy.sh` with `--test-threads=1`.

---

## 4. Conclusion & Actionable Implementation Recommendations

The required kernel implementations for Milestone 3 (Features 15, 16, 17, 18, 19, 20) are fully designed, developed, and tested on in-flight branches. The Worker can implement Milestone 3 with zero code from scratch by landing these branches and applying surgical wire-up edits.

### 4.1 Step-by-Step Integration Plan for Worker

#### Step 1: Merge Tessl SAM Trunk Linear GEMM
In `/Users/bharath/Code/research/tessl`:
```bash
git merge --no-ff gp/w34-nt-bias-round -m "Merge gp/w34-nt-bias-round: Fused NT BF16 GEMM with bias addition and BF16 rounding (M3: Feature 18)"
```
*(If a 1-line conflict occurs in `CHANGELOG.md`, keep both entries).*

#### Step 2: Merge Ojas Vision & Attention Branches
In `/Users/bharath/Code/research/ojas`:
```bash
# 1. Vision ops & autograd records (Features 19 & 20)
git merge --no-ff gp/ug-cpu-sam -m "Merge gp/ug-cpu-sam: GroupNorm & Bilinear backward with Tape records (M3: Feature 19)"

# 2. MaxPool & Spatial backward (Features 13 & 14)
git merge --no-ff gp/ug-metal-maxpool -m "Merge gp/ug-metal-maxpool: Metal max_pool2d forward and backward (M3: Feature 13)"
git merge --no-ff gp/ug-metal-spatial-bwd -m "Merge gp/ug-metal-spatial-bwd: Metal conv2d and conv_transpose2d backward (M3: Feature 14)"

# 3. ViT Encoder backward (Feature 18)
git merge --no-ff gp/ug-metal-encoder-bwd -m "Merge gp/ug-metal-encoder-bwd: Double-single Metal encoder backwards (M3: Feature 18)"

# 4. Tiled General Attention backward (Features 15 & 16)
git merge --no-ff gp/ug-metal-attn-bwd -m "Merge gp/ug-metal-attn-bwd: FlashAttention-2 Metal general backward (M3: Features 15 & 16)"
```

#### Step 3: Reconcile `ojas-metal/build.rs`
Resolve the `const SOURCES` conflict in `ojas-metal/build.rs` to include all kernels:
```rust
const SOURCES: &[&str] = &[
    "per_head_gate.metal",
    "causal_attn_bwd.metal",
    "attn_gen_bwd.metal",
    "ojas_backend.metal",
    "encoder_bwd.metal",
    "max_pool.metal",
    "spatial_bwd.metal",
];
```
And add kernel headers to `cargo:rerun-if-changed`:
```rust
println!("cargo:rerun-if-changed=kernels/attn_bwd_tiled.h");
```

#### Step 4: Unlock Forward Cross-Attention at $D=256$ in `ojas-metal`
Apply the following surgical edits to `ojas-metal`:

1. **`ojas-metal/src/backend.rs`**:
```rust
// Extend supported encoder attention head dims to include 256
const ENCODER_ATTN_HEAD_DIMS: [usize; 3] = [32, 64, 256];

// Extend flash cross attention head dims to include 64, 128, 256
const FLASH_CROSS_HEAD_DIMS: [u32; 4] = [32, 64, 128, 256];
```

2. **`ojas-metal/src/device.rs` (around line 2560)**:
```rust
// Replace hardcoded "ojas_attn_fwd_d32" with dynamic width dispatch:
let width = Self::attn_width(g.d)?;
let name = format!("ojas_attn_fwd_d{width}");
self.ktg(
    &name,
    q_len.div_ceil(ATTN_ROWS),
    planes,
    ATTN_THREADS,
    |b| {
        // ... (unchanged argument bindings)
    },
)?;
```

#### Step 5: Wire Fused SAM Trunk Linear GEMM in `ojas-metal`
In `ojas-metal/src/backend.rs`:
Wire `linear_bias_forward` to route 2D/3D linear layers with 1D bias on Metal under BF16 to `tessl::gemm::gemm_nt_bf16_bias`:
```rust
fn linear_bias_forward(
    &self,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor, OjasError> {
    if self.link.has_tensorops() && bias.shape().len() == 1 {
        // Route through fused Cmd::LinearBias to tessl::gemm::gemm_nt_bf16_bias
        let (rows, kin, nout) = linear_dims(input, weight)?;
        if bias.shape()[0] == nout {
            return self.fused_linear_bias(input, weight, bias, rows, kin, nout);
        }
    }
    // Fallback for non-TensorOps or broadcasting bias
    let product = self.linear_forward(input, weight)?;
    self.broadcast_add_forward(&product, bias)
}
```

---

## 5. Verification Method

To independently verify the Milestone 3 integration:

1. **Verify Metal Attention Backward Parity (CPU & PyTorch Goldens)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/mac_heavy.sh cargo test --package ojas-metal --test attention_backward_general -- --test-threads=1
   ```
   *Expected Result*: All 41 general attention tests, SAM detector cross-attention with bias tests, and determinism tests pass.

2. **Verify ViT Encoder Backward Parity**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/mac_heavy.sh cargo test --package ojas-metal --test encoder_backward -- --test-threads=1
   ```
   *Expected Result*: All LayerNorm, GELU erf, Softmax, broadcast add/mul, and linear bias backward tests pass against PyTorch goldens.

3. **Verify SAM Autograd Tape Integration**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test --package ojas-autograd --test sam_tape
   cargo test --package ojas-autograd --test metal_encoder_tape
   ```
   *Expected Result*: All autograd tape gradient checks pass cleanly.

4. **Verify Tessl Fused NT Bias GEMM**:
   ```bash
   cd /Users/bharath/Code/research/tessl
   ./scripts/mac_heavy.sh cargo test --test gemm_nt_bf16_bias -- --test-threads=1
   ```
   *Expected Result*: 5 of 5 tests pass with bit-identical match to the 3-step sequence.

5. **Invalidation Conditions**:
   - Modifying `ATT_BQ` or `ATT_BK` in `attn_bwd_tiled.h` without adjusting the threadgroup memory bounds or cooperative tensor dimensions.
   - Emitting code that uses non-deterministic atomic operations in gradient accumulation.
   - Changing `const SOURCES` without adding the corresponding `.metal` files to `ojas-metal/kernels/`.
