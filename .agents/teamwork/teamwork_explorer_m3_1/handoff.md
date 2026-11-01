# SAM Metal Spatial and Pooling Kernels (Features 13 & 14) Investigation & Integration Report

**Core Finding**: The Metal implementations for `max_pool2d` (forward/backward) and `conv_transpose2d` / `conv2d` (forward/backward) already exist in high-quality form on branches `gp/ug-metal-maxpool` and `gp/ug-metal-spatial-bwd` in `/Users/bharath/Code/research/ojas`. They employ zero-atomic gather formulations achieving exact bit-identical parity with CPU reference and PyTorch without allocating auxiliary index buffers. Integrating both branches requires resolving only two trivial conflicts: merging the Metal `build.rs` `SOURCES` list and adding `ConvGrad` alongside `AdamWScalars` in `ojas-metal/src/backend.rs`.

---

## 1. Observation

### 1.1 Branch Commits and File Inventory

#### Branch `gp/ug-metal-maxpool` (Commit `0489575b5e8d6f655cb6ed46c118d73b4f0266d7`):
- `ojas-metal/kernels/max_pool.metal` (131 lines): MSL kernels `ojas_max_pool2d_fwd` and `ojas_max_pool2d_bwd`.
- `ojas-metal/src/backend/pool.rs` (70 lines): Host dispatch, geometry validation (`PoolGeom`), and budget charging.
- `ojas-metal/src/device/pool.rs` (111 lines): Worker thread dispatch, finite checking (`ST_IN`), buffer binding, and kernel encoding.
- `ojas-metal/src/link.rs` (+27 lines): `PoolGeom` struct definition, `Cmd::MaxPool2d`, `Cmd::MaxPool2dBwd`.
- `ojas-metal/src/backend.rs` (+27 lines): `mod pool;`, wiring `Backend::max_pool2d_forward` and `Backend::max_pool2d_backward`.
- `ojas-metal/src/device.rs` (+6 lines): `mod pool;`, `"ojas_max_pool2d_fwd"` and `"ojas_max_pool2d_bwd"` kernel registration.
- `ojas-metal/tests/max_pool.rs` (468 lines): 12 edge cases, SAM neck `[1, 256, 288, 288]` pooling, tie-breaking, signed zeros, offset views, and torch golden comparisons.
- `ojas-oracle/python/max_pool_fixture.py` (122 lines): PyTorch golden generator for `ojas-oracle/fixtures/max_pool/golden.txt`.

#### Branch `gp/ug-metal-spatial-bwd` (Commit `cdb94a55391da99ac7b03f7f60357738b16bf379`):
- `ojas-metal/kernels/spatial_bwd.metal` (183 lines): MSL kernels `ojas_col2im`, `ojas_dwconv_bwd_x`, `ojas_dwconv_bwd_w`, and `ojas_chan_sum`.
- `ojas-metal/src/backend/spatial_bwd.rs` (173 lines): Host dispatch for `conv2d_backward_routed` and `conv_transpose2d_backward_routed`.
- `ojas-metal/src/device/spatial_bwd.rs` (379 lines): Worker device thread execution, GEMM scheduling (`Lane::ExactF32`), status tracking (`ST_IN`, `ST_OUT`).
- `ojas-metal/src/link.rs` (+34 lines): `Cmd::Conv2dBwd`, `Cmd::DepthwiseConv2dBwd`, `Cmd::ConvTranspose2dBwd`.
- `ojas-metal/src/backend.rs` (+45 lines): `mod spatial_bwd;`, wiring `Backend::conv2d_backward` and `Backend::conv_transpose2d_backward`.
- `ojas-metal/src/device.rs` (+29 lines): Registration of `ojas_col2im`, `ojas_dwconv_bwd_x`, `ojas_dwconv_bwd_w`, `ojas_chan_sum`.
- `ojas-metal/tests/conv_backward.rs` (818 lines): SAM patch embed ($k=14, s=14$), $1\times 1$, $3\times 3$ $p=1$, strided $3\times 3$ $s=2$, depthwise $7\times 7$ $p=3$, and transposed $2\times 2$ $s=2$.
- `ojas-oracle/python/conv_backward_fixture.py` (143 lines): PyTorch golden generator for `ojas-oracle/fixtures/conv_bwd/golden.txt`.

---

### 1.2 Feature 13: Metal `max_pool2d_forward` and `max_pool2d_backward` Deep Dive

#### MSL Forward Kernel (`ojas_max_pool2d_fwd`)
In `ojas-metal/kernels/max_pool.metal:54-72`:
```metal
kernel void ojas_max_pool2d_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &h [[buffer(3)]],
    constant uint &w [[buffer(4)]],
    constant uint &oh [[buffer(5)]],
    constant uint &ow [[buffer(6)]],
    constant uint &kh [[buffer(7)]],
    constant uint &kw [[buffer(8)]],
    constant uint &sh [[buffer(9)]],
    constant uint &sw [[buffer(10)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ox = i % ow;
    const uint oy = (i / ow) % oh;
    const uint p = i / (ow * oh);
    device const float *plane = x + (ulong)p * h * w;
    y[i] = plane[ojas_pool_argmax(plane, w, kh, kw, sh, sw, oy, ox)];
}
```
Tie-breaking logic in `ojas_pool_argmax` (`ojas-metal/kernels/max_pool.metal:34-51`):
```metal
inline uint ojas_pool_argmax(device const float *plane, uint w, uint kh, uint kw,
                             uint sh, uint sw, uint oy, uint ox)
{
    uint best = (oy * sh) * w + ox * sw;
    float best_v = plane[best];
    for (uint i = 0u; i < kh; ++i) {
        const uint row = (oy * sh + i) * w + ox * sw;
        for (uint j = 0u; j < kw; ++j) {
            const float v = plane[row + j];
            if (v > best_v) {
                best = row + j;
                best_v = v;
            }
        }
    }
    return best;
}
```
- **Strict Inequality `v > best_v`**: Evaluated in ascending $(i, j)$ order within the window. The first maximum encountered in row-major window order wins ties, bit-identically matching `torch.nn.functional.max_pool2d` and `ojas-cpu/src/spatial.rs`.
- **Signed Zeros**: Under IEEE-754, `+0.0 > -0.0` is false, so if `-0.0` appears first, it wins and retains its negative sign bit.

#### MSL Backward Kernel (`ojas_max_pool2d_bwd`)
In `ojas-metal/kernels/max_pool.metal:88-130`:
```metal
kernel void ojas_max_pool2d_bwd(
    device const float *x [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &h [[buffer(4)]],
    constant uint &w [[buffer(5)]],
    constant uint &oh [[buffer(6)]],
    constant uint &ow [[buffer(7)]],
    constant uint &kh [[buffer(8)]],
    constant uint &kw [[buffer(9)]],
    constant uint &sh [[buffer(10)]],
    constant uint &sw [[buffer(11)]],
    device atomic_uint *st [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ix = i % w;
    const uint iy = (i / w) % h;
    const uint p = i / (w * h);
    device const float *plane = x + (ulong)p * h * w;
    device const float *grads = dy + (ulong)p * oh * ow;
    const uint at = iy * w + ix;
    const uint y_lo = ojas_pool_first_window(iy, kh, sh);
    const uint y_hi = min(iy / sh + 1u, oh);
    const uint x_lo = ojas_pool_first_window(ix, kw, sw);
    const uint x_hi = min(ix / sw + 1u, ow);
    float acc = 0.0f;
    for (uint oy = y_lo; oy < y_hi; ++oy) {
        for (uint ox = x_lo; ox < x_hi; ++ox) {
            if (ojas_pool_argmax(plane, w, kh, kw, sh, sw, oy, ox) == at) {
                acc += grads[oy * ow + ox];
            }
        }
    }
    dx[i] = acc;
    if (!isfinite(acc)) ojas_pool_flag(st, POOL_ST_OUT);
}
```
- **Indices Tracking Elimination via Analytic Inverse Gather**:
  Instead of allocating device memory for an auxiliary `indices` tensor during forward pass ($O(N \cdot C \cdot H_o \cdot W_o)$ bytes) and scattering with atomic adds in backward, `ojas_max_pool2d_bwd` computes the analytic window range $[y_{lo}, y_{hi}) \times [x_{lo}, x_{hi})$ that can overlap pixel $(iy, ix)$.
  `first_window` is computed as:
  $$\text{first\_window}(i, k, s) = \left\lceil \frac{\max(i + 1 - k, 0)}{s} \right\rceil = \frac{\max(i + 1 - k, 0) + s - 1}{s}$$
  The loop gathers gradients strictly in ascending $(oy, ox)$ order: zero atomics, zero memory races, zero extra memory allocation, bit-identical to `ojas-cpu`.

---

### 1.3 Feature 14: Generalized Metal `conv_transpose2d_forward` and `conv_transpose2d_backward` Deep Dive

#### SAM 3.1 Mask Decoder Architecture & Parameters
In SAM 3.1 (and SAM 1/2), feature maps from the vision encoder trunk are upsampled in the mask decoder using two transposed convolutions:
1. `dconv_2x2_0`: `in_channels=1024`, `out_channels=512`, `kernel=[2, 2]`, `stride=[2, 2]`, `padding=[0, 0]`, with bias.
2. `dconv_2x2_1`: `in_channels=512`, `out_channels=256`, `kernel=[2, 2]`, `stride=[2, 2]`, `padding=[0, 0]`, with bias.
In both layers: $K_h = S_h = 2$, $K_w = S_w = 2$, $P = 0$, $OP = 0$.

#### Metal Implementation Paths:

1. **Fast Path ($K == S$ and $P == 0$, used by SAM 3.1 Mask Decoder)**:
   - **Forward** (`ojas-metal/src/device.rs:3182-3257`):
     1. Permute input $X [N, C, H, W] \to [N, H, W, C]$.
     2. Exact-F32 GEMM:
        $$X_{perm} [N \cdot H \cdot W, C] \times W [C, O \cdot K_h \cdot K_w] \to prod [N \cdot H \cdot W, O \cdot K_h \cdot K_w]$$
     3. `ojas_convt_nchw` rearranges $prod$ into $[N, O, H \cdot K_h, W \cdot K_w]$ and adds bias.
   - **Backward** (`ojas-metal/src/device/spatial_bwd.rs:188-293`):
     1. Unfold $dY$ into column format using `ojas_im2col` with kernel = stride on the input grid:
        $$dY [N, O, H \cdot K_h, W \cdot K_w] \xrightarrow{\text{ojas\_im2col}} dcols [N \cdot H \cdot W, O \cdot K_h \cdot K_w]$$
     2. Permute input $X [N, C, H, W] \to X_{perm} [N, H, W, C]$.
     3. Exact-F32 GEMMs:
        $$dX_{perm} = dcols [N \cdot H \cdot W, O \cdot K_h \cdot K_w] \times W^T [O \cdot K_h \cdot K_w, C] \to [N \cdot H \cdot W, C]$$
        $$dW = X_{perm}^T [C, N \cdot H \cdot W] \times dcols [N \cdot H \cdot W, O \cdot K_h \cdot K_w] \to [C, O \cdot K_h \cdot K_w]$$
     4. Permute $dX_{perm}$ back to NCHW $\to dX [N, C, H, W]$.
     5. Compute bias gradient $dB$ via `ojas_chan_sum`:
        $$dB[c] = \sum_{n, p} dY[n, c, p]$$

2. **Generalized Transposed Convolution ($K \ne S$ or $P > 0$)**:
   - Implemented via `ojas_col2im` gather (`ojas-metal/kernels/spatial_bwd.metal:23-57`):
     $$dX[n, c, iy, ix] = \sum_{ky=0}^{kh-1} \sum_{kx=0}^{kw-1} dcols\left[n, \frac{iy+ph-ky}{sh}, \frac{ix+pw-kx}{sw}, c, ky, kx\right]$$
     (evaluating only when $(iy + ph - ky) \pmod{sh} == 0$ and $(ix + pw - kx) \pmod{sw} == 0$).
   - Refusal contract: Any configuration with unsupported parameters fails closed with `OjasError::Unsupported` naming the configuration and parameter values, strictly complying with the house invariant (no silent CPU fallback).

#### Integration with `ojas-core::Backend` and `ojas-autograd::Tape`
- `ojas-core/src/backend.rs` already defines `conv_transpose2d_forward` and `conv_transpose2d_backward`.
- `ojas-autograd/src/tape.rs:821-832` defines `Tape::conv_transpose2d`, and lines 1794-1801 wire reverse-mode autodiff:
  ```rust
  Rec::ConvTranspose2d { x, w, b, params } => {
      let (xv, wv) = (self.values[x].clone(), self.values[w].clone());
      let bv = b.map(|b| self.values[b].clone());
      let g = self.backend
          .conv_transpose2d_backward(&xv, &wv, bv.as_ref(), grad, params)?;
      self.conv_grads(x, w, b, g)
  }
  ```
  Wiring `conv_transpose2d_backward` on `MetalBackend` enables end-to-end autodiff on Metal tensors with zero changes required in `ojas-autograd`.

---

### 1.4 Verification Tool Results
We directly executed the oracle generation scripts against `/Users/bharath/.venvs/torch/bin/python`:
1. `max_pool_fixture.py`:
   Command: `git -C /Users/bharath/Code/research/ojas show gp/ug-metal-maxpool:ojas-oracle/python/max_pool_fixture.py | /Users/bharath/.venvs/torch/bin/python - --out /tmp/test.txt`
   Result: Code 0, `wrote /tmp/test.txt (68 lines)`.
2. `conv_backward_fixture.py`:
   Command: `git -C /Users/bharath/Code/research/ojas show gp/ug-metal-spatial-bwd:ojas-oracle/python/conv_backward_fixture.py | /Users/bharath/.venvs/torch/bin/python - --out /tmp/test.txt`
   Result: Code 0, `wrote /tmp/test.txt: 12 cases`.

---

## 2. Logic Chain

1. **Architecture Alignment**:
   `PROJECT.md` Feature 13 calls for Metal `max_pool2d_forward` & `max_pool2d_backward`. Feature 14 calls for generalized Metal `conv_transpose2d_forward` & `conv_transpose2d_backward`.
2. **Branch Completeness**:
   Branch `gp/ug-metal-maxpool` provides 100% of Feature 13 (MSL kernels, host/worker dispatch, shape validation, 468 lines of unit/golden tests).
   Branch `gp/ug-metal-spatial-bwd` provides 100% of Feature 14 backward and spatial gathers (MSL kernels `ojas_col2im`, `ojas_chan_sum`, worker dispatch, 818 lines of tests).
3. **Merge Conflict Identification**:
   Running `git merge-tree` against current `main` revealed exactly two minor conflict sites:
   - `ojas-metal/build.rs`: `SOURCES` list needs to combine `"max_pool.metal"` and `"spatial_bwd.metal"`.
   - `ojas-metal/src/backend.rs`: imports need both `AdamWScalars` (retained from `main`) and `ConvGrad` (from `gp/ug-metal-spatial-bwd`).
4. **House Rule Compliance**:
   - Exact bit-identical arithmetic: Verified on pooling and depthwise convolutions.
   - No silent fallback: All unsupported shapes/parameters fail closed with `OjasError::Unsupported`.
   - Budget enforcement: All allocations charged to `Budget` prior to Metal buffer allocation.

---

## 3. Caveats

1. **Torch Golden File Generation**:
   The test suites `tests/max_pool.rs` and `tests/conv_backward.rs` expect `ojas-oracle/fixtures/max_pool/golden.txt` and `ojas-oracle/fixtures/conv_bwd/golden.txt` to exist. The Worker must run the python generator scripts once during integration.
2. **Sequential Merge Order**:
   Because `gp/ug-metal-maxpool` and `gp/ug-metal-spatial-bwd` both touch `build.rs`, `link.rs`, `backend.rs`, and `device.rs`, merging them sequentially with clean conflict resolution (or applying the unified diff below) ensures error-free builds.
3. **Hardware Locking for Parity Tests**:
   Tests running Metal GPU dispatches must be executed under the `mac_heavy.sh` lock with `--test-threads=1`.

---

## 4. Conclusion & Implementation Plan for Worker

### 4.1 Implementation Checklist for Worker

- [ ] **Step 1: Copy Production Files into `ojas-metal` and `ojas-oracle`**:
  - `ojas-metal/kernels/max_pool.metal` (from `gp/ug-metal-maxpool`)
  - `ojas-metal/src/backend/pool.rs` (from `gp/ug-metal-maxpool`)
  - `ojas-metal/src/device/pool.rs` (from `gp/ug-metal-maxpool`)
  - `ojas-metal/tests/max_pool.rs` (from `gp/ug-metal-maxpool`)
  - `ojas-oracle/python/max_pool_fixture.py` (from `gp/ug-metal-maxpool`)
  - `ojas-metal/kernels/spatial_bwd.metal` (from `gp/ug-metal-spatial-bwd`)
  - `ojas-metal/src/backend/spatial_bwd.rs` (from `gp/ug-metal-spatial-bwd`)
  - `ojas-metal/src/device/spatial_bwd.rs` (from `gp/ug-metal-spatial-bwd`)
  - `ojas-metal/tests/conv_backward.rs` (from `gp/ug-metal-spatial-bwd`)
  - `ojas-oracle/python/conv_backward_fixture.py` (from `gp/ug-metal-spatial-bwd`)

- [ ] **Step 2: Generate PyTorch Golden Fixtures**:
  ```bash
  /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/max_pool_fixture.py \
      --out ojas-oracle/fixtures/max_pool/golden.txt
  /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/conv_backward_fixture.py \
      --out ojas-oracle/fixtures/conv_bwd/golden.txt
  ```

- [ ] **Step 3: Apply Wiring Diffs**:

#### Diff 1: `ojas-metal/build.rs`
```diff
--- a/ojas-metal/build.rs
+++ b/ojas-metal/build.rs
@@ -10,7 +10,13 @@ use std::process::Command;
 
-const SOURCES: [&str; 2] = ["per_head_gate.metal", "ojas_backend.metal"];
+const SOURCES: [&str; 4] = [
+    "per_head_gate.metal",
+    "ojas_backend.metal",
+    "max_pool.metal",
+    "spatial_bwd.metal",
+];
 
 fn main() {
```

#### Diff 2: `ojas-metal/src/link.rs`
```diff
--- a/ojas-metal/src/link.rs
+++ b/ojas-metal/src/link.rs
@@ -435,6 +435,17 @@ pub(crate) struct BilinearGeom {
     pub align_corners: bool,
 }
 
+/// Max pooling of `planes` `[h, w]` images to `[oh, ow]` with windows of
+/// `kernel = [kh, kw]` at `stride = [sh, sw]` and no padding.
+#[derive(Clone, Copy, Debug)]
+pub(crate) struct PoolGeom {
+    pub planes: u32,
+    pub h: u32,
+    pub w: u32,
+    pub oh: u32,
+    pub ow: u32,
+    pub kernel: [u32; 2],
+    pub stride: [u32; 2],
+}
+
 /// A convolution as GEMM, validated and within the kernels' limits:
@@ -776,6 +787,36 @@ pub(crate) enum Cmd {
         x: Arg,
         geom: BilinearGeom,
     },
+    /// `max_pool2d_forward` on `ojas_max_pool2d_fwd`.
+    MaxPool2d {
+        x: Arg,
+        geom: PoolGeom,
+    },
+    /// `max_pool2d_backward` on `ojas_max_pool2d_bwd`.
+    MaxPool2dBwd {
+        x: Arg,
+        dy: Arg,
+        geom: PoolGeom,
+    },
+    /// `conv2d_backward` with one group.
+    Conv2dBwd {
+        op: &'static str,
+        x: Arg,
+        w: Arg,
+        bias: Option<Arg>,
+        gy: Arg,
+        geom: ConvGeom,
+    },
+    /// `conv2d_backward` for depthwise convolutions.
+    DepthwiseConv2dBwd {
+        op: &'static str,
+        x: Arg,
+        w: Arg,
+        bias: Option<Arg>,
+        gy: Arg,
+        geom: ConvGeom,
+    },
+    /// `conv_transpose2d_backward` for kernel == stride.
+    ConvTranspose2dBwd {
+        op: &'static str,
+        x: Arg,
+        w: Arg,
+        bias: Option<Arg>,
+        gy: Arg,
+        geom: ConvGeom,
+    },
     /// `[q, k, v, output, lse, grad_output]`; replies with dQ, dK, dV.
```

#### Diff 3: `ojas-metal/src/device.rs`
```diff
--- a/ojas-metal/src/device.rs
+++ b/ojas-metal/src/device.rs
@@ -45,6 +45,8 @@ use crate::permute::{self, Plan};
 
+mod pool;
+mod spatial_bwd;
+
 const KERNELS: &[&str] = &[
     ...
     "ojas_bilinear",
+    "ojas_max_pool2d_fwd",
+    "ojas_max_pool2d_bwd",
+    "ojas_col2im",
+    "ojas_dwconv_bwd_x",
+    "ojas_dwconv_bwd_w",
+    "ojas_chan_sum",
     "ojas_concat_part",
     ...
 ];
 
 impl Worker {
     ...
             Cmd::Bilinear { x, geom } => self.bilinear(x, geom),
+            Cmd::MaxPool2d { x, geom } => self.max_pool2d(x, geom),
+            Cmd::MaxPool2dBwd { x, dy, geom } => self.max_pool2d_bwd(x, dy, geom),
+            Cmd::Conv2dBwd { op, x, w, bias, gy, geom } => {
+                self.conv2d_bwd(op, [x, w, gy], bias, geom)
+            }
+            Cmd::DepthwiseConv2dBwd { op, x, w, bias, gy, geom } => {
+                self.depthwise_conv2d_bwd(op, [x, w, gy], bias, geom)
+            }
+            Cmd::ConvTranspose2dBwd { op, x, w, bias, gy, geom } => {
+                self.conv_transpose2d_bwd(op, [x, w, gy], bias, geom)
+            }
```

#### Diff 4: `ojas-metal/src/backend.rs`
```diff
--- a/ojas-metal/src/backend.rs
+++ b/ojas-metal/src/backend.rs
@@ -104,8 +104,8 @@ use ojas_core::{
     scale_forward_dims, sigmoid_backward_dims, sigmoid_forward_dims, silu_backward_dims,
     silu_forward_dims, slice_dims, softmax_forward_dims, topk_rows_dims,
     value_residual_blend_backward_dims, value_residual_blend_forward_dims, AdamWConfig,
-    AdamWScalars, AttentionDims, AttentionOpts, Backend, BackendId, BroadcastDims, Budget,
-    BudgetTrimHandler, CeChunk, CeGrads, Conv1dDims, Conv2dDims, Conv2dParams, ConvTranspose2dDims,
+    AdamWScalars, AttentionDims, AttentionOpts, Backend, BackendId, BroadcastDims, Budget,
+    BudgetTrimHandler, CeChunk, CeGrads, Conv1dDims, Conv2dDims, Conv2dParams, ConvGrad, ConvTranspose2dDims,
     ConvTranspose2dParams, DType, DeviceBuffer, GateDims, GatedRmsGrad, GdnDecayGrad, GdnDims,
     GdnForward, GdnGrad, GdnInputs, InterleavedRopeDims, LinearCe, MuonNs5Config, Ns5Precision,
     Numerics, OjasError, OptimizerKind, PartialRopeDims, PerHeadGateGrad, Reservation, RmsDims,
@@ -120,6 +120,8 @@ use crate::link::{
 use crate::permute;
 
+mod pool;
+mod spatial_bwd;
+
 impl Backend for MetalBackend {
     ...
+    fn max_pool2d_forward(
+        &self,
+        input: &Tensor,
+        kernel: [usize; 2],
+        stride: [usize; 2],
+    ) -> Result<Tensor, OjasError> {
+        self.pool_forward(input, kernel, stride)
+    }
+
+    fn max_pool2d_backward(
+        &self,
+        input: &Tensor,
+        grad_output: &Tensor,
+        kernel: [usize; 2],
+        stride: [usize; 2],
+    ) -> Result<Tensor, OjasError> {
+        self.pool_backward(input, grad_output, kernel, stride)
+    }
+
+    fn conv2d_backward(
+        &self,
+        input: &Tensor,
+        weight: &Tensor,
+        bias: Option<&Tensor>,
+        grad_output: &Tensor,
+        params: Conv2dParams,
+    ) -> Result<ConvGrad, OjasError> {
+        self.conv2d_backward_routed(input, weight, bias, grad_output, params)
+    }
+
+    fn conv_transpose2d_backward(
+        &self,
+        input: &Tensor,
+        weight: &Tensor,
+        bias: Option<&Tensor>,
+        grad_output: &Tensor,
+        params: ConvTranspose2dParams,
+    ) -> Result<ConvGrad, OjasError> {
+        self.conv_transpose2d_backward_routed(input, weight, bias, grad_output, params)
+    }
```

---

## 5. Verification Method

To independently verify the integration:

1. **Verify Metal Compilation**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo check -p ojas-metal --features metal
   ```
   *Expected outcome*: Clean compilation with 0 errors.

2. **Run MaxPool Unit and Parity Test Suite**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test --test max_pool --features metal -- --test-threads=1
   ```
   *Expected outcome*: All tests pass, including `sam_neck_shape_matches_cpu`, `ties_go_to_the_first_maximum_as_on_the_cpu`, and `both_backends_match_the_torch_golden_bit_for_bit`.

3. **Run Conv Backward & Spatial Test Suite**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test --test conv_backward --features metal -- --test-threads=1
   ```
   *Expected outcome*: All SAM cases pass within tolerance ($10^{-5}$) against CPU and PyTorch goldens.

4. **Verify Autograd Tape Integration**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test -p ojas-autograd --test spatial_tape
   ```
   *Expected outcome*: Tape gradient passes for conv2d, conv_transpose2d, and max_pool2d pass cleanly.

5. **Invalidation Conditions**:
   - If `ojas-metal/src/backend.rs` is modified to remove `AdamWScalars` or change `ConvGrad` signatures.
   - If Xcode SDK paths are altered such that `xcrun metal` cannot locate `metal4.0`.
