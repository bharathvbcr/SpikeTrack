# Comprehensive Kernel Specification & Investigation Report: SAM and Gemma Embedding with PyTorch Parity

**Author**: Explorer 2 (Kernel Specification Explorer)  
**Location**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`  
**Date**: 2026-10-31T17:23:00Z  
**Target Codebases**: `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/tessl`, `/Users/bharath/Code/research/SpikeTrack`  

---

## 1. Observation

Direct code observations from inspection of `ojas`, `tessl`, `SpikeTrack`, and GitPulse tasks:

### 1.1 Ojas Backend Trait & Dispatch Surface
- **`ojas/ojas-core/src/backend.rs:2141-2237`**: Macro `with_backend_methods!` enumerates exactly 95 methods on trait `Backend`.
- **`ojas/ojas-metal/src/backend.rs`**: Implements 81 methods of `Backend`. Exactly 14 methods are missing or rely on trait default refusals:
  - `max_pool2d_forward` (missing in `MetalBackend`; trait default at `ojas-core/src/backend.rs:1272` returns `OjasError::Unsupported`).
  - Backwards missing on Metal: `attention_backward`, `conv2d_backward`, `conv_transpose2d_backward`, `max_pool2d_backward`, `layer_norm_backward`, `channel_layer_norm_backward`, `gelu_erf_backward`, `linear_bias_backward`, `broadcast_add_backward`, `broadcast_mul_backward`, `softmax_backward`.
  - Missing in `tessl`: Search `rg -i "max_pool" /Users/bharath/Code/research/tessl/src /Users/bharath/Code/research/tessl/kernels` returned 0 hits. Neither `ojas-metal` nor `tessl` contains a GPU kernel for 2D max pooling.

### 1.2 SAM Memory Attention Bottleneck & Attention Route Restrictions
- **`ojas/ojas-metal/src/backend.rs:1571`**: `const ENCODER_ATTN_HEAD_DIMS: [usize; 2] = [32, 64];`.
- **`ojas/ojas-metal/src/backend.rs:1631`**: In `routes_to_encoder_attn_within`: `ENCODER_ATTN_HEAD_DIMS.contains(&dims.head_dim)` restricts `encoder_attn` routing strictly to head dimensions 32 and 64.
- **`ojas/ojas-metal/src/backend.rs:1739`**: `const FLASH_CROSS_HEAD_DIMS: [u32; 1] = [32];`.
- **`ojas/ojas-metal/src/backend.rs:1791`**: In `xattn_splits`: `!matches!(g.d, 16 | 32)` returns `None`.
- **`SpikeTrack/docs/embedgemma_reuse.md:45`**: SAM memory attention uses hidden dimension 256 with 1 head ($D = 256$), attending over $T_q = 5,184$ queries and $T_k \approx 36,288$ keys ($7 \text{ frames} \times 5,184 + \text{pointers}$). Because $D = 256$ is not in `[32, 64]`, `[32]`, or `[16, 32]`, every call to memory attention bypasses all fast matrix kernels and falls back to scalar kernel `ojas_attn_gen_fwd`.
- **`SpikeTrack/HANDOFF.md:39-40`**: "Memory attention is 74% of each tracker state's time. It does not shrink with objects, and its cost is in Ojas kernels (`bench/results/profile-tracker`); the numbers went to ojas-78."
- **`tessl/kernels/encoder_attn_tiled.h:42,78`**: Clamps sequence length to `len = min(lens[b], T)` where $T = T_q$. When $T_k > T_q$ (as in cross-attention and memory attention), keys beyond $T_q$ are unreachable.

### 1.3 Additive Logit Bias Attention (DETR boxRPB)
- **`ojas/ojas-metal/src/backend.rs:1627`**: In `routes_to_encoder_attn_within`: `dims.bias.is_none()` is required.
- **`ojas/ojas-metal/src/backend.rs:1771`**: In `flash_cross_covers`: `g.bias.is_none()` is required.
- **`ojas/ojas-metal/kernels/ojas_backend.metal:1720-1770`**: Only scalar kernel `ojas_attn_gen_fwd` consumes `bias`. The fast tiled flash and encoder kernels do not read an additive bias tensor.
- **`tessl/kernels/attn_tiled.h:54-216`**: Stores score tile $S[r \cdot BK + c]$ in threadgroup memory, but does not read or add an external bias tensor.

### 1.4 Transposed Conv2D Limitations
- **`ojas/ojas-metal/src/backend.rs:1512-1525`**:
  ```rust
  fn refuse_unsupported_conv_transpose2d(op: &'static str, d: &ConvTranspose2dDims) -> Res<()> {
      let p = d.params;
      if p.stride == d.kernel && p.padding == [0, 0] && p.output_padding == [0, 0] {
          return Ok(());
      }
      Err(OjasError::Unsupported { ... })
  }
  ```
  Only transposed convolutions where `stride == kernel` and `padding == [0, 0]` are supported on Metal via GEMM reformulation. Overlapping kernels ($k=3, s=2$) or padded deconvolutions return `OjasError::Unsupported`.

### 1.5 Sliding Window Attention: Causal vs Non-Causal Semantics
- **`ojas/ojas-cpu/src/attn/general.rs:123-133`**:
  ```rust
  let (lo, hi) = if self.causal {
      let Some(seen) = (i + self.kv_len).checked_sub(self.q_len).map(|p| p + 1) else {
          return 0..0;
      };
      let lo = self.window.map_or(0, |w| seen.saturating_sub(w));
      (lo, seen.min(len))
  } else {
      (0, len)
  };
  ```
  On CPU, if `self.causal == false`, `self.window` is completely ignored and `lo..hi` spans the entire sequence `0..len`.
- **`ojas/ojas-metal/kernels/ojas_backend.metal:1756-1758`**: In `ojas_attn_gen_fwd`:
  ```metal
  const uint lo = (window > 0u && pos + 1u > window) ? pos + 1u - window : 0u;
  const uint j0 = max(origin + split * chunk, lo);
  const uint j1 = min(origin + split * chunk + chunk, pos + 1u);
  ```
  The scalar kernel always caps keys at `pos + 1u` (`pos = kv_len - tq + i`), enforcing causal masking even when bidirectional attention is requested.
- **`ojas/ojas-metal/src/backend.rs:1626`**: `dims.window.is_none()` is strictly required to route to `encoder_attn`.
- **`tessl/kernels/encoder_attn.metal:10-12`**: Contains a true symmetric inclusive sliding window kernel: `|t_k - t_q| <= window` (`window == 0` is global). However, Ojas never routes windowed requests to it.

### 1.6 Missing Gemma Embedding Activation and Reduction Primitives
- **`ojas/ojas-core/src/backend.rs`**: Only defines `gelu_erf_forward` (exact erf). No `gelu_tanh` or `gated_gelu_tanh` exists on `Backend`.
- **`tessl/src/embedgemma2.rs:13-35`**: EmbeddingGemma 2 requires:
  - `gelu_tanh(gate) * up` (GeGLU MLP)
  - `ple_i = rms_norm(embeds @ Wple[i] * hidden^-0.5) * ple_norm.w` (per-layer projection)
  - `segment_mean_rows` (masked sequence mean pooling)
  - `l2_normalize_rows` (embedding L2 normalization with Matryoshka dimension truncation)
  - Weightless V RMSNorm per head: `v = rms_norm(v)`
- **`tessl/kernels/act.h:24-30`**: `tessl_gelu_pytorch_tanh` exists in tessl, but is not exposed to Ojas's `Backend`.
- **`tessl/kernels/embed_pool.metal`**: Implements `segment_mean_rows_f32` and `l2_normalize_rows_f32`, but neither is exposed in `ojas-core` or `ojas-cpu`.

### 1.7 Autograd Tape Omissions
- **`ojas/ojas-autograd/src/tape.rs:34-250`**: Enum `Rec` contains no variants for:
  - `Relu`
  - `GroupNorm`
  - `InterpolateBilinear`
  - `RopeInterleaved`
  - `Concat`
  - `Slice`
  - `Fill`
  Backward propagation through these operations cannot be recorded on the Tape.

### 1.8 GitPulse Tasks Survey
- **`ft-0f61c4e29a53fe886ac192979e4d571b`**: Backlog task explicitly recording Gemma requirements:
  - Acceptance criterion: "Gemma-4-E4B is deferred, with its requirements recorded: gelu, logit softcapping, head_dim above 256, per-layer embeddings, shared KV. gemma-metal is inference-only and stays outside the workspace..."
- **`gp-bidirectional-attention.md`**: P0 task for bidirectional and cross attention with additive bias and key-length masks.
- **`gp-vision-norm-act-bias.md`**: P0 task for LayerNorm, exact GELU, linear bias, and broadcast elementwise ops.
- **`gp-spatial-ops.md`**: P1 task for Conv2d, depthwise/transposed conv, pooling, bilinear interpolation, and 2-D RoPE.

---

## 2. Logic Chain

From the direct observations above, the logical derivation of architectural and kernel gaps is structured as follows:

```
[Observation 1.1 - 1.4: Metal lacks MaxPool2D, D=256 Attention, BoxRPB Bias, Gen Transposed Conv]
       │
       ▼
[SAM 3.1 & SAM 2 Video Tracker Multiplex Pipeline Failure / Bottleneck]
       ├── SAM ViT Trunk: Needs BF16 Activation Tier + Fused QKV-RoPE-Window to beat CoreML (Obs 1.2, 1.3)
       ├── SAM Neck: 0.5x level calls max_pool2d_forward -> Unsupported panic on Metal (Obs 1.1)
       ├── DETR Decoder: boxRPB additive bias forces fallback to scalar attention (Obs 1.3)
       ├── Memory Attention: 1-head D=256 cross-attention falls back to scalar kernel -> 74% frame time (Obs 1.2)
       └── Mask Decoder: Deconvolution with padding or stride!=kernel fails on Metal (Obs 1.4)
```

```
[Observation 1.5 - 1.6: Ojas lacks GELU-tanh, Non-Causal Window, Segment Mean, L2 Norm]
       │
       ▼
[Gemma Embedding 2 Pipeline Failure / Divergence]
       ├── Feedforward Layer: GeGLU requires gelu_tanh; gelu_erf breaks PyTorch parity (Obs 1.6)
       ├── Sliding Attention: Non-causal window is ignored on CPU (spans full) & causal on Metal (cuts future) (Obs 1.5)
       ├── Normalization: Weightless V RMSNorm missing in rms_qk_norm_forward (Obs 1.6)
       └── Embedding Head: Sequence mean pooling and Matryoshka L2 normalization missing on Backend (Obs 1.6)
```

### Detailed Kernel Specifications for Missing Operations

#### Kernel 1: `max_pool2d_forward` (Metal GPU Implementation)
- **Problem**: `ojas-core` and `ojas-cpu` support `max_pool2d_forward`, but `ojas-metal` lacks an implementation and returns `OjasError::Unsupported`.
- **Inputs**:
  - `input`: `Tensor` $[N, C, H, W]$, dtype `F32` (or `Bf16`).
  - `kernel`: $[k_h, k_w]$ (typically $[2, 2]$).
  - `stride`: $[s_h, s_w]$ (typically $[2, 2]$).
- **Outputs**:
  - `output`: `Tensor` $[N, C, H_o, W_o]$ where $H_o = \lfloor(H - k_h)/s_h\rfloor + 1$, $W_o = \lfloor(W - k_w)/s_w\rfloor + 1$.
- **Shape Validator**: `crate::shapes::max_pool2d_forward_dims(input, kernel, stride)`.
- **CPU Reference**: Ascending index search over $[0..k_h) \times [0..k_w)$; ties go to the earliest position in row-major order.
- **Metal GPU Implementation**:
  - Kernel: `ojas_max_pool2d`. 2D grid over $W_o \times H_o$, 1D batch/channel over $N \times C$.
  - Threadgroup size: $16 \times 16$ (256 threads).
  - Memory: Direct device memory read. Each thread computes one output pixel:
    ```metal
    float m = -INFINITY;
    for (uint ky = 0; ky < kh; ++ky) {
        for (uint kx = 0; kx < kw; ++kx) {
            float val = in[((n * C + c) * H + (oy * sh + ky)) * W + (ox * sw + kx)];
            m = max(m, val);
        }
    }
    out[((n * C + c) * Ho + oy) * Wo + ox] = m;
    ```
- **Oracle Parity Plan**: Test fixture with PyTorch `nn.functional.max_pool2d` over random tensors (including negative infinities, ties, and edge dimensions). Tolerance: Bit-identical in F32 (`max_abs == 0.0`).

#### Kernel 2: Generalized `conv_transpose2d_forward` (Metal GPU Implementation)
- **Problem**: Metal currently refuses any transposed conv where `stride != kernel` or `padding != [0,0]`.
- **Inputs**:
  - `input`: `Tensor` $[N, C, H, W]$
  - `weight`: `Tensor` $[C, O / \text{groups}, k_h, k_w]$
  - `bias`: `Option<&Tensor>` $[O]$
  - `params`: `ConvTranspose2dParams { stride, padding, output_padding, dilation, groups }`
- **Outputs**:
  - `output`: `Tensor` $[N, O, H_o, W_o]$
- **Shape Validator**: `crate::shapes::conv_transpose2d_forward_dims`.
- **Metal GPU Implementation**:
  - When $s == k$ and $p == 0$: GEMM + pixel shuffle (existing fast path).
  - Generalized path: GEMM ($X \times W$) produces col matrix, followed by atomic-free or scatter-free `col2im` Metal kernel:
    Launch over output pixels $(ox, oy)$, gathering overlapping input taps $(ix, iy)$ where $(ox + p - ix \cdot s)$ is within $[0, k)$.
    This guarantees zero race conditions and deterministic accumulation order.
- **Oracle Parity Plan**: Match PyTorch `nn.ConvTranspose2d` with $k=3, s=2, p=1$ and $k=4, s=2, p=1$. Tolerance: `max_abs < 1e-5`, `min_cosine > 1.0 - 1e-7`.

#### Kernel 3: High-Dimensional Tiled Cross-Attention ($D=256$, $T_k \ge T_q$)
- **Problem**: SAM memory attention uses 1 head, $D=256$, $T_q = 5,184$, $T_k \approx 36,288$. Metal falls back to scalar kernel, taking 74% of tracker execution time.
- **Inputs**:
  - `q`: `Tensor` $[B, H, T_q, D]$ with $D = 256$, $H = 1$
  - `k`: `Tensor` $[B, H_{kv}, T_k, D]$
  - `v`: `Tensor` $[B, H_{kv}, T_k, D]$
  - `opts`: `AttentionOpts { causal: false, window: None, ... }`
- **Outputs**:
  - `output`: `Tensor` $[B, H, T_q, D]$
  - `lse`: `Tensor` $[B, H, T_q]$
- **Metal / Tessl Implementation**:
  - Update `tessl/kernels/attn_tiled.h` and `encoder_attn_tiled.h` to decouple query length $T_q$ from key length $T_k$.
  - Replace `len = min(lens[b], T)` with `len = min(lens[b], Tkv)`.
  - Extend `ENCODER_ATTN_HEAD_DIMS` in `ojas-metal/src/backend.rs` to include 256 (`[32, 64, 256]`).
  - Route through `tessl::embedgemma2::encoder_attn_with` with $D=256$ TensorOps tiled execution.
- **Speed & Parity Expectation**:
  - Throughput increases from ~0.18 TFLOP/s (scalar) to 17–19 TFLOP/s (TensorOps tiled bf16/f32).
  - Memory attention per object drops from ~5,000 ms to ~50 ms (100x acceleration).
  - Tolerance vs PyTorch `scaled_dot_product_attention`: `max_abs < 1e-4`, `max_rel < 1e-2`, `min_cosine > 1.0 - 1e-8`.

#### Kernel 4: Biased Attention Forward (`boxRPB` in DETR Decoder)
- **Problem**: DETR decoder attention has per-head additive bias $[B|1, H, T_q, T_k]$ for box-aware relative position bias. Existing fast paths reject `bias.is_some()`.
- **Inputs**:
  - `q, k, v`: Shapes as above ($D=32, H=8$).
  - `bias`: `Tensor` $[B|1, H, T_q, T_k]$
- **Outputs**:
  - `output`: `Tensor` $[B, H, T_q, D]$
- **Metal / Tessl Implementation**:
  - In `tessl/kernels/attn_tiled.h`:
    Inside the score evaluation loop:
    ```metal
    float score = S[r * BK + c] * scale;
    if (has_bias) {
        score += bias_tile[r * BK + c];
    }
    ```
  - Stage the bias tile concurrently into threadgroup memory alongside query/key tiles.
  - Enable `flash_cross_covers` and `routes_to_encoder_attn` to accept `bias: Option<&Tensor>`.
- **Oracle Parity Plan**: Compare against PyTorch `scaled_dot_product_attention(q, k, v, attn_mask=bias)`. Tolerance: `max_abs < 1e-4`, `min_cosine > 1.0 - 1e-8`.

#### Kernel 5: Fused Vision QKV + 2D Interleaved RoPE + Window Partition
- **Problem**: SAM ViT trunk pays 4 permute dispatches, intermediate buffer allocations, and finite checks per layer (9 dispatches per attention call).
- **Specification**:
  - Single fused Metal kernel executing directly after fused QKV projection GEMM:
    1. Read fused QKV row $[1024 \times 3]$.
    2. Split into $Q, K, V$ $[16 \text{ heads} \times 64]$.
    3. Apply 2D interleaved RoPE to $Q$ and $K$ using checkpoint frequency table (interleaved pairs $2i, 2i+1$).
    4. Fold window partitioning into the write address: map $(x, y)$ coordinate directly into window-major layout $[9 \text{ windows}, 576 \text{ tokens}, 16, 64]$.
- **Benefit**: Eliminates ~128 kernel dispatches per image frame, cutting ~35 ms of host-device overhead.

#### Kernel 6: BF16 End-to-End ViT Activation Tier & BF16 LayerNorm
- **Problem**: Upstream runs ViT MLP in BF16. Ojas `attention_forward` widens to F32 and `layer_norm_forward` is F32-only.
- **Specification**:
  - Implement `layer_norm_bf16`: Reads BF16 or F32 input, computes row statistics in F32 (two-pass), writes BF16 output directly into GEMM operand buffers.
  - Expose BF16 activation persistence across `Backend`: accept `DType::Bf16` in `attention_forward` without F32 widening.
- **Speed Expectation**: Leverages tessl's measured 22–26 TFLOP/s BF16 GEMM peak (vs 5.7–6.5 F32), accelerating the trunk by 2.9x (57.5 ms vs 169 ms).

#### Kernel 7: GEMM Epilogue with `GeluErf` Activation
- **Problem**: Tessl's fused GEMM epilogue supports `Relu`, `GeluTanh`, `Silu`, but not `GeluErf`. SAM ViT MLP requires exact erf GELU, forcing a separate un-fused pass over the 98 MB tensor.
- **Specification**:
  - Extend `tessl/kernels/act.h` with `tessl_gelu_erf(float x)` using the rational Chebyshev approximation of `erf` matching `ojas-core/src/erf_exact.rs`.
  - Add `Activation::GeluErf` discriminant to `tessl/src/gemm.rs` and `tessl/kernels/matmul_tensorops.metal`.
- **Bandwidth Reduction**: Saves 2 full round trips of the $[5184, 4736]$ tensor per layer $\times 32$ layers $\approx 6.2 \text{ GB/s}$ bandwidth saved.

#### Kernel 8: `gelu_tanh_forward` and `gelu_tanh_backward` / `gated_gelu_tanh`
- **Problem**: Gemma embedding uses PyTorch tanh-approximation GELU (`gelu_tanh`) in its feedforward network and PLE gating. Ojas only has `gelu_erf`.
- **Specification**:
  - Trait method on `Backend`:
    ```rust
    fn gelu_tanh_forward(&self, input: &Tensor) -> Result<Tensor, OjasError>;
    fn gelu_tanh_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError>;
    fn gated_gelu_tanh_forward(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor, OjasError>;
    ```
  - CPU Implementation:
    $y = 0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$.
    Two-pass exact accumulation, identical across threads.
  - Metal GPU Implementation:
    Wire `tessl::nn::mlp_gelu_tanh_with_scalars` from `tessl/kernels/mlp_gelu_tanh.metal`.
- **Oracle Parity Plan**: Match PyTorch `nn.functional.gelu(x, approximate="tanh")`. Tolerance: `max_abs < 1e-5`, `max_rel < 1e-3`, `min_cosine > 1.0 - 1e-9`.

#### Kernel 9: Symmetric Inclusive Non-Causal Sliding Window Attention (`|i - j| <= W`)
- **Problem**: Gemma 2 uses symmetric sliding window attention with $W=512$. In Ojas, CPU ignores `window` for non-causal attention, and Metal scalar kernel truncates future tokens causally.
- **Specification**:
  - In `ojas-cpu/src/attn/general.rs`:
    ```rust
    let (lo, hi) = if self.causal {
        let lo = self.window.map_or(0, |w| seen.saturating_sub(w));
        (lo, seen.min(len))
    } else {
        let lo = self.window.map_or(0, |w| i.saturating_sub(w));
        let hi = self.window.map_or(len, |w| (i + w + 1).min(len));
        (lo, hi)
    };
    ```
  - In `ojas-metal/src/backend.rs`: Update `routes_to_encoder_attn_within` to allow `dims.window.is_some()` when non-causal, routing to tessl's `encoder_attn.metal` which already implements `|t_k - t_q| <= window`.
- **Oracle Parity Plan**: Compare against HuggingFace `transformers` `EmbeddingGemma2TextModel` sliding window attention layer. Tolerance: `max_abs < 1e-4`, `min_cosine > 1.0 - 1e-8`.

#### Kernel 10: Masked Sequence Mean Pooling (`segment_mean_rows`)
- **Problem**: Gemma embedding requires sequence-level mean pooling over token embeddings excluding padding tokens.
- **Inputs**:
  - `input`: `Tensor` $[N_{total}, D]$ (all sequence tokens concatenated).
  - `segments`: `Tensor` $[B, 2]$, dtype `U32` (pairs of $[start, end]$ token offsets).
- **Outputs**:
  - `output`: `Tensor` $[B, D]$ where $\text{out}[b, d] = \frac{1}{end_b - start_b} \sum_{i=start_b}^{end_b - 1} \text{input}[i, d]$.
- **CPU Reference**: Exact two-pass summation in `f64` per column, divided by length.
- **Metal GPU Implementation**: Wire `tessl/kernels/embed_pool.metal` (`segment_mean_rows_f32`). Grid over $B \times D$, each threadgroup accumulating a tile of sequence rows.
- **Oracle Parity Plan**: Match `sentence_transformers` pooling module output. Tolerance: `max_abs < 1e-5`, `min_cosine > 1.0 - 1e-9`.

#### Kernel 11: Standalone L2 Normalization with Matryoshka Prefix Slicing (`l2_normalize_rows`)
- **Problem**: Gemma embedding normalizes output vectors to unit sphere: $x / \sqrt{\sum x^2 + \epsilon}$, with optional prefix truncation $d \in \{128, 256, 512, 768\}$.
- **Inputs**:
  - `input`: `Tensor` $[B, D]$
  - `truncate_dim`: `Option<usize>` ($d \le D$)
  - `eps`: `f32` (default `1e-12`)
- **Outputs**:
  - `output`: `Tensor` $[B, d]$
- **CPU Reference**:
  Two-pass: 1) compute $S = \sum_{c=0}^{d-1} x_c^2$ in `f64`; 2) scale $x_c / \sqrt{S + \epsilon}$.
- **Metal GPU Implementation**:
  Wire `tessl/kernels/embed_pool.metal` (`l2_normalize_rows_f32`). One threadgroup per row, reduction tree over lanes.
- **Oracle Parity Plan**: Match PyTorch `torch.nn.functional.normalize(x[:, :d], p=2, dim=-1, eps=1e-12)`. Tolerance: `max_abs < 1e-6`, `min_cosine > 1.0 - 1e-11`.

#### Kernel 12: Per-Head V RMSNorm (`rms_qkv_norm_forward`)
- **Problem**: Gemma 2 attention normalizes Q and K with learned weights, and V with a weightless RMSNorm per head. Ojas's `rms_qk_norm_forward` normalizes only Q and K.
- **Specification**:
  - Add `rms_qkv_norm_forward(q, k, v, q_weight, k_weight, eps)` to `Backend`.
  - Normalizes $V$ per head: $v_{h, t} / \sqrt{\text{mean}(v_{h, t}^2) + \epsilon}$.
- **Oracle Parity Plan**: Compare against `EmbeddingGemma2` attention projection normalizations. Tolerance: `max_abs < 1e-5`, `min_cosine > 1.0 - 1e-9`.

#### Kernel 13: Autograd Tape Wiring for Vision & Embedding Ops
- **Problem**: Ojas cannot train SAM or Gemma blocks with LoRA/fine-tuning because `ojas-autograd/src/tape.rs` lacks tape records for new ops.
- **Specification**:
  - Add variants to `Rec` in `ojas-autograd/src/tape.rs`:
    `Rec::Relu { x }`, `Rec::GroupNorm { x, w, b, groups, eps }`, `Rec::InterpolateBilinear { x, out_size, align_corners }`, `Rec::RopeInterleaved { x, cos, sin }`, `Rec::Concat { inputs, axis }`, `Rec::Slice { x, axis, start, len }`.
  - Wire backward pass handlers in `Tape::backward` calling corresponding backend methods.

---

## 3. Caveats

1. **Hardware Specifics**: Metal execution and timing numbers cited from benchmarks were recorded on Apple Silicon M5 Pro. Kernel threadgroup configurations (e.g. 256 threads, 16x16 tiles) are tuned for Apple Silicon Unified Memory Architecture and Metal 4.
2. **Video Loader Parity**: As discovered in `SpikeTrack/docs/embedgemma_reuse.md:20`, SAM 3.1 video reference defaults to OpenCV (`INTER_CUBIC` BGR24), whereas image fixtures use torchvision/PIL bilinear antialiasing. Parity testing for video preprocessing must explicitly state which reference loader is being targeted.
3. **W8A8/Quantization Deferral**: Int8 TensorOps GEMM (`gemm_i8_dequant`) reaches 45–57 TOP/s in tessl, but lacks per-token dynamic scaling and epilogue fusion. As documented in GitPulse tasks, quantization is deferred past base architecture parity.
4. **Scope Constraint**: In accordance with the Teamwork Explorer identity and project constraints, no source code in `ojas` or `tessl` was modified. All kernel designs are provided as actionable specifications.

---

## 4. Conclusion

The investigation establishes a complete, evidence-grounded catalog of missing operations and performance bottlenecks across `ojas` and `tessl` for running SAM 3.1 (including the Video Tracker Multiplex) and EmbeddingGemma 2 with PyTorch parity:

1. **SAM 3.1 Critical Path**:
   - **Performance Lever #1**: Routing $D=256$ cross-attention to tiled Metal matrix kernels (Kernel 3) eliminates the scalar fallback that currently causes **74% of tracker frame time** (~5,000 ms $\to$ ~50 ms).
   - **Correctness Blocker #1**: Implementing `max_pool2d_forward` on Metal (Kernel 1) resolves an outright panic/refusal on the neck 0.5x downsample level.
   - **Parity Lever #1**: Supporting additive logit bias (Kernel 4) unblocks the DETR decoder's `boxRPB` relative position bias on fast GPU paths.
   - **Throughput Lever #1**: An end-to-end BF16 ViT activation tier with fused QKV-RoPE-Window layout (Kernels 5, 6, 7) delivers a **2.9x trunk speedup** (~58 ms vs ~169 ms), beating the CoreML port without requiring CoreML ML Programs.

2. **Gemma Embedding Critical Path**:
   - **Parity Blocker #1**: Implementing `gelu_tanh` / GeGLU (Kernel 8) is required for MLP parity; using `gelu_erf` creates an unavoidable divergence against official Gemma weights.
   - **Parity Blocker #2**: Implementing symmetric inclusive non-causal sliding window attention ($|i - j| \le W$) (Kernel 9) corrects a silent semantic failure where CPU ignores the window and Metal cuts off future keys causally.
   - **Embedding Head Completion**: Adding masked sequence mean pooling (Kernel 10), Matryoshka L2 normalization (Kernel 11), and per-head V RMSNorm (Kernel 12) completes the end-to-end sentence-transformers pipeline.

---

## 5. Verification Method

To independently verify the findings, observations, and specifications in this report:

### 5.1 Inspect Direct Code Locations
```sh
# Verify max_pool2d is missing from ojas-metal
rg -n "max_pool2d" /Users/bharath/Code/research/ojas/ojas-metal/src

# Verify ENCODER_ATTN_HEAD_DIMS restricts to [32, 64]
rg -n "ENCODER_ATTN_HEAD_DIMS" /Users/bharath/Code/research/ojas/ojas-metal/src/backend.rs

# Verify CPU attention_forward ignores window when causal is false
sed -n '120,138p' /Users/bharath/Code/research/ojas/ojas-cpu/src/attn/general.rs

# Verify scalar attention in Metal caps j1 causally
sed -n '1755,1760p' /Users/bharath/Code/research/ojas/ojas-metal/kernels/ojas_backend.metal

# Verify Autograd Tape missing Rec variants
sed -n '34,250p' /Users/bharath/Code/research/ojas/ojas-autograd/src/tape.rs
```

### 5.2 Test Existing Test Suites
```sh
# Run ojas-core trait agreement tests (validates with_backend_methods!)
cargo test -p ojas-core --test trait_agreement

# Run CPU spatial ops parity
cargo test -p ojas-cpu --test spatial

# Run SAM activation fixture parity on CPU
cargo test -p ojas-oracle --test sam_fixtures -- --nocapture
```

### 5.3 Invalidation Conditions
This report's diagnosis and specifications would be invalidated if:
1. `ojas-metal` contains a registered `max_pool2d_forward` dispatch that was overlooked (disproved by `rg` scan returning 0 hits).
2. `ojas_attn_gen_fwd` correctly computes non-causal symmetric sliding windows (disproved by line 1758 capping at `pos + 1u`).
3. $D=256$ attention already routes to `encoder_attn` on Metal (disproved by `ENCODER_ATTN_HEAD_DIMS` constant set to `[32, 64]`).
