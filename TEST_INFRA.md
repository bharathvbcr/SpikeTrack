# TEST_INFRA.md — Comprehensive 4-Tier E2E Test Infrastructure & Test Plan

**Author**: Teamwork E2E Test Writer (`teamwork_test_writer`)  
**Target Repositories**: `/Users/bharath/Code/research/SpikeTrack`, `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/tessl`  
**Governing Documents**: `PROJECT.md`, `ORIGINAL_REQUEST.md`, `AGENTS.md`  
**Status**: APPROVED & READY FOR EXECUTION  

---

## 1. Testing Philosophy & Core Architecture

The Ojas, Tessl, and SpikeTrack test infrastructure is designed around **opaque-box, requirement-driven, mathematically authoritative verification**. All test suites satisfy the following core principles:

1. **Deterministic Oracles & Independent Expected Values**:
   - Every test compares against an authoritative external oracle: PyTorch 2.14 / HuggingFace reference models, offline reference fixtures (`.safetensors` in `fixtures/`), or closed-form analytical mathematical properties.
   - Tests assert against pinned tolerance structures ([`Tolerance`]), never circular or tautological self-comparisons.
2. **"No Silent Fallback" Invariant Enforcement**:
   - The framework enforces fail-closed placement checks.
   - Operations dispatched on a GPU backend (`BackendId::Metal`) MUST return explicit errors (`OjasError::Unsupported` or `OjasError::Placement`) if a kernel is missing or incompatible. It is strictly forbidden to fall back to CPU execution behind the caller's back.
   - CPU backend routines refuse GPU tensors with `OjasError::Placement`.
3. **Exact vs. Fast Numerics Contract**:
   - Under `Numerics::Exact`, operations on `CpuBackend` run sequentially or with deterministic row-level partitioning, producing bit-identical IEEE 754 results regardless of thread count, batch size, or host hardware.
   - Under `Numerics::Fast` and GPU backends, results are held to strict numerical tolerances covering maximum absolute error ($\le 10^{-4}$ to $10^{-6}$), relative error ($\le 10^{-2}$ to $10^{-3}$), and cosine similarity ($\ge 1.0 - 10^{-8}$).
4. **Unified Memory & Root Budget Invariant**:
   - All memory allocations across CPU host leases and Tessl GPU `BufferPool` allocations are bound to a single authoritative root ledger: `ojas_core::Budget`.
   - Exceeding the assigned budget returns `OjasError::CapacityExceeded` without triggering process crashes or memory leaks.
5. **Progressive Testability**:
   - Test suites are organized in progressive layers. Unit shape validators run sub-millisecond without hardware requirements; isolated op fixtures verify single kernel correctness; and end-to-end integration suites evaluate complete real-world pipelines.

---

## 2. 4-Tier E2E Test Plan Specification

```
+-----------------------------------------------------------------------------------------------+
|                                      4-TIER E2E TEST MATRIX                                   |
+-----------------------------------------------------------------------------------------------+
|  TIER 1: Feature Coverage (>=5 test cases per feature in isolation)                            |
|          - SAM 3.1: RoPE, Window/Global Attn, LN/LN2d, Conv2d/TConv2d, GN, Cross-Attn, BoxRPB |
|          - Gemma: GeGLU (gelu_tanh), Non-Causal Window, Segment Mean, L2 Norm, Head V RMSNorm  |
|          - Shared Memory: Zero-copy UMA, Timeline Fences, Placement Refusals, Budget Ledger    |
+-----------------------------------------------------------------------------------------------+
|  TIER 2: Boundary, Corner & Adversarial Robustness (>=5 test cases per feature)               |
|          - Empty tensors (0 tokens, 0 batch), unit dimensions, extreme aspect ratios           |
|          - Strict NaN / Inf / non-finite rejection across all operands before compute         |
|          - Budget exhaustion & memory pool pressure trim ladder                                |
+-----------------------------------------------------------------------------------------------+
|  TIER 3: Pairwise Combinatorial Interactions                                                  |
|          - Cross-feature interactions: RoPE + Windowed Attention, Conv2d + LayerNorm2d,       |
|            GeGLU + Segment Mean Pooling, MappedBuffer + Timeline Barrier Wait                  |
+-----------------------------------------------------------------------------------------------+
|  TIER 4: End-to-End Real-World Production Workloads                                           |
|          - Scenario A: SAM 3.1 Video Tracker Frame Multiplex Pipeline (ViT -> Neck -> Memory)  |
|          - Scenario B: Gemma Embedding Inference Pipeline (Token -> Trans -> Pool -> L2 Norm)  |
+-----------------------------------------------------------------------------------------------+
```

---

### Tier 1 — Core Feature Coverage (>= 5 Cases per Feature)

Every feature in the `PROJECT.md` scope carries at least 5 isolated, independent test cases verifying its primary execution paths against PyTorch or exact analytical formulas.

#### Feature 1: 2D Axial RoPE Tables & Interleaved Application
- **Test 1.1**: ViT Block 0 local window table generation ($[24, 24]$, scale $1.0$, theta $10,000$).
- **Test 1.2**: ViT Block 7 global trunk table generation ($[72, 72]$, scale $24/72$, theta $10,000$).
- **Test 1.3**: Interleaved rotation on Query tensor ($[B, H, T, D]$ where $D=64$).
- **Test 1.4**: Interleaved rotation on Key tensor ($[B, H_{kv}, T_k, D]$).
- **Test 1.5**: Inversion property: applying RoPE with negative frequencies restores original coordinates within `ROPE_TOL`.

#### Feature 2: SAM 3.1 Windowed & Global Attention
- **Test 2.1**: Local window attention ($9 \text{ windows}, 576 \text{ tokens}, 16 \text{ heads}, D=64$).
- **Test 2.2**: Full global attention ($5,184 \text{ tokens}, 16 \text{ heads}, D=64$).
- **Test 2.3**: Key-length padding mask attention with variable sequence lengths.
- **Test 2.4**: Attention softmax scaling check ($\text{scale} = 1/\sqrt{64} = 0.125$).
- **Test 2.5**: Attention with Log-Sum-Exp (LSE) auxiliary output verification against PyTorch.

#### Feature 3: LayerNorm & LayerNorm2d
- **Test 3.1**: ViT 1D LayerNorm over hidden dimension ($D=1024$, eps $10^{-6}$).
- **Test 3.2**: FPN Neck 2D LayerNorm over channel dimension ($C=256$, $[N, C, H, W]$).
- **Test 3.3**: Affine parameters test: scaling with learned weight $\gamma$ and bias $\beta$.
- **Test 3.4**: Two-pass variance numerical stability under large scalar mean offsets ($x + 10^5$).
- **Test 3.5**: BF16 activation input with F32 reduction and BF16 output persistence.

#### Feature 4: Conv2d (im2col + GEMM) & Generalized Transposed Conv2d
- **Test 4.1**: SAM patch embedding Conv2d ($k=14, s=14, p=0$, in=3, out=1024).
- **Test 4.2**: FPN neck conv 3x3 ($k=3, s=1, p=1$, in=256, out=256).
- **Test 4.3**: Transposed Conv2d stride 2, no padding ($k=2, s=2, p=0$).
- **Test 4.4**: Transposed Conv2d overlapping kernel with padding ($k=3, s=2, p=1$).
- **Test 4.5**: Transposed Conv2d with output padding ($k=4, s=2, p=1, \text{out\_pad}=1$).

#### Feature 5: GroupNorm & Bilinear Interpolation
- **Test 5.1**: Pixel decoder GroupNorm ($G=32, C=256$, eps $10^{-5}$).
- **Test 5.2**: Pixel decoder GroupNorm ($G=32, C=128$, eps $10^{-5}$).
- **Test 5.3**: Bilinear 2x upsampling with `align_corners = false` ($144 \times 144 \to 288 \times 288$).
- **Test 5.4**: Bilinear upsampling with `align_corners = true`.
- **Test 5.5**: Backward propagation through GroupNorm and Bilinear interpolation on autograd `Tape`.

#### Feature 6: High-Dimensional Memory Cross-Attention ($D=256, T_k \gg T_q$)
- **Test 6.1**: SAM memory cross-attention single object ($B=1, H=1, T_q=5,184, T_k=36,288, D=256$).
- **Test 6.2**: SAM memory cross-attention multi-object batch ($B=4, H=1, T_q=5,184, T_k=10,368, D=256$).
- **Test 6.3**: Memory cross-attention with decoupled query/key sequence lengths ($T_q < T_k$).
- **Test 6.4**: Metal TensorOps tiling verification ensuring no scalar kernel fallback occurs.
- **Test 6.5**: Numerical parity against PyTorch `scaled_dot_product_attention` within `ATTENTION_TOL`.

#### Feature 7: Additive Logit Bias Attention (`boxRPB` in DETR Decoder)
- **Test 7.1**: DETR decoder attention with head-specific relative position bias $[1, 8, 100, 100]$.
- **Test 7.2**: Broadcastable additive bias $[1, 1, T_q, T_k]$ across multiple heads.
- **Test 7.3**: Negative infinity masked bias elements (hard causal/positional exclusion).
- **Test 7.4**: Metal tiled flash attention path consuming bias tile into threadgroup memory.
- **Test 7.5**: Parity against PyTorch `scaled_dot_product_attention(q, k, v, attn_mask=bias)`.

#### Feature 8: 2D Max Pooling (`max_pool2d_forward`)
- **Test 8.1**: Standard $2 \times 2$ max pool with stride $2$ over $[1, 256, 144, 144]$.
- **Test 8.2**: Overlapping $3 \times 3$ max pool with stride $2$ and padding $1$.
- **Test 8.3**: Tie-breaking verification: deterministic earliest row-major position selection on equal maximums.
- **Test 8.4**: All-negative tensor verification ensuring negative maxima are preserved.
- **Test 8.5**: Metal GPU execution vs CPU reference bit-identical result in F32.

#### Feature 9: Gemma Embedding `gelu_tanh` / GeGLU Activation
- **Test 9.1**: Pointwise `gelu_tanh` forward against PyTorch `approximate="tanh"` formula.
- **Test 9.2**: `gelu_tanh_backward` derivative accuracy against central differences.
- **Test 9.3**: Fused `gated_gelu_tanh(gate, up)` evaluating $\text{gelu\_tanh}(\text{gate}) \times \text{up}$.
- **Test 9.4**: Monotonicity and asymptotic behavior at large negative ($x < -10 \to 0$) and positive ($x > 10 \to x$) values.
- **Test 9.5**: Parity check across 65,536 random values in $[-5.0, 5.0]$ meeting `GELU_TOL`.

#### Feature 10: Symmetric Non-Causal Sliding Window Attention ($|i-j| \le W$)
- **Test 10.1**: Symmetric sliding window $W=512$ with full sequence length $T=2048$ (bidirectional).
- **Test 10.2**: Edge token tokens attending to exact $[0, i + W]$ and $[i - W, T-1]$ spans.
- **Test 10.3**: Short sequence boundary where $T < W$ (degenerating to full dense attention).
- **Test 10.4**: Metal GPU execution routing to `encoder_attn` sliding window kernel.
- **Test 10.5**: PyTorch parity against HuggingFace `EmbeddingGemma2` non-causal attention layer.

#### Feature 11: Masked Sequence Mean Pooling (`segment_mean_rows`)
- **Test 11.1**: Uniform sequence lengths (batch size 4, length 128, dim 768).
- **Test 11.2**: Highly variable sequence lengths with padding masks ($[10, 50, 256, 4]$ tokens).
- **Test 11.3**: Single-token sequences ($start_b + 1 = end_b$).
- **Test 11.4**: Metal GPU parallel reduction tree vs CPU two-pass f64 reference.
- **Test 11.5**: Cosine similarity $\ge 1.0 - 10^{-9}$ against `sentence_transformers` pooling.

#### Feature 12: Matryoshka Prefix L2 Normalization (`l2_normalize_rows`)
- **Test 12.1**: Full dimension normalization ($D=768$, unit norm output $\|y\|_2 = 1.0 \pm 10^{-6}$).
- **Test 12.2**: Matryoshka dimension truncation to $d=512$.
- **Test 12.3**: Matryoshka dimension truncation to $d=256$.
- **Test 12.4**: Matryoshka dimension truncation to $d=128$.
- **Test 12.5**: Epsilon stabilization with near-zero input rows ($\|x\| \approx 0$).

#### Feature 13: Per-Head V RMSNorm (`rms_qkv_norm_forward`)
- **Test 13.1**: Weightless RMSNorm on V per head ($H=8, D=64$, eps $10^{-6}$).
- **Test 13.2**: Simultaneous learned Q-norm, learned K-norm, and weightless V-norm.
- **Test 13.3**: Orthogonal head scaling stability.
- **Test 13.4**: Metal kernel fusion verifying no intermediate tensor roundtrips to VRAM.
- **Test 13.5**: Parity against HuggingFace Gemma 2 attention normalization.

#### Feature 14: Shared Memory Architecture (Zero-Copy UMA & HAL Traits)
- **Test 14.1**: `MappedBuffer` implementation validation on `SharedBuf` (checking `host_ptr` and `gpu_address`).
- **Test 14.2**: Timeline fence monotonically increasing and waiting on command completion.
- **Test 14.3**: Zero-copy `to_host()` on Apple Silicon verifying identical underlying physical pointer without memcpy.
- **Test 14.4**: Zero-copy `upload()` of mapped buffer into GPU tensor without re-allocation.
- **Test 14.5**: Thread-safe concurrent host lease acquisition (`HostLease`) across multiple reader threads.

#### Feature 15: No Silent Fallback & Placement Invariants
- **Test 15.1**: CPU backend calling a GPU tensor returns `OjasError::Placement`.
- **Test 15.2**: Metal backend calling an op with unsupported dimensions returns `OjasError::Unsupported` rather than host compute.
- **Test 15.3**: Missing kernel invocation explicitly fails closed.
- **Test 15.4**: Tensor host access on un-downloaded device tensor returns `OjasError::Placement`.
- **Test 15.5**: Multi-device graph prevents implicit placement transfers.

#### Feature 16: Budget Ledger Enforcement & Memory Trim
- **Test 16.1**: Root `Budget::try_reserve` accurately accounts for host allocations.
- **Test 16.2**: Tessl `BufferPool` memory reservations charged against child budget.
- **Test 16.3**: Exceeding total budget returns `OjasError::CapacityExceeded`.
- **Test 16.4**: Cache trim ladder clears freed buffers upon memory pressure signal.
- **Test 16.5**: Zero memory leaks across 1,000 allocate-free cycles in `BufferPool`.

---

### Tier 2 — Boundary, Corner & Adversarial Robustness (>= 5 Cases per Feature)

Tier 2 attacks operations with pathological, degenerate, and malicious boundary inputs:

1. **Zero & Degenerate Dimensions**:
   - Zero sequence tokens ($T=0$ query or key).
   - Zero batch dimension ($B=0$).
   - Unit dimensions ($1 \times 1 \times 1 \times 1$).
   - Prime number shapes and non-tile-aligned dimensions ($T=37, D=43$).
   - Segment pooling with empty span ($start_b == end_b$).
2. **Extreme Dimension & Resource Bounds**:
   - Giant sequence cross-attention ($T_k = 65,536$).
   - Hidden dimensions exceeding standard tiles ($D = 4096$).
   - Conv2d where kernel size exceeds image size ($k > H$).
   - Transposed Conv2d with asymmetric padding larger than kernel.
   - Matryoshka truncation where $d > D$ (refusal check).
3. **Strict Non-Finite (NaN / Inf / Denormal) Rejection**:
   - Single NaN in input tensor must cause immediate refusal with `OjasError::NonFinite`.
   - Positive and negative infinities in weights or activations refused.
   - Non-finite check must occur *before* GPU kernel dispatch or memory reservation.
   - Denormal float handling without silent zero flush divergence.
   - Softmax inputs with large values ($\ge 10^4$) preventing overflow through numerical stability checks.
4. **Adversarial Memory & Budget Stress**:
   - Allocation request of $2^{60}$ bytes fails cleanly with `OjasError::CapacityExceeded` without integer overflow.
   - Allocation of exactly remaining budget bytes succeeds; subsequent 1-byte request fails cleanly.
   - Rapid allocation and deallocation under multi-threaded contention (16 threads).
   - Dropping `Tensor` drops `MappedBuffer` immediately, returning capacity to root `Budget`.
   - Device lost emulation returns typed `AllocError::DeviceLost`.

---

### Tier 3 — Pairwise Combinatorial Interactions

Tier 3 executes end-to-end composite layers combining pairs of features to ensure seams, memory layouts, and strides integrate seamlessly:

1. **2D Axial RoPE + Windowed Attention**:
   - Applies 2D RoPE to Q and K, partitions into 9 windows, dispatches windowed attention, and unpartitions back to spatial layout. Verifies stride handling and coordinate preservation.
2. **Conv2d Patch Embed + LayerNorm2d**:
   - 3-channel input passes through $14 \times 14$ strided convolution, followed by 2D LayerNorm over the 1024 channels. Verifies contiguous stride assumptions between im2col output and channel norm.
3. **GeGLU Activation + Sequence Mean Pooling**:
   - Gemma MLP feedforward ($\text{GeGLU}(x W_{gate}) W_{down}$) followed immediately by masked token sequence pooling. Verifies elementwise non-linear output piping into reduction buffers.
4. **Transposed Conv2d + MaxPool2d**:
   - Deconvolution upsamples $72 \times 72 \to 144 \times 144$, followed by $2 \times 2$ max pooling down to $72 \times 72$. Verifies spatial geometry conservation and boundary pixel padding.
5. **Zero-Copy MappedBuffer + Timeline Barrier**:
   - Metal kernel writes into `SharedBuf`; CPU acquires `HostLease` after timeline wait, reads bytes without copy, modifies mask, and re-submits to GPU. Verifies coherence and race-free execution.
6. **Fused QKV Projection + 2D RoPE + Window Partition**:
   - GEMM output row is sliced into Q, K, V, rotated via RoPE, and partitioned into window memory in a single pipeline.
7. **Additive Logit Bias + Symmetric Sliding Window Attention**:
   - Sliding window mask ($|i-j| \le W$) combined with additive relative position bias tensor ($S_{ij} = \frac{Q_i K_j^T}{\sqrt{D}} + B_{ij} + M_{window}$). Verifies bias addition inside window bounds.
8. **Multi-layer Projection (PLE) + Matryoshka L2 Normalization**:
   - Concatenated layer embeddings projected through learned linear layer, followed by prefix truncation to 256 dimensions and L2 normalization.

---

### Tier 4 — End-to-End Real-World Production Workloads

#### Scenario A: SAM 3.1 Video Tracker Frame Multiplex Pipeline
- **Workload Description**:
  Simulates one complete step of the SpikeTrack real-time video tracker on a VISEM endoscopic video frame ($1008 \times 1008$ pixels, BGR24 format).
- **Execution Pipeline**:
  1. Image preprocessing and patch embedding: Conv2d ($k=14, s=14, p=0, C=3 \to 1024$).
  2. 32-layer ViT Trunk execution:
     - 28 windowed attention blocks ($24 \times 24$ windows in $3 \times 3$ grid) with 2D axial RoPE.
     - 4 global attention blocks ($5,184$ tokens) with global RoPE tables.
     - Pointwise exact GELU (erf) MLP layers.
  3. FPN Neck multi-scale feature assembly (0.5x, 1x, 2x, 4x levels) utilizing `conv_transpose2d` and `max_pool2d`.
  4. SAM 2/3.1 Memory Attention:
     - 1-head, $D=256$ cross-attention attending current frame queries ($T_q=5,184$) over 7 historical memory frames plus object pointer tokens ($T_k \approx 36,288$).
  5. DETR Box & Mask Decoder:
     - Additive `boxRPB` relative position bias attention.
     - GroupNorm and Bilinear upsampling to generate high-resolution binary mask outputs.
- **Pass Criteria**:
  - IoU $\ge 0.999$ and cosine similarity $\ge 1.0 - 10^{-7}$ against PyTorch SAM 3.1 reference tracker.
  - No CPU-GPU silent fallback.
  - Zero memory growth over 10 consecutive frame iterations.

#### Scenario B: Gemma Embedding Production Inference Pipeline
- **Workload Description**:
  Simulates the HuggingFace `EmbeddingGemma 2` text embedding pipeline across a batch of variable-length queries and documents.
- **Execution Pipeline**:
  1. Token embedding lookup ($V=256,000, D=2048$).
  2. Transformer encoder stack:
     - Alternating local symmetric sliding window attention ($W=512$, non-causal $|i-j| \le W$) and global attention.
     - Per-head V RMSNorm in attention projection.
     - GeGLU feedforward layers (`gelu_tanh(gate) * up`).
  3. Multi-layer embedding extraction (PLE) with RMSNorm projection.
  4. Masked sequence mean pooling (`segment_mean_rows`) over valid tokens.
  5. Matryoshka prefix truncation ($D=2048 \to d \in \{128, 256, 512, 768\}$) and L2 normalization (`l2_normalize_rows`).
- **Pass Criteria**:
  - Cosine similarity $\ge 1.0 - 10^{-7}$ against HuggingFace Gemma embedding reference.
  - Unit sphere normalization verified: $\|y\|_2 = 1.0 \pm 10^{-6}$.
  - Bit-identical CPU results under `Numerics::Exact`.

---

## 3. Authoritative Reference Oracles & Tolerance Standards

All test cases are judged strictly against the following pre-registered tolerances:

| Op Domain | Constant | Max Abs Error | Max Rel Error | Min Cosine Similarity | Justification |
|---|---|---|---|---|---|
| RoPE Tables | `TABLE_TOL` | $1 \times 10^{-6}$ | $1 \times 10^{-3}$ | $1.0 - 1 \times 10^{-9}$ | Analytical f64 double-precision vs f32 IEEE 754 rounding. |
| RoPE Application | `ROPE_TOL` | $1 \times 10^{-5}$ | $1 \times 10^{-3}$ | $1.0 - 1 \times 10^{-9}$ | 2 multiplies + 1 add per element across activation scale. |
| GELU (erf) | `GELU_TOL` | $1 \times 10^{-5}$ | $1 \times 10^{-3}$ | $1.0 - 1 \times 10^{-9}$ | Rational Chebyshev approximation vs PyTorch CPU erf. |
| GeGLU (`gelu_tanh`) | `GELU_TANH_TOL` | $1 \times 10^{-5}$ | $1 \times 10^{-3}$ | $1.0 - 1 \times 10^{-9}$ | Exact PyTorch `approximate="tanh"` formula. |
| LayerNorm / LayerNorm2d | `LAYER_NORM_TOL` | $1 \times 10^{-4}$ | $1 \times 10^{-3}$ | $1.0 - 1 \times 10^{-9}$ | Two-pass f64 row reduction vs PyTorch f32 sum variance. |
| GroupNorm | `GROUP_NORM_TOL` | $1 \times 10^{-4}$ | $1 \times 10^{-3}$ | $1.0 - 1 \times 10^{-9}$ | Normalization across spatial groups up to 2.6M elements. |
| Attention (Window/Global) | `ATTENTION_TOL` | $1 \times 10^{-4}$ | $1 \times 10^{-2}$ | $1.0 - 1 \times 10^{-8}$ | Tiled GEMM reduction order difference on long key sequences. |
| Cross-Attention ($D=256$) | `XATTN_TOL` | $1 \times 10^{-4}$ | $1 \times 10^{-2}$ | $1.0 - 1 \times 10^{-8}$ | Tiled TensorOps accumulation order vs PyTorch SDPA. |
| MaxPool2D | `MAXPOOL_TOL` | $0.0$ (Bit-Exact) | $0.0$ | $1.0$ | Exact order-independent max reduction on IEEE 754 floats. |
| L2 Normalization | `L2_NORM_TOL` | $1 \times 10^{-6}$ | $1 \times 10^{-4}$ | $1.0 - 1 \times 10^{-11}$ | Vector norm division with epsilon stabilization. |
| CPU Exact Contract | `EXACT_TOL` | $0.0$ (Bit-Exact) | $0.0$ | $1.0$ | Bit-identical across threads under `Numerics::Exact`. |

---

## 4. Test Harness Directory Layout & Execution Matrix

### Layout
```
/Users/bharath/Code/research/
├── ojas/
│   └── ojas-oracle/
│       ├── src/
│       │   ├── lib.rs              # Exported oracle modules
│       │   ├── parity.rs           # Core tolerance gates & model parity
│       │   ├── sam_fixtures.rs     # SAM 3.1 isolated op parity harness
│       │   └── gemma_fixtures.rs   # Gemma embedding isolated op parity harness
│       └── tests/
│           ├── sam_fixtures_cpu.rs # CPU SAM op parity tests
│           ├── e2e_tier1_features.rs # Tier 1 Feature Coverage Suite
│           ├── e2e_tier2_boundaries.rs # Tier 2 Boundary & Adversarial Suite
│           ├── e2e_tier3_combinatorial.rs # Tier 3 Cross-Feature Suite
│           └── e2e_tier4_production.rs # Tier 4 Full Production Workloads
├── SpikeTrack/
│   ├── fixtures/                   # Reference fixtures (.safetensors)
│   ├── TEST_INFRA.md               # This specification
│   └── TEST_READY.md               # Readiness declaration
```

### Execution Commands

```sh
# 1. Tier 1: Core Feature Coverage Suite
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  -p ojas-oracle --test e2e_tier1_features -- --nocapture

# 2. Tier 2: Boundary, Corner & Adversarial Suite
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  -p ojas-oracle --test e2e_tier2_boundaries -- --nocapture

# 3. Tier 3: Combinatorial & Cross-Feature Interaction Suite
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  -p ojas-oracle --test e2e_tier3_combinatorial -- --nocapture

# 4. Tier 4: End-to-End Real-World Production Workloads
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  -p ojas-oracle --test e2e_tier4_production -- --nocapture

# 5. Full E2E Test Suite Run
cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  -p ojas-oracle --test e2e_tier1_features --test e2e_tier2_boundaries \
  --test e2e_tier3_combinatorial --test e2e_tier4_production --release -- --nocapture

# 6. SpikeTrack Application Workspace Regression Pass
cargo test --workspace -- --nocapture
```

---

## 5. Defect Escalation & QA Protocol

When any test fails during milestone implementation:
1. **Determine Fix Direction**:
   - If the code has a bug (e.g. op produces incorrect values outside tolerance, panics unexpectedly, falls back silently), **escalate to the implementation worker**. Test writers do NOT modify implementation code.
   - If the test encodes an incorrect expectation violating the specification, update the test with documented rationale.
2. **Defect Report Schema**:
   Include:
   - Failing test name and location (`file:line`).
   - Exact input tensor dimensions and dtype.
   - Observed output vs expected oracle output.
   - Measured error metric (`max_abs`, `max_rel`, `cosine`) vs tolerance gate.
   - Traceback / error variant.
