# Speed plan (2026-10-28)

Goal: close the gap to the CoreML port (fp16, tracker only) on the M5 Pro without giving up the
exact path. The f32 path stays the default and the parity reference. Faster numerics ship as a
`--fast` tier, gated on D3 kept-sets, VISEM 22 session IoU and held-out HOTA. Every speed number
is paired, interleaved, min-of-N, with the GPU utilization from other sessions recorded.

Labels: [V] measured or read in code, [I] inferred, [U] unverified.

## Where a frame goes (profile-cbd5565, the pin before 8b88328; refresh in `speed-baseline/`)

One tracker frame at F = 7 memory frames, every op synced: 2380 ms over 961 calls [V].

| op class | calls | ms | share |
|---|---|---|---|
| linear_bias_forward (all f32) | 276 | 1198 | 50% |
| attention_forward | 47 | 553 | 23% |
| permute | 255 | 144 | 6% |
| conv2d_forward | 17 | 134 | 6% |
| gelu_erf_forward | 45 | 105 | 4% |
| residual_add + layer_norm + rope | 281 | 152 | 6% |

The trunk alone is 1430 ms of it [V]. Its matmuls are about 4.6 TFLOP per frame [I].

## The machine

- M5 Pro: 20 GPU cores, each with a matrix unit ("neural accelerator") that only TensorOps / MPP
  reach; 307 GB/s; 32 KB threadgroup memory [V, Apple].
- Exact f32 runs on the SIMD ALUs. tessl's exact-f32 NT GEMM is 6.3 TFLOP/s at our shapes, its
  ceiling [V, ojas `bench/results/2026-10-27-linear-gemm`].
- bf16 NT at our shapes reaches 16-19 TFLOP/s [V]; tessl's bf16 NN reaches 24.8-26.6 TFLOP/s at
  4096³ and mlp shapes [V, tessl `docs/benchmarking.md`]. Community data on base M5 puts fp16 and
  bf16 at the same rate, int8 at 2x [C]. The 20-core peak is unmeasured, plausibly ~30 [I].
- tessl emits a device barrier after every dispatch by default (`ab_flags::hazard_barriers`,
  off) [V], so independent ops never overlap on the GPU.
- The ANE is not worth chasing for a ViT-L at 1008: community reports GPU 4.6x faster than the
  ANE on M5 Max and allocation failures at >= 768 px [C].

## What CoreML does that we do not

fp16 everywhere on the matrix units [I: 4.6 TFLOP in 0.272 s is ~17 TFLOP/s]. Its output matches
upstream to cosine 0.99995 [V], so it is not exact either.

## Experiments (each with the result that would refute it)

| id | change | repo | prediction | refuted if |
|---|---|---|---|---|
| E0 | baseline: occupancy, profile f32 and bf16, CoreML CPU_AND_GPU / ALL / CPU_AND_NE | SpikeTrack | — | — |
| E1 | count Autocast's per-call input/output rounding in the bf16 profile | ojas (read) | >= 100 ms per trunk pass | < 30 ms: the bf16 gap is elsewhere |
| E2 | one fused qkv linear `[3072, 1024]` instead of three | SpikeTrack | -10% f32, -30% bf16 on those calls | < 3% |
| E3 | bf16 GEMM at our shapes: NN on pre-transposed weights vs NT, tile and split-K sweep | tessl | >= 1.3x on fc1 | < 1.1x: tessl's bf16 is at its ceiling |
| E4 | bias (+GELU) fused in `gemm_epilogue` (NN only: pre-transpose weights at load) | ojas | -8% f32 fc1, -20% bf16 fc1 | < 3% |
| E5 | hazard-tracked barriers; detector and tracker as independent work; frame t+1 trunk under frame t tracker | ojas/tessl | >= 1.2x per frame if steady-state GPU occupancy < 70% | occupancy > 90%: skip |
| E6 | permute and elementwise passes at bandwidth (permute measures ~70 GB/s of 307); fold head split into attention strides | ojas/tessl | permute >= 3x faster, most of 255 calls gone | < 1.5x |
| E7 | 672 input (48x48 tokens, 2304 vs 5184): VISEM is 640x480 | SpikeTrack | trunk >= 2x faster; held-out HOTA >= 0.47 at 16 and 64 objects | HOTA < 0.47 |
| E8 | memory cross-attention at F = 7 on 8b88328 | ojas | read the fresh profile first | — |
| E9 | bf16 trunk as `--fast` once E1-E4 land | SpikeTrack | trunk <= 500 ms | D3 or session gate fails |

Not this week: replacing or distilling the backbone (no parity gate for a different trunk, and no
time to train one), int8 W8A8, the ANE.

E7 scope [V, read]: windowed blocks keep their 576-token RoPE tables; the 4 global blocks load a
5184-row `freqs_cis` from the checkpoint and need it regenerated for 48x48; 118 places in
`crates/` hard-code 72 / 288 / 1008 / 5184.
