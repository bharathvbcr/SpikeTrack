# SpikeTrack — code components and changes

> Third audit (2026-10-25): the model runs on tessl directly; Ojas's Backend has causal attention only and is not used for SAM. See HANDOFF.md §10.

Written 2026-10-25. Every component the port needs, where it lives, and whether Ojas/tessl/sparsl already have it.
Labels: **[V]** verified (paper, docs or a grep of the local repos on 2026-10-25), **[I]** inferred, **[U]** check against the reference code on D1.
Architecture facts are from the SAM 3 paper (arXiv 2511.16719) [V] unless marked; SAM 2-derived tracker internals are [I] until gp-reference-oracle dumps the module tree.

## Rule for changes to the pre-existing libraries

Implement everything in this repo first (crate `st-ops`). Do **not** modify Ojas, tessl or sparsl during the hackathon unless a kernel cannot be written outside them; ask Bharath before any such change. Anything worth upstreaming is listed in §9 for after Nov 4.

## 1. Workspace layout

| Crate | Role | Task |
|---|---|---|
| `st-weights` | Checkpoint → typed Rust structs | gp-weights-loader |
| `st-ops` | Ops missing from Ojas/tessl (§2) | gp-vision-backbone, gp-detector-and-mask-head, gp-tracker-multiplex |
| `st-model` | Image model: backbone, text, prompts, detector, heads | gp-vision-backbone, gp-prompt-encoders, gp-detector-and-mask-head |
| `st-track` | Video tracker, memory, multiplex, association, motion model | gp-tracker-multiplex, gp-motion-aware-memory |
| `st-gate` | Event gating: change maps, window cache, policies | gp-event-gating |
| `st-video` | Decode, resize/normalize, overlay render, encode | gp-cli-and-no-python-proof |
| `st-eval` | J&F, HOTA, dataset loaders | gp-eval-harness |
| `spiketrack` | CLI binary | gp-cli-and-no-python-proof |
| `tools/reference` | Python, offline only: fixtures, baselines, optional LoRA training | gp-reference-oracle, gp-domain-lora |
| `bench/` | FPS-vs-objects, gating ablation, packaging numbers | gp-tensorops-hot-path, gp-event-gating |

## 2. `st-ops` — ops to build (none of these exist locally)

| Op | Used by | Approach | Local status |
|---|---|---|---|
| Patch embed (14×14, stride 14) | Backbone | im2col reshape + tessl `gemm` | im2col: none in ojas/tessl [V] |
| Window partition / unpartition (3×3 grid of 24×24 tokens) | Backbone, 28 of 32 layers | index permutation, zero-copy where possible | none [V] |
| Transposed conv (stride 2) | Segmentation head, SAM 2 mask decoder [I] | GEMM + col2im / pixel-shuffle | none [V] |
| Small strided conv 3×3 | Memory encoder mask downsampler [I] | im2col + GEMM | none [V] |
| Depthwise conv 7×7 | Memory encoder fuser blocks [I] | direct Metal kernel or CPU SIMD (not a GEMM) | none [I] |
| 2D (axial) RoPE | Backbone (every layer) [V], memory attention [I] | positions buffer, per-axis frequencies | tessl `axial_qkv_norm_rope` exists but is fused with RMS QK-norm [V]; copy a RoPE-only variant into this repo if PE has no QK-norm |
| Fourier / sine positional encodings | Prompt encoder, DETR | tables from checkpoint matrices | none [V] |
| Bilinear resize (antialiased for input, plain for mask upsample) | Pre-processing, masks | separable kernel | tessl media has image resize paths — check reuse [I] |
| Box ops: cxcywh↔xyxy, IoU, NMS (if the reference uses it) | Detector post-process, association | CPU | none [V] |
| LayerNorm2d (channel-wise LN on NCHW) | Necks, memory encoder [I] | wrap existing LayerNorm with layout change | LayerNorm exists in ojas and tessl [V] |

## 3. Reused from the existing libraries (no change)

| Capability | Where [V unless noted] |
|---|---|
| GEMM, bf16 operands, fused epilogues | tessl `gemm`, `gemm_bf16` |
| Bidirectional attention (non-causal SDPA) | tessl `embedgemma2::encoder_attn(_bf16)`: right-padded batches, head dim 64/256/512 on TensorOps [V]. Ojas has causal SDPA only [V], so it is not used for the model |
| LayerNorm, exact GELU | tessl `bert::layer_norm`, `bias_residual_layer_norm`, `bias_gelu_erf` [V] |
| Softmax, sigmoid | tessl / ojas |
| Safetensors read (mmap, bf16 widening) | ojas-io `SafeTensors` |
| BPE tokenizer core | ojas-data `bpe.rs` [I — vocab format must match SAM 3's tokenizer] |
| Video decode (AVFoundation / VideoToolbox), image decode | tessl `media/video.rs`, `vt.rs`, `image.rs` |
| Q4/Q8 weight kernels | tessl (MLX Q4/Q8 entry points, `qwen35_q4`) — for gp-weight-quantization |
| LoRA adapters | Ojas LoRA is Qwen3.5-only [V]; gp-domain-lora trains in PyTorch offline instead |
| Bitpacked spike vectors, gather/scatter, prefix scans | sparsl `spikes.rs`, `scan.rs`, `sparse.rs` |
| Zero-copy device buffers + MTLSharedEvent sync | sparsl ↔ tessl `BufferKind::External` |

## 4. `st-model` — image model components

| Component | Size / shape [V] | Notes |
|---|---|---|
| Pre-processing | 1008×1008 input, stretch resize (no pad), PIL bilinear, /255, mean 0.5 std 0.5 [V processor_config.json] | weights and configs at Code/research/models/sam3.1 |
| Vision encoder PE-L+ | ~450M params, 32 layers, patch 14 → 72×72 = 5,184 tokens, 28 windowed + 4 global layers, RoPE every layer | output downsampled 2×2 to 1,296 tokens; multi-scale neck/FPN ("dual ViT-Det") for detector and tracker [V docs] |
| Text encoder | causal transformer, ~300M params, 32-token context | run once per prompt; cache embeddings in memory |
| Geometry / exemplar encoder | points, boxes, masks, image exemplars | exemplar boxes enable gp-exemplar-prompts |
| Fusion encoder | 6 layers, image tokens conditioned on prompt tokens | |
| DETR decoder | 6 layers, learned queries, box-aware relative position bias (boxRPB), iterative box refinement | query count [U] |
| Presence head | one presence token: "is the concept in this image" | the calibration hook for "I don't know" |
| Segmentation head | pixel decoder + per-query mask embedding | upsampling via transposed conv [I] |

## 5. `st-track` — video components

| Component | Notes |
|---|---|
| Memory encoder | mask downsampler (strided convs) + fuser (depthwise-conv blocks) [I, SAM 2 design] |
| Memory bank | num_maskmem = 7: 1 conditioning + 6 recent frames [V] |
| Memory attention | self + cross attention over memory with 2D RoPE [I] |
| SAM 2-style mask decoder | two-way transformer, IoU head, object-score head, object pointers [I] |
| Object Multiplex | fixed-capacity buckets processed jointly with shared memory [V]; bucket size 16 [U, Appendix H] |
| Detector↔tracker association | births, deaths, re-identification heuristics from the reference |
| Motion model (improvement) | Kalman filter + motion-aware memory selection (gp-motion-aware-memory) |
| Session API | streaming frames in, tracks out, bounded memory |

## 6. `st-gate` — event gating

| Component | Notes |
|---|---|
| Change map | per-frame |Δ| over a threshold → bitpacked bits per 14×14 patch and per 24×24-token window (sparsl) |
| Window cache | per-window outputs of the 28 windowed layers, reused when the window's input is unchanged |
| Global-layer policy | recompute the 4 global layers always, or skip the frame when total change < threshold |
| Slot policy | skip multiplex slots whose masks overlap no changed window |
| Statistics | per-window change rates for choosing the showcase clip (run on D1–D2, no model) |

## 7. `st-video`, `st-eval`, CLI

| Component | Notes |
|---|---|
| Decode | tessl media (AVFoundation) [V] |
| Overlay + encode | mask colour blend; MP4 writer via AVFoundation [U — tessl may be decode-only]; PNG sequence fallback |
| J&F, HOTA | Rust, parity with DAVIS tools and TrackEval on fixtures |
| VISEM loader | YOLO-format boxes + per-video track-ID files; classes 0 sperm, 1 cluster, 2 small/pinhead; CC BY 4.0; Zenodo record 7293726 [V] |
| DAVIS 2017 / SA-V loaders | DAVIS is the fallback if SA-V access is slow |
| CLI | `spiketrack image|track|eval|bench` |

## 8. Python, offline only (`tools/reference/`)

Fixture dumping, TrackEval/DAVIS metric parity, CoreML/MPS baselines, and (only if gp-domain-lora runs in PyTorch) LoRA training. Never imported or executed by any Rust crate.

## 9. Candidate upstream changes (after Nov 4, not during)

- tessl: batched bidirectional windowed attention; depthwise conv; transposed-conv GEMM epilogue.
- sparsl: 2D frame-difference → bitpacked window-mask kernel.
- Ojas: conv/im2col family, if the hackathon versions prove out.
