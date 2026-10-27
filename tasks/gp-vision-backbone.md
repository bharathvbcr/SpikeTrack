---
id: "gp-vision-backbone"
title: "Vision backbone forward on Metal (patch embed as im2col GEMM)"
status: "in-progress"
priority: 0
severity: "critical"
type: "feature"
owner: "unassigned"
due: "2026-10-27"
labels:
  - "backbone"
  - "vit"
  - "metal"
  - "parity"
  - "kill-gate"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/backbone.rs"
  - "crates/st-model/src/im2col.rs"
  - "crates/st-model/src/attention.rs"
  - "crates/st-model/tests/backbone_parity.rs"
acceptance_criteria:
  - "Backbone output cosine > 0.999 vs fixture on all 5 images, f32 [V]"
  - "Per-stage error table (embed, each block group, neck) committed under bench/results/parity-backbone/"
  - "bf16-operand path cosine > 0.995 with the drop documented"
  - "Single-image latency on M5 Pro recorded (median of 5, f32 and bf16)"
---
# Task brief v1

## Title
Vision backbone forward on Metal (patch embed as im2col GEMM)

Task: gp-vision-backbone
Type: feature
Status: in-progress
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-10-27
Labels: backbone, vit, metal, parity, kill-gate

## Repositories
- spiketrack

## Description
Port the shared vision encoder (detector and tracker share it [V]).

- Patch embedding as im2col + tessl GEMM — **do not add a general conv2d to Ojas** (Ojas has none [V]; not needed).
- Reuse Ojas attention (flash-attn, sdpa, LayerNorm, RoPE exist [V]) for windowed and global blocks; add only what the reference shows is missing (e.g. 2D RoPE, relative position) and note it.
- f32 first for parity, bf16 operands second.
- **Kill gate (end D2):** cosine similarity > 0.999 and max-abs error recorded per stage on 5 images. If missed, stop and record the failing layer in gaps.jsonl before continuing.

### Second audit (2026-10-25)
- Shape of the work [V, arXiv 2511.16719]: 32 layers, 28 windowed (24x24-token windows, 3x3 grid) + 4 global, RoPE every layer, 1008 px / patch 14. Port windowed attention as batched attention over 9 windows; this is also what makes window-level event gating exact in 28 of 32 layers.
- Compute estimate [I]: ~2 x 450M x 5,184 ≈ 4.7 TFLOP of matmul per frame, so at 10-15 effective TFLOP/s the backbone alone is ~0.3-0.5 s/frame on the M5 Pro. Treat ~2-3 frames/s as the realistic ungated ceiling and never call the result real-time.
- The D2 gate needs backbone fixtures by D1 evening; if they are late, move the gate, not the threshold.

### Third audit (2026-10-25)
- Build on tessl, not Ojas's Backend (Ojas attention is causal-only [V]). Reuse: `embedgemma2::encoder_attn(_bf16)` for both window (batch 9 × 576 tokens) and global (1 × 5,184) layers; `bert::layer_norm` / `bias_residual_layer_norm`; `bert::bias_gelu_erf` (not tanh-GELU); tessl GEMM for patch embed and MLPs.
- 2D RoPE: tessl's `axial_qkv_norm_rope` is fused with an RMS QK-norm; if PE-L+ has no QK-norm, copy it into a RoPE-only kernel in this repo's own .metal file. Match PE's axis split and theta first [U].
- The multi-scale neck likely needs transposed convs (×4, ×2) and pooling [I]: build them here.
- Confirm head dim from the checkpoint (expected 64 [I]); 64/256/512 keep the TensorOps path.

### Preprocessing and tokenizer (processor_config.json, tokenizer_config.json) [V]
- Images and video frames: convert to RGB, resize straight to 1008x1008 (no aspect-preserving pad, no crop) with resample = 2 (PIL bilinear), rescale by 1/255, normalize with mean 0.5 and std 0.5 per channel. Masks come out at 288x288 (low_res_mask_size). The tessl bicubic resize is the wrong filter here: match torch/PIL bilinear (antialias behaviour [U]: check Sam3ImageProcessorFast).
- Tokenizer: CLIP BPE, vocab 49,408, merges.txt (48,895 lines), `<|startoftext|>` 49406 / `<|endoftext|>` 49407, context 32. CLIP's BPE lowercases, cleans whitespace and uses `</w>` word-end suffixes; check that ojas-data's BPE reproduces it token for token before reusing it.

See HANDOFF.md for rules, schedule and dependencies.

### D1 result (2026-10-26)
- CPU gate passed: st-model on ojas d2922fb CpuBackend, cosine >= 0.9999999 at all 95 stages on 5 images (bench/results/parity-backbone/cpu-f32.md, commit 3debd07). The brief stays open: Metal (conv2d/convT/rope Metal paths due D2 from ojas-78), bf16 and latency are not done.

## Progress
- 2026-10-27: D2 gate passes on CPU and on Metal (Ojas on tessl, d5d33cc): worst cosine 1.0000000 over 95 stages on 5 images (bench/results/module-gates). ViT LayerNorm eps corrected to 1e-5 (was 1e-6) with a regression test; gates re-queued at cbd5565. Open: half precision (Autocast bf16 D2 check queued).

## Acceptance criteria
- [ ] Backbone output cosine > 0.999 vs fixture on all 5 images, f32 [V]
- [ ] Per-stage error table (embed, each block group, neck) committed under bench/results/parity-backbone/
- [ ] bf16-operand path cosine > 0.995 with the drop documented
- [ ] Single-image latency on M5 Pro recorded (median of 5, f32 and bf16)

## Planned files
- crates/st-model/src/backbone.rs
- crates/st-model/src/im2col.rs
- crates/st-model/src/attention.rs
- crates/st-model/tests/backbone_parity.rs
