---
id: "gp-reference-oracle"
title: "Offline PyTorch reference fixtures and baselines"
status: "ready"
priority: 0
severity: "critical"
type: "research"
owner: "unassigned"
due: "2026-10-27"
labels:
  - "reference"
  - "parity"
  - "fixtures"
  - "baseline"
repositories:
  - "spiketrack"
planned_files:
  - "tools/reference/dump_fixtures.py"
  - "tools/reference/README.md"
  - "tools/reference/modal_app.py"
  - "fixtures/MANIFEST.json"
  - "bench/results/baselines/"
acceptance_criteria:
  - "tools/reference/README.md documents the exact architecture from the checkpoint (module tree, shapes, resolution, multiplex bucket size) with evidence labels"
  - "fixtures/MANIFEST.json lists every fixture with shape, dtype, SHA-256 and generating command; fixtures are git-ignored and re-creatable"
  - "Activations exist for 5 images, 3 SA-V clips (30 frames) and 2 VISEM clips (60 frames)"
  - "CoreML port measured on the M5 Pro at multiplex 16 (backbone and tracker s/frame, median of 5) in bench/results/baselines/"
  - "PyTorch-MPS baseline measured or its failure recorded with the error"
  - "No Python file is reachable from any Rust crate's build"
---
# Task brief v1

## Title
Offline PyTorch reference fixtures and baselines

Task: gp-reference-oracle
Type: research
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-10-27
Labels: reference, parity, fixtures, baseline

## Repositories
- spiketrack

## Description
Produce the ground truth every parity gate checks against. Python lives **only** here.

- Run the official `facebookresearch/sam3` code (Python 3.12+, PyTorch 2.7+, CUDA 12.6+ [V]) on a CUDA GPU: Modal partner credits ($100) preferred; the RTX 3070 Ti (8 GB) is a fallback if fp16/bf16 inference fits [U].
- Record the real architecture from code and checkpoint (not memory): module tree, parameter names/shapes/dtypes, input resolution, patch size, backbone type (believed Perception Encoder ViT [I]), windowed vs global attention layers, text encoder and tokenizer, detector/presence head, mask head upsampling ops, tracker memory sizes, and how Object Multiplex buckets objects (paper Appendix H, arXiv 2511.16719) [U].
- Convert the checkpoint to safetensors offline if it is not already (record SHA-256); do not commit it.
- Dump per-module activations (safetensors or .npy) for: 5 images (backbone, text embedding for 3 prompts, detector outputs, masks), 3 SA-V clips × 30 frames (tracker states, masks), 2 VISEM clips × 60 frames ('sperm' text prompt). Fixed seeds; fp32.
- Baselines on the M5 Pro: CoreML port `sdkv2/sam3.1-coreml` s/frame for backbone and tracker at multiplex 16 (re-measure; published 2.05 + 0.7 s on a 16 GB Mac [V]); PyTorch-MPS if the official code runs on MPS [U]. Median of 5, with spread.

### Second audit (2026-10-25)
Labels: [V] checked against a primary source this pass; [U] unverified.
- Architecture now partly known [V, SAM 3 paper arXiv 2511.16719]: vision encoder PE-L+, ~450M params, 1008x1008 input, patch 14 (72x72 = 5,184 tokens), 32 ViT layers, windowed attention in 28 layers (3x3 non-overlapping 336 px windows of 24x24 tokens) and global in 4, RoPE in every layer, outputs downsampled 2x2 to 1,296 tokens. Text encoder: causal transformer, ~300M params, context 32 tokens. Detector: DETR, 6 encoder + 6 decoder layers, presence token. Tracker: SAM 2-derived, num_maskmem = 7 (1 conditioning + 6 recent). Multiplex details (Appendix H) still [U].
- New fallback for end-to-end ground truth: SAM 3.1 is served on the Meta Model API as `sam-3.1` (boxes, masks, stable IDs across frames) [V, Mind-Verse report]. Use it for end-to-end mask/track references if the CUDA run slips; per-layer fixtures still need the PyTorch run. Using it also shows Meta Model API use.
- CUDA is a stated requirement of the official code [V]; Modal credits go to "select participants" [V], so they may not arrive. Try CPU PyTorch on the Mac for the 5-image backbone fixtures first (slow but enough), RTX 3070 Ti second, Modal third.
- Backbone fixtures are on the critical path for the D2 gate: produce them first, before video fixtures.

### Checkpoint config facts (2026-10-25, from facebook/sam3.1 config.json) [V]
- Vision ViT: hidden 1024, 16 heads (head dim 64), MLP 4736, GELU, qkv bias, LayerNorm eps 1e-6, no layer scale, patch 14, image 1008, window 24, global attention at layers 7, 15, 23, 31, RoPE theta 10000, pretrain size 336.
- Neck (FPN): 256 channels, scale factors 4, 2, 1, 0.5 with kernel 2 / stride 2, so the x4/x2 upsamplers are non-overlapping transposed convs (pure GEMM + pixel shuffle) and 0.5 is a 2x2 pool. Feature sizes 288, 144, 72.
- Text: CLIP text model, 24 layers, hidden 1024, 16 heads (head dim 64), MLP 4096, GELU, LayerNorm eps 1e-5, context 32, vocab 49,408 (BOS 49406, EOS 49407), projection 512.
- DETR encoder and decoder: 6 layers each, hidden 256, 8 heads (head dim 32), FFN 2048, ReLU, 200 queries, presence token, box RPB mode "log". Geometry encoder: 3 layers, same widths, RoI size 7.
- Image mask decoder: 3 upsampling stages, hidden 256, 8 heads.
- Tracker: memory attention 4 layers, hidden 256, config.json reads 1 head (head dim 256) but the multiplex model runs 8 heads of 32 (`tools/reference/README.md:111`, `gaps.jsonl:5`), FFN 2048 ReLU, 2-D RoPE over 72x72, theta 10000. Memory encoder 256 -> 64 channels; mask downsampler kernel 3, stride 2, total stride 16 (4 stages), GELU; fuser 2 layers, depthwise 7x7, padding 3, GELU, MLP 1024, layer scale 1e-6. SAM mask decoder: 2 layers, hidden 256, 8 heads, attention downsample rate 2 (head dim 16 in cross-attention), MLP 2048, 3 multimask outputs, IoU head depth 3. num_maskmem 7, max_cond_frame_num 4, max_object_pointers_in_encoder 16, low_res_mask_size 288.
- Video heuristics: new_det_thresh 0.7, score_threshold_detection 0.5, det_nms_thresh 0.1, assoc_iou_thresh 0.1, trk_assoc_iou_thresh 0.5, hotstart_delay 15, recondition_every_nth_frame 16, max_trk_keep_alive 30, fill_hole_area 16.
- Checkpoint: sam3.1_multiplex.pt, a PyTorch pickle (not safetensors), 3.5 GB, SHA-256 verified; at Code/research/models/sam3.1/. Load only with torch.load(weights_only=True) and convert offline.


### Preprocessing and tokenizer (processor_config.json, tokenizer_config.json) [V]
- Images and video frames: convert to RGB, resize straight to 1008x1008 (no aspect-preserving pad, no crop) with resample = 2 (PIL bilinear), rescale by 1/255, normalize with mean 0.5 and std 0.5 per channel. Masks come out at 288x288 (low_res_mask_size). The tessl bicubic resize is the wrong filter here: match torch/PIL bilinear (antialias behaviour [U]: check Sam3ImageProcessorFast).
- Tokenizer: CLIP BPE, vocab 49,408, merges.txt (48,895 lines), `<|startoftext|>` 49406 / `<|endoftext|>` 49407, context 32. CLIP's BPE lowercases, cleans whitespace and uses `</w>` word-end suffixes; check that ojas-data's BPE reproduces it token for token before reusing it.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] tools/reference/README.md documents the exact architecture from the checkpoint (module tree, shapes, resolution, multiplex bucket size) with evidence labels
- [ ] fixtures/MANIFEST.json lists every fixture with shape, dtype, SHA-256 and generating command; fixtures are git-ignored and re-creatable
- [ ] Activations exist for 5 images, 3 SA-V clips (30 frames) and 2 VISEM clips (60 frames)
- [ ] CoreML port measured on the M5 Pro at multiplex 16 (backbone and tracker s/frame, median of 5) in bench/results/baselines/
- [ ] PyTorch-MPS baseline measured or its failure recorded with the error
- [ ] No Python file is reachable from any Rust crate's build

## Planned files
- tools/reference/dump_fixtures.py
- tools/reference/README.md
- tools/reference/modal_app.py
- fixtures/MANIFEST.json
- bench/results/baselines/
