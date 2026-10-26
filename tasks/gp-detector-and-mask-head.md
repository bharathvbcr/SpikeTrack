---
id: "gp-detector-and-mask-head"
title: "DETR detector, presence head and mask head (image mode)"
status: "in-progress"
priority: 0
severity: "critical"
type: "feature"
owner: "unassigned"
due: "2026-10-28"
labels:
  - "detector"
  - "mask-head"
  - "image-mode"
  - "kill-gate"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/detector.rs"
  - "crates/st-model/src/mask_head.rs"
  - "crates/st-model/src/upsample.rs"
  - "crates/st-model/tests/image_parity.rs"
acceptance_criteria:
  - "Mean mask IoU ≥ 0.95 vs reference on 20 frames; per-frame table committed [V]"
  - "Box/score agreement: same instances above threshold on ≥ 19/20 frames"
  - "`spiketrack image <img> --prompt <text>` writes masks and an overlay PNG"
---
# Task brief v1

## Title
DETR detector, presence head and mask head (image mode)

Task: gp-detector-and-mask-head
Type: feature
Status: in-progress
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-10-28
Labels: detector, mask-head, image-mode, kill-gate

## Repositories
- spiketrack

## Description
Complete image-mode SAM 3.1: concept prompt → boxes, scores, masks.

- Transposed convolutions as GEMM + pixel-shuffle (no general conv2d).
- Post-processing (score threshold, NMS if present in the reference) in Rust.
- **Kill gate (end D3):** mean mask IoU ≥ 0.95 vs reference on 20 frames (5 images + 15 video frames). If missed by end of D3, ship image-only and move the remaining days to speed + demo; record the decision in HANDOFF.md.

See HANDOFF.md for rules, schedule and dependencies.

## Progress
- 2026-10-26: D3 gate passed on CPU f32 (bench/results/d3-gate/cpu-f32.log): mean mask IoU 0.9999 over 20 VISEM frames, kept sets identical on 20/20, worst single mask 0.984. Detector stage parity: 96 stages, worst cosine 0.9999998 (bench/results/parity-detector).
- Open: the `spiketrack image` command (lands with gp-cli-and-no-python-proof); Metal run of the same gate.

## Acceptance criteria
- [x] Mean mask IoU ≥ 0.95 vs reference on 20 frames; per-frame table committed [V]
- [x] Box/score agreement: same instances above threshold on ≥ 19/20 frames
- [ ] `spiketrack image <img> --prompt <text>` writes masks and an overlay PNG

## Planned files
- crates/st-model/src/detector.rs
- crates/st-model/src/mask_head.rs
- crates/st-model/src/upsample.rs
- crates/st-model/tests/image_parity.rs
