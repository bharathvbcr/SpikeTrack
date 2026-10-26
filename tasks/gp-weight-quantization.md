---
id: "gp-weight-quantization"
title: "Stretch: Q8/Q4 backbone weights via tessl"
status: "backlog"
priority: 3
severity: "low"
type: "performance"
owner: "unassigned"
due: "none"
labels:
  - "stretch"
  - "quantization"
  - "tessl"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/quant.rs"
  - "bench/results/quant/"
acceptance_criteria:
  - "Mask IoU within 0.01 of bf16 on the 20 parity frames, or the drop reported"
  - "Backbone ms/frame and peak memory, bf16 vs Q8 vs Q4"
---
# Task brief v1

## Title
Stretch: Q8/Q4 backbone weights via tessl

Task: gp-weight-quantization
Type: performance
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: stretch, quantization, tessl

## Repositories
- spiketrack

## Description
Only if gates are green by D8. Use tessl's existing Q4/Q8 kernels [V] for the backbone's linear layers; keep norms and heads in bf16.

See HANDOFF.md, COMPONENTS.md and IMPROVEMENTS.md.

## Acceptance criteria
- [ ] Mask IoU within 0.01 of bf16 on the 20 parity frames, or the drop reported
- [ ] Backbone ms/frame and peak memory, bf16 vs Q8 vs Q4

## Planned files
- crates/st-model/src/quant.rs
- bench/results/quant/
