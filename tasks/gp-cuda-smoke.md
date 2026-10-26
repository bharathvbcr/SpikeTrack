---
id: "gp-cuda-smoke"
title: "Stretch: CUDA smoke path via ojas-cuda on RTX 3070 Ti"
status: "backlog"
priority: 3
severity: "low"
type: "feature"
owner: "unassigned"
due: "none"
labels:
  - "stretch"
  - "cuda"
  - "portability"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/backend_cuda.rs"
  - "bench/results/cuda/"
acceptance_criteria:
  - "Backbone parity (cosine > 0.999) on the RTX 3070 Ti, or the blocking gap recorded in gaps.jsonl"
---
# Task brief v1

## Title
Stretch: CUDA smoke path via ojas-cuda on RTX 3070 Ti

Task: gp-cuda-smoke
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: stretch, cuda, portability

## Repositories
- spiketrack

## Description
Only if every gate is green by D8. ojas-cuda kernels are not run on the Mac (NVRTC) and every CUDA `Qwen35Step` method refuses until kernels back it [V] — scope this to backbone parity on the 3070 Ti only.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] Backbone parity (cosine > 0.999) on the RTX 3070 Ti, or the blocking gap recorded in gaps.jsonl

## Planned files
- crates/st-model/src/backend_cuda.rs
- bench/results/cuda/
