---
id: "gp-tensorops-hot-path"
title: "tessl Metal 4 TensorOps for hot GEMMs and FPS-vs-objects curve"
status: "backlog"
priority: 1
severity: "high"
type: "performance"
owner: "unassigned"
due: "2026-10-31"
labels:
  - "performance"
  - "tessl"
  - "tensorops"
  - "benchmark"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/backend.rs"
  - "crates/st-track/src/backend.rs"
  - "bench/fps_vs_objects.rs"
  - "bench/results/fps/"
acceptance_criteria:
  - "FPS-vs-objects table (median of 5, spread) for SpikeTrack, CoreML port and PyTorch-MPS on the M5 Pro [V]"
  - "Ratio vs CoreML at 16 objects stated with the command that produced it"
  - "All parity tests still pass on the optimized path"
  - "Peak memory and cold-start time recorded"
---
# Task brief v1

## Title
tessl Metal 4 TensorOps for hot GEMMs and FPS-vs-objects curve

Task: gp-tensorops-hot-path
Type: performance
Status: backlog
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: 2026-10-31
Labels: performance, tessl, tensorops, benchmark

## Repositories
- spiketrack

## Description
Route the heaviest matmuls (backbone MLPs/attention projections, memory attention) through tessl MPP TensorOps (`gemm`, `gemm_bf16` [V]); fuse epilogues where tessl supports them.

- Benchmark FPS at 1/4/16/64/128 objects on the M5 Pro, paired against the CoreML port and PyTorch-MPS baselines from gp-reference-oracle.
- Parity gates from earlier tasks must still pass after every change.
- **Gate (end D6):** ≥3× the CoreML port's frames/s at 16 objects on the same machine. If missed, report the measured ratio honestly — do not change the target wording after the fact.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] FPS-vs-objects table (median of 5, spread) for SpikeTrack, CoreML port and PyTorch-MPS on the M5 Pro [V]
- [ ] Ratio vs CoreML at 16 objects stated with the command that produced it
- [ ] All parity tests still pass on the optimized path
- [ ] Peak memory and cold-start time recorded

## Planned files
- crates/st-model/src/backend.rs
- crates/st-track/src/backend.rs
- bench/fps_vs_objects.rs
- bench/results/fps/
