---
id: "gp-motion-aware-memory"
title: "Motion-aware memory and Kalman association (SAMURAI-style)"
status: "backlog"
priority: 1
severity: "high"
type: "feature"
owner: "unassigned"
due: "2026-10-31"
labels:
  - "tracker"
  - "kalman"
  - "training-free"
  - "improvement"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-track/src/kalman.rs"
  - "crates/st-track/src/memory_select.rs"
  - "crates/st-track/tests/motion.rs"
  - "bench/results/motion/"
acceptance_criteria:
  - "Off by default; parity tests unchanged with the flag off"
  - "HOTA and ID switches on 15 held-out VISEM videos, with and without the motion model"
  - "Per-frame overhead measured (< 2% of frame time)"
---
# Task brief v1

## Title
Motion-aware memory and Kalman association (SAMURAI-style)

Task: gp-motion-aware-memory
Type: feature
Status: backlog
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: 2026-10-31
Labels: tracker, kalman, training-free, improvement

## Repositories
- spiketrack

## Description
Training-free tracker improvement from SAMURAI [V, arXiv 2411.11922]: a Kalman filter on each object's box (position, size, velocities); each candidate mask scored by KF-IoU blended with the mask affinity score (α_kf = 0.15 best in their ablation); a frame enters the memory bank only if mask affinity, object score and KF score all pass thresholds; N_mem = 7, which matches SAM 3's num_maskmem [V]. Reported +7.1% AUC on LaSOT-ext and +3.5% AO on GOT-10k with no retraining and no compute overhead.

- Behind a flag so the reference-parity path stays exact.
- Tune α and thresholds on 5 VISEM videos, report on the other 15.

See HANDOFF.md, COMPONENTS.md and IMPROVEMENTS.md.

## Acceptance criteria
- [ ] Off by default; parity tests unchanged with the flag off
- [ ] HOTA and ID switches on 15 held-out VISEM videos, with and without the motion model
- [ ] Per-frame overhead measured (< 2% of frame time)

## Planned files
- crates/st-track/src/kalman.rs
- crates/st-track/src/memory_select.rs
- crates/st-track/tests/motion.rs
- bench/results/motion/
