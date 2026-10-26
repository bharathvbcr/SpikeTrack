---
id: "gp-event-gating"
title: "Training-free sparsl event gating"
status: "backlog"
priority: 1
severity: "high"
type: "feature"
owner: "unassigned"
due: "2026-11-01"
labels:
  - "sparsl"
  - "event-gating"
  - "novelty"
  - "kill-gate"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-gate/src/change_map.rs"
  - "crates/st-gate/src/policy.rs"
  - "crates/st-gate/tests/gating.rs"
  - "bench/results/gating/"
acceptance_criteria:
  - "Ablation table: each policy's FPS and HOTA/J&F on VISEM and SA-V [V]"
  - "Chosen policy meets ≥1.5× FPS at ≤1 HOTA lost, or the miss is reported and gating is off by default"
  - "Two runs on the same video give identical output hashes"
  - "Change-map cost per frame measured (must be < 5% of frame time)"
---
# Task brief v1

## Title
Training-free sparsl event gating

Task: gp-event-gating
Type: feature
Status: backlog
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: 2026-11-01
Labels: sparsl, event-gating, novelty, kill-gate

## Repositories
- spiketrack

## Description
The novel layer: skip work where the scene did not change.

- Per frame, compute a bitpacked 1-bit change map per patch/window (sparsl bitpacked vectors [V]); share buffers zero-copy with tessl via MTLSharedEvent [V].
- Policies, simplest first: (1) skip whole-frame backbone recompute below a change threshold; (2) recompute only changed windows in windowed-attention blocks, carrying cached outputs (global blocks make this approximate — document it); (3) skip re-decoding multiplex slots whose masks overlap no changed region.
- Deterministic: same video → same output bits.
- **Gate (end D7):** ≥1.5× FPS at ≤1 HOTA point lost on VISEM; otherwise ship ungated and report the ablation.

### Second audit (2026-10-25)
- Gate on the backbone's own windows [I]: 28 of 32 layers attend within 3x3 windows of 24x24 tokens [V], so an unchanged window's output in those layers can be reused exactly given unchanged inputs; only the 4 global layers mix windows. Policy (2) should be window-aligned, not patch-aligned.
- Risk [I]: in sperm-motility microscopy, cells move across the whole field, so most windows may change every frame and gating may save little on VISEM. Measure this on D1-D2 with no model at all: per-window change rate on the 20 VISEM videos and on a static-camera lab clip. If VISEM windows change > 80% of frames, showcase gating on the lab clip and report VISEM ungated.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] Ablation table: each policy's FPS and HOTA/J&F on VISEM and SA-V [V]
- [ ] Chosen policy meets ≥1.5× FPS at ≤1 HOTA lost, or the miss is reported and gating is off by default
- [ ] Two runs on the same video give identical output hashes
- [ ] Change-map cost per frame measured (must be < 5% of frame time)

## Planned files
- crates/st-gate/src/change_map.rs
- crates/st-gate/src/policy.rs
- crates/st-gate/tests/gating.rs
- bench/results/gating/
