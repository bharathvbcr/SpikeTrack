---
id: "gp-tracker-multiplex"
title: "Video tracker with SAM 3.1 Object Multiplex"
status: "in-progress"
priority: 0
severity: "critical"
type: "feature"
owner: "unassigned"
due: "2026-10-30"
labels:
  - "tracker"
  - "multiplex"
  - "video"
  - "kill-gate"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-track/src/memory.rs"
  - "crates/st-track/src/memory_attention.rs"
  - "crates/st-track/src/multiplex.rs"
  - "crates/st-track/src/associate.rs"
  - "crates/st-track/src/session.rs"
  - "crates/st-track/tests/video_parity.rs"
acceptance_criteria:
  - "J&F within 2 points of reference on 3 SA-V clips (30 frames each) [V]"
  - "Object IDs stable vs reference on both VISEM fixture clips (ID switches counted)"
  - "Multiplex output equals one-object-per-pass output within tolerance at 16 objects (the bucketing changes speed, not answers, beyond documented tolerance)"
  - "Memory stays bounded over a 1,500-frame VISEM video (peak RSS recorded)"
---
# Task brief v1

## Title
Video tracker with SAM 3.1 Object Multiplex

Task: gp-tracker-multiplex
Type: feature
Status: in-progress
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-10-30
Labels: tracker, multiplex, video, kill-gate

## Repositories
- spiketrack

## Description
Port the SAM 2-derived tracker [V] and SAM 3.1's Object Multiplex (fixed-capacity buckets processed jointly, shared memory [V]; bucket size from the reference — believed 16 [U]).

- Memory encoder, memory bank, memory attention, object pointers, propagation, detector↔tracker association, new-object births and track deaths as in the reference.
- Multiplex buckets with deterministic object-to-slot assignment.
- Streaming session API: frames in, per-frame tracks out, bounded memory.
- **Kill gate (end D5):** J&F within 2 points of the PyTorch reference on 3 SA-V clips.

### Second audit (2026-10-25)
- num_maskmem = 7 (1 conditioning + 6 recent frames) [V, SAM 3 architecture docs]; the memory bank is bounded by design.
- SAM 3 reports ~30 FPS for ~5 objects only by splitting work across multiple H200s [V, arXiv 2511.16719]: our speed claims are relative to the CoreML port on the same Mac, never absolute.

### Third audit (2026-10-25)
- Before Appendix H is readable, inspect the CoreML port's `track_step` input/output spec (multiplex 16) to learn the multiplex state layout [V that it exists].
- Memory attention and the SAM 2-style mask decoder are non-causal: use tessl `encoder_attn`, not Ojas.

See HANDOFF.md for rules, schedule and dependencies.

## Progress
- 2026-10-26: module parity on CPU vs upstream's recorded inputs (DAVIS judo, 8 frames): memory attention, memory encoder and interactive decoder at cosine 1.0000000; multiplex mask decoder rerun queued after a fixture-key fix in the test.
- VOS driver (crates/st-model/src/tracker/vos.rs) wired; per-frame stage check (tests/vos_stages.rs) and the J&F gate (tests/vos_gate.rs) queued.
- Reference J&F (upstream, CPU): judo 0.9107, shooting 0.9456, gold-fish 0.9513, mean 0.9359 (bench/results/vos-reference).
- **Substitution:** the gate runs on 3 DAVIS 2017 val clips, not SA-V, because the SA-V access form is not yet submitted (GAP-SAV-ACCESS). It does not meet the SA-V criterion as written.
- Detect-track (text prompt) loop: spec in tools/reference/DETECT_TRACK_SPEC.md; oracle tools/reference/detect_track_reference.py queued; Rust port not started.

## Acceptance criteria
- [ ] J&F within 2 points of reference on 3 SA-V clips (30 frames each) [V]
- [ ] Object IDs stable vs reference on both VISEM fixture clips (ID switches counted)
- [ ] Multiplex output equals one-object-per-pass output within tolerance at 16 objects (the bucketing changes speed, not answers, beyond documented tolerance)
- [ ] Memory stays bounded over a 1,500-frame VISEM video (peak RSS recorded)

## Planned files
- crates/st-track/src/memory.rs
- crates/st-track/src/memory_attention.rs
- crates/st-track/src/multiplex.rs
- crates/st-track/src/associate.rs
- crates/st-track/src/session.rs
- crates/st-track/tests/video_parity.rs
