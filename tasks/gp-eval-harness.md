---
id: "gp-eval-harness"
title: "J&F, HOTA and dataset loaders in Rust"
status: "ready"
priority: 1
severity: "high"
type: "feature"
owner: "unassigned"
due: "2026-10-30"
labels:
  - "eval"
  - "metrics"
  - "hota"
  - "davis"
  - "datasets"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-eval/src/jf.rs"
  - "crates/st-eval/src/hota.rs"
  - "crates/st-eval/src/visem.rs"
  - "crates/st-eval/src/sav.rs"
  - "crates/st-eval/tests/metric_parity.rs"
acceptance_criteria:
  - "J&F matches the official DAVIS evaluation within 0.1 on fixtures [V]"
  - "HOTA matches TrackEval within 0.1 on fixtures [V]"
  - "VISEM loader reads all 20 videos; track-ID availability recorded with the chosen metric"
  - "`spiketrack eval` prints a table and writes JSON under bench/results/"
---
# Task brief v1

## Title
J&F, HOTA and dataset loaders in Rust

Task: gp-eval-harness
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: 2026-10-30
Labels: eval, metrics, hota, davis, datasets

## Repositories
- spiketrack

## Description
Metrics must be trustworthy before any number is quoted.

- J&F (DAVIS region similarity + boundary F) and HOTA (TrackEval definition) in Rust.
- Parity: compute the same metrics with the official Python tools offline in gp-reference-oracle and compare on fixtures.
- VISEM-Tracking is annotated with bounding boxes [V]: evaluate HOTA on boxes derived from our masks. If VISEM has no track IDs [U], fall back to detection AP + ID-free counts and say so.
- SA-V subset loader (clip list pinned in a file).

### Second audit (2026-10-25)
- SA-V needs a license form [U]; if access is not granted by D3, compute J&F on DAVIS 2017 val (openly downloadable) and say so. The kill gate wording becomes "within 2 J&F points of the reference on 3 DAVIS clips".

- Update [V, arXiv 2212.02842]: VISEM-Tracking has per-sperm track IDs, YOLO boxes, 3 classes, 45–50 fps, CC BY 4.0, Zenodo 7293726.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] J&F matches the official DAVIS evaluation within 0.1 on fixtures [V]
- [ ] HOTA matches TrackEval within 0.1 on fixtures [V]
- [ ] VISEM loader reads all 20 videos; track-ID availability recorded with the chosen metric
- [ ] `spiketrack eval` prints a table and writes JSON under bench/results/

## Planned files
- crates/st-eval/src/jf.rs
- crates/st-eval/src/hota.rs
- crates/st-eval/src/visem.rs
- crates/st-eval/src/sav.rs
- crates/st-eval/tests/metric_parity.rs
