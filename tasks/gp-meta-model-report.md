---
id: "gp-meta-model-report"
title: "Stretch: on-device motility report with Muse Glimmer"
status: "backlog"
priority: 3
severity: "low"
type: "feature"
owner: "unassigned"
due: "none"
labels:
  - "stretch"
  - "meta-models"
  - "glimmer"
repositories:
  - "spiketrack"
planned_files:
  - "crates/spiketrack/src/report.rs"
  - "docs/report-example.md"
acceptance_criteria:
  - "Report generated from a real VISEM run with every number traceable to the track JSON"
  - "Runs locally or the API dependency is stated; no medical-advice wording"
---
# Task brief v1

## Title
Stretch: on-device motility report with Muse Glimmer

Task: gp-meta-model-report
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: stretch, meta-models, glimmer

## Repositories
- spiketrack

## Description
Only if time remains. Turn track statistics (counts, speeds, motility classes) into a short plain-language report with Muse Glimmer running locally (Apache 2.0, ~30B, 4-bit ~16 GB [V]) or Muse Spark via the Meta Model API if the rules require API use [U]. Must not give medical advice; frame as a lab summary.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] Report generated from a real VISEM run with every number traceable to the track JSON
- [ ] Runs locally or the API dependency is stated; no medical-advice wording

## Planned files
- crates/spiketrack/src/report.rs
- docs/report-example.md
