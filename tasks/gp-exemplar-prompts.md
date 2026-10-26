---
id: "gp-exemplar-prompts"
title: "Image-exemplar prompts from the first frame"
status: "backlog"
priority: 1
severity: "medium"
type: "feature"
owner: "unassigned"
due: "2026-10-30"
labels:
  - "prompts"
  - "exemplars"
  - "training-free"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/exemplar.rs"
  - "crates/st-model/tests/exemplar_parity.rs"
acceptance_criteria:
  - "Exemplar embeddings match reference fixtures (max-abs error recorded)"
  - "Detection recall on VISEM for text / exemplar / both, same frames as gp-zero-shot-check"
---
# Task brief v1

## Title
Image-exemplar prompts from the first frame

Task: gp-exemplar-prompts
Type: feature
Status: backlog
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: 2026-10-30
Labels: prompts, exemplars, training-free

## Repositories
- spiketrack

## Description
SAM 3's detector is conditioned on text, geometry and image exemplars [V]. A few boxed sperm from frame 0 can replace or reinforce the text prompt with no training.

- Port the exemplar path of the geometry/exemplar encoder (gp-prompt-encoders covers points/boxes).
- CLI: `--exemplar x0,y0,x1,y1` repeated, or `--exemplars-from-labels <file>` for evaluation.
- Compare text-only, exemplar-only and both on VISEM.

See HANDOFF.md, COMPONENTS.md and IMPROVEMENTS.md.

## Acceptance criteria
- [ ] Exemplar embeddings match reference fixtures (max-abs error recorded)
- [ ] Detection recall on VISEM for text / exemplar / both, same frames as gp-zero-shot-check

## Planned files
- crates/st-model/src/exemplar.rs
- crates/st-model/tests/exemplar_parity.rs
