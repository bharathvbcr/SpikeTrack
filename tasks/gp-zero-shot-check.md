---
id: "gp-zero-shot-check"
title: "Zero-shot check: can SAM 3.1 find sperm at all?"
status: "ready"
priority: 0
severity: "critical"
type: "research"
owner: "unassigned"
due: "2026-10-26"
labels:
  - "zero-shot"
  - "risk"
  - "meta-model-api"
  - "visem"
repositories:
  - "spiketrack"
planned_files:
  - "tools/reference/zero_shot_check.py"
  - "bench/results/zero-shot/"
acceptance_criteria:
  - "Recall/precision at IoU 0.5 for each prompt on 50 frames, in bench/results/zero-shot/"
  - "False-positive rate for 2 absent-concept prompts recorded"
  - "Decision written into IMPROVEMENTS.md: recall ≥ 0.7 → LoRA stays stretch; below → gp-domain-lora becomes P0"
---
# Task brief v1

## Title
Zero-shot check: can SAM 3.1 find sperm at all?

Task: gp-zero-shot-check
Type: research
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-10-26
Labels: zero-shot, risk, meta-model-api, visem

## Repositories
- spiketrack

## Description
The showcase depends on SAM 3.1 detecting sperm. Zero-shot SAM 3 has collapsed on other niche domains (pixel IoU 0.017 before LoRA on crack images) [V, arXiv 2609.00469]. Check before building on it.

- Call the Meta Model API `sam-3.1` (Responses API, Bearer token) [V] on 50 frames sampled across 10 VISEM-Tracking videos (CC BY 4.0, Zenodo 7293726 [V]).
- Prompts: text "sperm", "sperm cell", "spermatozoon"; then a few exemplar boxes from the frame if the API accepts them [U].
- Score against VISEM boxes (class 0 sperm): recall and precision at IoU 0.5, and how often an absent-concept prompt ("car", "cell nucleus") still returns detections.
- This needs no port and no weights, so it can run while access is pending. Offline Python is allowed here.

See HANDOFF.md, COMPONENTS.md and IMPROVEMENTS.md.

## Acceptance criteria
- [ ] Recall/precision at IoU 0.5 for each prompt on 50 frames, in bench/results/zero-shot/
- [ ] False-positive rate for 2 absent-concept prompts recorded
- [ ] Decision written into IMPROVEMENTS.md: recall ≥ 0.7 → LoRA stays stretch; below → gp-domain-lora becomes P0

## Planned files
- tools/reference/zero_shot_check.py
- bench/results/zero-shot/
