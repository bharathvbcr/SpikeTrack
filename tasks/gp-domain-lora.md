---
id: "gp-domain-lora"
title: "LoRA domain adaptation on VISEM, with honest presence calibration"
status: "ready"
priority: 0
severity: "high"
type: "research"
owner: "unassigned"
due: "2026-11-01"
labels:
  - "lora"
  - "fine-tuning"
  - "calibration"
  - "improvement"
repositories:
  - "spiketrack"
planned_files:
  - "tools/reference/train_lora.py"
  - "crates/st-weights/src/lora.rs"
  - "bench/results/lora/"
acceptance_criteria:
  - "Detection recall/precision and HOTA on 5 held-out videos, zero-shot vs LoRA"
  - "Presence false-positive rate on absent-concept prompts before and after, plus calibration (ECE)"
  - "LoRA weights load in Rust; merged-weight parity vs PyTorch on fixtures"
  - "Write-up states that LoRA on SAM 3 is prior work (SAM3-LoRA, MedSAM3 LoRA)"
---
# Task brief v1

## Title
LoRA domain adaptation on VISEM, with honest presence calibration

Task: gp-domain-lora
Type: research
Status: in progress
Priority: 0 (Urgent; zero-shot recall < 0.7, 2026-10-26)
Severity: high
Owner: unassigned
Due: 2026-11-01
Labels: lora, fine-tuning, calibration, improvement

## Repositories
- spiketrack

## Description
Only if gp-zero-shot-check shows recall < 0.7 (then this becomes P0); otherwise stretch.

- Recipe from SAM3-LoRA [V, arXiv 2609.00469]: rank 16, alpha 32, dropout 0.1 on attention projections of the fusion encoder, detection decoder and segmentation-head prompt cross-attention ("Light", ~1M trainable params); text encoder frozen (adapting it weakened prompt discrimination).
- Supervision: VISEM YOLO boxes (box + class losses; masks not needed for the detector).
- Every absent class as a negative prompt (plus a few everyday objects) so the presence head does not fire for any prompt after adaptation — the failure SAM3-LoRA found and fixed.
- Train offline: PyTorch on the 3070 Ti or Modal (Python is allowed in tools/), or in Ojas on the Mac with backbone features cached [I] — pick by what runs on D4. Merge or load the LoRA weights in Rust; runtime stays Python-free.
- Train on 15 videos, evaluate on 5 held out by video, never by frame.

### Third audit (2026-10-25)
- Training in Ojas is not feasible in the window: Ojas LoRA exists only for Qwen3.5 and Ojas lacks non-causal attention and its backward [V]. Train in PyTorch offline; load merged weights in Rust.

See HANDOFF.md, COMPONENTS.md and IMPROVEMENTS.md.

## Acceptance criteria
- [ ] Detection recall/precision and HOTA on 5 held-out videos, zero-shot vs LoRA [recall/precision V: eval.md, "sperm" @0.3 recall 0.000 -> 0.761; HOTA only an 8-frame smoke so far (bench/results/track/box-convention), full held-out HOTA pending]
- [x] Presence false-positive rate on absent-concept prompts before and after, plus calibration (ECE) [V: bench/results/lora/visem-lora-r16/eval.md; absent VISEM classes now detect on 16% of items (zero-shot 0), negatives stay 0; presence ECE 0.177 -> 0.045, detection ECE 0.072 -> 0.110]
- [x] LoRA weights load in Rust; merged-weight parity vs PyTorch on fixtures [V: bench/results/lora/merge-parity.log, 3 tests pass, max |d| 1.5e-8 on the 5 fixture tensors; `spiketrack image|track --lora` (bench/results/lora/cli-check)]
- [x] Write-up states that LoRA on SAM 3 is prior work (SAM3-LoRA, MedSAM3 LoRA) [V: bench/results/lora/README.md]

## Planned files
- tools/reference/train_lora.py
- crates/st-weights/src/lora.rs
- bench/results/lora/
