---
id: "gp-prompt-encoders"
title: "Text concept encoder, tokenizer and point/box prompt encoder"
status: "backlog"
priority: 0
severity: "high"
type: "feature"
owner: "unassigned"
due: "2026-10-28"
labels:
  - "text-encoder"
  - "tokenizer"
  - "prompts"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-model/src/text.rs"
  - "crates/st-model/src/tokenizer.rs"
  - "crates/st-model/src/geometry_prompt.rs"
  - "crates/st-model/tests/prompt_parity.rs"
acceptance_criteria:
  - "Token IDs match the reference tokenizer for 50 phrases including 'sperm', 'cell', 'foot'"
  - "Text embeddings cosine > 0.999 vs fixtures for the 3 dumped prompts"
  - "Point and box prompt embeddings match fixtures (max-abs error recorded)"
---
# Task brief v1

## Title
Text concept encoder, tokenizer and point/box prompt encoder

Task: gp-prompt-encoders
Type: feature
Status: backlog
Priority: 0 (Urgent)
Severity: high
Owner: unassigned
Due: 2026-10-28
Labels: text-encoder, tokenizer, prompts

## Repositories
- spiketrack

## Description
SAM 3 detects all instances of a concept from a short phrase [V]; VISEM needs the prompt 'sperm'.

- Port the tokenizer in Rust (reuse `ojas-data` BPE if the vocabulary format matches [U]) and the text encoder.
- Port point/box/mask prompt encoding for interactive refinement and tracker initialisation.
- Do not precompute text embeddings in Python — that would break the no-Python claim.

### Preprocessing and tokenizer (processor_config.json, tokenizer_config.json) [V]
- Images and video frames: convert to RGB, resize straight to 1008x1008 (no aspect-preserving pad, no crop) with resample = 2 (PIL bilinear), rescale by 1/255, normalize with mean 0.5 and std 0.5 per channel. Masks come out at 288x288 (low_res_mask_size). The tessl bicubic resize is the wrong filter here: match torch/PIL bilinear (antialias behaviour [U]: check Sam3ImageProcessorFast).
- Tokenizer: CLIP BPE, vocab 49,408, merges.txt (48,895 lines), `<|startoftext|>` 49406 / `<|endoftext|>` 49407, context 32. CLIP's BPE lowercases, cleans whitespace and uses `</w>` word-end suffixes; check that ojas-data's BPE reproduces it token for token before reusing it.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] Token IDs match the reference tokenizer for 50 phrases including 'sperm', 'cell', 'foot'
- [ ] Text embeddings cosine > 0.999 vs fixtures for the 3 dumped prompts
- [ ] Point and box prompt embeddings match fixtures (max-abs error recorded)

## Planned files
- crates/st-model/src/text.rs
- crates/st-model/src/tokenizer.rs
- crates/st-model/src/geometry_prompt.rs
- crates/st-model/tests/prompt_parity.rs
