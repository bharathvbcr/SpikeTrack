---
id: "gp-weights-loader"
title: "SAM 3.1 checkpoint loader into Ojas/tessl tensors"
status: "done"
priority: 0
severity: "high"
type: "feature"
owner: "unassigned"
due: "2026-10-26"
labels:
  - "io"
  - "weights"
  - "rust"
repositories:
  - "spiketrack"
planned_files:
  - "crates/st-weights/src/lib.rs"
  - "crates/st-weights/src/names.rs"
  - "crates/st-weights/tests/load.rs"
acceptance_criteria:
  - "Every tensor in the checkpoint is claimed by exactly one field; a test fails on any unclaimed or missing name"
  - "Loads the full checkpoint on the M5 Pro; peak RSS and load time recorded"
  - "Shape/dtype of each field matches tools/reference manifest [V]"
---
# Task brief v1

## Title
SAM 3.1 checkpoint loader into Ojas/tessl tensors

Task: gp-weights-loader
Type: feature
Status: done
Priority: 0 (Urgent)
Severity: high
Owner: unassigned
Due: 2026-10-26
Labels: io, weights, rust

## Repositories
- spiketrack

## Description
Map every checkpoint tensor to typed Rust structs using `ojas-io` safetensors (`SafeTensors::open`, `read_f32_widened`, `read_bf16_bits` [V]).

- One struct per module family (backbone, text encoder, detector, mask head, tracker); names generated from the manifest in gp-reference-oracle, not hand-typed.
- Load as bf16 storage with f32 widening where a kernel needs it; zero-copy mmap where Ojas allows.
- Refuse unknown or missing tensors with a typed error naming them.

- Update [V]: the checkpoint is sam3.1_multiplex.pt (PyTorch pickle). gp-reference-oracle converts it to safetensors offline with torch.load(weights_only=True); this loader reads the safetensors.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [x] Every tensor in the checkpoint is claimed by exactly one field; a test fails on any unclaimed or missing name [V: crates/st-weights/tests/load.rs, schema test + full load, commit 297a4d8]
- [x] Loads the full checkpoint on the M5 Pro; peak RSS and load time recorded [V: 1.38 s, peak RSS 3.27 GiB, release build, 2026-10-26]
- [x] Shape/dtype of each field matches tools/reference manifest [V: schema test against tools/reference/weights_manifest.json]

## Planned files
- crates/st-weights/src/lib.rs
- crates/st-weights/src/names.rs
- crates/st-weights/tests/load.rs
