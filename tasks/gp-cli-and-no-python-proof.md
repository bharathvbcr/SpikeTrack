---
id: "gp-cli-and-no-python-proof"
title: "Single binary, CLI and no-Python proof"
status: "in-progress"
priority: 1
severity: "high"
type: "feature"
owner: "unassigned"
due: "2026-11-02"
labels:
  - "cli"
  - "packaging"
  - "proof"
repositories:
  - "spiketrack"
planned_files:
  - "crates/spiketrack/src/main.rs"
  - "scripts/no_python_check.sh"
  - "bench/results/packaging/"
acceptance_criteria:
  - "`otool -L` output committed; no Python, libtorch or ONNX runtime linked [V]"
  - "End-to-end run on a VISEM video with python removed from PATH succeeds"
  - "Binary size, cold-start time and peak memory recorded"
  - "README quickstart works from a fresh clone (weights fetched per NOTICE)"
---
# Task brief v1

## Title
Single binary, CLI and no-Python proof

Task: gp-cli-and-no-python-proof
Type: feature
Status: in-progress
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: 2026-11-02
Labels: cli, packaging, proof

## Repositories
- spiketrack

## Description
Ship one binary: `spiketrack track <video> --prompt <text> [--max-objects N] [--gate on|off]` → masks, tracks JSON, overlay MP4.

- Video decode via AVFoundation (as tessl's media path does [V]) — no ffmpeg/Python.
- Proof script: `otool -L` on the binary, plus a run with PATH stripped of python, showing only system frameworks.
- Record binary size, cold start, peak memory.

See HANDOFF.md for rules, schedule and dependencies.

## Progress
- 2026-10-27: crates/spiketrack: `image` and `eval-visem` built and tested under cargo (34 tests); native ImageIO/AVFoundation decode; own PNG writer. scripts/no_python_check.sh written (checked on toy binaries). Open: `track` (needs the Session parity run), `--backend metal`, the proof run on the real binary, overlay MP4 (no encoder; PNG frames).

## Acceptance criteria
- [ ] `otool -L` output committed; no Python, libtorch or ONNX runtime linked [V]
- [ ] End-to-end run on a VISEM video with python removed from PATH succeeds
- [ ] Binary size, cold-start time and peak memory recorded
- [ ] README quickstart works from a fresh clone (weights fetched per NOTICE)

## Planned files
- crates/spiketrack/src/main.rs
- scripts/no_python_check.sh
- bench/results/packaging/
