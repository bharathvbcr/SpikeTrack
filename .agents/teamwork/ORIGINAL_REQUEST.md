# Original User Request

## Initial Request — 2026-10-31T17:10:21Z

Create multiple Sonnet agents if required. Consult Fable for guidance and use Opus as orchestrator. Refactor the architecture so Ojas and Tessl systems are built around a unified structure to eliminate duplicate code and share memory pools across CPU and GPU, and generate all necessary kernels for SAM and Gemma embedding with PyTorch parity for production.

Working directory: /Users/bharath/Code/research/ojas
Integrity mode: development

## Requirements

### R1. SAM and Gemma Embedding Kernels with PyTorch Parity
Identify missing kernels and required ops for SAM (Segment Anything Model) and Gemma embedding by reviewing the codebase and GitPulse tasks, and implement them to achieve parity with PyTorch.

### R2. Unified Architecture and Shared Memory Pools Across Ojas and Tessl
Refactor the architecture so Ojas and Tessl are built around a unified structure to eliminate duplicate code and share memory pools across CPU and GPU. Tessl specializes in Metal and heavy Mac inference optimizations; Ojas serves as the universal system with a focus on Rust-based training.

### R3. GitPulse Tasks and Accelerated Workflows
Review GitPulse tasks for priority, missing items, and bugs. Develop multiple workflows to accelerate the process.

## Verification Resources
- In-repo test suites across `ojas` and `tessl`
- `ojas-oracle` reference fixtures and PyTorch parity test harnesses

## Acceptance Criteria

### Kernel Parity and Correctness
- [ ] All missing kernels identified for SAM and Gemma embedding are implemented.
- [ ] Kernel outputs match PyTorch reference outputs on equivalent inputs within `ojas-oracle` parity tolerances.
- [ ] Existing `cargo test --workspace` suites pass without regressions.

### Unified Architecture and Memory Sharing
- [ ] Duplicate code between Ojas and Tessl is eliminated into a unified structure.
- [ ] Memory pools are shared across CPU and GPU between Ojas and Tessl without silent fallback.

### GitPulse Task Tracking and Workflows
- [ ] GitPulse tasks are reviewed and updated with priority classifications, missing items, and bug status.
- [ ] Workflows for accelerated development and testing are documented and executable.


## Follow-up — 2026-10-31T18:49:06Z

User instruction: "Check other branches and continue their work" — please ensure the team checks any other branches / worktrees across the repositories and integrates/continues any in-flight or prior branch work as part of the milestones.

## Follow-up — 2026-10-31T19:48:42Z

Server restarted — please resume your orchestration from where you left off (Milestone 2 remediation of clippy/deny lints and proceeding through Milestone 3 SAM & Gemma kernels and final verification). Check the current state of your subagents and restart/revive any that were stopped by the server restart.

## Follow-up — 2026-11-01T04:17:15Z

Follow-up user request: "Commit and merge all branches and update docs" — please ensure all uncommitted changes, remaining unmerged branches/worktrees across `ojas`, `tessl`, `SpikeTrack` (and `sparsl` if applicable) are cleanly committed and merged into `main`, and update all documentation (including crate READMEs, `docs/*.md`, `docs/data/test-runs.json`, and `go run scripts/sitegen/main.go` where applicable).
