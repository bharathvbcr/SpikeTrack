# Explorer 1 Dispatch: Unified Architecture & Shared Memory Pools

## Target Scope
Investigate the codebase architecture of `ojas` (/Users/bharath/Code/research/ojas) and `tessl` (/Users/bharath/Code/research/tessl).
Investigate memory pools, CPU/GPU buffer representations, backend traits, duplicate code between ojas and tessl, how tessl Metal / Mac inference optimizations are wired, and how to structure a unified architecture where memory pools are shared across CPU and GPU without silent fallback.

## Authoritative User Request
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`.

## Output
Write your findings to `handoff.md` in your working directory:
`/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_1/handoff.md`.
Then send a completion message back.


## 2026-10-31T17:13:05Z
You are Explorer 1 (Architecture & Memory Explorer).
Your identity is teamwork_explorer.
Your assigned working directory is `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_1/`.
You own this folder exclusively. Write your progress to `progress.md` in your folder and your final report to `handoff.md` in your folder.

Authoritative User Request:
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`.
Read instructions in `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_1/DISPATCH.md`.

Target Codebases:
- `/Users/bharath/Code/research/ojas`
- `/Users/bharath/Code/research/tessl`

Tasks:
1. Examine the architecture of both ojas and tessl. Note that ojas has path dependencies to sibling checkouts `../tessl` and `../../devtools/gusset`. Read AGENTS.md in ojas and tessl if present.
2. Code Navigation: Use DevMap tools (`devmap_search`, `devmap_explore`, etc.) and GitPulse tools where available to inspect symbols, callers, and structure.
3. Identify all duplicate code, shared concepts, and architectural seams between ojas and tessl.
4. Analyze memory management: How CPU buffers and GPU (Metal) buffers are currently represented and allocated in ojas and tessl.
5. Investigate how unified architecture and shared memory pools across CPU and GPU can be implemented without silent fallback (e.g. unified shared storage / zero-copy unified memory buffers on Apple Silicon / explicit device contracts).
6. Detail the division of responsibilities: Tessl specializing in Metal and heavy Mac inference optimizations; Ojas serving as the universal system with a focus on Rust-based training.
7. Detail exact interface contracts, crate structures, file paths, and potential breaking changes.
8. Write a comprehensive, structured handoff report to `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_1/handoff.md`.
9. Send a message to the orchestrator when finished.
