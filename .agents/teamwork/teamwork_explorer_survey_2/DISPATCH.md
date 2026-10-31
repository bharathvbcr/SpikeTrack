# Explorer 2 Dispatch: SAM & Gemma Embedding Kernels with PyTorch Parity

## Target Scope
Investigate SAM (Segment Anything Model) and Gemma embedding requirements across `ojas` (/Users/bharath/Code/research/ojas) and `tessl` (/Users/bharath/Code/research/tessl).
Enumerate all missing ops and kernels needed for both architectures.
Investigate reference fixtures in `ojas-oracle`, PyTorch parity harnesses, numerical tolerance criteria (Numerics::Exact vs Fast), CPU and Metal kernel implementations, and autograd tape requirements.

## Authoritative User Request
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`.

## Output
Write your findings to `handoff.md` in your working directory:
`/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`.
Then send a completion message back.

## 2026-10-31T17:13:20Z
[Message from parent 7f1f02db-7cbf-40d1-8919-b2741643d465]
You are Explorer 2 (Kernel Specification Explorer).
Your identity is teamwork_explorer.
Your assigned working directory is `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/`.
You own this folder exclusively. Write your progress to `progress.md` in your folder and your final report to `handoff.md` in your folder.

Authoritative User Request:
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`.
Read instructions in `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/DISPATCH.md`.

Target Codebases:
- `/Users/bharath/Code/research/ojas`
- `/Users/bharath/Code/research/tessl`

Tasks:
1. Thoroughly investigate SAM (Segment Anything Model) and Gemma embedding architectures.
2. Review existing ops and kernels in `ojas` (across `ojas-core`, `ojas-cpu`, `ojas-simd`, `ojas-metal`, `ojas-kernels`, etc.) and `tessl`.
3. Enumerate all missing operations and kernels required to run SAM and Gemma embedding with PyTorch parity.
4. Check GitPulse tasks for any existing tasks or notes mentioning SAM, Gemma, missing kernels, or parity issues.
5. Inspect `ojas-oracle` reference fixtures, torch golden generators, and parity gate test harnesses.
6. Detail numerical tolerance expectations, handling of `Numerics::Exact` vs Fast tiers, bit-identical reduction rules, and autograd tape integration.
7. Provide a detailed specification for each missing kernel (op name, inputs/outputs, shape rules, CPU implementation plan, Metal/GPU implementation plan, oracle parity test plan).
8. Write a comprehensive, structured handoff report to `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`.
9. Send a message to the orchestrator when finished.
