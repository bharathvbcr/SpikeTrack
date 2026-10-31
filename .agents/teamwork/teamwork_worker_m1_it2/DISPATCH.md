# Worker Dispatch: Milestone 1 Iteration 2 — Parity Runner Fail-Closed Fix & Docs Alignment

## Assigned Scope
Implement the fixes and alignments designed by the Iteration 2 Explorers:
1. **Fix `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`**:
   - Apply the fail-closed verification implementation designed by Explorer 1 and Explorer 2 (`run_test_with_verification`).
   - Ensure live terminal output streaming via `tee`, POSIX signal trap cleanup, exit code preservation (`${PIPESTATUS[0]}`), and fail-closed exit with code 1 if an op filter is provided and matches 0 tests (`running 0 tests` or `0 passed; 0 failed`).
2. **Update Documentation in `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md`**:
   - Apply Explorer 3's clarification partitioning fixtures into "Required Activation Fixtures (MANDATORY)" (`visem_22_f0.safetensors`, 3.2 GB) and "Model Weights Checkpoint (OPTIONAL for op parity)" (`sam3.1_multiplex.safetensors`).
   - Ensure `go run scripts/sitegen/main.go -check` in `/Users/bharath/Code/research/ojas` passes with exit code 0.
3. **Execute Verification Tests**:
   - Test `./scripts/parity_check.sh --backend cpu --op non_existent_op` -> MUST exit with code 1!
   - Test `./scripts/parity_check.sh --backend metal --op non_existent_op` -> MUST exit with code 1!
   - Test `./scripts/parity_check.sh --backend cpu --op gelu_erf --release` -> MUST pass and exit 0!
   - Test `./scripts/dev_fast_check.sh ojas-core` -> MUST pass and exit 0!
   - Test `go run scripts/sitegen/main.go -check` -> MUST pass and exit 0!

## Exclusive Write Ownership
- `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`
- `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md`
- `/Users/bharath/Code/research/tessl/docs/WORKFLOWS.md`
- `/Users/bharath/Code/research/ojas/scripts/sitegen/main.go` (if updating docCategory)

## Mandatory Reading
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m1_1/handoff.md`
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_1/handoff.md`
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_2/handoff.md`
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_3/handoff.md`

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Completion Requirements
- Run and document all verification commands and outputs in your handoff report.
- Deliver your report to `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m1_it2/handoff.md`.
- Send a completion message back to the orchestrator.


## 2026-10-31T18:03:50Z
Dispatched by parent orchestrator (7f1f02db-7cbf-40d1-8919-b2741643d465).
Task: Parity Runner Fail-Closed Fix & Docs Alignment.
