# Worker Dispatch: Milestone 1 — GitPulse Tasks & Accelerated Workflows

## Assigned Scope
Implement and execute Milestone 1:
1. **GitPulse Tasks Audit & Updates**:
   - Review GitPulse tasks on `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/tessl`, and `/Users/bharath/Code/research/SpikeTrack`.
   - Update task priority classifications (P0 blockers, P1 architecture & kernels, P2 follow-ups) and bug status based on Explorer 3 findings in `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_3/handoff.md`.
2. **Accelerated Workflows Implementation**:
   - Create and verify the 5 executable workflows designed by Explorer 3:
     - `WF1`: Fast Inner Development Loop (`/Users/bharath/Code/research/ojas/scripts/dev_fast_check.sh` - sub-15s check for any crate). Make executable (`chmod +x`).
     - `WF2`: Tiered Testing Pipeline documented and verified in `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md`.
     - `WF3`: PyTorch Parity Verification Loop script/helper (`/Users/bharath/Code/research/ojas/scripts/parity_check.sh`).
     - `WF4`: Metal Test Execution Workflow (using `tools/mac_heavy.sh` with exclusive GPU locking).
     - `WF5`: Cross-workspace synchronization protocol.
   - Run verification commands for the newly created scripts to ensure they work without error.

## Exclusive Write Ownership
- `/Users/bharath/Code/research/ojas/scripts/dev_fast_check.sh`
- `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`
- `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md`
- `/Users/bharath/Code/research/tessl/docs/WORKFLOWS.md`
- GitPulse tasks via GitPulse MCP tools.

## Mandatory Reading
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_3/handoff.md`

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Completion Requirements
- Run syntax and execution checks on all created scripts.
- Document build and verification commands and results in your handoff report.
- Write your final handoff report to `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m1/handoff.md`.
- Send a completion message back to the orchestrator.


## 2026-10-31T17:25:44Z
Received dispatch message from parent (7f1f02db-7cbf-40d1-8919-b2741643d465):
Worker 1 (Workflows & Tasks Worker).
Scope: Milestone 1 — GitPulse Tasks & Accelerated Workflows
