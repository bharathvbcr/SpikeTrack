# Dispatch: Milestone 5 Tier 5 Adversarial Coverage Hardening (Challenger 2)

## Context
Milestone 5 Phase 2: Adversarial Coverage Hardening (Tier 5).
All 142/142 tests across Tiers 1-4 are passing. Now, as Challenger 2, perform a white-box source coverage audit across Metal kernels, UMA shared memory pools, timeline fences, host/device leases, and concurrency stress in `ojas-metal` and `tessl`.

## Task
1. Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md` and `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`.
2. Inspect the implementation source in `ojas-metal/src/`, `ojas-metal/kernels/`, and `tessl/src/`:
   - Inspect GPU buffer leases, `SharedBuf`, `MappedBuffer`, and `Timeline` fence implementations.
   - Inspect Metal kernel status word checking (`ST_IN`, `ST_OUT`), device error propagation, and placement invariants (`OjasError::Placement`).
   - Inspect freelist cache trim under memory pressure and multi-threaded contention.
   - Find untested paths, concurrency hazards, fence race conditions, OOM recovery branches, zero-length device allocations.
3. Author an adversarial test suite in `ojas-metal/tests/adversarial_tier5_metal.rs` (or `tessl/tests/adversarial_tier5_concurrency.rs`) that directly exercises these untested branches, concurrency stress, and fence boundaries under `mac_heavy.sh` lock.
4. Run the tests using cargo test under `mac_heavy.sh`.
5. Produce a comprehensive gap report and handoff in:
   `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m5_2/handoff.md`
   Report any exposed bugs or gaps found. If all paths pass cleanly and no gaps remain, declare `APPROVE` / `NO_GAPS`.
6. Send a message to orchestrator (`7f1f02db-7cbf-40d1-8919-b2741643d465`) when complete.


## 2026-11-01T02:52:17Z
You are Challenger 2 for Milestone 5 Phase 2: Adversarial Coverage Hardening (Tier 5).
Your working directory is /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m5_2.
Initialize BRIEFING.md and progress.md there.

MANDATORY FIRST STEPS:
1. Read /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md and /Users/bharath/Code/research/SpikeTrack/PROJECT.md.
2. Read your task assignment in /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m5_2/DISPATCH.md.

YOUR MISSION:
Perform a white-box source code coverage audit across `ojas-metal` and `tessl`.
- Inspect code in `ojas-metal/src/`, `ojas-metal/kernels/`, and `tessl/src/`.
- Identify untested branches in GPU buffer lifecycle, UMA memory pools, `MappedBuffer`, `SharedBuf`, timeline fences, status word checks (`ST_IN`, `ST_OUT`), freelist cache trim under load, and multi-threaded contention.
- Write an empirical adversarial test suite in `ojas-metal/tests/adversarial_tier5_metal.rs` exercising these edge cases.
- Execute the tests under mac_heavy.sh lock (`bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh tier5_metal cargo test -p ojas-metal --test adversarial_tier5_metal -- --nocapture`).
- Verify formatting (`./scripts/fmt_check.sh`), tracked targets (`./scripts/check_tracked_targets.sh`), and clippy (`cargo clippy --workspace --all-targets -- -D warnings`).
- Produce a structured handoff report in:
  /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m5_2/handoff.md
  Following standard sections: Observation, Logic Chain, Caveats, Conclusion (APPROVE or REQUEST_CHANGES if gaps require implementation fixes), and Verification Method.
- Send a message to orchestrator (7f1f02db-7cbf-40d1-8919-b2741643d465) when complete.
