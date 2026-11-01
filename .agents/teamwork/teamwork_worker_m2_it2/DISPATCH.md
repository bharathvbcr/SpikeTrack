# Dispatch: Milestone 2 Iteration 2 Remediation Worker

## Identity
- Working directory: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m2_it2`
- Role: `teamwork_worker` (Milestone 2 Iteration 2 Remediation Worker)
- Target Codebases: `/Users/bharath/Code/research/ojas` and `/Users/bharath/Code/research/tessl`

## Mandatory Integrity Warning
> DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Authoritative Context
- User Request: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`
- Master Plan: `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`
- Explorer 1 Handoff: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_1/handoff.md`
- Explorer 2 Handoff: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_2/handoff.md`
- Explorer 3 Handoff: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_3/handoff.md`
- Challenger 1 Feedback: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m2_1/handoff.md`
- Challenger 2 Feedback: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m2_2/handoff.md`

## Objectives
Implement the complete remediation plan for Milestone 2 Iteration 2 as detailed in the three Explorer reports:
1. Re-export `BudgetTrimHandler` in `ojas-core/src/lib.rs`.
2. Implement pool cache trimming across `tessl` (`GpuRuntime::trim_free_list`) and `ojas-metal` (`MetalTrimHandler`, `Cmd::TrimPool`, `Reply::Trimmed(u64)`, `NoOpTrim` for child budgets).
3. Implement `MappedBuffer`, `DeviceBuffer`, `HostLease`, and `HostLeaseMut` on `SharedBuf` in `tessl/src/shared_buf.rs` by wrapping `Arc<SharedBufInner>` holding `Option<Arc<PooledBuffer>>`, and implement `Timeline` for `HostFence` in `tessl/src/runtime/host_fence.rs`.
4. Fix the 8 clippy warnings in `ojas-io/src/half.rs` & `safetensors.rs`.
5. Fix the 2 clippy warnings in `tessl/src/loader.rs` & `tests/shared_runtime.rs`.
6. Fix the 2 wildcard dependency errors in `tessl/Cargo.toml` by specifying `version = "0.1.0"` for `ojas-core` and `ojas-io`.
7. Update benchmark plots and documentation via `go run scripts/benchreport/main.go plots && go run scripts/sitegen/main.go`.
8. Verify that all CI gates and tests pass: `cargo clippy`, `cargo deny check`, `./scripts/check_tracked_targets.sh`, `./scripts/fmt_check.sh`, `cargo test`, and `./scripts/ci_local.sh`.
9. Deliver comprehensive `handoff.md` in your working directory.

## 2026-10-31T20:01:54Z
You are the Milestone 2 Remediation Worker (Iteration 2) for the Ojas + Tessl unified architecture refactoring.

Your working directory is:
`/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m2_it2`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

First, read the authoritative context files:
1. User Request: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`
2. Master Project Plan: `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`
3. Dispatch Assignment: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m2_it2/DISPATCH.md`
4. Explorer 1 Handoff (BudgetTrimHandler & Cache Trim): `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_1/handoff.md`
5. Explorer 2 Handoff (MappedBuffer & SharedBuf Bridge): `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_2/handoff.md`
6. Explorer 3 Handoff (CI Gates & Lints): `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_it2_3/handoff.md`
7. Challenger 1 Feedback: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m2_1/handoff.md`
8. Challenger 2 Feedback: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m2_2/handoff.md`

Your tasks:
1. Re-export `BudgetTrimHandler`:
   In `/Users/bharath/Code/research/ojas/ojas-core/src/lib.rs` line 48:
   Change `pub use budget::{Budget, Reservation, Scratch};` to `pub use budget::{Budget, BudgetTrimHandler, Reservation, Scratch};`.

2. Freelist cache trim wiring:
   a. In `/Users/bharath/Code/research/tessl/src/runtime.rs`:
      Add `pub fn trim_free_list(&self) -> usize` on `GpuRuntime` using `self.pool.try_lock()` to safely trim and return bytes freed.
   b. In `/Users/bharath/Code/research/ojas/ojas-metal/src/link.rs`:
      Add `Trimmed(u64)` variant to `pub(crate) enum Reply`.
   c. In `/Users/bharath/Code/research/ojas/ojas-metal/src/device.rs`:
      Update `drop_pool(&self) -> u64` and `trim_pool(&self) -> Res<Reply>` to compute freed bytes via `before.saturating_sub(after)` and return `Reply::Trimmed(freed)`.
   d. In `/Users/bharath/Code/research/ojas/ojas-metal/src/backend.rs`:
      Implement `MetalTrimHandler` (implementing `BudgetTrimHandler`) calling `Cmd::TrimPool`, and register it on root `budget` in `MetalBackend::new(budget)` via `budget.set_trim_handler(...)`.
   e. In `/Users/bharath/Code/research/ojas/ojas-metal/src/pool_ledger.rs`:
      Install a `NoOpTrim` handler on `live` and `cache` child budgets in `BudgetLedger::new()` so child allocations don't recurse into root trim.

3. Bridge `tessl::shared_buf::SharedBuf` to `ojas_core::hal::MappedBuffer`:
   a. In `/Users/bharath/Code/research/tessl/src/runtime/host_fence.rs`:
      Implement `ojas_core::hal::Timeline for HostFence`.
   b. In `/Users/bharath/Code/research/tessl/src/shared_buf.rs`:
      Refactor `SharedBuf` to wrap `Arc<SharedBufInner>` (holding `_mtl`, `_pooled: Option<Arc<crate::tensor::PooledBuffer>>`, `fence: HostFence`, `last_use`, `last_write`, `lease`).
      Implement `DeviceBuffer`, `MappedBuffer`, `HostLease`, and `HostLeaseMut` for `SharedBuf` (and leases) following Explorer 2's concrete code.
      Add `into_shared_buf(self) -> SharedBuf` and `shared_buf(&self) -> SharedBuf` on `SharedAlloc`.

4. Fix Quality Gates and Lints:
   a. In `/Users/bharath/Code/research/ojas/ojas-io/src/half.rs`:
      Apply Explorer 3's Patch 1 (convert `chunks_exact_mut(4)` to `as_chunks_mut::<4>()` and `chunks_exact(2)` to `as_chunks::<2>()`).
   b. In `/Users/bharath/Code/research/ojas/ojas-io/src/safetensors.rs`:
      Apply Explorer 3's Patch 2 (convert `checked_mul` to `saturating_mul` at line 205, and convert `chunks_exact` to `as_chunks` at lines 529, 543, 557, 605, 648).
   c. In `/Users/bharath/Code/research/tessl/src/loader.rs`:
      Apply Explorer 3's Patch 3 (move `open_regular_file` before line 362, before `mod tests`).
   d. In `/Users/bharath/Code/research/tessl/tests/shared_runtime.rs`:
      Apply Explorer 3's Patch 4 (replace `.err().expect(...)` with `.expect_err(...)` at line 102).
   e. In `/Users/bharath/Code/research/tessl/Cargo.toml`:
      Apply Explorer 3's Patch 5 (add `version = "0.1.0"` to `ojas-core` and `ojas-io` dependencies).
   f. From `/Users/bharath/Code/research/ojas`:
      Run `go run scripts/benchreport/main.go plots && go run scripts/sitegen/main.go` to update committed plot artifacts.

5. Build & Test Verification:
   Run the following verification suite and document commands and output:
   - `cargo clippy --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml --workspace --all-targets -- -D warnings`
   - `cargo clippy --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml --all-targets -- -D warnings`
   - `cargo deny --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml check`
   - `cd /Users/bharath/Code/research/ojas && ./scripts/check_tracked_targets.sh`
   - `cd /Users/bharath/Code/research/ojas && ./scripts/fmt_check.sh`
   - `cargo test --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml`
   - `cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml --workspace --release -- --test-threads=1`
   - `cd /Users/bharath/Code/research/ojas && ./scripts/ci_local.sh`

Deliver your comprehensive handoff report at `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m2_it2/handoff.md`.
Use `send_message` to notify me when complete.

## 2026-10-31T20:12:18Z
**Context**: Milestone 2 Iteration 2 Remediation Implementation
**Content**: Status query: checking in on progress across Tasks 1-5 (BudgetTrimHandler, tessl MappedBuffer bridge, cache trim wiring, clippy & deny fixes, and test runs).
**Action**: Please report your current progress, any blockers encountered, and update your progress.md with latest timestamp.

## 2026-10-31T20:22:54Z
**Context**: Milestone 2 Iteration 2 Task 5 Test Verification
**Content**: Checking in on Task 5 progress (tessl and ojas workspace test runs and ci_local.sh). Please provide a status update and update your progress.md timestamp.
**Action**: Report test status and deliver handoff.md once test suite concludes.

## 2026-10-31T20:33:33Z
**Context**: Milestone 2 Iteration 2 Task 5 Test Verification
**Content**: Status check on Task 5 test verification (tessl test, ojas release test, and ci_local.sh). Please update your progress.md with current timestamp and status.
**Action**: Notify when test verification concludes and deliver handoff.md.

## 2026-10-31T20:34:15Z
**Context**: Milestone 2 Iteration 2 Task 5 Test Verification
**Content**: Understood. Awaiting completion of tessl and ojas test suites and delivery of handoff.md.
**Action**: Notify when handoff.md is delivered.

## 2026-10-31T20:49:20Z
**Context**: Milestone 2 Iteration 2 Task 5 Test Verification
**Content**: Status check on release test execution (tessl and ojas workspace). Please update progress.md with latest timestamp and current test suite status.
**Action**: Report progress and notify when handoff.md is ready.

## 2026-10-31T20:50:04Z
**Context**: Milestone 2 Iteration 2 Task 5 Test Verification
**Content**: Progress noted. Awaiting completion of tessl and ojas release test suites and delivery of handoff.md.
**Action**: Notify when handoff.md is ready.
