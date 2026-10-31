# Adversarial Verification & Challenge Report: Milestone 1 Workflow Scripts

**Author**: Challenger 1 (Adversarial Critic, `teamwork_challenger_m1_1`)  
**Target Codebases**: `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/tessl`, `/Users/bharath/Code/research/SpikeTrack`  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_challenger_m1_1`  
**Verdict**: **`REQUEST_CHANGES`**  

---

## Challenge Summary

**Overall risk assessment**: **HIGH** (Silent false-positive pass on invalid/typo op filter in `parity_check.sh`).

---

## 1. Observation

### 1.1 `dev_fast_check.sh` Adversarial Attacks
Target: `/Users/bharath/Code/research/ojas/scripts/dev_fast_check.sh`

1. **Non-existent crate name**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/dev_fast_check.sh non_existent_crate_xyz`
   - Exit code: `101`
   - Output:
     ```
     ==> [Fast Check] Target: non_existent_crate_xyz
     ==> [Stage 1/3] Syntax & Typecheck (cargo check)
     error: package ID specification `non_existent_crate_xyz` did not match any packages
     ```
   - Behavior: Fails immediately at Stage 1, does not proceed to Stages 2 or 3. Fail-closed verified.

2. **Corrupted / unknown flags**:
   - Command: `./scripts/dev_fast_check.sh --invalid-flag` -> Exit code `1`, outputs usage help.
   - Command: `./scripts/dev_fast_check.sh -x` -> Exit code `1`, outputs usage help.
   - Command: `./scripts/dev_fast_check.sh ""` -> Exit code `101` (`error: package name cannot be empty`).
   - Command: `./scripts/dev_fast_check.sh '; echo "pwned"'` -> Exit code `101` (`error: invalid character ';' in package name`). Safely quoted, no shell injection.

3. **Multi-crate mixed valid/invalid arguments**:
   - Command: `./scripts/dev_fast_check.sh ojas-core non_existent` -> Exit code `101`. Fails closed at Stage 1.

4. **Performance & timing under simulated 4-core CPU load**:
   - Test harness: 4 concurrent busy-loop Python threads running alongside fast check.
   - Target `ojas-core`: Completed in `0.49s` wall-clock (budget: < 15s).
   - Target `ojas-cpu`: Completed in `11.31s` wall-clock (budget: < 15s).

---

### 1.2 `parity_check.sh` Adversarial Attacks
Target: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`

1. **Invalid backend argument**:
   - Command: `./scripts/parity_check.sh --backend invalid_backend`
   - Exit code: `1`
   - Output: `ERROR: Invalid backend 'invalid_backend'. Must be 'cpu', 'metal', or 'all'.`
   - Command: `./scripts/parity_check.sh -b ""` -> Exit code `1`.

2. **Corrupted / unknown options**:
   - Command: `./scripts/parity_check.sh --corrupted-flag`
   - Exit code: `1`, displays usage manual.

3. **Non-existent / corrupted fixture paths**:
   - Command: `./scripts/parity_check.sh --fixtures /tmp/does_not_exist.safetensors`
   - Exit code: `1`, prints explicit error:
     ```
     ERROR: SAM 3.1 fixture file not found!
     Looked for: visem_22_f0.safetensors in SpikeTrack/fixtures/backbone/
     Set OJAS_SAM_FIXTURES or pass --fixtures <path>.
     ```
   - Command: `touch /tmp/corrupt_fixture.safetensors && ./scripts/parity_check.sh --fixtures /tmp/corrupt_fixture.safetensors --op gelu_erf`
   - Exit code: `101` (cargo test panic: `truncated file: header length needs 8 bytes, the file has 0`). Fail-closed verified.

4. **Non-existent / misspelled op filter (CRITICAL BUG)**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend cpu --op non_existent_op_foo_bar_xyz`
   - Exit code: `0`
   - Verbatim Output:
     ```
     ======================================================================
     ==> [Parity Check] PyTorch Oracle Parity Runner
         Backend:           cpu
         Op Filter:         non_existent_op_foo_bar_xyz
         Profile:           release
         Fixtures:          /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors (3.2G)
         GroupNorm:         /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors
     ======================================================================

     ==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu...
         Command: cargo test --release -p ojas-oracle --test sam_fixtures_cpu non_existent_op_foo_bar_xyz -- --ignored --nocapture
         Finished `release` profile [optimized] target(s) in 0.03s
          Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

     running 0 tests

     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s


     ======================================================================
     ==> [Parity Check] SUCCESS in 0s
     ======================================================================
     ```
   - Same behavior confirmed on Metal backend:
     Command: `./scripts/parity_check.sh --backend metal --op non_existent_op_foo_bar_xyz`
     Exit code: `0`
     Output: `running 0 tests`, `test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out`, `mac_heavy: finished with exit 0`, `==> [Parity Check] SUCCESS in 22s`.

---

### 1.3 Concurrent Lock Contention under `mac_heavy.sh`
Target: `/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh`

- Two concurrent processes executed under `mac_heavy.sh`:
  - Job 1 (`contention-holder`): Holds lock for 4 seconds.
  - Job 2 (`contention-waiter`): Launched 1 second after Job 1 while lock is active.
- Observed Log Output:
  - Job 1: `mac_heavy: 2026-10-31T17:48:53Z 'contention-holder' holds the lock (waited 0s; load 13.06, 1128 processes)`
  - Job 1 start/end timestamps: `HOLDER_LOCK_HELD 1793468934`, `HOLDER_LOCK_DONE 1793468938`
  - Job 2: `mac_heavy: 2026-10-31T17:48:54Z 'contention-waiter' waiting 0s for the lock; holder: label=contention-holder pid=68819 ...`
  - Job 2 waited until Job 1 released lock at 17:48:58Z:
    `mac_heavy: 2026-10-31T17:48:59Z 'contention-waiter' holds the lock (waited 4s; load 12.41, 1129 processes)`
  - Job 2 start/end timestamps: `CONTENDER_LOCK_HELD 1793468939`, `CONTENDER_LOCK_DONE 1793468940`
- Result: Strict mutual exclusion and queue serialization confirmed with zero race conditions.

---

### 1.4 GitPulse Task Reconciliation
- Called `gitpulse_list_tasks(repo_path: "/Users/bharath/Code/research/SpikeTrack", limit: 50)`:
  Returned `total: 29` active tasks on the board. Reconciliation verified.

---

## 2. Logic Chain

1. In `parity_check.sh`, lines 198-204 and lines 245-256 execute `cargo test` passing `${OP_FILTER}` as a test filter.
2. In `cargo test`, when an argument filter does not match any test function names, cargo tests 0 items (`running 0 tests`, `0 passed; 0 failed; 12 filtered out`) and exits with status code `0`.
3. Because `parity_check.sh` relies solely on `set -e` and cargo test's exit code, it interprets exit code `0` as success.
4. Consequently, passing a misspelled op name (e.g. `--op gelu_erff`, `--op non_existent_op`) reports `==> [Parity Check] SUCCESS in 0s` with exit `0`.
5. This violates the core system invariant: *"A check that could not run must never report the same result as a check that ran and passed — that is how 'approved' comes to mean 'unexamined.'"*
6. It also violates the Milestone 1 Challenger instruction in `DISPATCH.md`: *"Non-existent op name (must report test failure or not found cleanly)."*

---

## 3. Caveats

1. `dev_fast_check.sh` is robust: syntax checks, clippy lints, unit tests, flag handling, and timing under load are verified and pass all requirements without issues.
2. `mac_heavy.sh` mutual exclusion and lock queuing are robust and prevent concurrent GPU thrashing.
3. The defect in `parity_check.sh` is localized to test result validation when `--op` filter is supplied.

---

## 4. Conclusion & Required Changes

**Verdict**: **`REQUEST_CHANGES`**

Worker 1 must update `/Users/bharath/Code/research/ojas/scripts/parity_check.sh` to ensure fail-closed behavior when an op filter does not match any test:

### Required Remediation in `parity_check.sh`:
Inspect `cargo test` output in `run_cpu_parity()` and `run_metal_parity()`. If an `OP_FILTER` was provided and cargo runs 0 tests (e.g. matches `running 0 tests` or `0 passed; 0 failed`), the script must print an explicit error message and exit with a non-zero exit code (e.g., `exit 1`):

```bash
# Example fix for run_cpu_parity and run_metal_parity:
local output rc=0
output=$("${cmd[@]}" 2>&1) || rc=$?
echo "$output"
if [[ $rc -ne 0 ]]; then
  exit $rc
fi
if [[ -n "$OP_FILTER" ]] && echo "$output" | grep -q "running 0 tests"; then
  echo "======================================================================" >&2
  echo "ERROR: No parity tests matched op filter '$OP_FILTER'!" >&2
  echo "Available CPU parity tests in ojas-oracle/tests/sam_fixtures_cpu.rs:" >&2
  echo "  rope_tables, rope_interleaved, gelu_erf, windowed_attention, global_attention," >&2
  echo "  layer_norm, group_norm, linear_bias, conv2d, conv_transpose2d" >&2
  echo "======================================================================" >&2
  exit 1
fi
```

Once Worker 1 applies this fix and verifies that `./scripts/parity_check.sh --op non_existent_op` exits non-zero (`exit 1`), Challenger 1 will re-verify for final approval.

---

## 5. Verification Method

To independently verify the failure and proposed fix:

1. **Reproduce the Bug**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op non_existent_op_foo_bar_xyz
   echo "Exit status: $?"
   ```
   *Expected Current (Buggy) Output*: Exits with code `0` and prints `[Parity Check] SUCCESS in 0s`.

2. **Verify Correct Fail-Closed Behavior (Post-Fix)**:
   ```bash
   ./scripts/parity_check.sh --backend cpu --op non_existent_op_foo_bar_xyz
   echo "Exit status: $?"
   ```
   *Expected Post-Fix Output*: Exits with code `1`, prints `ERROR: No parity tests matched op filter 'non_existent_op_foo_bar_xyz'!`.

3. **Verify Valid Op Filter Still Passes**:
   ```bash
   ./scripts/parity_check.sh --backend cpu --op gelu_erf --release
   echo "Exit status: $?"
   ```
   *Expected Output*: Exits with code `0`, executes `gelu_erf_matches_torch`, and reports test results.
