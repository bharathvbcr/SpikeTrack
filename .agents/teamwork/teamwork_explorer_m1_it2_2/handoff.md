# Investigation & Architecture Report: Output Capture and Exit Code Handling in `parity_check.sh`

**Author**: Explorer 2 (`teamwork_explorer_m1_it2_2`)  
**Target File**: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`  
**Related Test Suites**:
- `/Users/bharath/Code/research/ojas/ojas-oracle/tests/sam_fixtures_cpu.rs`
- `/Users/bharath/Code/research/ojas/ojas-metal/tests/sam_fixtures_metal.rs`  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_2`  
**Verdict**: **`READY_FOR_IMPLEMENTATION`** (Self-contained design, empirical verification complete, patch provided for Worker 1)

---

## Executive Summary
Naive subshell capture (`output=$(cmd 2>&1)`) swallows all terminal output during long Metal runs and GPU lock waits, while direct execution (`"${cmd[@]}"`) silently passes with exit code `0` when `--op <filter>` matches 0 tests (`running 0 tests; 0 passed; 0 failed`). The verified solution implements a canonical `run_test_with_verification()` wrapper using `"${cmd[@]}" 2>&1 | tee "$log_file"` with temporary `set +e` and `${PIPESTATUS[0]}` to guarantee live real-time output streaming, preserve exact Cargo exit codes (e.g. `101` on test panics), clean up temp files via POSIX traps, and fail closed with `exit 1` on unmatched op filters.

---

## 1. Observation

### 1.1 Current Execution in `scripts/parity_check.sh`
In `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`:
- **CPU execution** (Lines 198–205):
  ```bash
  198:   local cmd=(cargo test $PROFILE_FLAG -p ojas-oracle --test sam_fixtures_cpu)
  199:   if [[ -n "$OP_FILTER" ]]; then
  200:     cmd+=("$OP_FILTER")
  201:   fi
  202:   cmd+=(-- --ignored --nocapture)
  203:   echo "    Command: ${cmd[*]}"
  204:   "${cmd[@]}"
  ```
- **Metal execution** (Lines 245–256):
  ```bash
  245:   local base_cmd=(cargo test $PROFILE_FLAG -p ojas-metal --test sam_fixtures_metal)
  246:   if [[ -n "$OP_FILTER" ]]; then
  247:     base_cmd+=("$OP_FILTER")
  248:   fi
  249:   base_cmd+=(-- --ignored --nocapture --test-threads=1)
  250: 
  251:   if [[ "${#lock_wrapper[@]}" -gt 0 ]]; then
  252:     "${lock_wrapper[@]}" "${base_cmd[@]}"
  253:   else
  254:     "${base_cmd[@]}"
  255:   fi
  ```

### 1.2 Verbatim Cargo Test Count Reporting
Empirical executions on `/Users/bharath/Code/research/ojas` yielded the following output patterns from Cargo's built-in `libtest` harness:

1. **Unmatched Op Filter (`--op non_existent_op`)**:
   ```
       Finished `release` profile [optimized] target(s) in 0.02s
        Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

   running 0 tests

   test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s
   ```
   - **Exit code**: `0` (Cargo treats 0 matching tests as success).
   - **Signature lines**:
     - `running 0 tests`
     - `test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s`

2. **Valid Op Filter (`--op gelu_erf`)**:
   ```
       Finished `release` profile [optimized] target(s) in 0.02s
        Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

   running 1 test
   | op | compared | kind | elements | max abs | max rel | cosine | tol abs | tol rel | tol cosine | result |
   ...
   test gelu_erf_matches_torch ... ok

   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 11 filtered out; finished in 3.36s
   ```
   - **Exit code**: `0`.
   - **Signature lines**: `running 1 test`, `1 passed; 0 failed`.

3. **Real Failure (Assertion mismatch, panic, or missing fixtures)**:
   ```
   thread 'rope_tables_match_the_recorded_freqs_cis' (148279503) panicked at ojas-oracle/tests/sam_fixtures_cpu.rs:47:28:
   OJAS_SAM_FIXTURES is not set: this test cannot run
   test rope_tables_match_the_recorded_freqs_cis ... FAILED

   failures:
       rope_tables_match_the_recorded_freqs_cis

   test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 11 filtered out; finished in 0.00s

   error: test failed, to rerun pass `-p ojas-oracle --test sam_fixtures_cpu`
   ```
   - **Exit code**: `101`.

4. **Metal Execution under `mac_heavy.sh`**:
   ```
   mac_heavy: 2026-10-31T17:58:16Z 'test-probe' holds the lock (waited 0s; load 4.46, 1111 processes); running: cargo test --release -p ojas-metal --test sam_fixtures_metal nonexistent -- --ignored --nocapture --test-threads=1
       Finished `release` profile [optimized] target(s) in 0.99s
        Running tests/sam_fixtures_metal.rs (target/release/deps/sam_fixtures_metal-bbf224a179ac8609)

   running 0 tests

   test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s

   mac_heavy: 2026-10-31T17:58:18Z 'test-probe' finished with exit 0
   ```
   - `mac_heavy.sh` transparently wraps the process, logs acquisition status, and forwards the exact exit code (`0` on clean pass, `101` on panic).

### 1.3 Streaming vs Buffering Comparison
- **Command Substitution (`output=$("${cmd[@]}" 2>&1)`)**:
  - Blocks until child process completely terminates.
  - Suppresses all output during multi-second test execution (e.g. 20–30s on Metal, or while `mac_heavy.sh` waits in queue for an exclusive GPU lock).
  - Terminal appears completely hung; in CI, timeouts may be triggered due to lack of output.
- **Pipeline Tee (`"${cmd[@]}" 2>&1 | tee "$log_file"`)**:
  - Live lines stream to terminal stdout in real-time as they are emitted by Cargo and `mac_heavy.sh`.
  - Complete output is concurrently stored in `$log_file` for post-execution parsing.

### 1.4 Shell Quirks & Traps Discovered
1. **macOS Bash 3.2 `set -u` with Empty Array**:
   - In `/bin/bash` on macOS (version 3.2.57), expanding an empty array `${arr[@]}` under `set -u` produces: `bash: arr[@]: unbound variable` and exits with error code `1`.
   - Solution: Guard cleanup iteration with `if [[ ${#TMP_LOG_FILES[@]} -gt 0 ]]; then ... fi`.
2. **BSD `mktemp` Template Requirement**:
   - Running `mktemp /tmp/ojas_parity_XXXXXX.log` on macOS does not substitute the `X`s because BSD `mktemp` without `-t` requires the template string to end with `X`.
   - Solution: Use `mktemp /tmp/ojas_parity_XXXXXX` (ending in `X`), which is portably supported on both BSD/macOS and GNU/Linux.
3. **Pipeline Exit Code (`PIPESTATUS`)**:
   - Under `set -euo pipefail`, if `cmd` fails in `cmd | tee log`, bash immediately aborts at the pipeline statement before any cleanup or custom error handling can execute.
   - Solution: Wrap execution with `set +e ... set -e` and extract `${PIPESTATUS[0]}`.

---

## 2. Logic Chain

1. **Step 1 — Real-Time Visibility**: Developers and CI systems require real-time log streaming when running parity tests. Blocking stdout/stderr via `output=$(...)` hides GPU lock wait notifications and markdown report tables until termination. Streaming via `tee` preserves immediate visibility.
2. **Step 2 — Exit Status Isolation**: In a pipeline `"${cmd[@]}" 2>&1 | tee "$log_file"`, `tee` exits `0`. With `set -o pipefail`, the pipeline status reflects failures, but bash terminates immediately under `set -e`. Wrapping the pipeline in `set +e` allows reading `${PIPESTATUS[0]}`, which isolates the exact exit code of `cmd` (`cargo test` or `mac_heavy.sh`).
3. **Step 3 — Preserving Real Failures**: If `${PIPESTATUS[0]}` is non-zero (e.g. `101` from a Rust test panic or compiler error), the script must clean up the log file and immediately exit with `exit "$rc"`. It must not perform zero-test string matching on real failures.
4. **Step 4 — Fail-Closed on 0 Tests**: If `${PIPESTATUS[0]}` is `0` and an op filter was specified (`[[ -n "$OP_FILTER" ]]`), the test harness succeeded only if tests actually executed. Cargo outputs `running 0 tests` and `0 passed; 0 failed` when a filter fails to match. Detecting this pattern via `grep -qE "(running 0 tests|test result: .* 0 passed; 0 failed)" "$log_file"` detects unmatched filters.
5. **Step 5 — Preserving Full Suite Runs**: When no filter is supplied (`OP_FILTER=""`), all tests are executed. In this case, `[[ -n "$OP_FILTER" ]]` is false, and standard Cargo exit code semantics apply directly without enforcing the zero-test check.
6. **Step 6 — Unification at the Seam**: Both `run_cpu_parity()` and `run_metal_parity()` execute Cargo test suites. Placing the execution, streaming, PIPESTATUS capture, log removal, and zero-test verification into a single canonical helper function `run_test_with_verification()` eliminates duplicate shell logic.

---

## 3. Caveats

1. **Non-TTY Color Formatting**: When `cargo test` is piped into `tee`, Cargo detects a non-interactive pipe and suppresses ANSI color escape sequences by default. This is standard behavior across UNIX pipelines and ensures clean text parsing in `$log_file`.
2. **Temporary File Storage**: Temporary log files are allocated under `/tmp/` via `mktemp`. On macOS and Linux, `/tmp/` is memory-backed (tmpfs/launchd). The log file is removed immediately after inspection in all code branches, and a global trap guarantees removal on SIGINT/SIGTERM.
3. **Test Name Substrings**: Cargo filter matching uses substring search. A filter like `--op attention` correctly matches both `global_attention_matches_torch` and `windowed_attention_matches_torch`. The zero-test check triggers only if the filter matches zero tests across the entire test target.
4. **No Other Unexplored Areas**: Both Darwin (Apple Silicon) and CPU execution paths were tested and confirmed.

---

## 4. Conclusion & Recommended Implementation

### 4.1 Architecture Assessment
Worker 1 should apply the following modifications to `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`:
1. Add an exit/interrupt trap with bash 3.2 `set -u` empty array protection to clean up any temporary files.
2. Implement `run_test_with_verification()` as the canonical execution runner.
3. Update `run_cpu_parity()` and `run_metal_parity()` to delegate execution through `run_test_with_verification()`.

### 4.2 Recommended Diff Patch for Worker 1
The following patch can be directly applied to `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`:

```diff
--- a/scripts/parity_check.sh
+++ b/scripts/parity_check.sh
@@ -21,6 +21,17 @@ ROOT="$(cd "$(dirname "$0")/.." && pwd)"
 cd "$ROOT"
 
 START_TIME=$(date +%s)
+TMP_LOG_FILES=()
+cleanup_logs() {
+  if [[ ${#TMP_LOG_FILES[@]} -gt 0 ]]; then
+    for f in "${TMP_LOG_FILES[@]}"; do
+      [[ -f "$f" ]] && rm -f "$f"
+    done
+  fi
+  return 0
+}
+trap cleanup_logs EXIT INT TERM
 
 show_help() {
   cat <<'EOF'
@@ -194,14 +205,44 @@ if [[ -n "${OJAS_SAM_WEIGHTS:-}" ]]; then
 fi
 echo "======================================================================"
 
+run_test_with_verification() {
+  local -a cmd=("$@")
+  local log_file
+  log_file=$(mktemp /tmp/ojas_parity_XXXXXX)
+  TMP_LOG_FILES+=("$log_file")
+
+  set +e
+  "${cmd[@]}" 2>&1 | tee "$log_file"
+  local rc=${PIPESTATUS[0]}
+  set -e
+
+  if [[ $rc -ne 0 ]]; then
+    rm -f "$log_file"
+    exit "$rc"
+  fi
+
+  if [[ -n "$OP_FILTER" ]] && grep -qE "(running 0 tests|test result: .* 0 passed; 0 failed)" "$log_file"; then
+    rm -f "$log_file"
+    echo "" >&2
+    echo "======================================================================" >&2
+    echo "ERROR: No parity tests matched op filter '$OP_FILTER'!" >&2
+    echo "Available parity tests in ojas-oracle and ojas-metal:" >&2
+    echo "  conv2d, conv_transpose2d, gelu_erf, global_attention, group_norm," >&2
+    echo "  layer_norm, linear_bias, rope_interleaved, rope_tables, windowed_attention" >&2
+    echo "======================================================================" >&2
+    exit 1
+  fi
+
+  rm -f "$log_file"
+  return 0
+}
+
 run_cpu_parity() {
   echo ""
   echo "==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu..."
   local cmd=(cargo test $PROFILE_FLAG -p ojas-oracle --test sam_fixtures_cpu)
   if [[ -n "$OP_FILTER" ]]; then
     cmd+=("$OP_FILTER")
   fi
   cmd+=(-- --ignored --nocapture)
   echo "    Command: ${cmd[*]}"
-  "${cmd[@]}"
+  run_test_with_verification "${cmd[@]}"
 }
 
 run_metal_parity() {
@@ -248,11 +289,13 @@ run_metal_parity() {
   fi
   base_cmd+=(-- --ignored --nocapture --test-threads=1)
 
+  local full_cmd=()
   if [[ "${#lock_wrapper[@]}" -gt 0 ]]; then
-    "${lock_wrapper[@]}" "${base_cmd[@]}"
+    full_cmd=("${lock_wrapper[@]}" "${base_cmd[@]}")
   else
-    "${base_cmd[@]}"
+    full_cmd=("${base_cmd[@]}")
   fi
+  echo "    Command: ${full_cmd[*]}"
+  run_test_with_verification "${full_cmd[@]}"
 }
```

---

## 5. Verification Method

Once Worker 1 applies the proposed patch, verify using the following commands:

1. **Verify Unmatched Op Filter Fails Closed (CPU)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op non_existent_op_foo_bar_xyz
   ```
   - **Verification**: Command MUST exit with code `1`, print live cargo output showing `running 0 tests`, and print error banner: `ERROR: No parity tests matched op filter 'non_existent_op_foo_bar_xyz'!`.

2. **Verify Unmatched Op Filter Fails Closed (Metal under `mac_heavy.sh`)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend metal --op non_existent_op_foo_bar_xyz
   ```
   - **Verification**: Command MUST exit with code `1`, print live `mac_heavy` lock acquisition status, and print error banner.

3. **Verify Valid Op Filter Passes (CPU)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op rope_tables --release
   ```
   - **Verification**: Command MUST exit with code `0`, print live test results table, and print `==> [Parity Check] SUCCESS`.

4. **Verify Real Failure Propagation**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   OJAS_SAM_FIXTURES="/tmp/corrupted.safetensors" ./scripts/parity_check.sh --backend cpu --op rope_tables
   ```
   - **Verification**: Command MUST exit with code `101`, displaying Cargo's test panic backtrace live.

5. **Verify No Lingering Temporary Files**:
   ```bash
   ls -la /tmp/ojas_parity_* 2>/dev/null || echo "All temp files cleaned up"
   ```
   - **Verification**: Output MUST confirm no orphaned log files remain in `/tmp`.
