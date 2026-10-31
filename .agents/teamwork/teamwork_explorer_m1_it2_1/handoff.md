# Handoff Report: Fail-Closed Parity Check Strategy

**Author**: Explorer 1 (`teamwork_explorer_m1_it2_1`)  
**Target Codebase**: `/Users/bharath/Code/research/ojas`  
**Target Script**: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_1`  
**Date**: 2026-10-31T18:03:00Z  
**Type**: Hard Handoff (Investigation & Architecture Design Complete)

---

## 1. Observation

### 1.1 Verbatim Bug Reproduction in `parity_check.sh`
Target File: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh` (lines 195–205 and 207–256).

When an invalid, misspelled, or non-existent op name is passed to `./scripts/parity_check.sh`, the script reports complete success and exits with code `0`.

**Command**:
```bash
cd /Users/bharath/Code/research/ojas
./scripts/parity_check.sh --backend cpu --op non_existent_op_foo_bar_xyz
echo "Exit status: $?"
```

**Verbatim Output**:
```text
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
    Finished `release` profile [optimized] target(s) in 0.02s
     Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s


======================================================================
==> [Parity Check] SUCCESS in 0s
======================================================================
Exit status: 0
```

**Metal Backend Reproduction**:
```bash
./scripts/parity_check.sh --backend metal --op non_existent_op_foo_bar_xyz
echo "Exit status: $?"
```
```text
==> [Metal Parity] Running ojas-metal sam_fixtures_metal...
    GPU Lock: Wrapping with /Users/bharath/Code/research/ojas/../Lappi-decision/tools/mac_heavy.sh
mac_heavy: 2026-10-31T17:55:56Z 'ojas-parity-check' holds the lock (waited 0s; load 6.70, 1099 processes); running: cargo test --release -p ojas-metal --test sam_fixtures_metal non_existent_op_foo_bar_xyz -- --ignored --nocapture --test-threads=1
    Finished `release` profile [optimized] target(s) in 1.15s
     Running tests/sam_fixtures_metal.rs (target/release/deps/sam_fixtures_metal-bbf224a179ac8609)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s

mac_heavy: 2026-10-31T17:55:58Z 'ojas-parity-check' finished with exit 0

======================================================================
==> [Parity Check] SUCCESS in 2s
======================================================================
Exit status: 0
```

### 1.2 Inspection of Parity Suites
Examined tests in:
- `ojas-oracle/tests/sam_fixtures_cpu.rs`
- `ojas-metal/tests/sam_fixtures_metal.rs`

Both test binaries contain identical 12 test functions matching the following available op filters:
1. `rope_tables` (`rope_tables_match_the_recorded_freqs_cis`)
2. `rope_interleaved` (`rope_interleaved_matches_torch`)
3. `gelu_erf` (`gelu_erf_matches_torch`)
4. `windowed_attention` (`windowed_attention_matches_torch`)
5. `global_attention` (`global_attention_matches_torch`)
6. `layer_norm` (`layer_norm_rows_match_torch_with_the_affine_part_recovered` and `layer_norm_with_the_true_affine_matches_torch`)
7. `the_ops_the_fixture_cannot_check_are_recorded`
8. `linear_bias` (`linear_bias_matches_torch`)
9. `conv2d` (`conv2d_matches_torch`)
10. `conv_transpose2d` (`conv_transpose2d_matches_torch`)
11. `group_norm` (`group_norm_matches_torch_at_both_pixel_decoder_sizes`)

### 1.3 Analysis of Challenger 1's Proposal & Subtle Pitfalls
Challenger 1 proposed:
```bash
local output rc=0
output=$("${cmd[@]}" 2>&1) || rc=$?
echo "$output"
if [[ $rc -ne 0 ]]; then
  exit $rc
fi
if [[ -n "$OP_FILTER" ]] && echo "$output" | grep -q "running 0 tests"; then
  ...
  exit 1
fi
```
Investigation identified two critical limitations in this pattern:
1. **Loss of Real-time Streaming**: Subshell variable assignment `output=$(...)` buffers all stdout and stderr in memory until the test command exits. A full parity run across multi-gigabyte safetensors takes 20–60+ seconds. Buffering hides live execution tables and progress from the developer, and if interrupted or hung, nothing has been displayed.
2. **BSD macOS `mktemp` Suffix Behavior**: In exploring a file-based streaming alternative with `tee`, BSD `mktemp` on macOS requires template strings to end strictly with `XXXXXX`. Templates like `mktemp /tmp/ojas_XXXXXX.log` do not expand on macOS and write literal static filenames, causing silent path collisions. The template must be `mktemp -t ojas_parity.XXXXXX`.

---

## 2. Logic Chain

1. **Root Cause**:
   - In Rust's libtest runner (`cargo test`), filtering tests by an argument pattern is a selection mechanism. If no tests match the pattern, the test runner prints `running 0 tests`, prints `test result: ok. 0 passed; 0 failed; ...`, and exits with return code `0`.
   - `parity_check.sh` runs under `set -euo pipefail`. Because `cargo test` exits with code `0`, `set -e` does not trigger, and the script proceeds to the final block:
     `echo "==> [Parity Check] SUCCESS in ${TOTAL_TIME}s"`.
   - This violates the core invariant: *"A check that could not run must never report the same result as a check that ran and passed — that is how 'approved' comes to mean 'unexamined.'"*

2. **Requirements for Robust Remediation**:
   - **Fail-Closed**: If 0 tests execute (whether an op filter was misspelled, invalid, or no tests were selected), the script must fail with exit code `1` and output an explicit error listing the available op filters.
   - **Real-Time Streaming**: Parity tests produce substantial multi-row Markdown tables as they compute. The solution must stream output in real time so developers have immediate visibility.
   - **Signal and Failure Exit Code Preservation**: If a compilation fails or an assertion fails, the exact non-zero exit code (`101`, `1`, etc.) or `mac_heavy` lock status (`70`, `74`, `75`) must be preserved and propagated.
   - **Resource Safety / Zero Leakage**: Any temporary log or capture buffer must be cleaned up on all exit paths, including normal completion, test failure, `set -e` exits, and signals (`SIGINT` Ctrl+C, `SIGTERM`).
   - **DRY / Single Canonical Owner**: `run_cpu_parity()` and `run_metal_parity()` share the same test execution contract. Rather than copy-pasting checking logic across both, a unified helper (`run_test_suite`) must manage execution, streaming, logging, zero-test verification, and failure formatting.

3. **Technical Architecture of the Solution**:
   - Define a global `TMP_LOG` variable and a POSIX signal trap:
     ```bash
     TMP_LOG=""
     cleanup() {
       if [[ -n "${TMP_LOG:-}" && -f "$TMP_LOG" ]]; then
         rm -f "$TMP_LOG"
       fi
     }
     trap cleanup EXIT INT TERM
     ```
   - In `run_test_suite`:
     - Allocate temporary log: `TMP_LOG=$(mktemp -t ojas_parity.XXXXXX)`
     - Stream execution directly to terminal while recording to `$TMP_LOG`:
       `"${cmd[@]}" 2>&1 | tee "$TMP_LOG" || rc=$?`
       Under `set -o pipefail`, if `cmd` fails (e.g. exit 101), the pipeline returns 101.
       The `|| rc=$?` construct prevents `set -e` from aborting before log inspection.
     - If `$rc -ne 0`, clean up `$TMP_LOG` and exit with `exit $rc`.
     - Inspect `$TMP_LOG` for `running 0 tests`. If matched, print the error block with available tests and exit with `exit 1`.
     - Remove `$TMP_LOG` on successful runs.

---

## 3. Caveats

1. **Rust libtest output format**: This fix relies on cargo test printing `running 0 tests` when no tests are executed. This has been the invariant of Rust's standard libtest harness across all stable compiler releases.
2. **Scope boundary**: This investigation is read-only and targets the shell runner `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`. No changes to Rust test code or safetensors fixtures are required.
3. **Mac Heavy Lock Script**: In `run_metal_parity()`, `mac_heavy.sh` executes as a wrapper. Testing confirmed that `mac_heavy.sh` passes cargo test's exit code through untouched, so the pipeline properly captures and inspects output under GPU locking as well as unlocked modes.

---

## 4. Conclusion & Proposed Implementation

The proposed fix introduces `run_test_suite()`, unifies `run_cpu_parity()` and `run_metal_parity()`, provides real-time streaming, guarantees temporary file cleanup, and enforces fail-closed behavior when `--op` matches 0 tests.

### Proposed Code Changes in `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`

#### Before (Lines 195–256):
```bash
run_cpu_parity() {
  echo ""
  echo "==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu..."
  local cmd=(cargo test $PROFILE_FLAG -p ojas-oracle --test sam_fixtures_cpu)
  if [[ -n "$OP_FILTER" ]]; then
    cmd+=("$OP_FILTER")
  fi
  cmd+=(-- --ignored --nocapture)
  echo "    Command: ${cmd[*]}"
  "${cmd[@]}"
}

run_metal_parity() {
  echo ""
  echo "==> [Metal Parity] Running ojas-metal sam_fixtures_metal..."
  if [[ "$(uname -s)" != "Darwin" ]]; then
    if [[ "${OJAS_ALLOW_NO_GPU:-0}" == "1" ]]; then
      echo "    [SKIP] Not running on Darwin and OJAS_ALLOW_NO_GPU=1."
      return 0
    else
      echo "    ERROR: Metal backend tests require macOS Darwin (Apple Silicon)." >&2
      exit 1
    fi
  fi

  # Check GPU lock
  local lock_wrapper=()
  if [[ -n "${MAC_HEAVY_LABEL:-}" ]]; then
    echo "    GPU Lock: Active under MAC_HEAVY_LABEL='$MAC_HEAVY_LABEL'"
  elif [[ "$NO_GPU_LOCK" -eq 1 ]]; then
    echo "    GPU Lock: Bypassed (--no-gpu-lock)"
  else
    # Check for mac_heavy.sh in candidate paths
    local LOCK_SCRIPTS=(
      "$ROOT/../Lappi-decision/tools/mac_heavy.sh"
      "$ROOT/../tessl/tools/mac_heavy.sh"
      "${HOME}/Code/research/Lappi-decision/tools/mac_heavy.sh"
    )
    for s in "${LOCK_SCRIPTS[@]}"; do
      if [[ -x "$s" ]]; then
        lock_wrapper=("$s" "ojas-parity-check")
        echo "    GPU Lock: Wrapping with $s"
        break
      fi
    done
    if [[ "${#lock_wrapper[@]}" -eq 0 ]]; then
      echo "    Notice: mac_heavy.sh not found. Running with --test-threads=1 serialization."
    fi
  fi

  local base_cmd=(cargo test $PROFILE_FLAG -p ojas-metal --test sam_fixtures_metal)
  if [[ -n "$OP_FILTER" ]]; then
    base_cmd+=("$OP_FILTER")
  fi
  base_cmd+=(-- --ignored --nocapture --test-threads=1)

  if [[ "${#lock_wrapper[@]}" -gt 0 ]]; then
    "${lock_wrapper[@]}" "${base_cmd[@]}"
  else
    "${base_cmd[@]}"
  fi
}
```

#### After (Replace Lines 195–256 with):
```bash
TMP_LOG=""
cleanup() {
  if [[ -n "${TMP_LOG:-}" && -f "$TMP_LOG" ]]; then
    rm -f "$TMP_LOG"
  fi
}
trap cleanup EXIT INT TERM

run_test_suite() {
  local suite_label="$1"
  local test_file="$2"
  shift 2
  local cmd=("$@")

  TMP_LOG=$(mktemp -t ojas_parity.XXXXXX)
  local rc=0
  "${cmd[@]}" 2>&1 | tee "$TMP_LOG" || rc=$?

  if [[ $rc -ne 0 ]]; then
    rm -f "$TMP_LOG"
    TMP_LOG=""
    exit $rc
  fi

  if grep -q "running 0 tests" "$TMP_LOG"; then
    rm -f "$TMP_LOG"
    TMP_LOG=""
    echo "" >&2
    echo "======================================================================" >&2
    if [[ -n "$OP_FILTER" ]]; then
      echo "ERROR: No parity tests matched op filter '$OP_FILTER' in $suite_label suite!" >&2
    else
      echo "ERROR: 0 parity tests were executed in $suite_label suite!" >&2
    fi
    echo "Available parity tests in $test_file:" >&2
    echo "  rope_tables, rope_interleaved, gelu_erf, windowed_attention," >&2
    echo "  global_attention, layer_norm, group_norm, linear_bias," >&2
    echo "  conv2d, conv_transpose2d" >&2
    echo "======================================================================" >&2
    exit 1
  fi

  rm -f "$TMP_LOG"
  TMP_LOG=""
}

run_cpu_parity() {
  echo ""
  echo "==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu..."
  local cmd=(cargo test $PROFILE_FLAG -p ojas-oracle --test sam_fixtures_cpu)
  if [[ -n "$OP_FILTER" ]]; then
    cmd+=("$OP_FILTER")
  fi
  cmd+=(-- --ignored --nocapture)
  echo "    Command: ${cmd[*]}"

  run_test_suite "CPU" "ojas-oracle/tests/sam_fixtures_cpu.rs" "${cmd[@]}"
}

run_metal_parity() {
  echo ""
  echo "==> [Metal Parity] Running ojas-metal sam_fixtures_metal..."
  if [[ "$(uname -s)" != "Darwin" ]]; then
    if [[ "${OJAS_ALLOW_NO_GPU:-0}" == "1" ]]; then
      echo "    [SKIP] Not running on Darwin and OJAS_ALLOW_NO_GPU=1."
      return 0
    else
      echo "    ERROR: Metal backend tests require macOS Darwin (Apple Silicon)." >&2
      exit 1
    fi
  fi

  # Check GPU lock
  local lock_wrapper=()
  if [[ -n "${MAC_HEAVY_LABEL:-}" ]]; then
    echo "    GPU Lock: Active under MAC_HEAVY_LABEL='$MAC_HEAVY_LABEL'"
  elif [[ "$NO_GPU_LOCK" -eq 1 ]]; then
    echo "    GPU Lock: Bypassed (--no-gpu-lock)"
  else
    # Check for mac_heavy.sh in candidate paths
    local LOCK_SCRIPTS=(
      "$ROOT/../Lappi-decision/tools/mac_heavy.sh"
      "$ROOT/../tessl/tools/mac_heavy.sh"
      "${HOME}/Code/research/Lappi-decision/tools/mac_heavy.sh"
    )
    for s in "${LOCK_SCRIPTS[@]}"; do
      if [[ -x "$s" ]]; then
        lock_wrapper=("$s" "ojas-parity-check")
        echo "    GPU Lock: Wrapping with $s"
        break
      fi
    done
    if [[ "${#lock_wrapper[@]}" -eq 0 ]]; then
      echo "    Notice: mac_heavy.sh not found. Running with --test-threads=1 serialization."
    fi
  fi

  local base_cmd=(cargo test $PROFILE_FLAG -p ojas-metal --test sam_fixtures_metal)
  if [[ -n "$OP_FILTER" ]]; then
    base_cmd+=("$OP_FILTER")
  fi
  base_cmd+=(-- --ignored --nocapture --test-threads=1)

  local full_cmd=()
  if [[ "${#lock_wrapper[@]}" -gt 0 ]]; then
    full_cmd=("${lock_wrapper[@]}" "${base_cmd[@]}")
  else
    full_cmd=("${base_cmd[@]}")
  fi
  echo "    Command: ${full_cmd[*]}"

  run_test_suite "Metal" "ojas-metal/tests/sam_fixtures_metal.rs" "${full_cmd[@]}"
}
```

### Artifacts Delivered in Working Directory:
1. **Diff Patch File**:
   `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_1/parity_check_fail_closed.patch`
2. **Complete Replacement File**:
   `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_1/proposed_parity_check.sh`

---

## 5. Verification Method

Once Worker 1 applies the patch or replaces lines 195–256 in `/Users/bharath/Code/research/ojas/scripts/parity_check.sh`, execute the following verification steps:

### Test Case 1: CPU Parity with Misspelled / Unmatched Op Filter
```bash
cd /Users/bharath/Code/research/ojas
./scripts/parity_check.sh --backend cpu --op non_existent_op_foo_bar_xyz
echo "Exit: $?"
```
- **Expected Result**: Exits with code `1`. Prints:
  ```text
  ERROR: No parity tests matched op filter 'non_existent_op_foo_bar_xyz' in CPU suite!
  Available parity tests in ojas-oracle/tests/sam_fixtures_cpu.rs:
    rope_tables, rope_interleaved, gelu_erf, windowed_attention, ...
  ```
- **Failure Condition**: Exits with code `0` or prints `SUCCESS`.

### Test Case 2: Metal Parity with Misspelled / Unmatched Op Filter
```bash
cd /Users/bharath/Code/research/ojas
./scripts/parity_check.sh --backend metal --op non_existent_op_foo_bar_xyz
echo "Exit: $?"
```
- **Expected Result**: Exits with code `1`. Prints:
  ```text
  ERROR: No parity tests matched op filter 'non_existent_op_foo_bar_xyz' in Metal suite!
  Available parity tests in ojas-metal/tests/sam_fixtures_metal.rs:
    rope_tables, rope_interleaved, gelu_erf, windowed_attention, ...
  ```
- **Failure Condition**: Exits with code `0` or prints `SUCCESS`.

### Test Case 3: All Backends with Misspelled / Unmatched Op Filter
```bash
cd /Users/bharath/Code/research/ojas
./scripts/parity_check.sh --backend all --op non_existent_op_foo_bar_xyz
echo "Exit: $?"
```
- **Expected Result**: Fails immediately on CPU suite with exit code `1`, does not proceed to Metal, and does not print `SUCCESS`.

### Test Case 4: Valid Op Filter Passes with Full Streaming
```bash
cd /Users/bharath/Code/research/ojas
./scripts/parity_check.sh --backend cpu --op gelu_erf
echo "Exit: $?"
```
- **Expected Result**: Streams live output table, exits with code `0`, and reports `SUCCESS`.

### Test Case 5: Zero Leftover Files
```bash
ls -la /tmp/ojas_parity.* 2>/dev/null || echo "No leak"
```
- **Expected Result**: No leftover files in `/tmp` or `$TMPDIR`.
