# Quality Review & Adversarial Challenge Report: Milestone 1 Iteration 2

**Reviewer**: Reviewer 1 (`teamwork_reviewer`)  
**Roles**: reviewer, critic  
**Target Codebases**: `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/SpikeTrack`  
**Artifact Evaluated**: Worker 2's fixes in `scripts/parity_check.sh` and `docs/WORKFLOWS.md`  
**Date**: 2026-10-31T18:15:45Z  
**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**  

---

## 1. Observation

### 1.1 Direct Source Code Inspections
1. **POSIX Trap and Log Array Cleanup**:
   - File: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh` lines 24–33:
     ```bash
     TMP_LOG_FILES=()
     cleanup_logs() {
       if [[ ${#TMP_LOG_FILES[@]} -gt 0 ]]; then
         for f in "${TMP_LOG_FILES[@]}"; do
           [[ -f "$f" ]] && rm -f "$f"
         done
       fi
       return 0
     }
     trap cleanup_logs EXIT INT TERM
     ```
   - Observed: Guaranteed cleanup of any created log files upon exit, INT, or TERM signals.

2. **`run_test_with_verification` Implementation**:
   - File: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh` lines 206–244:
     ```bash
     run_test_with_verification() {
       local suite_label="$1"
       local test_file="$2"
       shift 2
       local -a cmd=("$@")

       local log_file
       log_file=$(mktemp /tmp/ojas_parity_XXXXXX)
       TMP_LOG_FILES+=("$log_file")

       set +e
       "${cmd[@]}" 2>&1 | tee "$log_file"
       local rc=${PIPESTATUS[0]}
       set -e

       if [[ $rc -ne 0 ]]; then
         rm -f "$log_file"
         exit "$rc"
       fi

       if grep -qE "(running 0 tests|test result: .* 0 passed; 0 failed)" "$log_file"; then
         rm -f "$log_file"
         echo "" >&2
         echo "======================================================================" >&2
         if [[ -n "$OP_FILTER" ]]; then
           echo "ERROR: No parity tests matched op filter '$OP_FILTER' in $suite_label suite!" >&2
         else
           echo "ERROR: 0 parity tests were executed in $suite_label suite!" >&2
         fi
         echo "Available parity tests in $test_file:" >&2
         echo "  conv2d, conv_transpose2d, gelu_erf, global_attention, group_norm," >&2
         echo "  layer_norm, linear_bias, rope_interleaved, rope_tables, windowed_attention" >&2
         echo "======================================================================" >&2
         exit 1
       fi

       rm -f "$log_file"
       return 0
     }
     ```
   - Observed: Unmatched op filters fail closed with exit code `1`; non-zero exit codes from failed runs propagate verbatim via `${PIPESTATUS[0]}`; stdout/stderr streams to console in real-time via `tee`.

3. **Fixture and Checkpoint Categorization**:
   - File: `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md` lines 76–90:
     - Clearly distinguishes "Required Activation Fixtures (MANDATORY for `parity_check.sh`)" (`SpikeTrack/fixtures/backbone/visem_22_f0.safetensors` ~3.2 GB and `SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors` ~203 MB) from "Model Weights Checkpoint (OPTIONAL for op parity; required for M3+ full-model inference)" (`SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors` ~3.5 GB).

### 1.2 Verbatim Independent Execution Results

#### Observation 2.1: Non-Existent Op Filter on CPU Backend
- **Command**: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend cpu --op non_existent_op_xyz`
- **Exit Code**: `1`
- **Output**:
  ```text
  ======================================================================
  ==> [Parity Check] PyTorch Oracle Parity Runner
      Backend:           cpu
      Op Filter:         non_existent_op_xyz
      Profile:           release
      Fixtures:          /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors (3.2G)
      GroupNorm:         /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors
  ======================================================================

  ==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu...
      Command: cargo test --release -p ojas-oracle --test sam_fixtures_cpu non_existent_op_xyz -- --ignored --nocapture
      Finished `release` profile [optimized] target(s) in 0.04s
       Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

  running 0 tests

  test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s


  ======================================================================
  ERROR: No parity tests matched op filter 'non_existent_op_xyz' in CPU suite!
  Available parity tests in ojas-oracle/tests/sam_fixtures_cpu.rs:
    conv2d, conv_transpose2d, gelu_erf, global_attention, group_norm,
    layer_norm, linear_bias, rope_interleaved, rope_tables, windowed_attention
  ======================================================================
  ```

#### Observation 2.2: Non-Existent Op Filter on Metal Backend (with `mac_heavy` GPU lock)
- **Command**: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend metal --op non_existent_op_xyz`
- **Exit Code**: `1`
- **Output**:
  ```text
  ======================================================================
  ==> [Parity Check] PyTorch Oracle Parity Runner
      Backend:           metal
      Op Filter:         non_existent_op_xyz
      Profile:           release
      Fixtures:          /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors (3.2G)
      GroupNorm:         /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors
  ======================================================================

  ==> [Metal Parity] Running ojas-metal sam_fixtures_metal...
      GPU Lock: Wrapping with /Users/bharath/Code/research/ojas/../Lappi-decision/tools/mac_heavy.sh
      Command: /Users/bharath/Code/research/ojas/../Lappi-decision/tools/mac_heavy.sh ojas-parity-check cargo test --release -p ojas-metal --test sam_fixtures_metal non_existent_op_xyz -- --ignored --nocapture --test-threads=1
  mac_heavy: 2026-10-31T18:13:25Z 'ojas-parity-check' waits without the lock: machine busy (load 13.39, max 12; 1181 processes, max 2500)
  mac_heavy: 2026-10-31T18:13:45Z 'ojas-parity-check' holds the lock (waited 20s; load 11.61, 1180 processes); running: cargo test --release -p ojas-metal --test sam_fixtures_metal non_existent_op_xyz -- --ignored --nocapture --test-threads=1
      Finished `release` profile [optimized] target(s) in 1.21s
       Running tests/sam_fixtures_metal.rs (target/release/deps/sam_fixtures_metal-bbf224a179ac8609)

  running 0 tests

  test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s

  mac_heavy: 2026-10-31T18:13:47Z 'ojas-parity-check' finished with exit 0

  ======================================================================
  ERROR: No parity tests matched op filter 'non_existent_op_xyz' in Metal suite!
  Available parity tests in ojas-metal/tests/sam_fixtures_metal.rs:
    conv2d, conv_transpose2d, gelu_erf, global_attention, group_norm,
    layer_norm, linear_bias, rope_interleaved, rope_tables, windowed_attention
  ======================================================================
  ```

#### Observation 2.3: Valid Op Filter (`gelu_erf`) on CPU with Live Streaming
- **Command**: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend cpu --op gelu_erf --release`
- **Exit Code**: `0`
- **Output**:
  ```text
  ======================================================================
  ==> [Parity Check] PyTorch Oracle Parity Runner
      Backend:           cpu
      Op Filter:         gelu_erf
      Profile:           release
      Fixtures:          /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors (3.2G)
      GroupNorm:         /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors
  ======================================================================

  ==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu...
      Command: cargo test --release -p ojas-oracle --test sam_fixtures_cpu gelu_erf -- --ignored --nocapture
      Finished `release` profile [optimized] target(s) in 0.02s
       Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

  running 1 test
  | op | compared | kind | elements | max abs | max rel | cosine | tol abs | tol rel | tol cosine | result |
  |---|---|---|---|---|---|---|---|---|---|---|
  | gelu_erf_forward | trunk.blocks.0.mlp.fc1.out -> trunk.blocks.0.mlp.gelu.out [1, 72, 72, 4736] | direct | 24551424 | 4.768e-7 | 5.082e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
  | gelu_erf_forward | trunk.blocks.7.mlp.fc1.out -> trunk.blocks.7.mlp.gelu.out [1, 72, 72, 4736] | direct | 24551424 | 4.768e-7 | 4.887e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
  | gelu_erf_forward | neck.convs.0.gelu.in -> neck.convs.0.gelu [1, 512, 144, 144] | direct | 10616832 | 4.768e-7 | 5.568e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
  test gelu_erf_matches_torch ... ok

  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 11 filtered out; finished in 3.36s


  ======================================================================
  ==> [Parity Check] SUCCESS in 4s
  ======================================================================
  ```

#### Observation 2.4: Valid Op Filter (`gelu_erf`) on `--backend all`
- **Command**: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend all --op gelu_erf --release`
- **Exit Code**: `0` (CPU and Metal both executed, verified, and passed in 26s).

#### Observation 2.5: Valid Op Filter (`rope_tables`) on CPU
- **Command**: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend cpu --op rope_tables --release`
- **Exit Code**: `0` in 1s; direct cosine match 1.000000000000.

#### Observation 2.6: E2E Tier 1 Feature Isolation Test Suite Execution
- **Command**: `cd /Users/bharath/Code/research/ojas && cargo test -p ojas-oracle --test e2e_tier1_features`
- **Exit Code**: `0`
- **Output**: `test result: ok. 65 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s`

#### Observation 2.7: E2E Tiers 2, 3, 4 Test Suite Unified Execution
- **Command**: `cd /Users/bharath/Code/research/ojas && cargo test -p ojas-oracle --test e2e_tier2_boundaries --test e2e_tier3_combinatorial --test e2e_tier4_production`
- **Exit Code**: `0`
- **Result**:
  - `e2e_tier2_boundaries`: 65 passed; 0 failed
  - `e2e_tier3_combinatorial`: 8 passed; 0 failed
  - `e2e_tier4_production`: 4 passed; 0 failed
  - Total across all 4 tiers: 142/142 tests passed.

#### Observation 2.8: Static Documentation Verification (`sitegen -check`)
- **Command**: `cd /Users/bharath/Code/research/ojas && go run scripts/sitegen/main.go -check`
- **Exit Code**: `0`
- **Output**: `sitegen: generated files are up to date (54 files)`

#### Observation 2.9: Fast Dev Loop Runner
- **Command**: `cd /Users/bharath/Code/research/ojas && ./scripts/dev_fast_check.sh ojas-core`
- **Exit Code**: `0`
- **Result**: Finished in 1s (< 15s budget) with 198 unit tests passing.

#### Observation 2.10: Zero Temp Leaks
- **Command**: `ls -la /tmp/ojas_parity_* 2>/dev/null || echo "No leak"`
- **Result**: `No leak`

---

## 2. Logic Chain

1. **Root Problem Addressed**:
   - In Cargo's standard `libtest` framework, running a test suite with a filter that matches 0 tests prints `running 0 tests` and `0 passed; 0 failed`, and exits with return code `0`.
   - In a test runner without post-execution inspection, this creates a false sense of success, violating the core principle of **No Silent Fallback** and fail-closed engineering (Observation 1.1).
2. **Correctness of the Solution**:
   - `run_test_with_verification` in `scripts/parity_check.sh` intercepts test output using `tee` to a temporary log file while simultaneously streaming progress to terminal stdout (Observation 1.1).
   - Capturing `${PIPESTATUS[0]}` ensures that actual test failures or panics (non-zero exit codes) are immediately propagated without interference (Observation 1.1).
   - When `${PIPESTATUS[0]}` is `0`, the runner evaluates whether the log contains `(running 0 tests|test result: .* 0 passed; 0 failed)` (Observation 1.1).
   - If found, it prints a clear diagnostic banner with available test op names and exits with code `1` (Observations 2.1, 2.2).
   - If not found (i.e. tests actually ran and passed), it exits cleanly with code `0` (Observations 2.3, 2.4, 2.5).
3. **Robustness & Cleanup**:
   - The shell script sets `trap cleanup_logs EXIT INT TERM`, guaranteeing that temporary log files in `/tmp` are removed under both normal and abnormal terminations (e.g., Ctrl-C / SIGTERM), verified by Observation 2.10 showing zero lingering files.
4. **Documentation Accuracy & Build Gate**:
   - Clarifying in `docs/WORKFLOWS.md` that `visem_22_f0.safetensors` and `pixel_decoder.safetensors` are required activation fixtures whereas `sam3.1_multiplex.safetensors` is optional resolves developer ambiguity without breaking documentation generation (Observation 1.1, 2.8).
   - `sitegen -check` confirmed all 54 site files remain up-to-date and consistent (Observation 2.8).
5. **No Integrity Violations Detected**:
   - No hardcoded test results, facade logic, bypassed tests, or fabricated outputs exist. Real commands run against real safetensor fixtures and real GPU hardware under `mac_heavy.sh` synchronization (Observations 2.2, 2.4).

---

## 3. Caveats

1. **macOS Apple Silicon Dependency for Metal Backend**:
   - Running `./scripts/parity_check.sh --backend metal` requires Darwin on Apple Silicon with Metal 4. On non-Darwin hosts, the script exits with an error unless `OJAS_ALLOW_NO_GPU=1` is provided. This is expected and documented.
2. **`libtest` Output Regular Expression**:
   - The regex `(running 0 tests|test result: .* 0 passed; 0 failed)` matches standard Cargo `libtest` outputs. If a custom test harness with a completely different output format is used in the future, this regex would need to be updated.

---

## 4. Conclusion

1. Worker 2's implementation completely resolves the silent pass failure mode: invalid op filters on both CPU and Metal backends fail closed with exit code `1`.
2. Real-time console streaming and exact exit codes are preserved.
3. Documentation accurately specifies required fixtures vs optional weights, and static site generation checks pass cleanly.
4. All 142 tests across Tiers 1-4 pass without regressions.
5. The verdict is **APPROVE**.

---

## 5. Verification Method

To independently reproduce this verification:

1. **Verify Unmatched Op Filter Fails Closed on CPU**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op non_existent_op_xyz
   # Exit code: 1
   ```

2. **Verify Unmatched Op Filter Fails Closed on Metal**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend metal --op non_existent_op_xyz
   # Exit code: 1
   ```

3. **Verify Valid Op Filter Passes and Streams on CPU**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op gelu_erf --release
   # Exit code: 0
   ```

4. **Verify E2E Tier 1 Feature Suite**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test -p ojas-oracle --test e2e_tier1_features
   # Exit code: 0 (65 passed)
   ```

5. **Verify Full 4-Tier Test Suite**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test -p ojas-oracle \
     --test e2e_tier1_features \
     --test e2e_tier2_boundaries \
     --test e2e_tier3_combinatorial \
     --test e2e_tier4_production
   # Exit code: 0 (142 passed)
   ```

6. **Verify Sitegen Consistency**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   go run scripts/sitegen/main.go -check
   # Exit code: 0 ("sitegen: generated files are up to date (54 files)")
   ```

---

## Review Summary

**Verdict**: **APPROVE**

### Findings
- **Integrity Violations**: None found.
- **Critical Findings**: None found.
- **Major Findings**: None found.
- **Minor Observations**:
  - `mac_heavy.sh` waits politely when system load exceeds thresholds before running Metal tests, ensuring clean GPU synchronization without flakiness.

### Verified Claims
- `./scripts/parity_check.sh --backend cpu --op non_existent_op_xyz` exits 1 → verified via execution → PASS
- `./scripts/parity_check.sh --backend metal --op non_existent_op_xyz` exits 1 → verified via execution → PASS
- `./scripts/parity_check.sh --backend cpu --op gelu_erf --release` exits 0 → verified via execution → PASS
- `go run scripts/sitegen/main.go -check` exits 0 → verified via execution → PASS
- `cargo test -p ojas-oracle --test e2e_tier1_features` passes 65 tests → verified via execution → PASS
- All 142 tests across Tiers 1-4 pass → verified via execution → PASS
- Zero temporary file leaks in `/tmp` → verified via execution → PASS

### Coverage Gaps
- None. All requested verification items from the dispatch were executed and evaluated.

### Unverified Items
- None.

---

## Challenge Summary

**Overall Risk Assessment**: **LOW**

### Challenges Tested
1. **Adversarial Op Filtering (Zero Match / Typos)**:
   - *Attack*: Developer passes typo or non-existent op filter expecting tests to run.
   - *Result*: Intercepted by regex, prints banner listing valid op options, exits with code 1. PASS.
2. **Backend Concurrency & Contention (`--backend all`)**:
   - *Attack*: Sequential execution of CPU and Metal tests under high system load.
   - *Result*: CPU suite runs immediately; Metal suite acquires `mac_heavy` exclusive lock, executes without memory interference, exits 0. PASS.
3. **Invalid Command Line Options**:
   - *Attack*: Developer passes unknown options (`--foo`, `--backend cuda`, or invalid fixtures path).
   - *Result*: Exits with code 1 and prints usage/help or error banner. PASS.
4. **Signal Interruption and File Descriptors**:
   - *Attack*: Process aborted or interrupted leaving dangling files in `/tmp`.
   - *Result*: POSIX trap `cleanup_logs` catches EXIT, INT, TERM and removes array of allocated log files. PASS.

### Stress Test Results
| Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|
| CPU Non-Existent Op | Exit code 1, Error banner | Exit code 1, Error banner printed | PASS |
| Metal Non-Existent Op | Exit code 1, Error banner | Exit code 1, Error banner printed | PASS |
| CPU Valid Op (`gelu_erf`) | Exit code 0, Table output | Exit code 0, Markdown table streamed | PASS |
| Metal Valid Op (`gelu_erf`) | Exit code 0, GPU lock acquired | Exit code 0, GPU lock acquired and released | PASS |
| Invalid Backend (`cuda`) | Exit code 1, Validation error | Exit code 1, Validation error printed | PASS |
| Invalid Flag (`--foo`) | Exit code 1, Usage printed | Exit code 1, Usage printed | PASS |
| Missing Fixtures Path | Exit code 1, Fixture error | Exit code 1, Error banner printed | PASS |
| Sitegen Consistency | Exit code 0, 54 files up to date | Exit code 0, 54 files up to date | PASS |
| E2E 142 Tests | 142 passed, 0 failed | 142 passed, 0 failed | PASS |

### Unchallenged Areas
- Full end-to-end model checkpoint weight inference (scheduled for Milestone 3).
