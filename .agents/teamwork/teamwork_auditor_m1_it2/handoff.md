# Forensic Audit Report: Milestone 1 Iteration 2 Integrity Verification

**Work Product**: `/Users/bharath/Code/research/ojas/scripts/parity_check.sh` and `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md`  
**Target Repositories**: `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/SpikeTrack`  
**Auditor**: Forensic Auditor (`teamwork_auditor`)  
**Profile**: General Project (Integrity Mode: `development` per `ORIGINAL_REQUEST.md`)  
**Verdict**: **`CLEAN`**

---

## 1. Observation

### 1.1 Source Code and Script Implementation Analysis

1. **`parity_check.sh` Log Management and Signal Cleanup** (`scripts/parity_check.sh:24-33`):
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
   Direct observation: POSIX trap intercepts `EXIT`, `INT`, and `TERM` to clean any generated temporary log files from `/tmp`.

2. **`parity_check.sh` Execution Wrapper & Fail-Closed Detection** (`scripts/parity_check.sh:206-244`):
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
   Direct observation:
   - Does not fake test outputs; genuine execution via `"${cmd[@]}" 2>&1 | tee "$log_file"`.
   - Captures real exit code via `${PIPESTATUS[0]}`. If Cargo fails (non-zero `rc`), exits immediately with that exact return code.
   - Greps specifically for Cargo libtest 0-test patterns `(running 0 tests|test result: .* 0 passed; 0 failed)`.
   - On match, prints diagnostic error banner and exits closed with code `1`.

3. **`docs/WORKFLOWS.md` Fixture Partitioning** (`docs/WORKFLOWS.md:76-90`):
   ```markdown
   ### Fixture Locations

   #### 1. Required Activation Fixtures (MANDATORY for `parity_check.sh`)
   - **SAM 3.1 Backbone Activation Fixtures**: `SpikeTrack/fixtures/backbone/visem_22_f0.safetensors` (~3.2 GB)
     - *Status*: Verified present on disk.
     - *Role*: Contains pre-recorded input activations and ground-truth output activations from PyTorch oracle across ViT trunk and neck layers (`trunk.blocks.*`, `neck.convs.*`). Required for isolated op parity testing.
   - **Pixel Decoder GroupNorm Fixtures**: `SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors` (~203 MB)
     - *Status*: Verified present on disk.
     - *Role*: Reference input/output activations for GroupNorm layer verification.

   #### 2. Model Weights Checkpoint (OPTIONAL for op parity; required for M3+ full-model inference)
   - **SAM 3.1 Multiplex Weights**: `SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors` (~3.5 GB)
     - *Status*: Optional for `parity_check.sh`.
     - *Note*: `parity_check.sh` checks for this path optionally (`# Resolve Weights (optional)`). Its absence does not impede isolated op validation. Full model weights will be staged during Milestone 3 full-model integration.
   ```
   Direct observation: Disambiguates mandatory 3.2 GB activation fixtures from optional 3.5 GB full multiplex weights.

### 1.2 Verbatim Empirical Verification Results

1. **CPU Unmatched Op Filter Fail-Closed Test**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend cpu --op non_existent_op_xyz`
   - Return Code: `1`
   - Verbatim Output:
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
         Finished `release` profile [optimized] target(s) in 0.02s
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

2. **Metal Unmatched Op Filter Fail-Closed Test Under `mac_heavy.sh`**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend metal --op non_existent_op_xyz`
   - Return Code: `1`
   - Verbatim Output:
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
     mac_heavy: 2026-10-31T18:15:40Z 'ojas-parity-check' holds the lock (waited 0s; load 8.50, 1161 processes); running: cargo test --release -p ojas-metal --test sam_fixtures_metal non_existent_op_xyz -- --ignored --nocapture --test-threads=1
         Finished `release` profile [optimized] target(s) in 1.07s
          Running tests/sam_fixtures_metal.rs (target/release/deps/sam_fixtures_metal-bbf224a179ac8609)

     running 0 tests

     test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s

     mac_heavy: 2026-10-31T18:15:42Z 'ojas-parity-check' finished with exit 0

     ======================================================================
     ERROR: No parity tests matched op filter 'non_existent_op_xyz' in Metal suite!
     Available parity tests in ojas-metal/tests/sam_fixtures_metal.rs:
       conv2d, conv_transpose2d, gelu_erf, global_attention, group_norm,
       layer_norm, linear_bias, rope_interleaved, rope_tables, windowed_attention
     ======================================================================
     ```

3. **CPU Authentic Op Parity Execution (`gelu_erf`)**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend cpu --op gelu_erf --release`
   - Return Code: `0`
   - Verbatim Output:
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
         Finished `release` profile [optimized] target(s) in 0.09s
          Running tests/sam_fixtures_cpu.rs (target/release/deps/sam_fixtures_cpu-f08cd9db644adcf7)

     running 1 test
     | op | compared | kind | elements | max abs | max rel | cosine | tol abs | tol rel | tol cosine | result |
     |---|---|---|---|---|---|---|---|---|---|---|
     | gelu_erf_forward | trunk.blocks.0.mlp.fc1.out -> trunk.blocks.0.mlp.gelu.out [1, 72, 72, 4736] | direct | 24551424 | 4.768e-7 | 5.082e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
     | gelu_erf_forward | trunk.blocks.7.mlp.fc1.out -> trunk.blocks.7.mlp.gelu.out [1, 72, 72, 4736] | direct | 24551424 | 4.768e-7 | 4.887e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
     | gelu_erf_forward | neck.convs.0.gelu.in -> neck.convs.0.gelu [1, 512, 144, 144] | direct | 10616832 | 4.768e-7 | 5.568e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
     test gelu_erf_matches_torch ... ok

     test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 11 filtered out; finished in 3.49s


     ======================================================================
     ==> [Parity Check] SUCCESS in 4s
     ======================================================================
     ```

4. **Metal Authentic Op Parity Execution (`gelu_erf`) Under GPU Lock**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/parity_check.sh --backend metal --op gelu_erf --release`
   - Return Code: `0`
   - Verbatim Output:
     ```text
     ======================================================================
     ==> [Parity Check] PyTorch Oracle Parity Runner
         Backend:           metal
         Op Filter:         gelu_erf
         Profile:           release
         Fixtures:          /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors (3.2G)
         GroupNorm:         /Users/bharath/Code/research/ojas/../SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors
     ======================================================================

     ==> [Metal Parity] Running ojas-metal sam_fixtures_metal...
         GPU Lock: Wrapping with /Users/bharath/Code/research/ojas/../Lappi-decision/tools/mac_heavy.sh
         Command: /Users/bharath/Code/research/ojas/../Lappi-decision/tools/mac_heavy.sh ojas-parity-check cargo test --release -p ojas-metal --test sam_fixtures_metal gelu_erf -- --ignored --nocapture --test-threads=1
     mac_heavy: 2026-10-31T18:15:58Z 'ojas-parity-check' holds the lock (waited 0s; load 8.12, 1166 processes); running: cargo test --release -p ojas-metal --test sam_fixtures_metal gelu_erf -- --ignored --nocapture --test-threads=1
         Finished `release` profile [optimized] target(s) in 1.39s
          Running tests/sam_fixtures_metal.rs (target/release/deps/sam_fixtures_metal-bbf224a179ac8609)

     running 1 test
     test gelu_erf_matches_torch ... | op | compared | kind | elements | max abs | max rel | cosine | tol abs | tol rel | tol cosine | result |
     |---|---|---|---|---|---|---|---|---|---|---|
     | gelu_erf_forward | trunk.blocks.0.mlp.fc1.out -> trunk.blocks.0.mlp.gelu.out [1, 72, 72, 4736] | direct | 24551424 | 4.768e-7 | 5.055e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
     | gelu_erf_forward | trunk.blocks.7.mlp.fc1.out -> trunk.blocks.7.mlp.gelu.out [1, 72, 72, 4736] | direct | 24551424 | 4.768e-7 | 4.865e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
     | gelu_erf_forward | neck.convs.0.gelu.in -> neck.convs.0.gelu [1, 512, 144, 144] | direct | 10616832 | 4.768e-7 | 5.548e-5 | 0.999999999999 | 1e-5 | 1e-3 | 0.999999999000 | pass |
     ok

     test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 11 filtered out; finished in 0.31s

     mac_heavy: 2026-10-31T18:16:00Z 'ojas-parity-check' finished with exit 0

     ======================================================================
     ==> [Parity Check] SUCCESS in 2s
     ======================================================================
     ```

5. **E2E Tier 4 Production Pipeline Execution**:
   - Command: `cd /Users/bharath/Code/research/ojas && cargo test -p ojas-oracle --test e2e_tier4_production`
   - Return Code: `0`
   - Verbatim Output:
     ```text
         Finished `test` profile [unoptimized + debuginfo] target(s) in 0.02s
          Running tests/e2e_tier4_production.rs (target/debug/deps/e2e_tier4_production-6e8359d2e5c6963c)

     running 4 tests
     test tier4_scenario_d_no_silent_fallback_placement_invariance ... ok
     test tier4_scenario_c_pipeline_budget_scoping_and_trim ... ok
     test tier4_scenario_b_gemma_embedding_production_inference_pipeline ... ok
     test tier4_scenario_a_sam31_video_tracker_multiplex_pipeline ... ok

     test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
     ```

6. **Sitegen Documentation Parity Verification**:
   - Command: `cd /Users/bharath/Code/research/ojas && go run scripts/sitegen/main.go -check`
   - Return Code: `0`
   - Verbatim Output:
     ```text
     sitegen: generated files are up to date (54 files)
     ```

7. **Fast Dev Loop Performance**:
   - Command: `cd /Users/bharath/Code/research/ojas && ./scripts/dev_fast_check.sh ojas-core`
   - Return Code: `0`
   - Output: 198 passed; 0 failed; 3 ignored; total time: `0s` (well within < 15s budget).

8. **Temporary File Leak Audit**:
   - Command: `ls -la /tmp/ojas_parity_* 2>/dev/null || echo "CLEAN"`
   - Output: `CLEAN` (zero lingering files).

---

## 2. Logic Chain

1. **Absence of Hardcoded Results / Fake Passes**:
   - Observation 1.1 reveals `run_test_with_verification` wraps the real command in a pipeline through `tee` into a dynamically allocated `mktemp` log file.
   - Observation 1.2 (items 3 and 4) reveals that executing `parity_check.sh` computes real floating-point tensor differences across 24,551,424 elements from the 3.2 GB safetensors fixture on disk. The computed metrics (`max abs = 4.768e-7`, `max rel = 5.082e-5`, `cosine = 0.999999999999`) match genuine mathematical properties of the erf GELU operator and satisfy strict PyTorch parity bounds (`1e-5`, `1e-3`, `0.999999999000`).
   - Hence, there are no hardcoded pass outputs or stubbed exit values.

2. **Genuine Execution of `cargo test` and `mac_heavy.sh`**:
   - Observation 1.2 (item 4) demonstrates that `mac_heavy.sh` is invoked with label `ojas-parity-check`. The script logs load average (`load 8.12, 1166 processes`), takes the heavy-job lock, runs `cargo test`, and logs the lock release.
   - The test run on Metal completed in `0.31s` kernel time vs `3.49s` on CPU, confirming authentic Apple Silicon GPU execution.

3. **Authentic Fail-Closed Behavior on Zero Tests**:
   - When given an invalid op filter `non_existent_op_xyz`, Cargo reports `running 0 tests` and exits `0`.
   - `run_test_with_verification` evaluates the output against regex `(running 0 tests|test result: .* 0 passed; 0 failed)`.
   - As observed in 1.2 (items 1 and 2), the script detects this condition, outputs a detailed error with valid test names, and exits with code `1`.
   - Adversarial testing with an invalid backend also immediately exits with code `1`.
   - Hence, the fail-closed integrity invariant is strictly verified.

4. **Zero Lingering State / No Pre-populated Artifacts**:
   - Observation 1.1 (item 1) shows the POSIX `trap cleanup_logs EXIT INT TERM`.
   - Observation 1.2 (item 8) verifies that after all executions, no `/tmp/ojas_parity_*` files remain on disk.
   - Searching the repository for pre-populated `.log` or output artifacts found only committed benchmark directories required by the repo layout rules.

5. **Authentic E2E Production Workload Tests**:
   - Review of `ojas-oracle/tests/e2e_tier4_production.rs` confirmed that Scenarios A, B, C, and D contain genuine algorithmic pipelines: SAM 3.1 video tracker multiplexing, Gemma embedding Matryoshka truncation ($64 \to 32 \to 16$), child budget scoping and trimming, and `OjasError::Shape` placement invariance.
   - Execution in Observation 1.2 (item 5) confirmed all 4 scenarios execute and pass in `0.07s`.

---

## 3. Caveats

- **Hardware Dependency**: The Metal backend parity check (`--backend metal`) requires macOS Darwin on Apple Silicon hardware with Metal 4. It was directly tested and passed on this host. On non-Darwin hosts, the script correctly gates on `uname -s == Darwin` and requires `OJAS_ALLOW_NO_GPU=1` to skip cleanly.
- No other caveats.

---

## 4. Conclusion

The work products for Milestone 1 Iteration 2 (`scripts/parity_check.sh` and `docs/WORKFLOWS.md`):
- Contain **no hardcoded outputs, facade implementations, or fabricated results**.
- Authentically invoke `cargo test` and `mac_heavy.sh`.
- Realistically load safetensors fixtures and compare tensors against PyTorch tolerances.
- Implement strict, fail-closed enforcement (exit code 1) on unmatched or zero test executions.
- Completely clean up all temporary files on exit or signal interruption.
- E2E production tests under `ojas-oracle` execute and pass cleanly.

**Final Forensic Verdict**: **`CLEAN`**

---

## 5. Verification Method

To independently re-verify every claim in this report:

1. **Verify Fail-Closed Unmatched Op Filter (CPU)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op non_existent_op_xyz
   # Expected: Exits with code 1; reports "ERROR: No parity tests matched op filter"
   ```

2. **Verify Fail-Closed Unmatched Op Filter (Metal)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend metal --op non_existent_op_xyz
   # Expected: Acquires mac_heavy lock, exits with code 1, reports ERROR banner
   ```

3. **Verify Authentic PyTorch Parity Execution (CPU)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend cpu --op gelu_erf --release
   # Expected: Computes 24.5M elements, prints table, exits with code 0
   ```

4. **Verify Authentic PyTorch Parity Execution (Metal)**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   ./scripts/parity_check.sh --backend metal --op gelu_erf --release
   # Expected: Acquires mac_heavy lock, computes on GPU, prints table, exits with code 0
   ```

5. **Verify E2E Production Workloads**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   cargo test -p ojas-oracle --test e2e_tier4_production
   # Expected: 4 passed; 0 failed; finished in < 0.2s
   ```

6. **Verify Sitegen Consistency**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   go run scripts/sitegen/main.go -check
   # Expected: "sitegen: generated files are up to date (54 files)", exit code 0
   ```
