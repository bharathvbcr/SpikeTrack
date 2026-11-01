# Branch, Worktree, and In-Flight Commit Audit Report

## 1. Observation

A full enumeration and audit of Git branches, worktrees, stashes, and in-flight commits was conducted across the three project workspaces:
- `/Users/bharath/Code/research/SpikeTrack`
- `/Users/bharath/Code/research/tessl`
- `/Users/bharath/Code/research/ojas`

### 1.1 SpikeTrack Audit Summary
- **Current Branch**: `main` (commit `ff7c97b`, 0 ahead, 0 behind remote).
- **Worktrees**: 1 (`/Users/bharath/Code/research/SpikeTrack`).
- **Stashes**: 0 (`git stash list` output is empty).
- **Working Tree State**: Clean on tracked source code; local changes exist only within orchestration directories (`.agents/teamwork/`).

---

### 1.2 Tessl Audit Summary
- **Current Branch**: `main` (commit `4bd4c9d`, 3 ahead of `origin/main`).
- **Stashes**: 0 across all worktrees.
- **Local Branches (11)** and Merge-Base Divergence from `main` (`4bd4c9d`):

| Branch Name | Merge Base | Ahead | Behind | Latest Commit & Summary |
|---|---|---|---|---|
| `gp/w34-nt-bias-round` | `c9adc87` | 5 | 5 | `139306d`: `bench/results: tests/gemm_nt_bf16_bias.rs run at f535dd3 under the mac_heavy lock, 5 of 5 pass (M5 Pro)` |
| `ug/sharedbuf` | `c4526bf` | 2 | 2 | `47bad14`: `test(shared_buf): the stamp test names its one-command-buffer assumption` |
| `ug/tessl-shared` | `c4526bf` | 2 | 2 | `84615fe`: `fix(runtime): tests stage commits through the Shared clock; changelog entry for Shared, with_shared and Timeline` |
| `ug/pool-ledger` | `c4526bf` | 1 | 2 | `c15dbd5`: `feat(runtime): PoolLedger hook on the buffer pool (charge the bucket before newBuffer, cached balance, trim-retry-refuse ladder, reclaim)` |
| `ug/alloc-error` | `c4526bf` | 1 | 2 | `51d583a`: `feat(runtime): tessl owns the reading of its allocation failures (is_allocation_failure, AllocFailure)` |
| `ug/bucket-dedup` | `c4526bf` | 1 | 2 | `72ec718`: `test(runtime): allocated_bytes_for equals the pool's buffer length across every bucket threshold` |
| `ug/const-dedup` | `c4526bf` | 2 | 2 | `319335f`: `docs(changelog): the exported layout constants` |
| `ug/retire-train-prep` | `c4526bf` | 3 | 2 | `aada668`: `docs(python): mark the retired training surface in the tessl_torch README` (Removes 20,620 lines: `qwen35_train.rs`, duplicate AdamW, 13 C-ABI training symbols) |
| `ug/docs` | `c4526bf` | 1 | 2 | `be1e310`: `docs: tessl's role as ojas's Metal extension and the Mac inference engine` |
| `ug/adamw-msl` | `c4526bf` | 0 | 2 | `c4526bf`: `CHANGELOG: c9539a1's runtime changes under Changed and Fixed` |
| `main` | `4bd4c9d` | 0 | 0 | `4bd4c9d`: `gitignore: match report files and directories by name in any form` |

- **Worktree Paths (10)**:
  - `/Users/bharath/Code/research/tessl` (`4bd4c9d` [main], dirty: modified `docs/benchmarking.md`, untracked `docs/WORKFLOWS.md`)
  - `/Users/bharath/Code/research/tessl-ug-docs` (`be1e310` [ug/docs], clean)
  - `/Users/bharath/Code/ug-adamw-msl/tessl` (`c4526bf` [ug/adamw-msl], clean)
  - `/Users/bharath/Code/ug-alloc-error/tessl` (`51d583a` [ug/alloc-error], clean)
  - `/Users/bharath/Code/ug-bucket-dedup/tessl` (`72ec718` [ug/bucket-dedup], clean)
  - `/Users/bharath/Code/ug-const-dedup/tessl` (`319335f` [ug/const-dedup], clean)
  - `/Users/bharath/Code/ug-pool-ledger/tessl` (`c15dbd5` [ug/pool-ledger], clean)
  - `/Users/bharath/Code/ug-retire-train-prep/tessl` (`aada668` [ug/retire-train-prep], clean)
  - `/Users/bharath/Code/ug-sharedbuf/tessl` (`47bad14` [ug/sharedbuf], clean)
  - `/Users/bharath/Code/ug-tessl-shared/tessl` (`84615fe` [ug/tessl-shared], clean)

---

### 1.3 Ojas Audit Summary
- **Current Branch**: `main` (commit `4fa6986`, 10 ahead of `origin/main` `c389342`).
- **Stashes**: 0 across all worktrees.
- **Local Branches (27)** and Merge-Base Divergence from `main` (`4fa6986`):

| Branch Name | Merge Base | Ahead | Behind | Latest Commit & Summary |
|---|---|---|---|---|
| `gp/ug-metal-maxpool` | `c389342` | 1 | 10 | `0489575`: `feat(metal): max_pool2d forward and backward kernels` (+964 lines, `kernels/max_pool.metal`, `tests/max_pool.rs`) |
| `gp/ug-metal-spatial-bwd` | `c389342` | 1 | 10 | `cdb94a5`: `feat(metal): conv2d and conv_transpose2d backward on the device` (+1,801 lines, `kernels/spatial_bwd.metal`, `tests/conv_backward.rs`) |
| `gp/ug-metal-attn-bwd` | `c389342` | 3 | 10 | `15d9159`: `refactor(metal): attention backward launch geometry in one function...` (includes `f4b68da`: FlashAttention-2 general backward on Metal, +2,199 lines) |
| `gp/ug-metal-encoder-bwd` | `c389342` | 2 | 10 | `dae0cd7`: `fix(metal): encoder backward review fixes` (includes `b381d41`: +3,245 lines, `kernels/encoder_bwd.metal`, `tests/encoder_backward.rs`) |
| `gp/ug-cpu-sam` | `c389342` | 2 | 10 | `177a424`: `feat(cpu): group_norm_backward and interpolate_bilinear_backward, their tape records, goldens and gradchecks` (+1,797 lines) |
| `gp/ug-core` | `c389342` | 1 | 10 | `b5abe74`: `feat(core): six EmbeddingGemma 2 and SAM-training Backend methods, AttentionOpts scale and radius` |
| `gp/ug-cpu-gemma` | `c389342` | 1 | 10 | `b5abe74`: Base commit + **UNCOMMITTED IN-FLIGHT WORKTREE** (see 1.4) |
| `gp/ug-metal-gemma` | `c389342` | 1 | 10 | `b5abe74`: Base commit + **UNCOMMITTED IN-FLIGHT WORKTREE** (see 1.4) |
| `gp/ug-pool-ledger` | `c389342` | 1 | 10 | `bef3d74`: `feat(metal): BudgetLedger, tessl's PoolLedger over two Budget children` (+496 lines) |
| `gp/ug-sharedbuf` | `c389342` | 1 | 10 | `d2c3515`: `docs(agents): name tessl's SharedBuf as the sanctioned unsafe owner for CPU/GPU sharing` |
| `gp/ug-alloc-error` | `c389342` | 1 | 10 | `eeaa4c1`: `refactor(metal): classify allocation failures with tessl::is_allocation_failure` |
| `gp/ug-adamw-msl` | `c389342` | 3 | 10 | `f7ac192`: `test(metal): the AdamW parity reference reads and writes only the first n elements of a pooled buffer` (+652 lines) |
| `gp/ug-adamw-scalars` | `c389342` | 1 | 10 | `fc1495f`: `refactor(core): AdamWScalars is the one owner of the host AdamW scalar formation` (+525 lines) |
| `gp/ug-bucket-dedup` | `c389342` | 2 | 10 | `c264c15`: `test(metal): keep the param_bytes test name the docs cite; its message names pool rounding` |
| `gp/ug-const-dedup` | `c389342` | 2 | 10 | `de2e386`: `refactor(metal): pin the stub literals to tessl, pin the Metal source defines, drop the unused SQ_PIECE import` |
| `gp/ug-geometry-dedup` | `c389342` | 3 | 10 | `54188de`: `refactor(core): oracle, train and tape shape products delegate to ojas_core::checked_product` |
| `gp/ug-io-sink` | `c389342` | 2 | 10 | `fefe15e`: `test(io): tighten hostile-header assertions...` (includes `925ddf7`: slab sink for safetensors, +944 lines) |
| `gp/ug-readback-clean` | `c389342` | 3 | 10 | `32abb3a`: `docs(core): wrap the to_host readback doc comment` (includes `c9bb1d7`: unzeroed storage readback perf, +424 lines) |
| `gp/ug-retire-train-prep` | `c389342` | 2 | 10 | `18a4999`: `docs(site): regenerate ojas-qwen35 reference page` (includes `f678ba7`: retire tessl whole-step shell in ojas) |
| `gp/ug-tiny-step` | `c389342` | 3 | 10 | `a9dc9cb`: `docs(metal): drop the retired tiny step from live docs` (includes `b7b26c6`: retires legacy `ojas-metal/src/gpu.rs`, -3,735 lines) |
| `gp/ug-one-runtime` | `c389342` | 2 | 10 | `8ba95f7`: `test(metal): pin MetalBackend::with_runtime (one runtime, error passthrough, panic poisons)` (+514 lines) |
| `gp/ug-ab-harness` | `c389342` | 2 | 10 | `a5db7f4`: `bench(qwen35): A/B harness tiny check uses tessl's fixture` (includes `6a15bf5`: +1,025 lines, Tape vs tessl provider A/B) |
| `gp/ug-docs-ojas` | `c389342` | 1 | 10 | `30aef8e`: `docs: direction - ojas trains everywhere, tessl is the Metal extension and Mac inference engine` |
| `gp/ug-plan-doc` | `c389342` | 2 | 10 | `6c39eba`: `docs(plan): map the sprint lanes onto plan items, marked inferred` (+40k bytes `docs/unification-plan.md`) |
| `gp/ug-tessl-shared` | `c389342` | 0 | 10 | `c389342`: Merged base commit |
| `integrate/2026-10-31` | `4fa6986` | 1 | 0 | `e1ec371`: `docs(bench): regenerate bench-plots for the result folders added since 1e39ad8` (Fixes `ci_local.sh`) |
| `main` | `4fa6986` | 0 | 0 | `4fa6986`: `fix(core): the step module's declaration comment is a plain comment...` |

---

### 1.4 In-Flight Worktree Deep Dive (Uncommitted Code Verified)

1. **`/Users/bharath/Code/research/ojas-ug-cpu-gemma`** (Branch `gp/ug-cpu-gemma`):
   - **Tracked Modifications**:
     - `ojas-autograd/src/tape.rs` (+59 lines: `TapeRecord::GeluTanh`, backward dispatch)
     - `ojas-cpu/src/backend.rs` (+41 lines: `gelu_tanh_forward`, `gelu_tanh_backward`, `segment_mean_rows`, `l2_normalize_rows`, `rms_qkv_norm_forward`)
     - `ojas-cpu/src/attn/general.rs` (+14 lines: symmetric non-causal sliding window attention implementation)
     - `ojas-cpu/tests/attention_general.rs` (+149 lines: sliding window attention verification)
   - **Untracked Production Files**:
     - `ojas-core/src/gelu_tanh.rs` (8,966 bytes: exact `gelu_tanh` and its derivative)
     - `ojas-cpu/src/gemma_ops.rs` (5,137 bytes: CPU implementations)
     - `ojas-cpu/tests/gemma_ops.rs` (18,508 bytes: 14 tests)
     - `ojas-autograd/tests/gelu_tanh_tape.rs` (6,241 bytes: 4 tests)
   - **Verification Execution**:
     - `cargo test --test gemma_ops` ran in 4.02s: **14 passed, 0 failed**.
     - `cargo test --test gelu_tanh_tape` ran in 17.63s: **4 passed, 0 failed**.

2. **`/Users/bharath/Code/research/ojas-ug-metal-gemma`** (Branch `gp/ug-metal-gemma`):
   - **Tracked Modifications**:
     - `ojas-metal/build.rs`, `kernels/ojas_backend.metal`, `src/backend.rs`, `src/device.rs`, `src/link.rs`
   - **Untracked Production Files**:
     - `ojas-metal/kernels/gemma_ops.metal` (Metal compute kernels for `gelu_tanh`, `segment_mean_rows`, `l2_normalize_rows`, `rms_qkv_norm`)
     - `ojas-metal/src/backend/gemma_host.rs`, `ojas-metal/src/device/gemma_dev.rs`, `ojas-metal/src/gemma_ops.rs`
   - **Verification Execution**:
     - `cargo check -p ojas-metal` built cleanly in 10.00s with 0 errors.
     - `cargo test --test attention_general` ran: **12 tests passed, 0 failed** on CPU, **10 tests passed, 0 failed** on Metal.

---

### 1.5 Tessl Branch `gp/w34-nt-bias-round` Deep Dive
- Located in `/Users/bharath/Code/research/tessl` on branch `gp/w34-nt-bias-round` (5 commits ahead of merge-base `c9adc87`).
- Implements:
  - `kernels/matmul_tensorops.metal` & `kernels/gemm_coop.h`: `matmul2d_tensorops_nt_bf16_f32_bias`, a fused BF16 NT GEMM with F32 bias addition and optional BF16 rounding in register epilogue.
  - `src/gemm.rs`: `gemm_nt_bf16_bias` dispatch helper.
  - `src/bin/bench_gemm_routes.rs`: Benchmarks tailored directly to SpikeTrack SAM 3.1 trunk linears ($M=5184, 2304$).
  - `tests/gemm_nt_bf16_bias.rs`: 5 tests pass under `mac_heavy.sh` lock on M5 Pro (`bench/results/gemm_nt_bf16_bias_tests_m5pro.txt`).

---

## 2. Logic Chain

1. **Mapping to Project Milestones**:
   - The user's request and `PROJECT.md` define five distinct milestones: M1 (Workflows & Tasks), M2 (Unified Architecture & UMA Memory Pools), M3 (SAM 3.1 & Vision Kernels), M4 (Gemma Embedding Kernels), and M5 (Parity Verification & Coverage Hardening).
   - Our branch and worktree audit reveals that work for **M2, M3, and M4 is already largely or entirely written** across specialized Git branches and worktrees created during sprint waves, waiting for systematic integration.

2. **Milestone M2 (Unified Architecture & UMA Memory Pools) Evidence Chain**:
   - `PROJECT.md` Features 7, 8, 9, 10, 11, 12 require:
     - HAL traits (`MappedBuffer`, `Timeline`, `AllocError`) in `ojas-core`: already on `main` (`e74f1e0`), supplemented by `tessl: ug/sharedbuf` (implements `SharedBuf` zero-copy UMA and `HostFence`), `tessl: ug/tessl-shared` (implements `Timeline` and `Shared` commit clock), and `tessl: ug/alloc-error` (`is_allocation_failure`).
     - Root budget ledger with pool cache trim ladder: implemented in `tessl: ug/pool-ledger` (`PoolLedger`) and `ojas: gp/ug-pool-ledger` (`BudgetLedger`).
     - SafeTensors read-into-slab: implemented in `ojas: gp/ug-io-sink` (`slab.rs` and `tests/slab_sink.rs`).
     - AdamW scalar and kernel dedup: implemented in `ojas: gp/ug-adamw-scalars` (`AdamWScalars` in `ojas-core`) and `ojas: gp/ug-adamw-msl` (sharing tessl's `qwen35_adamw_math.h`).
     - Training orchestration dedup: `ojas: gp/ug-tiny-step` retires 3,735 lines of duplicate `gpu.rs`, and `ug/retire-train-prep` in both repos prepares for the full retirement of `qwen35_train.rs`.

3. **Milestone M3 (SAM 3.1 & Vision Kernels) Evidence Chain**:
   - `PROJECT.md` Features 13, 14, 15, 16, 17, 18, 19, 20 require max pooling, generalized conv_transpose2d, high-dimensional tiled cross-attention, boxRPB additive bias, and vision autograd ops.
   - Every single one of these has a dedicated branch with tests and fixtures:
     - Feature 13: `gp/ug-metal-maxpool` (`kernels/max_pool.metal`, `tests/max_pool.rs`).
     - Feature 14: `gp/ug-metal-spatial-bwd` (`kernels/spatial_bwd.metal`, `tests/conv_backward.rs`).
     - Features 15 & 16: `gp/ug-metal-attn-bwd` (`kernels/attn_bwd_tiled.h`, `tests/attention_backward_general.rs`).
     - Feature 18: `tessl: gp/w34-nt-bias-round` (`gemm_nt_bf16_bias` on SAM trunk shapes).
     - Features 18 & 19: `gp/ug-metal-encoder-bwd` (`kernels/encoder_bwd.metal`, `tests/encoder_backward.rs`).
     - Features 19 & 20: `gp/ug-cpu-sam` (`group_norm_backward`, `interpolate_bilinear_backward`, `tests/sam_tape.rs`).

4. **Milestone M4 (Gemma Embedding Kernels) Evidence Chain**:
   - `PROJECT.md` Features 21, 22, 23, 24, 25, 26 require `gelu_tanh` forward/backward, symmetric non-causal sliding window attention, `segment_mean_rows`, `l2_normalize_rows`, and per-head V RMSNorm (`rms_qkv_norm_forward`).
   - The in-flight uncommitted code in `ojas-ug-cpu-gemma` and `ojas-ug-metal-gemma` implements all of these, passes 14 unit/property tests, 4 tape autograd tests, and 22 attention tests.

5. **Mergeability and Conflict Analysis**:
   - Executing `git merge-tree` simulations against current `main` (`4fa6986` in ojas, `4bd4c9d` in tessl) proved that:
     - In `tessl`: `ug/sharedbuf`, `ug/tessl-shared`, `ug/pool-ledger`, `ug/alloc-error`, `ug/bucket-dedup`, `ug/const-dedup`, `ug/docs` **all merge with zero conflicts**. `gp/w34-nt-bias-round` has a trivial 1-line conflict in `CHANGELOG.md` only.
     - In `ojas`: `gp/ug-metal-maxpool`, `gp/ug-metal-spatial-bwd`, `gp/ug-metal-attn-bwd`, `gp/ug-metal-encoder-bwd`, `gp/ug-cpu-sam`, `gp/ug-pool-ledger`, `gp/ug-sharedbuf`, `gp/ug-alloc-error`, `gp/ug-adamw-msl`, `gp/ug-adamw-scalars`, `gp/ug-tiny-step`, `gp/ug-io-sink`, `gp/ug-readback-clean`, `integrate/2026-10-31` **all merge with zero conflicts**. `gp/ug-geometry-dedup` has a trivial 3-line re-export conflict in `ojas-core/src/lib.rs`.

---

## 3. Caveats

1. **Owner Decision D5 Gate on Qwen3.5 Training Retirement**:
   - Tessl branch `ug/retire-train-prep` and Ojas branch `gp/ug-retire-train-prep` delete `qwen35_train.rs` and 13 C-ABI exports. The commit messages explicitly specify: `DO-NOT-MERGE-BEFORE-AB`. Owner decision D5 requires validating parity with `gp/ug-ab-harness` (`scripts/ab_provider.sh`) before landing the full retirement of the C-ABI training symbols.
2. **Uncommitted Worktrees Must Be Committed Before Worktree Removal**:
   - `/Users/bharath/Code/research/ojas-ug-cpu-gemma` and `/Users/bharath/Code/research/ojas-ug-metal-gemma` contain working tree modifications that must be committed (or formatted into diff patches) before merging. Removing the worktree without committing would lose code.
3. **Hardware Lock Requirements for Metal Kernel Parity**:
   - Tests marked with Metal device dispatches or GPU execution must be run under the existing project lock `mac_heavy.sh` with `--test-threads=1` to prevent GPU thread contention and test timeouts.

---

## 4. Conclusion

The audit demonstrates that instead of writing kernels and memory sharing from scratch, the vast majority of required features for Milestones M2, M3, and M4 have already been developed on dedicated branches and in-flight worktrees.

Integrating these branches will accelerate project delivery by multiple days.

### Milestone Mapping & Integration Plan

#### Phase 1: Land Milestone M2 (Unified Architecture & UMA Memory Pools)
**Tessl Integration**:
```bash
cd /Users/bharath/Code/research/tessl
git merge --no-ff ug/const-dedup -m "Merge ug/const-dedup into main (M2)"
git merge --no-ff ug/alloc-error -m "Merge ug/alloc-error into main (M2)"
git merge --no-ff ug/bucket-dedup -m "Merge ug/bucket-dedup into main (M2)"
git merge --no-ff ug/tessl-shared -m "Merge ug/tessl-shared into main (M2)"
git merge --no-ff ug/sharedbuf -m "Merge ug/sharedbuf into main (M2)"
git merge --no-ff ug/pool-ledger -m "Merge ug/pool-ledger into main (M2)"
```

**Ojas Integration**:
```bash
cd /Users/bharath/Code/research/ojas
git merge --no-ff gp/ug-alloc-error -m "Merge gp/ug-alloc-error into main (M2)"
git merge --no-ff gp/ug-bucket-dedup -m "Merge gp/ug-bucket-dedup into main (M2)"
git merge --no-ff gp/ug-const-dedup -m "Merge gp/ug-const-dedup into main (M2)"
git merge --no-ff gp/ug-adamw-scalars -m "Merge gp/ug-adamw-scalars into main (M2)"
git merge --no-ff gp/ug-adamw-msl -m "Merge gp/ug-adamw-msl into main (M2)"
git merge --no-ff gp/ug-pool-ledger -m "Merge gp/ug-pool-ledger into main (M2)"
git merge --no-ff gp/ug-sharedbuf -m "Merge gp/ug-sharedbuf into main (M2)"
git merge --no-ff gp/ug-readback-clean -m "Merge gp/ug-readback-clean into main (M2)"
git merge --no-ff gp/ug-io-sink -m "Merge gp/ug-io-sink into main (M2)"
git merge --no-ff gp/ug-tiny-step -m "Merge gp/ug-tiny-step into main (M2: retire duplicate gpu.rs)"
```

---

#### Phase 2: Land Milestone M3 (SAM 3.1 & Vision Kernels with PyTorch Parity)
**Tessl Integration (SAM Trunk Linear Kernel)**:
```bash
cd /Users/bharath/Code/research/tessl
# Cherry-pick or merge gp/w34-nt-bias-round, resolving the minor CHANGELOG.md conflict
git cherry-pick daff7db e326f33 3c428db f535dd3 139306d
```

**Ojas Integration**:
```bash
cd /Users/bharath/Code/research/ojas
git merge --no-ff gp/ug-metal-maxpool -m "Merge gp/ug-metal-maxpool into main (M3: Feature 13)"
git merge --no-ff gp/ug-metal-spatial-bwd -m "Merge gp/ug-metal-spatial-bwd into main (M3: Feature 14)"
git merge --no-ff gp/ug-metal-attn-bwd -m "Merge gp/ug-metal-attn-bwd into main (M3: Features 15 & 16)"
git merge --no-ff gp/ug-metal-encoder-bwd -m "Merge gp/ug-metal-encoder-bwd into main (M3: Features 18 & 19)"
git merge --no-ff gp/ug-cpu-sam -m "Merge gp/ug-cpu-sam into main (M3: Features 19 & 20)"
```

---

#### Phase 3: Land Milestone M4 (Gemma Embedding Kernels with PyTorch Parity)
**Commit & Integrate CPU & Metal Gemma Ops**:
```bash
# 1. Commit in-flight CPU Gemma ops in worktree
cd /Users/bharath/Code/research/ojas-ug-cpu-gemma
git add ojas-core/src/gelu_tanh.rs ojas-cpu/src/gemma_ops.rs ojas-cpu/tests/gemma_ops.rs ojas-autograd/tests/gelu_tanh_tape.rs
git commit -am "feat(cpu,autograd): Gemma embedding ops (gelu_tanh, segment_mean_rows, l2_normalize, symmetric window attention)"

# 2. Commit in-flight Metal Gemma ops in worktree
cd /Users/bharath/Code/research/ojas-ug-metal-gemma
git add ojas-metal/kernels/gemma_ops.metal ojas-metal/src/backend/gemma_host.rs ojas-metal/src/device/gemma_dev.rs ojas-metal/src/gemma_ops.rs
git commit -am "feat(metal): Gemma embedding kernels (gelu_tanh, segment_mean_rows, l2_normalize, rms_qkv_norm)"

# 3. Merge into main
cd /Users/bharath/Code/research/ojas
git merge --no-ff gp/ug-cpu-gemma -m "Merge gp/ug-cpu-gemma into main (M4)"
git merge --no-ff gp/ug-metal-gemma -m "Merge gp/ug-metal-gemma into main (M4)"
```

---

#### Phase 4: Land Milestone M5 (CI Local Fix & Parity Hardening)
```bash
cd /Users/bharath/Code/research/ojas
git merge --no-ff integrate/2026-10-31 -m "Merge integrate/2026-10-31 into main (M5: bench-plots & sitegen fix for ci_local.sh)"
```

---

## 5. Verification Method

To independently verify all findings and test results:

1. **Verify Clean Mergeability**:
   ```bash
   cd /Users/bharath/Code/research/ojas
   for b in gp/ug-metal-maxpool gp/ug-metal-spatial-bwd gp/ug-metal-attn-bwd gp/ug-metal-encoder-bwd gp/ug-cpu-sam gp/ug-pool-ledger gp/ug-sharedbuf gp/ug-alloc-error gp/ug-adamw-msl gp/ug-adamw-scalars gp/ug-tiny-step gp/ug-io-sink gp/ug-readback-clean integrate/2026-10-31; do
       git merge-tree $(git merge-base main $b) main $b | grep -c "<<<<<<<"
   done
   # Expected result: all return 0 (no conflict markers)
   ```

2. **Verify Gemma CPU Tests in Worktree**:
   ```bash
   cd /Users/bharath/Code/research/ojas-ug-cpu-gemma
   cargo test --test gemma_ops
   cargo test --test gelu_tanh_tape
   # Expected result: 14 passed in gemma_ops, 4 passed in gelu_tanh_tape
   ```

3. **Verify Gemma Metal Build & Attention Tests in Worktree**:
   ```bash
   cd /Users/bharath/Code/research/ojas-ug-metal-gemma
   cargo check -p ojas-metal
   cargo test --test attention_general
   # Expected result: compiles clean, 12 CPU + 10 Metal attention tests pass
   ```

4. **Verify Tessl SAM Kernel (`gemm_nt_bf16_bias`)**:
   ```bash
   cd /Users/bharath/Code/research/tessl
   git log -1 gp/w34-nt-bias-round --stat
   cat bench/results/gemm_nt_bf16_bias_tests_m5pro.txt
   # Expected result: commit 139306d records 5 of 5 tests passing on M5 Pro
   ```

5. **Invalidation Conditions**:
   - If any commit is made directly to `main` that modifies function signatures in `ojas-metal/src/backend.rs` or `ojas-core/src/backend.rs`, `git merge-tree` may encounter merge conflicts requiring re-simulation.
