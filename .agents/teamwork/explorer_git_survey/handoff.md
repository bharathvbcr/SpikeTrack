# Comprehensive Git Survey Report Across Research Repositories

**Date**: 2026-11-01  
**Investigator**: `explorer_git_survey`  
**Target Repositories**:
1. `/Users/bharath/Code/research/ojas`
2. `/Users/bharath/Code/research/tessl`
3. `/Users/bharath/Code/research/SpikeTrack`
4. `/Users/bharath/Code/research/sparsl`

---

## 1. Observation

Direct CLI inspection and GitPulse Insights (`gitpulse_insights`) were executed across all four repositories. Below are the verbatim observations, exact branches, commit hashes, dirty file lists, and worktree states.

### 1.1 Repository: `/Users/bharath/Code/research/ojas`
- **Current HEAD**: `faf2d1b2d3ba6be374acdcf696faa032dc50d211` on branch `main` (ahead of `origin/main` by 76 commits).
- **Working Tree State in Root (`/Users/bharath/Code/research/ojas`)**:
  - Exactly **45 staged files** (`Changes to be committed`), 0 unstaged modifications, 0 untracked files (`git diff --cached --stat`: +7,461 lines, -343 lines across 45 files).
  - Staged files include the full SAM and Gemma embedding ops (`ojas-core/src/gelu_tanh.rs`, `ojas-cpu/src/gemma_ops.rs`, `ojas-metal/src/gemma_ops.rs`, `ojas-metal/kernels/gemma_ops.metal`, `ojas-oracle/src/gemma_golden.rs`, `ojas-autograd/src/tape.rs`, plus comprehensive PyTorch parity tests and tier-5 adversarial tests).
- **Worktrees (28 total)**:
  - 25 worktrees are clean.
  - 2 worktrees have uncommitted changes:
    - `/Users/bharath/Code/research/ojas-ug-cpu-gemma` (branch `gp/ug-cpu-gemma`, HEAD `b5abe74`): 9 modified files, 4 untracked files (`ojas-autograd/tests/gelu_tanh_tape.rs`, `ojas-core/src/gelu_tanh.rs`, `ojas-cpu/src/gemma_ops.rs`, `ojas-cpu/tests/gemma_ops.rs`).
    - `/Users/bharath/Code/research/ojas-ug-metal-gemma` (branch `gp/ug-metal-gemma`, HEAD `b5abe74`): 7 modified files, 4 untracked files (`ojas-metal/kernels/gemma_ops.metal`, `ojas-metal/src/backend/`, `ojas-metal/src/device/`, `ojas-metal/src/gemma_ops.rs`).
  - *Direct Diff Verification*: `diff -u` between these worktrees and root `/Users/bharath/Code/research/ojas` shows that all files in `ojas-ug-cpu-gemma` and `ojas-ug-metal-gemma` are either byte-for-byte identical (e.g. `ojas-cpu/src/gemma_ops.rs`) or the version in `main` is a rustfmt-formatted superset with additional ops and tests.
- **Branch Inventory (27 local branches)**:
  - **19 branches already merged into `main`** (`git branch --merged main`):
    `gp/ug-adamw-msl`, `gp/ug-adamw-scalars`, `gp/ug-alloc-error`, `gp/ug-bucket-dedup`, `gp/ug-const-dedup`, `gp/ug-core`, `gp/ug-cpu-gemma`, `gp/ug-cpu-sam`, `gp/ug-io-sink`, `gp/ug-metal-attn-bwd`, `gp/ug-metal-encoder-bwd`, `gp/ug-metal-gemma`, `gp/ug-metal-maxpool`, `gp/ug-metal-spatial-bwd`, `gp/ug-pool-ledger`, `gp/ug-readback-clean`, `gp/ug-sharedbuf`, `gp/ug-tessl-shared`, `gp/ug-tiny-step`.
  - **7 branches unmerged into `main`** (`git branch --no-merged main`):
    1. `gp/ug-ab-harness` (2 commits ahead of main: `a5db7f4`, `6a15bf5`):
       - Commits: `bench(qwen35): A/B harness tiny check uses tessl's fixture; explicit collect type`, `bench(qwen35): paired A/B harness for TapeStepProvider vs the tessl provider, with a gradient-parity row`.
       - Merge test (`git merge-tree --write-tree main gp/ug-ab-harness`): Clean merge, **0 conflicts**.
    2. `gp/ug-docs-ojas` (1 commit ahead of main: `30aef8e`):
       - Commit: `docs: direction - ojas trains everywhere, tessl is the Metal extension and Mac inference engine`.
       - Merge test: Content changes in `AGENTS.md`, `README.md`, `docs/architecture.md`. Conflicts only in generated site files (`site/assets/reference-index.json`, `site/reference/index.html`), resolved by `go run scripts/sitegen/main.go`.
    3. `gp/ug-geometry-dedup` (3 commits ahead of main: `54188de`, `915f103`, `096a799`):
       - Commits: `refactor(core): oracle, train and tape shape products delegate to ojas_core::checked_product`, `refactor(wgpu): affine_f32 folds its grid with ojas_kernels::fold_grid; delete workgroup_grid and its duplicate test`, `refactor(geometry): one owner for the cached-attention window origin, split scratch and op-named shape product`.
       - Merge test: Textual conflicts in `ojas-core/src/lib.rs` and `ojas-wgpu/src/backend.rs` due to concurrent imports added on `main`.
    4. `gp/ug-one-runtime` (2 commits ahead of main: `8ba95f7`, `14de954`):
       - Commits: `test(metal): pin MetalBackend::with_runtime (one runtime, error passthrough, panic poisons); queue the verification commands`, `feat(metal,qwen35): run Qwen35Step on the MetalBackend device thread's runtime`.
       - Merge test: Textual conflict in `ojas-metal/src/device.rs` (`Worker::dispatch` match block).
    5. `gp/ug-plan-doc` (2 commits ahead of main: `6c39eba`, `0b4f9e7`):
       - Commits: `docs(plan): map the sprint lanes onto plan items, marked inferred`, `docs(plan): unification plan for ojas and tessl, owner decisions D1-D11 open`.
       - Adds `docs/unification-plan.md` (273 lines). Conflicts only in generated site files (`site/assets/reference-index.json`, `site/reference/index.html`).
    6. `gp/ug-retire-train-prep` (2 commits ahead of main: `18a4999`, `f678ba7`):
       - Commits: `docs(site): regenerate the ojas-qwen35 reference page after the step shell retirement`, `refactor(qwen35): retire the tessl whole-step shell (Qwen35Step, bank, host AdamW); keep config, names, groups, state format`.
       - Removes 3,966 lines of deprecated tessl whole-step shell code from `ojas-qwen35`. Conflicts only in generated site files.
    7. `integrate/2026-10-31` (1 commit ahead of main: `e1ec371`):
       - Commit: `docs(bench): regenerate bench-plots for the result folders added since 1e39ad8`.
       - Updates `docs/bench-plots.md`. Conflicts only in generated site files.

---

### 1.2 Repository: `/Users/bharath/Code/research/tessl`
- **Current HEAD**: `48937fbadd19f2c8a0663b37e4284a1f8a6dc6ec` on branch `main` (ahead of `origin/main` by 27 commits).
- **Working Tree State in Root (`/Users/bharath/Code/research/tessl`)**:
  - `git status`: **working tree clean** (0 staged, 0 unstaged, 0 untracked files).
- **Worktrees (10 total)**:
  - All 10 worktrees (`tessl`, `tessl-ug-docs`, and 8 under `/Users/bharath/Code/ug-*/tessl`) are **100% clean**.
- **Branch Inventory (11 local branches)**:
  - **9 branches already merged into `main`** (`git branch --merged main`):
    `gp/w34-nt-bias-round`, `ug/adamw-msl`, `ug/alloc-error`, `ug/bucket-dedup`, `ug/const-dedup`, `ug/pool-ledger`, `ug/sharedbuf`, `ug/tessl-shared`, `main`.
  - **2 branches unmerged into `main`** (`git branch --no-merged main`):
    1. `ug/docs` (1 commit ahead of main: `be1e310`):
       - Commit: `docs: tessl's role as ojas's Metal extension and the Mac inference engine, target versus current`.
       - Merge test (`git merge-tree --write-tree main ug/docs`): Clean merge, **0 conflicts**.
    2. `ug/retire-train-prep` (3 commits ahead of main: `aada668`, `0b3a8ee`, `102b89d`):
       - Commits: `docs(python): mark the retired training surface in the tessl_torch README`, `docs(qwen35): README, benchmarking and the docs-name-real-tools allowlist after the training retirement`, `refactor(qwen35)!: retire the Qwen3.5 training step, host AdamW and 13 training C-ABI exports (ABI 13 -> 14)`.
       - Merge test (`git merge-tree --write-tree main ug/retire-train-prep`): Clean merge, **0 conflicts**.

---

### 1.3 Repository: `/Users/bharath/Code/research/SpikeTrack`
- **Current HEAD**: `ff7c97b780cb018b419022d03ebc37aa1163654a` on branch `main`.
- **Branches**: Only `main`.
- **Worktrees**: Exactly 1 (`/Users/bharath/Code/research/SpikeTrack`).
- **Working Tree State**:
  - 1 modified tracked file: `PROJECT.md` (updates milestone table status for M1-M4 to DONE, M5 to IN_PROGRESS).
  - 8 modified agent coordination files in `.agents/teamwork/`: `ORIGINAL_REQUEST.md`, `orchestrator/BRIEFING.md`, `orchestrator/DISPATCH.md`, `orchestrator/GATE_STATUS.md`, `orchestrator/progress.md`, `sentinel/BRIEFING.md`, `teamwork_worker_m1_it2/BRIEFING.md`, `teamwork_worker_m1_it2/progress.md`.
  - Untracked files: subagent workspace folders and handoff records inside `.agents/teamwork/`.

---

### 1.4 Repository: `/Users/bharath/Code/research/sparsl`
- **Current HEAD**: `31b101495d1185d41374c73c55c95e7f4c3704f1` on branch `main` (`[ahead 2, behind 1]` vs `origin/main`).
- **Branches**: Only `main`.
- **Worktrees**: Exactly 1 (`/Users/bharath/Code/research/sparsl`).
- **Working Tree State**:
  - 1 modified file: `LICENSE-APACHE`.
  - Uncommitted diff: Replaces placeholder `Copyright [yyyy] [name of copyright owner]` with `Copyright 2026 Bharath Chandra Vaddaram`.
  - Upstream status: Remote `origin/main` commit `036b649d71c1862328c27a3f1807be01e2efb353` contains this verbatim change (`docs: fill in copyright owner in LICENSE-APACHE`). Reconciling with `origin/main` eliminates the uncommitted modification without data loss.

---

## 2. Logic Chain

1. **Staged Changes in `ojas` Main**:
   - 45 files (+7,461 lines) staged on `main` implement all SAM and Gemma ops required by Milestone 3 & Milestone 4 and the authoritative request.
   - The two dirty worktrees (`ojas-ug-cpu-gemma` and `ojas-ug-metal-gemma`) were the temporary working branches for earlier subagent lanes. Their code was merged into `main`, unified, formatted to pass rustfmt, and expanded with adversarial test coverage.
   - Therefore, committing the staged changes in `ojas` main safely captures all kernel work. The two worktrees can then be discarded or reset cleanly.

2. **Branch Integration Status**:
   - In `tessl`, all sprint branches except `ug/docs` and `ug/retire-train-prep` are already merged. Both unmerged branches merge cleanly into `main` with 0 conflicts.
   - In `ojas`, 19 of 26 sprint branches are already merged into `main`. Of the remaining 7 unmerged branches:
     - `gp/ug-ab-harness` merges cleanly with 0 conflicts.
     - 4 branches (`gp/ug-docs-ojas`, `gp/ug-plan-doc`, `gp/ug-retire-train-prep`, `integrate/2026-10-31`) conflict only in generated HTML and index artifacts (`site/reference/*.html` and `site/assets/reference-index.json`). Following the `ojas/AGENTS.md` rule ("The site under site/ is generated from the markdown. Edit the markdown, then run go run scripts/sitegen/main.go to regenerate it; never edit the HTML by hand"), these conflicts are resolved by accepting the source Markdown changes and regenerating the site.
     - 2 branches (`gp/ug-geometry-dedup` and `gp/ug-one-runtime`) contain small, straightforward textual conflicts in imports (`ojas-core/src/lib.rs`, `ojas-wgpu/src/backend.rs`) and a match block in `ojas-metal/src/device.rs`. Both can be resolved manually in a single edit session without losing any logic.

3. **C-ABI and Dependency Synchronization**:
   - `ug/retire-train-prep` in `tessl` bumps the C-ABI from 13 to 14 and retires the whole-step training shell.
   - `gp/ug-retire-train-prep` in `ojas` retires the corresponding consumer side in `ojas-qwen35`.
   - Merging `ug/retire-train-prep` in `tessl` first, followed by `gp/ug-retire-train-prep` in `ojas`, keeps both crates in lockstep.

4. **Working Tree Cleanliness**:
   - Once the staged changes in `ojas` and `SpikeTrack` are committed, the dirty worktrees in `ojas` are reset, and `sparsl` is aligned with `origin/main`, all 4 repositories will have completely clean working trees and all feature branches will be merged into `main`.

---

## 3. Caveats

1. **Generated Site Files**:
   - Merging doc branches into `ojas` will trigger git conflicts in `site/reference/*.html` and `site/assets/reference-index.json`. Do NOT manually resolve html diffs; always run `go run scripts/sitegen/main.go` to generate clean, authoritative artifacts.
2. **Worktree Directory Locks**:
   - In `ojas`, the worktrees `/Users/bharath/Code/research/ojas-ug-cpu-gemma` and `/Users/bharath/Code/research/ojas-ug-metal-gemma` are linked to local branches `gp/ug-cpu-gemma` and `gp/ug-metal-gemma`. Before deleting those branches with `git branch -d`, their worktrees must either be removed via `git worktree remove` or detached.
3. **MSRV and CI Gates**:
   - Any merges into `ojas` and `tessl` must be validated against `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace --release -- --test-threads=1`.

---

## 4. Conclusion & Recommended Action Plan

### Recommended Step-by-Step Execution Plan

#### Step 1: Commit Uncommitted & Staged Changes
1. **In `ojas`**:
   ```sh
   cd /Users/bharath/Code/research/ojas
   git commit -m "feat(sam,gemma): implement SAM and Gemma embedding kernels with PyTorch parity and tier-5 adversarial tests"
   ```
2. **Clean dirty worktrees in `ojas`**:
   ```sh
   cd /Users/bharath/Code/research/ojas-ug-cpu-gemma
   git checkout -- . && git clean -fd
   cd /Users/bharath/Code/research/ojas-ug-metal-gemma
   git checkout -- . && git clean -fd
   ```
3. **In `SpikeTrack`**:
   ```sh
   cd /Users/bharath/Code/research/SpikeTrack
   git add PROJECT.md .agents/teamwork/
   git commit -m "chore(milestones): update PROJECT.md milestone statuses and agent handoff records"
   ```
4. **In `sparsl`**:
   ```sh
   cd /Users/bharath/Code/research/sparsl
   git checkout LICENSE-APACHE
   git merge origin/main -m "Merge branch 'origin/main'"
   ```

#### Step 2: Merge Remaining Branches in `tessl`
```sh
cd /Users/bharath/Code/research/tessl
git checkout main
# 1. Merge docs
git merge ug/docs -m "docs: integrate tessl architecture role as ojas Metal extension"
# 2. Merge training retirement
git merge ug/retire-train-prep -m "refactor(qwen35)!: retire Qwen3.5 training step and update ABI to 14"
# 3. Verify clean build and tests
cargo test
# 4. Clean up merged branches
git branch -d ug/docs ug/retire-train-prep
```

#### Step 3: Merge Remaining Branches in `ojas`
```sh
cd /Users/bharath/Code/research/ojas
git checkout main

# 1. Merge A/B harness (clean, 0 conflicts)
git merge gp/ug-ab-harness -m "bench(qwen35): integrate paired A/B provider harness"

# 2. Merge docs branches (resolve by regenerating site)
git merge gp/ug-docs-ojas -m "docs: integrate framework direction and architecture roles"
# (If sitegen conflict occurs, run:)
go run scripts/sitegen/main.go
git add site/ docs/ AGENTS.md README.md && git commit --no-edit

git merge gp/ug-plan-doc -m "docs(plan): integrate ojas and tessl unification plan"
go run scripts/sitegen/main.go
git add site/ docs/ && git commit --no-edit

git merge integrate/2026-10-31 -m "docs(bench): update bench plots"
go run scripts/sitegen/main.go
git add site/ docs/ && git commit --no-edit

# 3. Merge training retirement in ojas (synced with tessl ABI 14)
git merge gp/ug-retire-train-prep -m "refactor(qwen35): retire tessl whole-step shell in ojas-qwen35"
go run scripts/sitegen/main.go
git add site/ ojas-qwen35/ && git commit --no-edit

# 4. Merge geometry deduplication
git merge gp/ug-geometry-dedup -m "refactor(geometry): deduplicate shape products and launch geometry"
# (Resolve import additions in ojas-core/src/lib.rs and ojas-wgpu/src/backend.rs)
git add ojas-core/src/lib.rs ojas-wgpu/src/backend.rs && git commit --no-edit

# 5. Merge one runtime
git merge gp/ug-one-runtime -m "feat(metal): integrate MetalBackend::with_runtime provider execution"
# (Resolve Worker::dispatch match block in ojas-metal/src/device.rs)
git add ojas-metal/src/device.rs && git commit --no-edit

# 6. Format and regenerate site
./scripts/fmt_check.sh --fix
go run scripts/sitegen/main.go
git add -u
git commit -m "chore: format and regenerate site after branch merges" || true
```

#### Step 4: Verification and Worktree Pruning
1. Prune/remove merged worktrees and branches across `ojas` and `tessl`.
2. Verify all 4 repositories:
   - `git -C /Users/bharath/Code/research/ojas status`
   - `git -C /Users/bharath/Code/research/tessl status`
   - `git -C /Users/bharath/Code/research/SpikeTrack status`
   - `git -C /Users/bharath/Code/research/sparsl status`
3. Execute final test suites:
   - `cd /Users/bharath/Code/research/ojas && cargo test --workspace --release -- --test-threads=1`
   - `cd /Users/bharath/Code/research/tessl && cargo test`

---

## 5. Verification Method

To independently verify the facts and recommendations in this report:

1. **Verify `ojas` Staged Files**:
   ```sh
   git -C /Users/bharath/Code/research/ojas status -s
   git -C /Users/bharath/Code/research/ojas diff --cached --stat
   ```
2. **Verify Unmerged Branches**:
   ```sh
   git -C /Users/bharath/Code/research/ojas branch --no-merged main
   git -C /Users/bharath/Code/research/tessl branch --no-merged main
   ```
3. **Verify Worktree Status**:
   ```sh
   git -C /Users/bharath/Code/research/ojas worktree list
   git -C /Users/bharath/Code/research/tessl worktree list
   ```
4. **Verify Dry-Run Branch Mergability**:
   ```sh
   git -C /Users/bharath/Code/research/ojas merge-tree --write-tree main gp/ug-ab-harness
   git -C /Users/bharath/Code/research/tessl merge-tree --write-tree main ug/docs
   git -C /Users/bharath/Code/research/tessl merge-tree --write-tree main ug/retire-train-prep
   ```
