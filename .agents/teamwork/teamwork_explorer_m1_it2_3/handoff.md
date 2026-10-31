# Handoff Report: Documentation and Sitegen Alignment (Milestone 1 Iteration 2)

**Agent**: Explorer 3 (`teamwork_explorer_m1_it2_3`)  
**Role**: Investigator, Synthesizer  
**Date**: 2026-10-31T18:03:00Z  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m1_it2_3/`  
**Target Systems**: `/Users/bharath/Code/research/ojas`, `/Users/bharath/Code/research/tessl`, `/Users/bharath/Code/research/SpikeTrack`  

---

## Executive Summary
This investigation resolves the three advisory and documentation issues identified during Milestone 1 reviews (Reviewer 1, `teamwork_reviewer_m1_1`) and adversarial challenges (Challenger 2, `teamwork_challenger_m1_2`):
1. **Sitegen Category Registration**: `scripts/sitegen/main.go` currently passes `-check` only because `docs/WORKFLOWS.md` is untracked in git. Once staged or committed, `sitegen -check` fails closed. We provide concrete recommendations to rename `docs/WORKFLOWS.md` to `docs/workflows.md` (conforming to 100% of lowercase kebab-case docs and URLs), register `"workflows": "status"` in `docCategory`, and regenerate reference HTML artifacts.
2. **Fixture vs Weights Clarification**: `docs/WORKFLOWS.md` lines 76–80 list `models/sam3.1/sam3.1_multiplex.safetensors` alongside activation fixtures without clarifying requirement status. We verify that `parity_check.sh` treats model weights as optional (`# Resolve Weights (optional)`), whereas `visem_22_f0.safetensors` (3.2 GB) is mandatory. We formulate exact markdown replacement text.
3. **Task Priority Reconciliation**: We confirm Challenger 2's finding that `tessl` task `ft-7f859b95f3683b11420a5128c77562ce` is Priority 1 (`severity: high`), not P0 as originally reported by Worker 1. We catalog the actual P0 blockers across `tessl` and `ojas`.

---

## 1. Observation

### 1.1 `scripts/sitegen/main.go` Mechanics and Tracking Gate
- **File**: `/Users/bharath/Code/research/ojas/scripts/sitegen/main.go`
- **Tracked Set Loading** (lines 98, 468–472):
  ```go
  func loadTracked() (trackedSet, error) {
      out, err := exec.Command("git", "ls-files", "-z").Output()
      ...
  ```
- **Document Collection Loop** (lines 276–289):
  ```go
  for _, p := range mds {
      if _, ok := tracked[filepath.ToSlash(p)]; !ok {
          continue
      }
      slug := strings.TrimSuffix(filepath.Base(p), ".md")
      if docSkip[slug] {
          continue
      }
      cat, ok := docCategory[slug]
      if !ok {
          return nil, fmt.Errorf("%s has no category: add it to docCategory in scripts/sitegen/main.go", p)
      }
      out = append(out, &doc{Slug: slug, Src: filepath.ToSlash(p), Cat: cat})
  }
  ```
- **Current Git Status**:
  - `git status --porcelain docs/` output:
    ```
     M docs/architecture.md
     M docs/dtype-policy.md
     M docs/op-coverage.md
    ?? docs/WORKFLOWS.md
    ```
  - Because `docs/WORKFLOWS.md` is currently untracked, `tracked["docs/WORKFLOWS.md"]` evaluates to `false`, and `collect()` skips the file.
  - Command: `cd /Users/bharath/Code/research/ojas && go run scripts/sitegen/main.go -check`
  - Result: Exits with code 0 (`sitegen: generated files are up to date (54 files)`).
- **Failure Mode on Git Staging**:
  - If `docs/WORKFLOWS.md` is added to git (`git add docs/WORKFLOWS.md`):
    - `tracked["docs/WORKFLOWS.md"]` becomes `true`.
    - `slug` is `"WORKFLOWS"`.
    - `docCategory["WORKFLOWS"]` lookup returns `ok = false`.
    - Resulting verbatim error:
      `docs/WORKFLOWS.md has no category: add it to docCategory in scripts/sitegen/main.go`

### 1.2 Slug and Casing Consistency Across the Ecosystem
- **All Tracked Markdown in `docs/`**:
  - Inspected `/Users/bharath/Code/research/ojas/docs/`:
    `adapter-checkpoint.md`, `adaptive-resources.md`, `architecture.md`, `audit-close.md`, `audit-cpu-hot.md`, `audit-larger-step.md`, `audit-phase2.md`, `audit-phase3.md`, `audit-resources.md`, `audit.md`, `backends.md`, `baseline.md`, `bench-cpu-vs-torch.md`, `bench-gpu-vs-torch.md`, `bench-plots.md`, `checkpoint-v1.md`, `cuda-backend-scoping.md`, `data-formats.md`, `dtype-policy.md`, `framework-design.md`, `lappi-lora-mac-runbook.md`, `metal-deferred-faults.md`, `op-coverage.md`, `pytorch-parity-plan.md`, `shape-contract.md`, `status-archive-2026-10-05.md`, `status.md`, `typed-storage-plan.md`, `web-and-domain.md`.
  - **100% of tracked markdown files are all-lowercase kebab-case**.
- **All Generated Reference HTML Pages**:
  - Inspected `/Users/bharath/Code/research/ojas/site/reference/`:
    All 36 HTML files are lowercase kebab-case (e.g. `architecture.html`, `lappi-lora-mac-runbook.html`).
  - Canonical URLs generated: `siteURL + "/reference/" + d.Slug + ".html"`.
  - Uppercase `WORKFLOWS.html` would be an anomaly in the generated reference tree.

### 1.3 `WORKFLOWS.md` Markdown Structural Validation
- **Table Parsing Verification**:
  - Ran table and fence parsing against `/Users/bharath/Code/research/ojas/docs/WORKFLOWS.md`.
  - Lines 9–20: ASCII workflow diagram enclosed in ` ``` ` block. Consumed by `reFence` before table matcher.
  - Lines 59–65: WF2 Tiered Testing table. Header has 5 cells, separator has 5 cells, all 5 rows have 5 cells. Ragged count = 0.
  - Lines 101–108: WF3 Tolerances table. Header has 4 cells, separator has 4 cells, all 6 rows have 4 cells. Ragged count = 0.
  - Verification formula (lines 111–114) is formatted as an ordered list (`1.`, `2.`, `3.`), not a table.
  - Total markdown links to resolve: 0 links. Unresolved count = 0.
  - Conclusion: Document syntax is 100% compliant with `sitegen`'s AST renderer.

### 1.4 Optional Weights Checkpoint vs Mandatory Activation Fixtures
- **Observation in `docs/WORKFLOWS.md` (lines 76–80)**:
  ```markdown
  ### Fixture Locations
  - **SAM 3.1 Backbone Fixtures**: `SpikeTrack/fixtures/backbone/visem_22_f0.safetensors` (~3.2 GB)
  - **Pixel Decoder GroupNorm Fixtures**: `SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors` (~203 MB)
  - **Weights Checkpoint**: `SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors` (~3.5 GB)
  ```
- **Filesystem Observation**:
  - `/Users/bharath/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors`: Exists (3.2 GB).
  - `/Users/bharath/Code/research/SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors`: Exists (203 MB).
  - `/Users/bharath/Code/research/SpikeTrack/models/`: **Does not exist on disk**.
- **Implementation Observation in `scripts/parity_check.sh`**:
  - Lines 114–140:
    ```bash
    # Resolve Fixtures path (visem_22_f0.safetensors)
    if [[ -z "${OJAS_SAM_FIXTURES:-}" || ! -f "$OJAS_SAM_FIXTURES" ]]; then
      echo "ERROR: SAM 3.1 fixture file not found!" >&2
      exit 1
    fi
    ```
    Mandatory: Script terminates with exit 1 if absent.
  - Lines 159–174:
    ```bash
    # Resolve Weights (optional)
    if [[ -n "$CUSTOM_WEIGHTS" ]]; then
      export OJAS_SAM_WEIGHTS="$CUSTOM_WEIGHTS"
    elif [[ -z "${OJAS_SAM_WEIGHTS:-}" ]]; then
      ...
    fi
    ```
    Optional: Script does NOT fail if `sam3.1_multiplex.safetensors` is missing.
  - Op parity checking uses pre-recorded intermediate activations (input/output tensors) from `visem_22_f0.safetensors` rather than requiring the full model weight tensor graph.

### 1.5 GitPulse Task Priority Audit
Direct empirical check using `gitpulse_get_task` and `gitpulse_list_tasks`:
- **`ft-7f859b95f3683b11420a5128c77562ce` (`tessl`)**:
  - Frontmatter / Board record: `priority: 1`, `severity: "high"`, `status: "review"`.
  - Title: "Main CI has not been green since 2026-09-29...".
  - Classification: **Priority 1 (P1)**. Worker 1 erroneously reported it as `p=0`.
- **True P0 Blockers on `tessl`**:
  - `ft-921d4808c849d8c57f0bc683474d97d2`: `priority: 0`, `status: "ready"` ("Push tessl and ojas main so every SHA the article cites resolves on GitHub").
  - `ft-c89dd818450b031bf3b9d1980b73adca`: `priority: 0`, `status: "ready"` ("Restore the MSRV 1.82 job and publish crate 0.3.0").
- **True P0 Blockers on `ojas`**:
  - `ft-b4f9ce4e61238a5853e72ee85f56fb03`: `priority: 0` ("Main CI is red on every push...").
  - `ft-53d639def8773f3c770269a46b53a4cf`: `priority: 0` ("Bidirectional and cross attention with additive bias...").
  - `ft-96baeabcc011dbc2ba67a695dadf999b`: `priority: 0` ("LayerNorm, exact GELU, linear bias...").
  - `ft-a050decae5826321f0428469472298b6`: `priority: 0` ("Metal GDN backward refuses every call...").
  - `ft-cab0fdf18f95a52f86e1074528638650`: `priority: 0` ("Device out-of-memory misclassified...").

---

## 2. Logic Chain

1. **Sitegen Invariant Enforcement**:
   - `scripts/sitegen/main.go` uses `loadTracked()` to build a `trackedSet` from `git ls-files -z` (Observation 1.1).
   - In `collect()`, any file in `docs/*.md` that is part of `trackedSet` must have an entry in `docCategory` (or `docSkip`), or `collect()` returns an error (Observation 1.1).
   - Currently, `sitegen -check` passes because `docs/WORKFLOWS.md` is untracked (Observation 1.1).
   - Once staged/committed, it will fail unless registered in `docCategory`.
   - All existing files in `docs/*.md` and generated files in `site/reference/*.html` use lowercase kebab-case (Observation 1.2).
   - Renaming `docs/WORKFLOWS.md` to `docs/workflows.md` and adding `"workflows": "status"` to `docCategory` preserves naming uniformity, URL structure, and categorizes the document under "Status and plans" alongside `lappi-lora-mac-runbook` and `pytorch-parity-plan`.

2. **Artifact Generation Requirement**:
   - `scripts/sitegen/main.go` generates HTML reference files (`outDir+"/"+d.Slug+".html"`), `site/reference/index.html`, `site/assets/reference-index.json`, and `site/sitemap.xml` (Observation 1.1).
   - If Worker 2 updates `main.go` and `docs/workflows.md` without running `go run scripts/sitegen/main.go`, `sitegen -check` will fail reporting stale files.
   - Therefore, Worker 2 must execute `go run scripts/sitegen/main.go` after registering the document.

3. **Weight vs Fixture Decoupling**:
   - `scripts/parity_check.sh` only tests isolated op kernels (`gelu_erf`, `attention`, etc.) against pre-recorded tensor slices stored in `visem_22_f0.safetensors` (Observation 1.4).
   - `sam3.1_multiplex.safetensors` represents full-model checkpoint weights and is explicitly treated as optional by `parity_check.sh` (Observation 1.4).
   - Documenting all three as "Fixtures" without distinction creates developer confusion because `/Users/bharath/Code/research/SpikeTrack/models/` does not exist on disk (Observation 1.4).
   - Clearly partitioning the section into "Required Activation Fixtures" (MANDATORY) and "Model Weights Checkpoint" (OPTIONAL / M3+) eliminates ambiguity.

4. **Task Priority Precision**:
   - Conflating `severity: high` with `priority: 0` causes incorrect blocker accounting (Observation 1.5).
   - Tessl's CI task `ft-7f859b95` is Priority 1, while its P0 blockers are `ft-921d4808` and `ft-c89dd818`.
   - Clear documentation prevents incorrect blocker escalation during multi-agent workflows.

---

## 3. Caveats

1. **Workspace Scope**:
   - `ojas` is the only repository with `sitegen`. `tessl/docs/WORKFLOWS.md` and `SpikeTrack` do not use `sitegen`, so their doc filenames do not trigger build gate failures. However, maintaining `docs/WORKFLOWS.md` in `tessl` is acceptable or can be mirrored if lowercase is preferred.
2. **Git Tracking Timing**:
   - If Worker 2 chooses to leave `docs/WORKFLOWS.md` untracked in git during Milestone 1, `sitegen -check` will continue to pass. However, doing so defers the issue to future milestones when documentation is committed. Resolving it now is the clean, idempotent path.
3. **Hardware Requirement for Metal Parity**:
   - As noted by Reviewer 1 and Challenger 2, `parity_check.sh --backend metal` requires Apple Silicon. Non-macOS environments must set `OJAS_ALLOW_NO_GPU=1`.

---

## 4. Conclusion & Concrete Recommendations for Worker 2

To maintain 100% build health, sitegen compliance, and documentation clarity, Worker 2 should perform the following actions:

### Recommendation 1: Align `WORKFLOWS.md` with `sitegen`

#### Step 1A: Rename File to Lowercase
In `/Users/bharath/Code/research/ojas`:
```bash
mv docs/WORKFLOWS.md docs/workflows.md
```

#### Step 1B: Register `"workflows"` in `scripts/sitegen/main.go`
In `/Users/bharath/Code/research/ojas/scripts/sitegen/main.go` lines 63–64:
```diff
--- a/scripts/sitegen/main.go
+++ b/scripts/sitegen/main.go
@@ -63,4 +63,4 @@ var docCategory = map[string]string{
 	"status": "status", "status-archive-2026-10-05": "status", "baseline": "status", "typed-storage-plan": "status",
-	"pytorch-parity-plan": "status", "cuda-backend-scoping": "status", "metal-deferred-faults": "status", "lappi-lora-mac-runbook": "status",
+	"pytorch-parity-plan": "status", "cuda-backend-scoping": "status", "metal-deferred-faults": "status", "lappi-lora-mac-runbook": "status", "workflows": "status",
 	"bench-cpu-vs-torch": "bench", "bench-gpu-vs-torch": "bench", "bench-plots": "bench",
```
*(Note: If maintaining uppercase `WORKFLOWS.md` is preferred, register `"WORKFLOWS": "status"` instead).*

#### Step 1C: Regenerate Static Reference Pages
```bash
cd /Users/bharath/Code/research/ojas
go run scripts/sitegen/main.go
go run scripts/sitegen/main.go -check
```

---

### Recommendation 2: Clarify Fixture vs Weights in `docs/workflows.md`

Replace lines 76–80 of `/Users/bharath/Code/research/ojas/docs/workflows.md` with:

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

---

### Recommendation 3: Record Accurate Task Priority Guidance

When documenting task boards or milestones:
- Record `tessl` task `ft-7f859b95f3683b11420a5128c77562ce` as **Priority 1 (High Severity)**.
- Record `tessl` true P0 blockers as `ft-921d4808c849d8c57f0bc683474d97d2` (Git SHA resolution) and `ft-c89dd818450b031bf3b9d1980b73adca` (MSRV 1.82 job & 0.3.0 publish).

---

## 5. Verification Method

To independently verify all observations and recommendations:

```bash
# 1. Verify sitegen check passes currently in ojas
cd /Users/bharath/Code/research/ojas
go run scripts/sitegen/main.go -check

# 2. Verify sitegen unit tests pass
go test scripts/sitegen/main.go scripts/sitegen/main_test.go

# 3. Verify presence of required activation fixtures and absence of models dir
ls -lh /Users/bharath/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors
ls -lh /Users/bharath/Code/research/SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors
ls -d /Users/bharath/Code/research/SpikeTrack/models 2>/dev/null || echo "models/ correctly absent"

# 4. Verify parity_check.sh treats weights as optional and passes gelu_erf on CPU
./scripts/parity_check.sh --backend cpu --op gelu_erf --release

# 5. Verify task priorities on tessl via GitPulse MCP
gitpulse_get_task(repo_path: "/Users/bharath/Code/research/tessl", task_id: "ft-7f859b95f3683b11420a5128c77562ce")
# Must verify: "priority": 1, "severity": "high"
```
