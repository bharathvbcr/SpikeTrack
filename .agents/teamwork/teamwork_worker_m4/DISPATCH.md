## 2026-11-01T01:15:24Z
You are the implementation worker for Milestone 4: `teamwork_worker_m4`.
Your role: Gemma Embedding Kernels Implementation Worker.
Your working directory: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m4`.
Your parent: Orchestrator (`7f1f02db-7cbf-40d1-8919-b2741643d465`).

MANDATORY FIRST STEP:
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md` and `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`.
Read the three explorer handoff reports:
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m4_1/handoff.md` (CPU & Autograd)
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m4_2/handoff.md` (Metal Kernels & Device Dispatch)
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m4_3/handoff.md` (PyTorch Parity Fixtures & Oracle Integration)

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

WRITE OWNERSHIP:
You have exclusive write ownership of source code in `/Users/bharath/Code/research/ojas`:
- `ojas-core/src/gelu_tanh.rs`
- `ojas-core/src/erf_exact.rs`
- `ojas-core/src/lib.rs`
- `ojas-core/src/shapes.rs`
- `ojas-core/src/autocast.rs`
- `ojas-core/src/backend.rs`
- `ojas-cpu/src/gemma_ops.rs`
- `ojas-cpu/src/backend.rs`
- `ojas-cpu/src/attn/general.rs`
- `ojas-cpu/src/lib.rs`
- `ojas-cpu/tests/gemma_ops.rs`
- `ojas-cpu/tests/attention_general.rs`
- `ojas-cpu/tests/autocast_sam_policy.rs`
- `ojas-autograd/src/tape.rs`
- `ojas-autograd/tests/gelu_tanh_tape.rs`
- `ojas-autograd/tests/attention_general.rs`
- `ojas-metal/build.rs`
- `ojas-metal/kernels/gemma_ops.metal`
- `ojas-metal/kernels/ojas_backend.metal`
- `ojas-metal/src/backend/gemma_host.rs`
- `ojas-metal/src/device/gemma_dev.rs`
- `ojas-metal/src/gemma_ops.rs`
- `ojas-metal/src/backend.rs`
- `ojas-metal/src/device.rs`
- `ojas-metal/src/lib.rs`
- `ojas-metal/src/link.rs`
- `ojas-metal/tests/gemma_ops.rs`
- `ojas-metal/tests/attention_general.rs`
- `ojas-oracle/python/gemma_ops_fixture.py`
- `ojas-oracle/fixtures/gemma_ops/golden.txt`
- `ojas-oracle/src/gemma_golden.rs`
- `ojas-oracle/src/lib.rs`
- `ojas-oracle/tests/e2e_tier4_production.rs`
- `ojas-model/tests/reach/mod.rs`
- `PROJECT.md` (milestone status tracking)

MISSION & PHASES:
Implement Milestone 4: Features 21–26 in `/Users/bharath/Code/research/ojas`:
- Feature 21: `gelu_tanh` / GeGLU forward & backward
- Feature 22: Symmetric non-causal sliding window attention ($|i - j| \le W$)
- Feature 23: Masked sequence mean pooling (`segment_mean_rows` / `seq_mean_forward`)
- Feature 24: L2 normalization with Matryoshka dimension truncation (`l2_normalize_rows` / `l2_normalize_forward`)
- Feature 25: Per-head V RMSNorm (`rms_qkv_norm_forward`)
- Feature 26: Gemma Embedding PyTorch Parity Verification & Oracle Golden Fixtures

EXECUTION DETAILS:
1. **Phase 1: Integrate CPU & Autograd files from `/Users/bharath/Code/research/ojas-ug-cpu-gemma`**:
   - Copy untracked files (`gelu_tanh.rs`, `gemma_ops.rs`, tests) into `ojas`.
   - Apply modifications to `erf_exact.rs`, `ojas-core/src/lib.rs`, `shapes.rs`, `autocast.rs`, `ojas-cpu/src/backend.rs`, `attn/general.rs`, `tape.rs`.
   - Update `autocast_sam_policy.rs` removing implemented ops from `REFUSED_ON_CPU_UNTIL_IMPLEMENTED`.
   - Verify CPU tests pass:
     `cargo test -p ojas-core -p ojas-cpu -p ojas-autograd`
     `cargo test -p ojas-cpu --test gemma_ops`
     `cargo test -p ojas-autograd --test gelu_tanh_tape`
     `cargo test -p ojas-cpu --test attention_general`

2. **Phase 2: Integrate Metal GPU files from `/Users/bharath/Code/research/ojas-ug-metal-gemma`**:
   - Copy `ojas-metal/kernels/gemma_ops.metal`, `src/backend/gemma_host.rs`, `src/device/gemma_dev.rs`, `src/gemma_ops.rs`.
   - Wire `build.rs` to compile `gemma_ops.metal`.
   - Wire `ojas_backend.metal` with sliding window flag bits `8u` (radius) and `16u` (padding rows).
   - Wire `ojas-metal/src/backend.rs`, `src/device.rs`, `src/link.rs`.
   - Implement `ojas-metal/tests/gemma_ops.rs` comparing Metal against CPU.
   - Enforce placement invariants: CPU accessors on Metal device tensors must fail closed with `OjasError::Placement`. Zero silent fallback.

3. **Phase 3: Wire Feature 25 (`rms_qkv_norm_forward`)**:
   - Add method to `Backend` trait, `with_backend_methods!`, autocast policy, shape validators, and implementations on `CpuBackend` and `MetalBackend`.

4. **Phase 4: Generate PyTorch Golden Fixtures & Parity Gates (Feature 26)**:
   - Create `ojas-oracle/python/gemma_ops_fixture.py`.
   - Run `/Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/gemma_ops_fixture.py --out ojas-oracle/fixtures/gemma_ops/golden.txt`.
   - Implement `ojas-oracle/src/gemma_golden.rs` and verify tolerances.
   - Un-stub and verify `tier4_scenario_b_gemma_embedding_production_inference_pipeline` in `ojas-oracle/tests/e2e_tier4_production.rs`.

5. **Phase 5: Full Workspace Quality Gates**:
   - Run tests under `bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh <tag>` with `--test-threads=1`.
   - Run `./scripts/fmt_check.sh` (`--fix` if needed).
   - Run `./scripts/check_tracked_targets.sh`.
   - Run `cargo clippy --workspace --all-targets -- -D warnings` (must be 0 warnings).
   - Run `./scripts/ci_local.sh` (must be exit code 0 `ci:local OK`).
   - Clean working tree: stage and commit all additions.

DELIVERABLE:
Write a comprehensive handoff report to:
`/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_worker_m4/handoff.md`
detailing all phases, files modified, test commands, and exact outputs.
When complete, notify your parent (`7f1f02db-7cbf-40d1-8919-b2741643d465`) via `send_message`.
