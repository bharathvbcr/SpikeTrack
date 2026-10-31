# E2E Test Writer Dispatch: Comprehensive 4-Tier Test Suite & Test Infra

## Assigned Scope
You are the E2E Test Writer for the project.
Your mandate is to design and establish the opaque-box, requirement-driven E2E test infrastructure and test suites covering:
1. **SAM 3.1 Operations & Parity**:
   - ViT trunk, windowed attention, 2D axial RoPE, exact GELU, LayerNorm/LayerNorm2d, Conv2d (im2col + GEMM), transposed conv2d, GroupNorm, memory cross-attention ($D=256$), DETR boxRPB additive bias.
2. **Gemma Embedding Operations & Parity**:
   - GeGLU (`gelu_tanh`), symmetric non-causal sliding window attention ($|i-j| \le W$), sequence mean pooling (`segment_mean_rows`), Matryoshka prefix L2 normalization (`l2_normalize_rows`), per-head V RMSNorm.
3. **Shared Memory & Architecture Invariants**:
   - Zero-copy shared memory buffer validation between CPU and GPU.
   - Fail-closed "No Silent Fallback" invariant checks: verifying that GPU tensors refuse CPU computation, and vice versa.
   - Single Budget ledger checks: verifying memory pool allocations charge against `ojas_core::Budget`.

## Test Methodology (4 Tiers):
- **Tier 1 — Feature Coverage**: >=5 test cases per feature covering representative happy-path inputs in isolation.
- **Tier 2 — Boundary & Corner Cases**: >=5 boundary/corner cases per feature (empty inputs, zero-length sequences, max dimensions, NaN/Inf rejection, non-finite handling).
- **Tier 3 — Cross-Feature Combinations**: Pairwise feature interactions (e.g. RoPE + window attention, conv2d + LayerNorm, GeGLU + sequence pooling, shared memory + timeline wait).
- **Tier 4 — Real-World Application Scenarios**: Realistic end-to-end inference scenarios (SAM 3.1 video tracker frame step, Gemma embedding text encoding pipeline).

## Deliverables
1. Create `/Users/bharath/Code/research/SpikeTrack/TEST_INFRA.md` following the Project Pattern template.
2. Implement or organize test fixtures and test cases in `ojas-oracle` / test crates.
3. Once all test cases for Tiers 1-4 are ready, generate `/Users/bharath/Code/research/SpikeTrack/TEST_READY.md`.

## Mandatory Reading
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`
- `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`

## Output
Write your handoff report to:
`/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_test_writer_e2e/handoff.md`
Then notify the orchestrator.


## 2026-10-31T17:25:52Z
[Message from orchestrator]
Sender: 7f1f02db-7cbf-40d1-8919-b2741643d465
Priority: MESSAGE_PRIORITY_HIGH
Content:
You are the E2E Test Writer.
Your identity is teamwork_test_writer.
Your assigned working directory is `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_test_writer_e2e/`.
You own this folder exclusively. Write your progress to `progress.md` in your folder and your final report to `handoff.md` in your folder.

Authoritative Instructions:
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/ORIGINAL_REQUEST.md`.
Read `/Users/bharath/Code/research/SpikeTrack/PROJECT.md`.
Read `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_test_writer_e2e/DISPATCH.md`.
Read Explorer 2's handoff at `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`.

Your Tasks:
1. Design and author `/Users/bharath/Code/research/SpikeTrack/TEST_INFRA.md` following the required template in PROJECT.md.
2. Structure the 4-tier E2E test plan covering:
   - Tier 1: Feature Coverage (>=5 per feature for SAM & Gemma ops, memory sharing, no silent fallback).
   - Tier 2: Boundary & Corner Cases (>=5 per feature: empty inputs, zero dims, max bounds, NaN/Inf rejection).
   - Tier 3: Pairwise Combinatorial Interactions (cross-feature pairs).
   - Tier 4: Real-world Workloads (SAM 3.1 video tracker multiplex and Gemma embedding inference pipeline).
3. Implement and verify test cases in `ojas-oracle` and test directories using PyTorch reference values and tolerance bounds.
4. When test suites are ready and executable, create `/Users/bharath/Code/research/SpikeTrack/TEST_READY.md`.
5. Write your handoff report to `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_test_writer_e2e/handoff.md` and send a message when finished.
