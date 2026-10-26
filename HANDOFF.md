# SpikeTrack — Meta Global AI Developer Hackathon handoff

Written 2026-10-25 for the hackathon window **Mon Oct 26 → Wed Nov 4, 9:00 PM ET (8:00 PM CT) submission**.
Track: **Indie**. Theme: **Push the Frontier** (accelerate a scientific experiment).
Evidence labels used everywhere: **[V]** verified against a primary source or a run, **[I]** inferred, **[U]** unverified — check before relying on it.

## 1. What we are building (one paragraph)

SpikeTrack runs Meta's **SAM 3.1** — including its **Object Multiplex** video tracker — as a **single Rust binary with no Python at runtime**, on Apple Silicon directly on tessl, Bharath's Metal 4 runtime (part of the Ojas stack), with sparsl for sparse/event compute. On top of the port it adds **training-free event gating**: bitpacked 1-bit pixel-change maps (sparsl) decide which image regions and which tracked-object slots are recomputed each frame. Showcase: many-object tracking in **clinical sperm-motility video (VISEM-Tracking)**, measured against the PyTorch reference.

The application promised: SAM 3.1 multiplex natively on Metal 4 tensor ops via Ojas/tessl; training-free sparsl event gating; evaluation on VISEM-Tracking (HOTA) and SA-V (J&F) against the PyTorch reference; target **≥3× the existing CoreML port's throughput at 16 objects** with published parity numbers.

## 2. Why it is new (keep this framing in every write-up)

- A pure-Rust SAM 3 exists (`rlx-sam3`, crates.io, v0.2.11 2026-07-06) but states **"SAM 3.1 multiplex is intentionally out of scope"** [V].
- The only non-PyTorch SAM 3.1 multiplex port is CoreML/Swift (`sdkv2/sam3.1-coreml`): ~2.05 s/frame backbone + ~0.7 s/frame tracker on a 16 GB M-series Mac, CPU+GPU only (ANE compile fails), multiplex 16 [V]. **Re-measure it on the M5 Pro before quoting any ratio** [U].
- Efficient-SAM2 (ICLR 2026) skips work by decoder saliency and needs training; EdgeTAM/TinySAM2 retrain small models. Ours is training-free pixel-event gating + neural-accelerator path + zero-copy sparsl↔tessl.
- Pitch: *"SAM 3.1 multi-object tracking on a laptop, zero Python, measured parity, several times faster than the existing Mac port."* Do not claim real-time or interactive speed (see Second audit). Not "a Rust port".

## 3. Hard rules (read before writing code)

1. **No Python at runtime.** Python is allowed only in `tools/reference/` to generate test fixtures and baselines offline. The shipped binary must pass `otool -L` with no Python/libtorch.
2. **New-work boundary.** Ojas, tessl and sparsl are pre-existing open-source libraries; use them as dependencies, do not present them as hackathon work. Everything built during Oct 26–Nov 4 lives in this repo and is listed in `CONTRIBUTIONS.md` (task gp-rules-repo-and-access). Confirm the official rules' wording on prior work on day 1.
3. **Never commit SAM weights or gated data.** Weights are gated on Hugging Face under the SAM License [V]; datasets have their own licenses. `.gitignore` them; document how to fetch.
4. **Parity before speed.** No speed number is quoted until the module it covers passes its parity gate.
5. **Mac resource limits.** Follow the Ojas conventions: heavy cargo runs under `mac_heavy.sh`, `-j 2`, `--test-threads=2`, `OJAS_ALLOW_NO_GPU` unset so a missing GPU fails instead of skipping.
6. **Record gaps.** Anything a tool could not answer goes into `gaps.jsonl` (same shape as `research/Lappi-decision/gaps.jsonl`).
7. **Ask before**: changing Ojas/tessl/sparsl public APIs, pushing to any remote, or spending partner GPU credits beyond the reference task.

## 4. Schedule and gates (D1 = Mon Oct 26)

| Day | Date | Lane A — model port | Lane B — data & eval | Gate at end of day |
|---|---|---|---|---|
| D0 | Sun Oct 25 | — | Request HF access to `facebook/sam3.1` (+ `facebook/sam3`) | Access requested |
| D1 | Mon Oct 26 | gp-weights-loader; start gp-vision-backbone | gp-rules-repo-and-access; gp-reference-oracle | Rules read; fixtures for 5 images exist |
| D2 | Tue Oct 27 | gp-vision-backbone | gp-eval-harness (J&F) | **Backbone cosine > 0.999 on 5 images** |
| D3 | Wed Oct 28 | gp-prompt-encoders, gp-detector-and-mask-head | gp-eval-harness (HOTA), VISEM loader | **Image masks mean IoU ≥ 0.95 on 20 frames — else ship image-only** |
| D4 | Thu Oct 29 | gp-tracker-multiplex | baselines (CoreML, torch-MPS) on M5 Pro | Single-object tracking runs end to end |
| D5 | Fri Oct 30 | gp-tracker-multiplex | SA-V subset prepared | **J&F within 2 points of reference on 3 SA-V clips** |
| D6 | Sat Oct 31 | gp-tensorops-hot-path | FPS-vs-objects harness | **≥3× CoreML port at 16 objects (same M5 Pro)** |
| D7 | Sun Nov 1 | gp-event-gating | gating ablation runs | **≥1.5× FPS at ≤1 HOTA lost — else ship ungated** |
| D8 | Mon Nov 2 | gp-cli-and-no-python-proof | full VISEM + SA-V numbers | Benchmark table frozen |
| D9 | Tue Nov 3 | gp-demo-and-submission | README, figures | Demo video cut |
| D10 | Wed Nov 4 | buffer | buffer | **Submit by 8 PM CT** |

Stretch (only if every gate above is green): gp-cuda-smoke, gp-meta-model-report.

## 5. Task index

| Order | Task | Priority | Status | Depends on |
|---|---|---|---|---|
| 0 | gp-rules-repo-and-access | 0 | ready | — |
| 1 | gp-reference-oracle | 0 | ready | access granted |
| 2 | gp-weights-loader | 0 | ready | access granted |
| 3 | gp-vision-backbone | 0 | ready | 1, 2 |
| 4 | gp-prompt-encoders | 0 | backlog | 2, 3 |
| 5 | gp-detector-and-mask-head | 0 | backlog | 3, 4 |
| 6 | gp-tracker-multiplex | 0 | backlog | 5 |
| 7 | gp-eval-harness | 1 | ready | 1 |
| 8 | gp-tensorops-hot-path | 1 | backlog | 6 |
| 9 | gp-event-gating | 1 | backlog | 6, 7 |
| 10 | gp-cli-and-no-python-proof | 1 | backlog | 6 |
| 11 | gp-demo-and-submission | 0 | backlog | 7–10 |
| 12 | gp-cuda-smoke | 3 | backlog | 8 (stretch) |
| 13 | gp-meta-model-report | 3 | backlog | 10 (stretch) |
| 14 | gp-zero-shot-check | 0 | ready | — (runs before weights arrive) |
| 15 | gp-exemplar-prompts | 1 | backlog | 4 |
| 16 | gp-motion-aware-memory | 1 | backlog | 6 |
| 17 | gp-domain-lora | 1 (P0 if zero-shot recall < 0.7) | backlog | 14, 2 |
| 18 | gp-weight-quantization | 3 | backlog | 8 (stretch) |

Component map: COMPONENTS.md. Improvement research and options: IMPROVEMENTS.md.

## 6. Numbers the submission must report

Mask IoU vs reference (image); J&F on the SA-V subset; HOTA on VISEM-Tracking; FPS vs object count (1/4/16/64/128) on M5 Pro against the CoreML port and PyTorch-MPS; peak memory; cold-start time; binary size; dependency list (`otool -L`). Every number cites the command and the result folder under `bench/results/`.

## 7. Start here (for an agent picking this up)

1. Read this file, then the task brief you were assigned in `tasks/`.
2. Check its dependencies are `done`; if not, stop and say which.
3. Work only in the brief's planned files; if you need others, note why in the brief's Description.
4. Close a criterion only with [V] evidence (command + output path). Update the brief's status and the index above.

## 8. Second audit (2026-10-25)

Labels as above. Findings, most serious first:

1. **Time [I]:** the plan assumes ten full days, but weekdays compete with a full-time job. Decide real hours per day; the weekend (Oct 31-Nov 1) is the only full block. If evenings only, cut gp-prompt-encoders to point/box prompts plus one text prompt, or move event gating to stretch.
2. **Weights access is a single point of failure [V gated]:** nothing in Lane A starts without it. Request tonight; escalate in the hackathon Discord if not granted by D2.
3. **Speed framing [I]:** the backbone is ~4.7 TFLOP of matmul per frame (450M params x 5,184 tokens), so ~2-3 frames/s is the realistic ungated ceiling on the M5 Pro. Keep the "≥3x the CoreML port" target; drop "interactive"/"real-time" wording everywhere.
4. **Gating may not help on VISEM [I]:** moving cells cover the whole field. Measure per-window change rates on D1-D2 (no model needed) and pick the gating showcase clip from the data.
5. **Reference path [V/U]:** official code needs CUDA [V]; Modal credits go to select participants [V]. Order: CPU PyTorch on the Mac for backbone fixtures, then RTX 3070 Ti, then Modal. The Meta Model API `sam-3.1` endpoint gives end-to-end masks/tracks as a second reference [V].
6. **Schedule clash:** the D2 backbone gate needs backbone fixtures on D1; fixtures for video can wait until D3-D4.
7. **SA-V access [U]:** fall back to DAVIS 2017 val for J&F if the form is not approved by D3.
8. **New-work rule:** only planning files exist before Oct 26; no code before kickoff.
9. **Architecture now known [V, arXiv 2511.16719]:** PE-L+ ~450M, 1008 px, patch 14, 32 layers (28 windowed 3x3, 4 global), RoPE; text encoder ~300M causal, 32 tokens; DETR 6+6; tracker num_maskmem = 7. Appendix H (multiplex) still [U].

Briefs updated with these notes: gp-reference-oracle, gp-vision-backbone, gp-event-gating, gp-eval-harness, gp-tracker-multiplex, gp-rules-repo-and-access.

## 9. Update (2026-10-25, later)

- VISEM-Tracking has per-sperm track IDs, YOLO-format boxes, classes (0 sperm, 1 cluster, 2 small/pinhead), 45–50 fps, CC BY 4.0, Zenodo record 7293726 [V, arXiv 2212.02842]. HOTA is computable.
- Supervised state of the art on VISEM: HOTA 74.3, MOTA 71.2 (YOLOv8 + DeepOCSORT) [V]. Report our zero-shot numbers as zero-shot, not head to head.
- New first task for D1: gp-zero-shot-check. Everything in the showcase assumes SAM 3.1 can find sperm; check it through the Meta Model API before the port depends on it.

## 10. Third audit — SAM 3.1 against Ojas and tessl (2026-10-25)

Labels: [V] read in the local code or a primary source this pass; [I] inferred; [U] confirm on D1. Line refs are to the repos as of 2026-10-25.

**Verdict: build the SAM 3.1 runtime on tessl directly, not through Ojas's Backend trait. Ojas stays a dependency only where it already fits (safetensors I/O).**

1. **Ojas has no bidirectional attention [V].** Its `Backend` trait exposes only `causal_sdpa_forward/backward` (ojas-core/src/backend.rs:788-816), and the Metal `sdpa_forward` is causal (ojas-metal/src/backend.rs:559). SAM's ViT, fusion encoder, DETR decoder and memory attention are all non-causal. Its trait also has no LayerNorm, conv2d or interpolation ops [V, docs/op-coverage.md]. Adding them means changing Ojas's public API, which the hackathon rules here forbid without asking.
2. **tessl already has most of a ViT encoder [V]:**
   - `embedgemma2::encoder_attn` / `encoder_attn_bf16`: bidirectional attention over right-padded batches with per-sequence lengths; TensorOps tiled kernels for head dim 64/256/512 (embedgemma2.rs:197-330).
   - **Windowed attention for free:** the 3×3 windows of 24×24 tokens are a batch of 9 sequences of 576 tokens, full length, no padding (72 = 3×24) [I]. Global layers are one sequence of 5,184 tokens.
   - `bert::layer_norm`, `bias_residual_layer_norm`, `bias_gelu_erf` (bert.rs:120-322): the LayerNorm and exact-GELU that ViTs use. Do not use `mlp_gelu_tanh`; tanh-GELU would break parity.
   - `embedgemma2_vision::axial_qkv_norm_rope` (embedgemma2_vision.rs:218): 2D axial RoPE driven by a positions buffer, but **fused with an RMS QK-norm**. If PE-L+ has no QK-norm [U], write a RoPE-only variant in SpikeTrack's own Metal file (copy, do not edit tessl). Also confirm PE's axis split and theta match `rope_inv_freq(head_dim/4, head_dim/2, theta)` [U].
   - `embedgemma2_pre::resize_rgb_u8` (embedgemma2_pre.rs:726): a torch-exact antialiased bicubic resize; SAM 3's own preprocessing interpolation must be matched exactly [U].
   - GEMM bf16 (26.6 TFLOP/s peak [V README]), Q4/Q8 GEMM/GEMV kernels for gp-weight-quantization [V nn.rs].
3. **Head dim [I]:** PE-L+ is ViT-L-like (1024 wide, 16 heads → head dim 64), which tessl's tiled encoder kernels cover. Confirm from the checkpoint on D1; a head dim outside 64/256/512 loses the TensorOps path.
4. **Transposed conv is needed earlier than planned [I]:** the "dual ViT-Det" neck that builds the multi-scale pyramid typically uses deconvolutions (×4, ×2) and pooling, so `st-ops` transposed conv lands in the backbone task, not only the mask head.
5. **LoRA training cannot run in Ojas during the hackathon [V]:** Ojas LoRA exists only for Qwen3.5 (ojas-model/tests/qwen35_lora*.rs) and needs the missing non-causal attention backward. gp-domain-lora trains in PyTorch offline (3070 Ti or Modal) and SpikeTrack loads merged weights.
6. **Multiplex interface hint [V]:** the CoreML port exports a stateful per-frame `track_step` with multiplex 16 (sdkv2/sam3.1-coreml). Read its model's input/output spec on D1 to learn the multiplex state layout before Appendix H is available.
7. **Pitch wording:** say "on tessl, my Metal 4 runtime (part of the Ojas stack)". Do not claim the model runs through Ojas's autograd or Backend API.

Briefs updated: gp-vision-backbone, gp-domain-lora, gp-tracker-multiplex. COMPONENTS.md updated.

## 11. Ojas ops (2026-10-25, Bharath's decision)

Bharath will build the missing vision ops in Ojas himself, so SAM 3.1 can run through Ojas's Backend (Metal via tessl kernels). Briefs in the Ojas repo: `tasks/gp-bidirectional-attention.md` (P0), `tasks/gp-vision-norm-act-bias.md` (P0), `tasks/gp-spatial-ops.md` (P1). This supersedes §3 rule 7 for these three briefs only. Until they land, SpikeTrack lanes may call tessl directly; switch to the Ojas ops as each passes parity. Record when each was built (before or during the hackathon) in CONTRIBUTIONS.md.
