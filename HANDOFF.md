# SpikeTrack — Meta Global AI Developer Hackathon handoff

Written 2026-10-25 for the hackathon window **Mon Oct 26 → Wed Nov 4, 9:00 PM ET (8:00 PM CT) submission**.
Track: **Indie**. Theme: **Push the Frontier** (accelerate a scientific experiment).
Evidence labels used everywhere: **[V]** verified against a primary source or a run, **[I]** inferred, **[U]** unverified — check before relying on it.

## 1. What we are building (one paragraph)

SpikeTrack runs Meta's **SAM 3.1** — including its **Object Multiplex** video tracker — as a **single Rust binary with no Python at runtime**, on Apple Silicon directly on tessl, Bharath's Metal 4 runtime (part of the Ojas stack), with sparsl for sparse/event compute. On top of the port it adds **training-free event gating**: bitpacked 1-bit pixel-change maps (sparsl) decide which image regions and which tracked-object slots are recomputed each frame. Showcase: many-object tracking in **clinical sperm-motility video (VISEM-Tracking)**, measured against the PyTorch reference.

The application promised: SAM 3.1 multiplex natively on Metal 4 tensor ops via Ojas/tessl; training-free sparsl event gating; evaluation on VISEM-Tracking (HOTA) and SA-V (J&F) against the PyTorch reference; target **≥3× the existing CoreML port's throughput at 16 objects** with published parity numbers.

**Speed target restated (2026-10-27, Bharath):** parity-or-better with the CoreML port on the tracker path at 16 objects, leading with what that port cannot do: text-prompted detection with births and stable IDs, upstream-exact parity on CPU and Metal, a single binary with no Python, and LoRA domain adaptation.

**Trunk default changed (2026-10-29, owner's call, E10 in docs/SPEED_PLAN.md):** on `--backend
metal`, `track` runs the ViT trunk as a CoreML fp32 ML Program that we export from the checkpoint
(`tools/baselines/coreml_trunk.py`). The frame is 0.80x, with the same id sets on all 1,500 held-out
frames and the same HOTA, 0.4922 at cap 64. With the default, the trunk is Apple's CoreML, not
Ojas/tessl. The necks, text encoder, detector and tracker still run on Ojas. `--trunk-precision f32`
is the all-Ojas path, and `--backend cpu` keeps it as the parity oracle.
**E12 (2026-10-30):** the detector's host DETR decoder is 337 -> 62 ms per frame, with predictions
byte-identical, and the frame is 0.86x on VISEM 22 (`bench/results/e12-host-decoder`). Large host
linears now use up to `host::MAX_THREADS` = 8 scoped threads each, so a run outside the mac_heavy
lock competes for CPU and gains less.
**Closing job (2026-10-30, `bench/results/closing-e15`):**
- E12 + E13 + E15 together make the frame 0.843x on 82 and 0.825x on 22 of 3ca1476 (min of 3,
  load 10-23).
- The 1,500 held-out frames are byte-identical to the committed CoreML fp32 run, so HOTA stays
  0.4922.
- The footprint stays flat through frame 300.
**E13 / E15 (2026-10-30):**
- Memory attention is 74% of each tracker state's time. It does not shrink with objects, and its
  cost is in Ojas kernels (`bench/results/profile-tracker`); the numbers went to ojas-78.
- E15 keeps memory attention's inputs on the device. The tokens phase is 64 / 50 -> 14 / 6 ms per
  state on 82, predictions are byte-identical, and the footprint is ~760 MiB lower
  (`bench/results/e15-memory-tokens`).
- API shape change: `st_model::tracker::MaskMem` (`nchw` on the host, the trace and parity surface;
  `tokens` on the device) replaces `Stored.maskmem: Vec<f32>`, and `FrameFeats.tokens` is now a
  device `Tensor`.
- Both live as long as their `Stored` entry or frame, the same lifetime the host `Vec`s had. Say this plainly in every
write-up that leads with Ojas/tessl.

## 2. Why it is new (keep this framing in every write-up)

- A pure-Rust SAM 3 exists (`rlx-sam3`, crates.io, v0.2.11 2026-07-06) but states **"SAM 3.1 multiplex is intentionally out of scope"** [V].
- The only non-PyTorch SAM 3.1 multiplex port is CoreML/Swift (`sdkv2/sam3.1-coreml` @ 9668c49), tracker only (no detector or text encoder), CPU+GPU (ANE compile fails), multiplex 16. **Measured on this M5 Pro (2026-10-27, bench/results/baselines):** backbone (trunk + propagation neck, fp16) 0.272 s/frame and the multiplex-16 track step 0.0775 s/frame, ~0.35 s/frame in all; its backbone output matches upstream at cosine 0.99995 (bench/results/baselines/coreml_parity.txt). The port's own figure (~2.05 s + ~0.7 s on a 16 GB M-series Mac) is ~8x slower than this machine and must not be quoted.
- Efficient-SAM2 (ICLR 2026) skips work by decoder saliency and needs training; EdgeTAM/TinySAM2 retrain small models. Ours is training-free pixel-event gating + neural-accelerator path + zero-copy sparsl↔tessl.
- Pitch: *"The whole of SAM 3.1 (text prompt, detector and multiplex tracker) on a laptop, zero Python, measured parity with the official code."* Not "several times faster than the existing Mac port": measured on this M5 Pro we are ~8x slower per frame than the CoreML port, which runs only the tracker (D6 restated, see the "Speed target restated" note). Lead with what the CoreML port cannot do (text prompts, the detector, births of new objects, parity evidence). Do not claim real-time or interactive speed (see Second audit). Not "a Rust port".

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
| D6 | Sat Oct 31 | gp-tensorops-hot-path | FPS-vs-objects harness | **≥ parity with the CoreML port on the tracker path at 16 objects (same M5 Pro)** — restated 2026-10-27 by Bharath from "≥3×", which the measured CoreML time (0.35 s/frame, fp16 at ~70% of the GPU's bf16 peak) puts out of reach (GAP-D6-COREML-DENOMINATOR) |
| D7 | Sun Nov 1 | gp-event-gating | gating ablation runs | **≥1.5× FPS at ≤1 HOTA lost — else ship ungated** |
| D8 | Mon Nov 2 | gp-cli-and-no-python-proof | full VISEM + SA-V numbers | Benchmark table frozen |
| D9 | Tue Nov 3 | gp-demo-and-submission | README, figures | Demo video cut |
| D10 | Wed Nov 4 | buffer | buffer | **Submit by 8 PM CT** |

Stretch (only if every gate above is green): gp-cuda-smoke, gp-meta-model-report.

## 5. Task index

| Order | Task | Priority | Status | Depends on |
|---|---|---|---|---|
| 0 | gp-rules-repo-and-access | 0 | in progress (repo, licences, NOTICE, CONTRIBUTIONS done; Official Rules unread) | — |
| 1 | gp-reference-oracle | 0 | in progress (CPU reference, safetensors, backbone + text fixtures done; video fixtures, CoreML/MPS baselines open) | access granted |
| 2 | gp-weights-loader | 0 | done | access granted |
| 3 | gp-vision-backbone | 0 | ready | 1, 2 |
| 4 | gp-prompt-encoders | 0 | backlog | 2, 3 |
| 5 | gp-detector-and-mask-head | 0 | backlog | 3, 4 |
| 6 | gp-tracker-multiplex | 0 | backlog | 5 |
| 7 | gp-eval-harness | 1 | in progress (HOTA = TrackEval, J&F port + oracle, VISEM loader; SA-V/DAVIS loaders and CLI open) | 1 |
| 8 | gp-tensorops-hot-path | 1 | backlog | 6 |
| 9 | gp-event-gating | 1 | backlog | 6, 7 |
| 10 | gp-cli-and-no-python-proof | 1 | backlog | 6 |
| 11 | gp-demo-and-submission | 0 | backlog | 7–10 |
| 12 | gp-cuda-smoke | 3 | backlog | 8 (stretch) |
| 13 | gp-meta-model-report | 3 | backlog | 10 (stretch) |
| 14 | gp-zero-shot-check | 0 | done (recall < 0.7) | — (runs before weights arrive) |
| 15 | gp-exemplar-prompts | 1 | backlog | 4 |
| 16 | gp-motion-aware-memory | 1 | backlog | 6 |
| 17 | gp-domain-lora | 0 (zero-shot recall < 0.7) | ready | 14, 2 |
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

## 12. D1 findings (2026-10-26, session tessl-45)

Labels as above. Detail: tools/reference/README.md; open items: gaps.jsonl.

1. **Reference runs on the Mac CPU [V].** Official sam3 f1647e8 plus `tools/reference/cpu_shims.py` and `sam3-cpu.patch`: the full 3.1 multiplex stack loads with 0 missing and 0 unexpected keys, and an fp32 trunk forward takes 6.6 s. CUDA is no longer needed for backbone fixtures. The video path on CPU is still untested.
2. **Backbone fixtures exist [V].** `fixtures/backbone/*.safetensors` (5 images, fp32, 5.0 GB, git-ignored) and `fixtures/MANIFEST.json`. `check_fixtures.py` recomputes 27 tensors with independent torch math; all pass. Sent to session ojas-78 for its Ojas op parity tests.
3. **Architecture corrections [V]:** three FPN necks (`convs`, `interactive_convs`, `propagation_convs`) with no 0.5 level. The 3x3 conv is part of the neck. Abs pos embed is tiled 3x3 from 24x24. `ln_pre` eps is 1e-5 (blocks use 1e-6). RoPE uses interleaved pairs and fractional positions in the global layers, and the tables are stored per block. There is no QK-norm. Multiplex bucket is 16. Video heuristics come from the official builder (0.4 / 0.65), not config.json (0.5 / 0.7).
4. **Upstream runs the ViT MLP in bf16 [V].** The fp32 fixtures are an fp32 oracle; see GAP-UPSTREAM-BF16-MLP.
5. **Gating ceiling [V, bench/results/gating-change-rates/summary.md].** At tau 8, about 1% of 14x14 patches change frame to frame, but 55% of the 3x3 attention windows do (20 videos, 300 frames each). Exact window-level skipping can save at most about 45% of the windowed layers. The ≥1.5x gate likely needs token-level (approximate) gating. Low-change showcase clips: 29, 23, 52.
6. **Ojas ops:** session ojas-78 owns gp-bidirectional-attention, gp-vision-norm-act-bias and gp-spatial-ops. This session holds no Ojas or tessl files.
