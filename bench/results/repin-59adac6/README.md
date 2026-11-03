# Repin ojas `286a444` -> `59adac6` (2026-11-03)

`59adac6` merges ojas branch `claude/flash-fwd-specialize-1103` (tip `b709353`) into ojas main. Its
tree equals the branch tip. For SpikeTrack the change that matters is `aade8fa`: the Metal f32
`linear_bias_forward` adds its bias in place on the GEMM output (11-13% faster per call with
identical output bits; ojas `bench/results/2026-11-03-linear-bias-inplace`). The end-to-end gain is
inside the noise (memory attention 0.93-0.98x against 8.6% noise), so no SpikeTrack speed claim is
made from it. tessl stays at `52bdd19`.

| Check on `59adac6` | Result | File |
|---|---|---|
| Module gates, CPU and Metal: text, detector, backbone, mask decoder, memory encoder, memory attention, interactive decoder, ViT layer-norm eps, Metal f32/bf16 D3 | pass, worst cosine 1.0000000 on every f32 stage; D3 mean mask IoU 1.0000 (f32), 0.9881 (bf16) | `../module-gates/repin-59adac6.log` |
| Metal bf16 backbone (gate > 0.999) | **fails, 0.9981223, identical at `286a444`** (A: detached worktrees in `/Users/bharath/Code/repin-check`; B: this tree), so not caused by the merge; recorded as `GAP-METAL-BF16-BACKBONE-GATE` | `bf16-backbone-A.log`, `bf16-backbone-B.log` |
| Detector fixtures (`fixtures/detector` was missing on disk) | regenerated with the manifest's command; all 4 sha256s match `fixtures/DETECTOR_MANIFEST.json` | `fixture-hashes.txt`, `dump-detector.log` |
| Detect-track parity, video 22 (the only `fixtures/detect_track` video on disk) | id sets equal on 32 of 32 yielded frames, mean per-id mask IoU 0.99712 over 470 pairs | `dt-parity-22.log` |
| Shipped predictions: v3 + `lorav3_exact_it_3p00.json`, train-5, 8 frames | byte-identical to the pre-merge run on all 5 videos (HOTA 0.5802) | `train5-0c.log`, `../weight-space/box-delta-train5/run-0c/` |

Jobs: `job.sh` (bf16 A/B and the prediction check), `fixtures-and-gates.sh`, then
`tools/locked.sh module-gates bash tools/jobs/module_gates.sh both`. The first gate run after the merge
(before the fixtures were restored) failed the detector and D3 gates only because `fixtures/detector`
was missing; that run's log was overwritten by the rerun above.
