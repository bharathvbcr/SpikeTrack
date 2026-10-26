# Improving SAM 3.1 for SpikeTrack

Written 2026-10-25. What can make SAM 3.1 better for many small, fast objects in microscopy video, ranked by value for effort. Labels as in COMPONENTS.md.

## The finding that changes the plan

Zero-shot SAM 3 can fail badly in niche domains: on a structural-crack dataset its pixel IoU was 0.017 and it scored zero on every detection metric of another, until LoRA fine-tuning raised IoU to 0.34–0.86 with only ~1M trainable parameters [V, SAM3-LoRA, arXiv 2609.00469]. The supervised state of the art on VISEM-Tracking is HOTA 74.3 / MOTA 71.2 with a trained YOLOv8 detector and DeepOCSORT [V, PMC11175353]. **So test zero-shot SAM 3.1 on VISEM on D1 (gp-zero-shot-check) before anything else depends on it.**

## Options

| # | Improvement | Training? | Expected effect | Cost | Task |
|---|---|---|---|---|---|
| 1 | Zero-shot check via the Meta Model API `sam-3.1` on 50 VISEM frames: text prompt "sperm" vs a few exemplar boxes | No | Decides whether 2–4 are needed | 2–3 h, D1 | gp-zero-shot-check |
| 2 | Image-exemplar prompts: give SAM 3 a few boxed sperm from the first frame instead of (or with) text | No | SAM 3 supports exemplar conditioning [V]; usually the cheapest fix for domain shift [I] | Small once prompts are ported | gp-exemplar-prompts |
| 3 | Motion-aware memory (SAMURAI-style): Kalman filter on each track's box, KF-IoU blended into mask selection (α = 0.15), memory frames admitted only when mask, object and motion scores pass | No | SAMURAI reports +7.1% AUC on LaSOT-ext and +3.5% AO on GOT-10k over SAM 2 with no retraining and no compute overhead [V, arXiv 2411.11922]; fits fast, near-linear sperm motion [I] | 1 day | gp-motion-aware-memory |
| 4 | LoRA "Light" on VISEM boxes: rank 16 on fusion-encoder, decoder and seg-head attention (~1M params); text encoder frozen; every absent class as a negative prompt so the presence head still says "not here" | Yes | Largest gain if zero-shot is weak [V for other domains]; negative prompts fix presence collapse [V, 2609.00469] | 1–2 days; PyTorch on 3070 Ti/Modal, or Ojas with cached backbone features [I] | gp-domain-lora |
| 5 | Presence-head calibration: measure how often the presence token fires on frames without the concept; report calibration (ECE) before/after 4 | No | The honest "I don't know" result, Lappi-style | Half a day | inside gp-domain-lora |
| 6 | Weight quantization of the backbone (Q8, then Q4) via tessl's existing Q4/Q8 kernels | No | Less memory traffic per frame [I]; must keep mask IoU within 0.01 | 1 day | gp-weight-quantization (stretch) |
| 7 | Distilled student backbones (EfficientSAM3) | Yes | Paper reports no results and no released weights [V, arXiv 2511.15833] | — | Not pursued |

## What not to claim

- Do not compare zero-shot SAM 3.1 HOTA to the supervised 74.3 as if like for like; report it as zero-shot vs supervised, and LoRA separately.
- LoRA on SAM 3 is not new (SAM3-LoRA, MedSAM3 LoRA); the new part is sperm-motility video, multiplex tracking, and running it with no Python on a laptop.

## Zero-shot check result (2026-10-26, gp-zero-shot-check) [V]

Local run of the official SAM 3.1 detector (multiplex checkpoint, CPU, upstream `Sam3Processor`; no Meta Model API key, see gaps.jsonl). 50 frames (5 from each of 10 videos, seed 20261026), 1,066 ground-truth sperm (class 0). Command and per-frame output: `tools/reference/zero_shot_check.py`, `bench/results/zero-shot/`. Pipeline check: it reproduces upstream's notebook oracle ("shoe" gives 12 objects).

| prompt | recall @IoU 0.5 (score ≥ 0.3) | precision | recall @IoU 0.1 | centre ≤ 10 px | predictions |
|---|---|---|---|---|---|
| sperm | 0.000 | 0.000 | 0.002 | 0.000 | 3 |
| sperm cell | 0.015 | 0.024 | 0.267 | 0.058 | 663 |
| spermatozoon | 0.000 | – | 0.000 | 0.000 | 0 |
| cell | 0.233 | 0.065 | 0.814 | 0.542 | 3,840 |
| car (absent) | – | – | – | – | 0 |
| cell nucleus (absent) | – | – | – | – | 0 |

**Decision: recall < 0.7 for every prompt, so gp-domain-lora becomes P0** (the rule above).

What the numbers say:
- The sperm vocabulary does not ground. "sperm" and "spermatozoon" return almost nothing.
- The generic "cell" does see most sperm: 54% within 10 px, 81% at IoU ≥ 0.1. It also fires on debris (3.6 predictions per sperm) and boxes tighter than VISEM's head-plus-neck boxes (median 18x19 vs 21x20 px).
- The presence head stays quiet on absent concepts (0 detections for "car" and "cell nucleus").
- The cheapest next test is exemplar prompts (gp-exemplar-prompts), boxed sperm from frame 0, which may lift precision without training. Run it in the reference before committing to LoRA.
- Report all of this as zero-shot, never against the supervised HOTA 74.3.
