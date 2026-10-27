# Detector LoRA on VISEM-Tracking

**Prior work.** LoRA fine-tuning of SAM 3 is not new: SAM3-LoRA (arXiv 2609.00469) adapts SAM 3 to
niche domains with about 1M trainable parameters, and MedSAM3's LoRA fine-tuning is further prior work
(cited from the plan, tasks/gp-domain-lora.md; not re-read here). The recipe here (rank 16, alpha 32, dropout 0.1 on
attention projections, text encoder frozen, absent classes as negative prompts) follows SAM3-LoRA's
"Light" setting. What is new is the target, sperm-motility video with the multiplex tracker, and
running the adapted model with no Python on a laptop.

## Run `visem-lora-r16` (2026-10-27)

- Adapter: 30 attention modules of the detector transformer (`detector.transformer.*`), each of q, k,
  v and out_proj its own LoRA target: 120 targets in 60 tensors, 983,040 trainable parameters.
  Segmentation head, tracker, vision trunk and text encoder are frozen.
- Data: `tools/reference/visem_split.json`, 15 training videos (960 frames) and 5 held-out videos
  (21, 22, 36, 47, 82; 120 frames), split by video, never by frame. Frozen trunk and text features cached
  in fp16 (`cache.log`, 2.67 GB).
- Training: 600 steps on MPS (this Mac), in 6 locked chunks of 100 with `--resume` (`visem-lora-r16/train.log`,
  `train_log.json`). Offline PyTorch only; the Rust binary merges the adapter at load (`st_weights::lora`).

### Held-out detection (`visem-lora-r16/eval.md`, `eval.json`)

Class-0 (sperm) detection at IoU 0.5 on the 120 held-out frames, processor score:

| prompt | threshold | zero-shot recall | zero-shot precision | LoRA recall | LoRA precision |
|---|---|---|---|---|---|
| sperm | 0.3 | 0.000 | – | 0.761 | 0.584 |
| sperm | 0.5 | 0.000 | – | 0.684 | 0.733 |
| cell | 0.3 | 0.425 | 0.117 | 0.606 | 0.218 |
| cell | 0.5 | 0.212 | 0.122 | 0.576 | 0.451 |

"sperm", which zero-shot SAM 3.1 does not ground at all, becomes the best prompt.

### Presence and calibration

| prompt role | items | any detection >= 0.5, zero-shot | LoRA | presence > 0.5, zero-shot | LoRA |
|---|---|---|---|---|---|
| absent VISEM class (cluster, small/pinhead) | 156 | 0.0 | **0.160** | 0.359 | 0.327 |
| seen negative | 360 | 0.0 | 0.0 | 0.0 | 0.0 |
| unseen negative | 240 | 0.0 | 0.0 | 0.0 | 0.0 |

The adapter introduces false detections for the absent VISEM classes on 16% of (frame, prompt) items
(zero-shot: none), while every negative prompt stays at zero. ECE (10 bins): presence head
0.177 -> 0.045 (better); class-0 detections 0.072 -> 0.110 (worse).

### Adapter correctness

- PyTorch: the merged weights against the LoRA-wrapped forward, max |logit| difference 6.1e-4, box
  9.9e-5, presence logit 1.9e-6 (end of `visem-lora-r16/train.log`).
- Rust: `spiketrack image --lora` merges the adapter's 120 targets into 60 tensors and records its base
  checkpoint hash, which matches `tools/reference/weights_manifest.json` (`cli-check/summary.txt`, step-200 adapter).
  Element-wise merge parity against PyTorch: `merge-parity.log` (crates/st-weights/tests/lora_merge.rs).

### Tracking (HOTA)

The adapter trains the detector; `track`'s upstream boxes come from the frozen tracker's
whole-sperm masks, so its effect shows only with `track --boxes detector`
(`../track/box-convention`). 8-frame held-out smoke, step-200 adapter, prompt "cell": HOTA 0.1838
(zero-shot, upstream boxes) -> 0.3003 (LoRA, detector boxes). Step-600 runs and the full held-out
HOTA are recorded in `../track/`.

## Earlier

- `smoke/`: a 10-step pipeline check (selftest, train, eval) before the real run.
- `failed-verify-2026-10-27/`: the first cache verification, which compared all 200 queries at an fp16
  tolerance and failed on low-probability ones; the verify now checks agreement on the queries that
  matter (`head_agreement` in `tools/reference/cache_detector_inputs.py`).
