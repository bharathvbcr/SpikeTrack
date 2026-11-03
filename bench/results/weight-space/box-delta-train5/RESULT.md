# Box-size correction for LoRA v3 + `lorav3_exact_it_3p00.json`: rejected at the held-out gate (2026-11-03)

**Why it was tried.** On 32 frames of train-5 (`../diag-train5/buckets.txt`), 62.5% of false positives
are localization errors and 63.7% of false negatives are near-misses (IoU 0.1-0.5). Predicted boxes
are smaller than the labels: median ln(gt/pred) for width/height is +0.061/+0.054 on matched pairs and
+0.335/+0.385 on near-misses; centers are within 0.03 of a box size. Concept errors (cluster/pinhead)
are 13.9% of FP, well under the half the plan required before trying a direction edit, so no
direction edit was made.

**Selection (train-5, 8 frames, `job.sh`, `eval-*.txt`).** `--post-box-delta 0,0,d,d`:

| d | 14 | 30 | 19 | 54 | 13 | all |
|---|---|---|---|---|---|---|
| 0 (two runs, identical) | 0.6918 | 0.6973 | 0.5062 | 0.4660 | 0.6256 | 0.5802 |
| 0.08 | 0.6769 | 0.6484 | 0.4690 | 0.5171 | 0.6522 | 0.5892 |
| **0.16** | 0.6430 | 0.5930 | 0.4263 | 0.5624 | 0.6646 | **0.5913** |
| 0.24 | 0.5734 | 0.5259 | 0.3804 | 0.5960 | 0.6458 | 0.5753 |
| 0.32 | | | | | | 0.5561 |

d = 0.16 met the pre-stated selection rule (+0.0111 over both d = 0 runs, not at the grid edge), but
only the two densest videos (54, 13: 566 of the 838 labelled boxes) improved; 14, 30 and 19 got worse.

**Held-out gate, one run (`../../track/held32-lorav3-it300-boxdelta016/`, 32 frames):**

| | 21 | 22 | 36 | 47 | 82 | all |
|---|---|---|---|---|---|---|
| v3 + edit (`../README.md`) | 0.674 | 0.680 | 0.575 | 0.560 | 0.555 | 0.6025 |
| + `--post-box-delta 0,0,0.16,0.16` | 0.5878 | 0.5749 | 0.4529 | 0.5956 | 0.4189 | **0.5017** |

Gate: > 0.6075. **Failed (-0.1008).** The correction is not shipped; the README's numbers stand.

**What it says.** The size gap is a property of how train videos 54 and 13 were labelled, not of the
model: the dense held-out videos (36, 82) lose the most when boxes grow. A single global box offset
cannot fit per-video labelling, and the aggregate train-5 HOTA hid that because it weights videos by
box count. Selection on box-weighted HOTA alone is unsafe for size calibration; sign consistency across
the selection videos should be part of any future rule.
