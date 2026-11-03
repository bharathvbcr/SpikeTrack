# `visem-lora-v4`: did not pass the held-out gate; v3 stays the shipped adapter (2026-11-03)

Recipe (PLAN.md, all changed at once): the cache at 128 frames per training video (2,040 frames with
the held-out 24s, 5.05 GB, 4/4 upstream verify checks ok, `cache.log`), warm start from v3 step 160,
400 more steps with v3's flags plus `--box-loss-weights 7.5 4.0` (`train-job.sh`, `train.log`,
`train_log.json`: 1,906 s on MPS). The box-weight option is covered by `lora_selftest.py`
(`selftest.log`, ALL PASS).

**Selection, train-5, 8 frames (`select-job.sh`, `select/eval-*.txt`), HOTA:**

| checkpoint | G 2.4 | G 3.0 | G 3.6 | G 4.2 |
|---|---|---|---|---|
| 100 | 0.5895 | 0.5925 | 0.5913 | |
| 200 | 0.5847 | 0.5839 | 0.5928 | |
| **300** | 0.5861 | 0.5921 | **0.5937** | 0.5927 |
| 400 | 0.5859 | 0.5897 | 0.5911 | |

v3 + `lorav3_exact_it_3p00.json` on the same protocol: 0.5802. The selected step 300 + G 3.6
(`exact_it_3p6.json`) per video against v3: 14 0.692 -> 0.697, 30 0.697 -> 0.689, 19 0.506 -> 0.517,
54 0.466 -> 0.484, 13 0.626 -> 0.647 (4 of 5 up).

**Held-out gate, one run, 32 frames (`../../track/held32-lorav4-s300-it360/`):**

| | 21 | 22 | 36 | 47 | 82 | all |
|---|---|---|---|---|---|---|
| v3 + edit (`../../weight-space/README.md`) | 0.674 | 0.680 | 0.575 | 0.560 | 0.555 | 0.6025 |
| v4 step 300 + G 3.6 | 0.6821 | 0.6785 | 0.5726 | 0.5801 | 0.5436 | **0.5995** |

Gate: >= 0.6125. **Failed.** The two are a tie within noise (2 videos up, 2 down, 1 flat; -0.003
overall), so this is not evidence that v4 is worse, only that the recipe change did not buy the
+0.01 the gate asked for. The +0.0135 on train-5 did not transfer: the extra frames come from the
same 15 videos, so they add correlated views rather than new scenes. The 1,500-frame rerun was not
run (it was conditional on the gate).
