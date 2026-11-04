# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | all16 | all16 / base (push, wall) |
|---|---|---|---|
| 1 | 865 / 995 | 750 / 839 | 0.867, 0.843 |
| 2 | 880 / 983 | 775 / 847 | 0.881, 0.862 |
| 3 | 880 / 966 | 720 / 831 | 0.818, 0.860 |

Median (and worst) over rounds of the per-round ratio: all16 push 0.867 (max 0.881), wall 0.860 (max 0.862) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | all16 | 31/32 | 0.9872 | 0.3640 | no |
| 2 | all16 | 31/32 | 0.9872 | 0.3640 | no |
| 3 | all16 | 31/32 | 0.9872 | 0.3640 | no |
