# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | e34 | e34 / base (push, wall) |
|---|---|---|---|
| 1 | 905 / 1029 | 920 / 1022 | 1.017, 0.993 |
| 2 | 940 / 1038 | 905 / 1034 | 0.963, 0.996 |
| 3 | 895 / 1021 | 930 / 1045 | 1.039, 1.024 |

Median (and worst) over rounds of the per-round ratio: e34 push 1.017 (max 1.039), wall 0.996 (max 1.024) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | e34 | 32/32 | 1.0000 | 1.0000 | yes |
| 2 | e34 | 32/32 | 1.0000 | 1.0000 | yes |
| 3 | e34 | 32/32 | 1.0000 | 1.0000 | yes |
