# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | dt16 | dt16 / base (push, wall) |
|---|---|---|---|
| 1 | 860 / 997 | 870 / 932 | 1.012, 0.935 |
| 2 | 870 / 970 | 770 / 880 | 0.885, 0.907 |
| 3 | 875 / 970 | 825 / 1049 | 0.943, 1.082 |

Median (and worst) over rounds of the per-round ratio: dt16 push 0.943 (max 1.012), wall 0.935 (max 1.082) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | dt16 | 31/32 | 0.9872 | 0.3640 | no |
| 2 | dt16 | 31/32 | 0.9872 | 0.3640 | no |
| 3 | dt16 | 31/32 | 0.9872 | 0.3640 | no |
