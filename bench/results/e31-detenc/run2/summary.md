# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | e16 | e16 / base (push, wall) |
|---|---|---|---|
| 1 | 1180 / 1282 | 1130 / 1425 | 0.958, 1.112 |
| 2 | 1260 / 1324 | 1050 / 1163 | 0.833, 0.878 |
| 3 | 1255 / 1485 | 1105 / 1204 | 0.880, 0.811 |

Median (and worst) over rounds of the per-round ratio: e16 push 0.880 (max 0.958), wall 0.878 (max 1.112) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | e16 | 32/32 | 0.9994 | 0.9656 | no |
| 2 | e16 | 32/32 | 0.9994 | 0.9656 | no |
| 3 | e16 | 32/32 | 0.9994 | 0.9656 | no |
