# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | e16 | e16 / base (push, wall) |
|---|---|---|---|
| 1 | 1180 / 1266 | 1080 / 1162 | 0.915, 0.918 |
| 2 | 1180 / 1258 | 1110 / 1140 | 0.941, 0.907 |
| 3 | 1185 / 1257 | 1085 / 1180 | 0.916, 0.939 |

Median (and worst) over rounds of the per-round ratio: e16 push 0.916 (max 0.941), wall 0.918 (max 0.939) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | e16 | 32/32 | 0.9994 | 0.9656 | no |
| 2 | e16 | 32/32 | 0.9994 | 0.9656 | no |
| 3 | e16 | 32/32 | 0.9994 | 0.9656 | no |
