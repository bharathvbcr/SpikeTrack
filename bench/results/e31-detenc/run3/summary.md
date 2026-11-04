# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | e16 | e16 / base (push, wall) |
|---|---|---|---|
| 1 | 1260 / 1360 | 1055 / 1112 | 0.837, 0.818 |
| 2 | 1155 / 1235 | 1085 / 1126 | 0.939, 0.912 |
| 3 | 1160 / 1228 | 1065 / 1131 | 0.918, 0.920 |

Median (and worst) over rounds of the per-round ratio: e16 push 0.918 (max 0.939), wall 0.912 (max 0.920) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | e16 | 32/32 | 0.9994 | 0.9656 | no |
| 2 | e16 | 32/32 | 0.9994 | 0.9656 | no |
| 3 | e16 | 32/32 | 0.9994 | 0.9656 | no |
