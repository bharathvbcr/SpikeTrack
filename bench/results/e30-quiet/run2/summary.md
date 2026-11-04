# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | t16m16 | t16m16n16 | t16m16n16 / t16m16 (push, wall) |
|---|---|---|---|
| 1 | 1265 / 1365 | 1180 / 1250 | 0.933, 0.916 |
| 2 | 1290 / 1361 | 1150 / 1222 | 0.891, 0.898 |
| 3 | 1285 / 1338 | 1110 / 1186 | 0.864, 0.886 |

Median (and worst) over rounds of the per-round ratio: t16m16n16 push 0.891 (max 0.933), wall 0.898 (max 0.916) over 3 round(s)

## Tracks against t16m16 (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | t16m16n16 | 32/32 | 0.9999 | 0.9907 | no |
| 2 | t16m16n16 | 32/32 | 0.9999 | 0.9907 | no |
| 3 | t16m16n16 | 32/32 | 0.9999 | 0.9907 | no |
