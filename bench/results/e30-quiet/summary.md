# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | t16m16 | t16m16n16 | o32 | t16m16n16 / t16m16 (push, wall) | o32 / t16m16 (push, wall) |
|---|---|---|---|---|---|
| 1 | 1280 / 1335 | 1110 / 1191 | 1260 / 1415 | 0.867, 0.892 | 0.984, 1.060 |
| 2 | 1465 / 1600 | 1380 / 1507 | 1295 / 1505 | 0.942, 0.942 | 0.884, 0.941 |
| 3 | 1370 / 1476 | 1145 / 1278 | 1395 / 1518 | 0.836, 0.866 | 1.018, 1.029 |

Median (and worst) over rounds of the per-round ratio: t16m16n16 push 0.867 (max 0.942), wall 0.892 (max 0.942) over 3 round(s); o32 push 0.984 (max 1.018), wall 1.029 (max 1.060) over 3 round(s)

## Tracks against t16m16 (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | t16m16n16 | 32/32 | 0.9999 | 0.9907 | no |
| 1 | o32 | 32/32 | 1.0000 | 1.0000 | yes |
| 2 | t16m16n16 | 32/32 | 0.9999 | 0.9907 | no |
| 2 | o32 | 32/32 | 1.0000 | 1.0000 | yes |
| 3 | t16m16n16 | 32/32 | 0.9999 | 0.9907 | no |
| 3 | o32 | 32/32 | 1.0000 | 1.0000 | yes |
