# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | o256 | o32 | o8 | o1 | o32 / o256 (push, wall) | o8 / o256 (push, wall) | o1 / o256 (push, wall) |
|---|---|---|---|---|---|---|---|
| 1 | 1380 / 1445 | 1355 / 1526 | 1620 / 1734 | 1470 / 1663 | 0.982, 1.056 | 1.174, 1.200 | 1.065, 1.151 |
| 2 | 1550 / 1742 | 1315 / 1512 | 1260 / 1341 | 1595 / 1688 | 0.848, 0.868 | 0.813, 0.770 | 1.029, 0.969 |
| 3 | 1580 / 1765 | 1460 / 1597 | 1465 / 1546 | 1455 / 1663 | 0.924, 0.905 | 0.927, 0.876 | 0.921, 0.943 |

Median (and worst) over rounds of the per-round ratio: o32 push 0.924 (max 0.982), wall 0.905 (max 1.056) over 3 round(s); o8 push 0.927 (max 1.174), wall 0.876 (max 1.200) over 3 round(s); o1 push 1.029 (max 1.065), wall 0.969 (max 1.151) over 3 round(s)

## Tracks against o256 (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | o32 | 32/32 | 1.0000 | 1.0000 | yes |
| 1 | o8 | 32/32 | 1.0000 | 1.0000 | yes |
| 1 | o1 | 32/32 | 1.0000 | 1.0000 | yes |
| 2 | o32 | 32/32 | 1.0000 | 1.0000 | yes |
| 2 | o8 | 32/32 | 1.0000 | 1.0000 | yes |
| 2 | o1 | 32/32 | 1.0000 | 1.0000 | yes |
| 3 | o32 | 32/32 | 1.0000 | 1.0000 | yes |
| 3 | o8 | 32/32 | 1.0000 | 1.0000 | yes |
| 3 | o1 | 32/32 | 1.0000 | 1.0000 | yes |
