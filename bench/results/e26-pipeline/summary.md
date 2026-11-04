# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | default | pipe | pipe-m16 | pipe-m16-seg32 | fast16 | pipe / default (push, wall) | pipe-m16 / default (push, wall) | pipe-m16-seg32 / default (push, wall) | fast16 / default (push, wall) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2920 / 3117 | 2370 / 2423 | 1855 / 1928 | 1910 / 2044 | 1425 / 1594 | 0.812, 0.778 | 0.635, 0.618 | 0.654, 0.656 | 0.488, 0.511 |
| 2 | 3035 / 2981 | 2645 / 2715 | 1990 / 2101 | 2075 / 2138 | 1375 / 1577 | 0.871, 0.911 | 0.656, 0.705 | 0.684, 0.717 | 0.453, 0.529 |
| 3 | 3240 / 3166 | 2745 / 2824 | 1870 / 2055 | 1880 / 2052 | 1220 / 1356 | 0.847, 0.892 | 0.577, 0.649 | 0.580, 0.648 | 0.377, 0.428 |

Median (and worst) over rounds of the per-round ratio: pipe push 0.847 (max 0.871), wall 0.892 (max 0.911) over 3 round(s); pipe-m16 push 0.635 (max 0.656), wall 0.649 (max 0.705) over 3 round(s); pipe-m16-seg32 push 0.654 (max 0.684), wall 0.656 (max 0.717) over 3 round(s); fast16 push 0.453 (max 0.488), wall 0.511 (max 0.529) over 3 round(s)

## Tracks against default (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | pipe | 32/32 | 1.0000 | 1.0000 | yes |
| 1 | pipe-m16 | 32/32 | 1.0000 | 0.9919 | no |
| 1 | pipe-m16-seg32 | 32/32 | 1.0000 | 0.9919 | no |
| 1 | fast16 | 32/32 | 0.9980 | 0.7945 | no |
| 2 | pipe | 32/32 | 1.0000 | 1.0000 | yes |
| 2 | pipe-m16 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | pipe-m16-seg32 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | fast16 | 32/32 | 0.9978 | 0.6837 | no |
| 3 | pipe | 32/32 | 1.0000 | 1.0000 | yes |
| 3 | pipe-m16 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | pipe-m16-seg32 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | fast16 | 32/32 | 0.9980 | 0.7945 | no |
