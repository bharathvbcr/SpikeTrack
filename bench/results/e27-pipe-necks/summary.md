# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | default | m16 | m16-n32 | fast16 | fast16-n16 | m16 / default (push, wall) | m16-n32 / default (push, wall) | fast16 / default (push, wall) | fast16-n16 / default (push, wall) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2975 / 2961 | 2335 / 2440 | 2225 / 2397 | 1650 / 1776 | 1280 / 1500 | 0.785, 0.824 | 0.748, 0.809 | 0.555, 0.600 | 0.430, 0.507 |
| 2 | 2850 / 2945 | 2240 / 2246 | 2110 / 2318 | 1575 / 1676 | 1315 / 1474 | 0.786, 0.763 | 0.740, 0.787 | 0.553, 0.569 | 0.461, 0.500 |
| 3 | 2715 / 2791 | 1940 / 2173 | 1995 / 2216 | 1480 / 1666 | 1405 / 1596 | 0.715, 0.779 | 0.735, 0.794 | 0.545, 0.597 | 0.517, 0.572 |

Median (and worst) over rounds of the per-round ratio: m16 push 0.785 (max 0.786), wall 0.779 (max 0.824) over 3 round(s); m16-n32 push 0.740 (max 0.748), wall 0.794 (max 0.809) over 3 round(s); fast16 push 0.553 (max 0.555), wall 0.597 (max 0.600) over 3 round(s); fast16-n16 push 0.461 (max 0.517), wall 0.507 (max 0.572) over 3 round(s)

## Tracks against default (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | m16 | 32/32 | 1.0000 | 0.9919 | no |
| 1 | m16-n32 | 32/32 | 1.0000 | 0.9919 | no |
| 1 | fast16 | 32/32 | 0.9981 | 0.7945 | no |
| 1 | fast16-n16 | 32/32 | 0.9980 | 0.7945 | no |
| 2 | m16 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | m16-n32 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | fast16 | 32/32 | 0.9981 | 0.7945 | no |
| 2 | fast16-n16 | 32/32 | 0.9980 | 0.7945 | no |
| 3 | m16 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | m16-n32 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | fast16 | 32/32 | 0.9981 | 0.7945 | no |
| 3 | fast16-n16 | 32/32 | 0.9980 | 0.7945 | no |
