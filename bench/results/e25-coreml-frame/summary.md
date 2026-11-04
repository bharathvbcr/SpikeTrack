# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | default | necks32 | mattn16 | fast16 | necks32 / default (push, wall) | mattn16 / default (push, wall) | fast16 / default (push, wall) |
|---|---|---|---|---|---|---|---|
| 1 | 2805 / 2966 | 2770 / 3163 | 2050 / 2113 | failed / failed | 0.988, 1.066 | 0.731, 0.712 | -, - |
| 2 | 2745 / 2792 | 2985 / 2973 | 2665 / 2799 | failed / failed | 1.087, 1.065 | 0.971, 1.003 | -, - |
| 3 | 2935 / 2931 | 2785 / 2842 | 2070 / 2186 | failed / failed | 0.949, 0.970 | 0.705, 0.746 | -, - |

Median (and worst) over rounds of the per-round ratio: necks32 push 0.988 (max 1.087), wall 1.065 (max 1.066) over 3 round(s); mattn16 push 0.731 (max 0.971), wall 0.746 (max 1.003) over 3 round(s); fast16 failed

## Tracks against default (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | necks32 | 32/32 | 1.0000 | 1.0000 | no |
| 1 | mattn16 | 32/32 | 1.0000 | 0.9919 | no |
| 1 | fast16 | failed | | | |
| 2 | necks32 | 32/32 | 1.0000 | 1.0000 | no |
| 2 | mattn16 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | fast16 | failed | | | |
| 3 | necks32 | 32/32 | 1.0000 | 1.0000 | no |
| 3 | mattn16 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | fast16 | failed | | | |
