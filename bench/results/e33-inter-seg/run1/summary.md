# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | is16 | is16 / base (push, wall) |
|---|---|---|---|
| 1 | 870 / 994 | 875 / 963 | 1.006, 0.968 |
| 2 | 910 / 1010 | 875 / 955 | 0.962, 0.945 |
| 3 | 920 / 1005 | 865 / 950 | 0.940, 0.946 |

Median (and worst) over rounds of the per-round ratio: is16 push 0.962 (max 1.006), wall 0.946 (max 0.968) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | is16 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | is16 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | is16 | 32/32 | 1.0000 | 0.9919 | no |
