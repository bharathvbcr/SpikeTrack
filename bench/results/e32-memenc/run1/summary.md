# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | me16 | me16 / base (push, wall) |
|---|---|---|---|
| 1 | 1165 / 1248 | 950 / 1080 | 0.815, 0.865 |
| 2 | 1165 / 1241 | 910 / 1044 | 0.781, 0.841 |
| 3 | 1130 / 1183 | 890 / 981 | 0.788, 0.829 |

Median (and worst) over rounds of the per-round ratio: me16 push 0.788 (max 0.815), wall 0.841 (max 0.865) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | me16 | 32/32 | 1.0000 | 0.9919 | no |
| 2 | me16 | 32/32 | 1.0000 | 0.9919 | no |
| 3 | me16 | 32/32 | 1.0000 | 0.9919 | no |
