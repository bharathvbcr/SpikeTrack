# E23 summary (steady frames >= 8)

| round | old tracker step ms | new | new / old |
|---|---|---|---|
| 1 | 1043.0 | 960.8 | 0.921 |
| 2 | 1179.2 | 929.4 | 0.788 |
| 3 | 1123.3 | 1001.8 | 0.892 |

Same-binary spread: old 13.1%, new 7.8%; noise 13.1%.
Speed (pre-stated: every ratio < 1 - noise): no speedup claimed.

Unprofiled controls, steady push_frame mean: old 4032.3 ms, new 3847.3 ms
- old: frame 8: metal allocated 3859 MiB, footprint 3660 MiB; frame 33: metal allocated 3950 MiB, footprint 3031 MiB
- new: frame 8: metal allocated 4139 MiB, footprint 4675 MiB; frame 33: metal allocated 4230 MiB, footprint 3229 MiB

Gate 7 (cached path ran): PASS, no tracker linear over more than one frame's rows
