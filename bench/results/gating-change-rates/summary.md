# VISEM per-window change rates (no model)

20 videos, 300 consecutive frames each from frame 0, resized to 1008x1008 grey.
Mean over videos of the per-frame fraction changed; window = one of the 3x3 attention windows.

| stride | tau | patches changed | windows changed | frames with all 9 changed | frames with none changed |
|---|---|---|---|---|---|
| 1 | 2 | 0.052 | 0.944 | 0.741 | 0.000 |
| 1 | 4 | 0.020 | 0.646 | 0.138 | 0.051 |
| 1 | 8 | 0.009 | 0.546 | 0.076 | 0.058 |
| 1 | 16 | 0.003 | 0.435 | 0.048 | 0.080 |
| 1 | 32 | 0.001 | 0.216 | 0.019 | 0.267 |
| 2 | 2 | 0.081 | 0.983 | 0.918 | 0.000 |
| 2 | 4 | 0.030 | 0.715 | 0.247 | 0.048 |
| 2 | 8 | 0.014 | 0.593 | 0.101 | 0.054 |
| 2 | 16 | 0.006 | 0.510 | 0.066 | 0.060 |
| 2 | 32 | 0.002 | 0.381 | 0.043 | 0.100 |
| 5 | 2 | 0.144 | 0.997 | 0.978 | 0.000 |
| 5 | 4 | 0.044 | 0.816 | 0.516 | 0.031 |
| 5 | 8 | 0.020 | 0.665 | 0.185 | 0.051 |
| 5 | 16 | 0.010 | 0.560 | 0.089 | 0.054 |
| 5 | 32 | 0.004 | 0.470 | 0.060 | 0.075 |

Per video, stride 1, tau 8 (windows changed):

- 11: windows 0.755, patches 0.013
- 12: windows 0.998, patches 0.034
- 13: windows 0.781, patches 0.007
- 14: windows 0.385, patches 0.005
- 15: windows 0.352, patches 0.002
- 19: windows 0.645, patches 0.011
- 21: windows 0.504, patches 0.006
- 22: windows 0.552, patches 0.006
- 23: windows 0.111, patches 0.001
- 24: windows 0.798, patches 0.016
- 29: windows 0.003, patches 0.000
- 30: windows 0.591, patches 0.010
- 35: windows 0.583, patches 0.009
- 36: windows 0.895, patches 0.013
- 38: windows 0.473, patches 0.007
- 47: windows 0.615, patches 0.011
- 52: windows 0.219, patches 0.002
- 54: windows 0.651, patches 0.008
- 60: windows 0.504, patches 0.007
- 82: windows 0.504, patches 0.005
