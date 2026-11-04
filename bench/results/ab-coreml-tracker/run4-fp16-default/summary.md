# Tracker step A/B summary (steady frames / steps >= 8)

## Precondition: stage runs against unprofiled controls

Controls' steady push_frame mean 917.9 ms over 52 frames.

- round 1: in-Session 1013.9 ms/frame (+10.5%) OUTSIDE 10%
- round 2: in-Session 809.0 ms/frame (-11.9%) OUTSIDE 10%
- round 3: in-Session 1072.1 ms/frame (+16.8%) OUTSIDE 10%

## Per round, ms

| round | SpikeTrack tracker median (min) | port fp16 step | port fp32 step | ratio vs fp16 | ratio vs fp32 | SpikeTrack backbone | port backbone fp16 |
|---|---|---|---|---|---|---|---|
| 1 | 214.2 (188.9) | 74.1 | 234.2 | 2.89 | 0.91 | 130.1 | 276.8 |
| 2 | 180.1 (172.3) | 71.0 | 212.8 | 2.54 | 0.85 | 113.3 | 275.7 |
| 3 | 275.8 (168.7) | 70.3 | 222.4 | 3.92 | 1.24 | 134.9 | 263.1 |

- round 1: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp16, births on 0 steady frames
- round 2: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp16, births on 0 steady frames
- round 3: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp16, births on 0 steady frames

## Noise

- round-to-round spread of the medians: SpikeTrack 53.1%, port fp16 5.4%
- port fp32 10.1%
- step cost against bank depth (port fp16, round 1, every timed call from step 2): 73.1, 74.5, 75.4, 76.0, 78.2, 78.4, 83.0, 77.0, 75.9, 74.7, 72.4, 73.5, 72.3, 72.3, 75.8, 73.5, 72.4, 73.4, 76.5, 73.8, 74.3, 73.9, 72.7, 73.4, 75.1, 73.9, 75.4, 72.3, 74.8, 78.0, 75.1, 77.0

## Decision (pre-stated)

NOT READ: a stage run is outside 10% of the controls, so its stage times are not taken literally.
- vs port fp32: ratios 0.91, 0.85, 1.24; noise 53.1%; MET (parity or better) (not read: precondition failed)
- vs port fp16: ratios 2.89, 2.54, 3.92; noise 53.1%; NOT MET (not read: precondition failed)
