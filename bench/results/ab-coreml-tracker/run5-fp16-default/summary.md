# Tracker step A/B summary (steady frames / steps >= 8)

## Precondition: stage runs against unprofiled controls

Controls' steady push_frame mean 874.8 ms over 52 frames.

- round 1: in-Session 1010.7 ms/frame (+15.5%) OUTSIDE 10%
- round 2: in-Session 1069.5 ms/frame (+22.3%) OUTSIDE 10%
- round 3: in-Session 939.9 ms/frame (+7.4%) ok

## Per round, ms

| round | SpikeTrack tracker median (min) | port fp16 step | port fp32 step | ratio vs fp16 | ratio vs fp32 | SpikeTrack backbone | port backbone fp16 |
|---|---|---|---|---|---|---|---|
| 1 | 203.2 (178.5) | 72.2 | 209.6 | 2.81 | 0.97 | 123.6 | 268.9 |
| 2 | 207.7 (183.5) | 69.9 | 210.2 | 2.97 | 0.99 | 124.7 | 314.1 |
| 3 | 204.9 (181.4) | 89.7 | 213.2 | 2.28 | 0.96 | 119.7 | 298.7 |

- round 1: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp16, births on 0 steady frames
- round 2: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp16, births on 0 steady frames
- round 3: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp16, births on 0 steady frames

## Noise

- round-to-round spread of the medians: SpikeTrack 2.2%, port fp16 28.3%
- port fp32 1.7%
- step cost against bank depth (port fp16, round 1, every timed call from step 2): 72.5, 72.8, 73.5, 74.8, 73.1, 72.0, 72.7, 71.2, 72.7, 71.4, 73.1, 73.8, 72.0, 72.5, 71.0, 74.5, 71.8, 72.3, 73.1, 73.5, 73.3, 72.2, 73.0, 73.3, 73.1, 71.5, 72.1, 70.7, 71.4, 71.0, 70.8, 71.6

## Decision (pre-stated)

NOT READ: a stage run is outside 10% of the controls, so its stage times are not taken literally.
- vs port fp32: ratios 0.97, 0.99, 0.96; noise 2.2%; MET (parity or better) (not read: precondition failed)
- vs port fp16: ratios 2.81, 2.97, 2.28; noise 28.3%; NOT MET (not read: precondition failed)
