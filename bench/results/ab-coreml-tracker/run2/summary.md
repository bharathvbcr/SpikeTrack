# Tracker step A/B summary (steady frames / steps >= 8)

## Precondition: stage runs against unprofiled controls

Controls' steady push_frame mean 1892.5 ms over 52 frames.

- round 1: in-Session 1768.6 ms/frame (-6.5%) ok
- round 2: in-Session 1757.3 ms/frame (-7.1%) ok
- round 3: in-Session 1880.4 ms/frame (-0.6%) ok

## Per round, ms

| round | SpikeTrack tracker median (min) | port fp16 step | port fp32 step | ratio vs fp16 | ratio vs fp32 | SpikeTrack backbone | port backbone fp16 |
|---|---|---|---|---|---|---|---|
| 1 | 407.8 (403.2) | 70.6 | 204.0 | 5.78 | 2.00 | 986.0 | 273.1 |
| 2 | 403.2 (400.0) | 68.6 | 206.0 | 5.88 | 1.96 | 982.0 | 256.8 |
| 3 | 425.0 (411.7) | 82.8 | 201.2 | 5.14 | 2.11 | 1020.5 | 353.0 |

- round 1: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames
- round 2: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames
- round 3: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames

## Noise

- round-to-round spread of the medians: SpikeTrack 5.4%, port fp16 20.6%
- port fp32 2.4%
- step cost against bank depth (port fp16, round 1, every timed call from step 2): 67.2, 67.8, 69.7, 69.9, 70.5, 69.3, 69.4, 68.5, 68.2, 68.5, 69.8, 69.2, 69.0, 69.8, 69.5, 70.7, 70.8, 68.9, 71.3, 71.1, 70.5, 70.6, 71.1, 70.2, 70.6, 71.4, 71.8, 71.8, 72.8, 71.4, 72.2, 72.7

## Decision (pre-stated)

- vs port fp32: ratios 2.00, 1.96, 2.11; noise 5.4%; NOT MET
- vs port fp16: ratios 5.78, 5.88, 5.14; noise 20.6%; NOT MET
