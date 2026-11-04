# Tracker step A/B summary (steady frames / steps >= 8)

## Precondition: stage runs against unprofiled controls

Controls' steady push_frame mean 1757.3 ms over 52 frames.

- round 1: in-Session 1928.8 ms/frame (+9.8%) ok
- round 2: in-Session 1887.1 ms/frame (+7.4%) ok
- round 3: in-Session 1812.8 ms/frame (+3.2%) ok

## Per round, ms

| round | SpikeTrack tracker median (min) | port fp16 step | port fp32 step | ratio vs fp16 | ratio vs fp32 | SpikeTrack backbone | port backbone fp16 |
|---|---|---|---|---|---|---|---|
| 1 | 430.7 (418.2) | 78.9 | 216.8 | 5.46 | 1.99 | 1048.5 | 283.6 |
| 2 | 425.2 (381.7) | 77.5 | 218.4 | 5.48 | 1.95 | 1029.9 | 273.8 |
| 3 | 390.7 (380.3) | 70.9 | 213.3 | 5.51 | 1.83 | 1029.0 | 267.1 |

- round 1: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames
- round 2: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames
- round 3: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames

## Noise

- round-to-round spread of the medians: SpikeTrack 10.2%, port fp16 11.3%
- port fp32 2.4%
- step cost against bank depth (port fp16, round 1, every timed call from step 2): 75.5, 70.7, 72.2, 71.8, 71.0, 72.8, 76.5, 79.2, 79.9, 77.7, 79.6, 79.7, 78.9, 79.5, 81.0, 80.3, 79.7, 80.1, 75.8, 79.0, 78.6, 79.7, 80.6, 79.5, 72.0, 74.2, 71.1, 75.5, 71.6, 71.2, 72.8, 73.3

## Decision (pre-stated)

- vs port fp32: ratios 1.99, 1.95, 1.83; noise 10.2%; NOT MET
- vs port fp16: ratios 5.46, 5.48, 5.51; noise 11.3%; NOT MET
