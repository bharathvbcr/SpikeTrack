# Tracker step A/B summary (steady frames / steps >= 8)

## Precondition: stage runs against unprofiled controls

Controls' steady push_frame mean 2054.0 ms over 52 frames.

- round 1: in-Session 2253.2 ms/frame (+9.7%) ok
- round 2: in-Session 2268.7 ms/frame (+10.4%) OUTSIDE 10%
- round 3: in-Session 2251.8 ms/frame (+9.6%) ok

## Per round, ms

| round | SpikeTrack tracker median (min) | port fp16 step | port fp32 step | ratio vs fp16 | ratio vs fp32 | SpikeTrack backbone | port backbone fp16 |
|---|---|---|---|---|---|---|---|
| 1 | 523.1 (440.6) | 79.8 | 216.3 | 6.55 | 2.42 | 1256.5 | 323.4 |
| 2 | 519.5 (434.5) | 99.3 | 274.3 | 5.23 | 1.89 | 1295.9 | 372.4 |
| 3 | 513.3 (484.9) | 104.0 | 275.7 | 4.94 | 1.86 | 1256.6 | 437.9 |

- round 1: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames
- round 2: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames
- round 3: SpikeTrack 26 frames, 16 tracks, cap 16, active 16-16 on steady frames, trunk coreml-fp32, births on 0 steady frames

## Noise

- round-to-round spread of the medians: SpikeTrack 1.9%, port fp16 30.3%
- port fp32 27.4%
- step cost against bank depth (port fp16, round 1, every timed call from step 2): 83.5, 78.1, 78.9, 76.5, 76.8, 77.8, 79.5, 79.7, 82.6, 79.2, 78.9, 79.7, 82.4, 78.7, 80.0, 77.5, 79.7, 80.1, 78.1, 79.7, 79.4, 79.9, 81.2, 80.7, 85.8, 88.4, 83.0, 79.3, 80.3, 81.4, 81.5, 77.2

## Decision (pre-stated)

NOT READ: a stage run is outside 10% of the controls, so its stage times are not taken literally.
- vs port fp32: ratios 2.42, 1.89, 1.86; noise 27.4%; NOT MET (not read: precondition failed)
- vs port fp16: ratios 6.55, 5.23, 4.94; noise 30.3%; NOT MET (not read: precondition failed)
