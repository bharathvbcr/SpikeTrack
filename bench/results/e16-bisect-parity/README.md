# E16: footprint bisect, and the parity suites run (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E16, b0c5437) before the run. One mac_heavy hold, 06:45-07:57Z,
load 2.2-9.3. Code b0c5437; binary shas in `versions.txt`; the script is `job.sh`.

## Footprint bisect: video 47, 300 frames, committed held-out settings

| build | predictions vs committed 47 | footprint slope, MiB/frame (16-299) | Metal allocated slope | steady mean |
|---|---|---|---|---|
| E11b (3ca1476) | byte-identical | -0.53 | +0.022 | 2122 ms |
| E12 B (5e13d74) | byte-identical | -0.85 | +0.017 | 1856 ms |
| E13 (6ba37d6) | byte-identical | -0.94 | +0.017 | 1861 ms |
| E15 (66c3a58) | byte-identical | **+2.46** | +0.024 | 1777 ms |
| E10 (b11bb69) | **differs: invalid arm** | -0.24 | +0.016 | 3026 ms |

- **The E10 arm is my setup error, not a finding.**
  - The saved E10 binary predates the CoreML default, so it ran the Ojas f32 trunk
    (`v47-e10/47.run.json`: `trunk_precision f32`).
  - The committed E10 run used `coreml-fp32`, which is why its predictions differ and it is slower.
  - It is not a control for the other builds.
  - The committed coreml-fp32 E10 run (closing-e15 README) already showed a flat footprint on 47
    (-0.03).
- **By the pre-stated rule, E15 owns the creep [V, n = 1].** E13 is at -0.94 (<= 0.2) and E15 is at
  +2.46 (>= 0.5); the step is 3.4, over the 0.3 floor. E11b through E13 all drift slightly down.
- **The shape is not "E15 sits higher."** Mean footprint by window, MiB:

  | build | 16-49 | 50-99 | 100-149 | 150-199 | 200-249 | 250-299 |
  |---|---|---|---|---|---|---|
  | E13 | 10134 | 10008 | 9849 | 9894 | 9970 | 9809 |
  | E15 | 8865 | 8777 | 8975 | 9106 | 9234 | 9393 |
  | E15, closing job | 9072 | 9056 | 9238 | 9215 | 9393 | 9381 |

  - E15 starts ~1.2 GB below E13, climbs from frame ~100 on, and ends ~400 MiB below it.
  - The closing job's E15 run on 47 has the same shape.
  - Metal allocated stays flat in every build, so the growth is not in Ojas's tracked device
    allocations [I].
  - Whether E15 levels off at or below E13's level (allocator or driver residency settling) or
    keeps climbing (a leak) is **not decided by 300 frames [unexplained]**.

## Parity suites, run [V]

- `tracker_parity`: all 8 ignored tests pass (judo; memory attention, memory encoder, multiplex
  and interactive decoders; CPU and Metal).
- `detect_track_parity`, video 22, 32 frames:
  - CPU: id-set misses 0, mean IoU 0.99854 over 470 pairs. The yield table and summary are
    identical to the committed `parity-tracker/cpu-f32-detect-track-visem22.log`. This is the CPU
    oracle through the E12-E15 type changes.
  - Metal: id-set misses 0, mean IoU 0.99855 over 470 pairs, identical to the committed log.
    - Every IoU in the yield table is identical.
    - Only the `prob dmax` column moved, 2.3e-6 -> 2.7e-6, uniformly on all 32 rows
      (`detect-track-metal.diff`).
    - The committed log is from 3658918, so this spans E12-E15 and the ojas-st changes since then
      [unattributed]. It is well inside the gate.

Files: the run directories and logs (masks removed), the test logs, the Metal diff, `job.sh` and
`versions.txt`.
