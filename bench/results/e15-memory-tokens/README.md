# E15: memory-attention inputs kept on the device (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E15) before the run. The binary is 66c3a58 and the control is the
E13 binary (6ba37d6); both shas are in `versions.txt`.

The setup is E13's: Metal, the default CoreML fp32 trunk, prompt "sperm", r16 LoRA,
`--boxes detector`, `--max-objects 64`, 16 frames of VISEM 82 and 22. `job.sh` ran in one mac_heavy
hold, in this order per video: control, E15, E15 with `--profile stages`, control, E15.

The job was launched with its stdout discarded (my mistake). It already held the lock, so it was
left to finish. Every summary below is recomputed from the per-run logs and predictions in this
directory, and the per-run load is not recorded. `uptime` read 46 when the job began and 11 just
after it ended.

## Gate [V]

- **Predictions are byte-identical** to control-1 for every E15 run and for control-2, on both
  videos (8 of 8 `cmp`, recomputed).
- **Tests:** spiketrack 75 and st-model 46 passed, and the Metal RPB test passed.
- **Stages validity (10% rule):**
  - 82: 2697 ms against controls of 2547 and 2488 (+8.4% of their mean).
  - 22: 1861 against 1871 and 1907 (-1.5%).

## Tracker phases, ms per steady frame (`--profile stages`)

| phase | E13, 82 state 0 | E15, 82 state 0 | E13, 82 state 1 | E15, 82 state 1 | E13, 22 | E15, 22 |
|---|---|---|---|---|---|---|
| memory + pointer tokens (host) + upload | 63.6 | 13.9 | 50.1 | 5.5 | 55.5 | 9.9 |
| memory attention (device) | 249.5 | 249.7 | 203.1 | 216.8 | 246.0 | 229.1 |
| multiplex decoder | 22.5 | 30.6 | 19.2 | 37.1 | 21.4 | 22.1 |

- **The tokens phase passes the pre-stated <= 15 ms on every state:** 13.9, 5.5 and 9.9.
  - What remains is the pointer block's upload and the device concat dispatch, which the next mark
    waits for.
- **The multiplex decoder is 8-18 ms higher on 82 [unexplained].** It is unchanged on 22.
  - E15 does not touch it.
  - The 82 stages run is also the slowest run of the hold, at +8.4% against the controls.
  - Read the rise as load until a rerun says otherwise.

## Frame time, unprofiled, steady mean (frames 1-15), ms

| | control | E15 | E15 / control, min of 2 | paired |
|---|---|---|---|---|
| 82 | 2547, 2488 | 2501, 2215 | 0.890 | -1.8%, -11.0% |
| 22 | 1871, 1907 | 1838, 1818 | 0.972 | -1.8%, -4.7% |

- **E15 is faster in all four pairs.** On 82 the min-of-2 ratio passes the pre-stated "3% or more".
- **The size is noisy.** The 82 pairs differ from each other by 9 points, and per-run load was not
  recorded.
  - The expected gain is the tokens phase's ~94 ms on 82 (about 3.5%) and ~46 ms on 22 (about 2.4%).
  - 22's -1.8% and -4.7% bracket that. 82's -11% pair is larger than E15 can explain alone.
- **HOTA** is inherited, not re-run. Predictions are byte-identical on 32 frames, and every changed
  step is a copy or the same single f32 add.

## Memory [V]

| | footprint at frame 15 (MiB) | across frames 1-15 |
|---|---|---|
| control (2 runs) | 9470, 9467 | climbs to ~10,300 by frame 12 |
| E15 (2 runs) | 8679, 8712 | stays at ~8,700-9,400 |

- **E15 runs about 760 MiB lower.** The bound was "within ~200 MiB, not climbing", and E15 is below
  the control and flat.
- The likely reason [I]: per frame, the host no longer builds and drops ~100 MB of token copies, and
  the per-frame `prop72` readback is gone.

Files: the run directories and logs, `v*-stages/*.profile-stages.md`, the test logs, `job.sh` and
`versions.txt`. Masks are not committed.
