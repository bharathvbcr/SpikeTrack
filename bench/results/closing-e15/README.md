# Closing job for E11b-E15 (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md ("Closing job for E11b-E15", 0f56e10) before the run. One
mac_heavy hold, 05:32-06:38Z; `uptime` read 19.0 at the start and 2.9 at the end. The code is 0f56e10
(the E15 code, 66c3a58, plus docs). Binary shas are in `versions.txt`: the E11b binary (3ca1476) is
`457ca88d...`, the E15 binary and `target/release/spiketrack` are both `8567e60a...`. The job script
is `job.sh`.

Setup for the speed rounds: Metal, the default CoreML fp32 trunk, prompt "sperm", r16 LoRA,
`--boxes detector`, `--max-objects 64`, 16 frames of VISEM 82 and 22. Each round runs old then new
on 82, then old then new on 22.

## 1. Every test target compiles [V]

`cargo test --release -p st-model -p spiketrack --tests --no-run` built all 14 test executables,
including the 12 integration suites that no hold had compiled since the E12-E15 type changes
(`tests-compile.log`). They were compiled, not run.

## 2. Cumulative speed, E15 against E11b [V]

Steady mean, frames 1-15, ms. The 1-minute load is read just before each run.

| round | 82 E11b | 82 E15 | 82 ratio | 22 E11b | 22 E15 | 22 ratio |
|---|---|---|---|---|---|---|
| 1 | 2695 (19.0) | 2287 (11.0) | 0.849 | 2325 (11.4) | 1927 (11.8) | 0.829 |
| 2 | 2824 (15.4) | 2351 (15.8) | 0.833 | 2391 (19.6) | 1919 (23.1) | 0.803 |
| 3 | 2711 (21.7) | 2271 (14.6) | 0.838 | 2419 (9.7) | 1923 (13.6) | 0.795 |
| min of 3 | 2695 | 2271 | **0.843** | 2325 | 1919 | **0.825** |

- E15 is faster in all 6 pairs. The paired ratios span 0.833-0.849 on 82 and 0.795-0.829 on 22.
- The load ran 9.7-23.1 across the 12 runs. This is the one cumulative ratio for E12 + E13 + E15,
  read at that load. The per-step ratios from earlier holds (0.86, 0.97) were at other loads and are
  not multiplied.
- E15's times are tighter than E11b's: 1919-1927 against 2325-2419 on 22 [V]. Why is
  [unexplained].
- All 6 E15 runs are byte-identical to E11b round 1 (`cmp`, 6 of 6).

## 3. Held-out 5 videos x 300 frames [V]

`tools/jobs/track_visem.sh 300` with the committed settings, the same settings as
`track/held300-lora600-sperm-detector-coreml-fp32-max64` (that run is b11bb69, E10).

- **All 5 videos' predictions are byte-identical** to the committed run (`cmp`, 5 of 5, 1500
  frames). That makes the identity gate hold from E10 through E15 on the full held-out set.
- **HOTA 0.4922** (DetA 0.3734, AssA 0.6903, LocA 0.7221). `held300/eval.txt` matches the committed
  `eval.txt` line for line.
- **Footprint: the device caches are bounded, but the process footprint creeps [V].** MiB at
  frames 15 / 99 / 199 / last:

  | video | 15 | 99 | 199 | 299 |
  |---|---|---|---|---|
  | 21 | 8414 | 8751 | 8771 | 8975 |
  | 22 | 9196 | 9143 | 9305 | 8779 |
  | 36 | 8952 | 8880 | 8918 | 8959 |
  | 47 | 9207 | 9311 | 9305 | 8698 |
  | 82 | 8872 | 9178 | 9054 | 9110 |

  The four samples hide a trend. Over all 300 frames, slope from frame 16, MiB per frame:

  | video | E15 footprint | E10 footprint (committed run) | E15 Metal allocated |
  |---|---|---|---|
  | 21 | +1.23 | -0.11 | +0.03 |
  | 22 | +0.87 | -0.93 | 0.00 |
  | 36 | +0.72 | -0.14 | +0.23 |
  | 47 | +1.55 | -0.03 | +0.02 |
  | 82 | +0.58 | -0.25 | +0.03 |

  - **Metal allocated is flat after ~frame 100** (36 rises while it gains tracker states). The
    device token caches E15 added are released with their frames [V]. They sit ~400 MiB above E10's
    Metal allocation (5557-5649 against 5185-5206 MiB at frame 16), and the process footprint at
    frames 250-299 is no higher than E10's.
  - **The process footprint keeps rising on every E15 video, and did not on E10.** Over frames
    250-299 it averages 80-177 MiB more than over 100-149. By the letter of the pre-stated rule, this
    is a footprint that keeps climbing past frame 16.
  - The growth is outside Metal allocations [I], and it is smaller than one leaked 5 MiB token
    tensor per frame. It is not yet tied to E12 (scoped host threads), E13 or E15 [unexplained].
  - At ~1.5 MiB per frame, a 3,000-frame video would gain ~4 GB, so this matters beyond 300 frames.
    It is recorded as GAP-E15-HOST-FOOTPRINT-CREEP.
- **Unpaired context only**, not a ratio: the steady held-out mean is 2165 ms per frame over the
  5 videos (21 2480, 22 1924, 36 2312, 47 1855, 82 2256). The committed E10 run averaged 3019. It ran
  on another day at another load, so the README and WRITEUP do not carry it.

## Also seen

- The build warns that `eval_visem::evaluate` is never used in the `spiketrack` bin
  (`build.log`). It predates this series: the file was last changed in 47a58df, and the series
  touched nothing in `crates/spiketrack/src`. It is not fixed here.

Files: `speed/` (run directories with predictions, plus the run logs), `held300/` (run, logs, eval),
the test and build logs, `held300.log`, `job.sh` and `versions.txt`. Masks are not committed.
