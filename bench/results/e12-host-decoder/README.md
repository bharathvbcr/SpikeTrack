# E12: the detector's host DETR decoder made faster, bit for bit (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E12) before the run.
- **A**: df8f066. `host::linear` computes 8 or 4 outputs per pass over the row, with no sum reordered
  and no FMA, and splits rows over up to 8 scoped threads.
- **B**: 5e13d74. A, plus the box-RPB bias as a device broadcast of its two axis tables. Before
  this, the host built the 33 MB bias and uploaded it six times per frame.
- **Control**: the E11b binary (3ca1476).

The setup is E11's: Metal, the default CoreML fp32 trunk, prompt "sperm", r16 LoRA,
`--boxes detector`, `--max-objects 64`, 16 frames of VISEM 82 and 22. `job.sh` ran it in one
mac_heavy hold. Load was 13.6 at the start and fell to 4.5 by the end (recorded per run in the job
output; binary shas in `versions.txt`).

## Gate [V]

- **Predictions are byte-identical to control-1** for every A and B run, profiled or not, on both
  videos (14 of 14 `cmp`). No HOTA rerun is needed.
- **Tests:** spiketrack 75 passed, and st-model 46 passed.
  - These include `blocked_threaded_linear_is_bit_identical_to_the_scalar_one` and the CPU RPB test.
  - `the_rpb_bias_matches_the_host_loop_bit_for_bit_on_metal` (ignored by default) passed.

## Detector host stages, ms per steady frame (`--profile stages`)

| stage | E11b (3ca1476), 82 | A, 82 | B, 82 | A, 22 | B, 22 |
|---|---|---|---|---|---|
| decoder host prep | 264.4 | 71.6 | 52.8 | 61.1 | 48.5 |
| decoder readback + box update | 47.7 | 6.4 | 6.3 | 6.1 | 6.2 |
| scoring | 25.2 | 2.9 | 2.8 | 2.5 | 2.6 |
| host decoder total | 337.3 | 80.9 | 61.9 | 69.7 | 57.3 |

Against the pre-stated readings:
- **A: prep is 71.6 ms on 82, at or under the 130 threshold.** The scalar MLP arithmetic was the bulk.
- **B: prep is 52.8 ms on 82 and 48.5 on 22, above the 40 ms "no longer the gap" line.** B takes
  19 ms more out of prep on 82 (13 on 22). What remains is the host MLPs themselves:
  `ref_point_head` and the two box-RPB MLPs, now threaded, and the sine embedding.

## Frame time, unprofiled, steady mean (frames 1-15), ms

Runs were interleaved per video in this order: control, A, B, control, A, B (after the two
profiled runs). n = 2 per arm.

| | control | A | B | A / control (min of 2) | B / control (min of 2) |
|---|---|---|---|---|---|
| 22 (load ~5) | 2204, 2160 | 1899, 1858 | 1866, 1848 | 0.860 | 0.856 |
| 82 (load 13.6 -> 4.6) | 2833, 2533 | 2485, 2239 | 2332, 2349 | 0.884 | 0.921 |

Reading [V for the numbers, I for the attribution]:
- **A cuts the frame by 12-14% with identical output.** On 22 the drop holds across both pairs
  and is far larger than the control-to-control spread (44 ms). On 82, load fell during the runs
  (control-1 to control-2 is 300 ms), so read 82 as a size, not as a ratio.
- **B's further gain does not show at frame level.** The stage gain is 13-19 ms, inside the noise
  of n = 2. The two B-over-A pairs on 82 have opposite signs. B stays in because it is exact,
  uploads 0.9 MB per layer instead of 33 MB, and is smaller in the stage table.
- **The trunk is now 46% of the steady frame on 22** (893 of 1924 ms in B's stages run) and 38% on
  82. That crosses E11's 40% line on 22, so the trunk is the largest single stage again.
- **Other large stages on 82 (B):**
  - tracker state 0: 344 ms
  - tracker state 1: 288 ms
  - memory encode: 211 ms device and 45 ms host
  - propagation neck: 140 ms
  - detector segmentation head: 124 ms
  - detector neck: 120 ms

Files: `v*-{control,a,b}-*` run directories and logs, `v*-{a,b}-stages/*.profile-stages.md`,
`unit-tests-b.log`, `rpb-metal-b.log`, `job.sh` and `versions.txt`. Masks are not committed.
