# E34 rules, written before any gate run (2026-11-04, 12:09 CT)

Two host-side changes on the main thread. Neither changes a computed value.
- **The output masks' non-overlap** (`Session::release`). Each pixel goes to the covering object
  with the highest positive tracker score, first index on ties. The old loop asked every object
  about every pixel: about 40 objects x 307,200 pixels a frame on video 36. `exclusive_masks` visits
  only set pixels. `Bitmask::from_logits` builds a word at a time instead of a bit at a time.
- **The frame's model input is made once** (`track`, pipelined trunk). The resize and
  normalisation made to prefetch frame t + 1 are kept for t + 1's own turn. Before this change they
  were made again.

## What was measured before these rules

- `exclusive_masks_equal_the_dense_loop_they_replaced`: `exclusive_masks` equals the old loop, kept
  in the test as its oracle. The test covers 100 random sets of masks at 5 sizes, with tied, zero,
  negative, NaN and infinite scores. Its first run failed: in the old loop, a NaN score on the first
  object blocks the pixels it covers, and the new code did not do that. The new code now does, and
  the test passes. It also checks `from_logits` against `from_fn(logits > 0)`, with NaN and -0.0.
- Profile of the E32 default (`../profile-e32/`): "outputs + release" is 41.6 ms a frame, of which
  3.5 ms is device ops; "decode + upload" is 14.2 ms a frame.

## Arms

- `base`: the build copy `SpikeTrack-pipe` binary (the E32 default; its only other difference from
  the main tree is the unused `--inter-neck` flag, whose default is ojas).
- `e34`: the `SpikeTrack-e34` binary, which is `SpikeTrack-pipe` plus the two changes.

Both run in the shipped configuration (LoRA v3 + `lorav3_exact_it_3p00.json`, prompt sperm,
`--boxes detector`, cap 64) with no placement flags.

## Gates, all required for adoption

- **H-same:** predictions byte-identical to `base`'s on the first 32 frames of all 5 held-out
  videos (`36.json` and the other ids' JSON, compared with `cmp`), and in every speed round.
- **H-speed:** video 36, first 32 frames, 3 rounds with the order rotated, steady frames >= 8,
  every arm started below load 6, and an arm with start or end load at or above 8 is not read.
  `e34`'s wall must not be slower than `base`'s: the median of the 3 ratios is below 1.00, and no
  round is above 1.02. A byte-identical change needs no 0.95 bar. Its gain is reported as measured,
  with the base spread beside it.

## What adoption changes

If both gates pass, the two changes move into the main tree as they are. There is no flag: the old
loop stays only as the test's oracle. If either gate fails, nothing moves, and this folder's README
says which gate failed and how.

Cutoff: 16:00 CT.
