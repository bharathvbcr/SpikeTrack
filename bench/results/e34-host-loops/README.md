# E34: two main-thread host loops, rewritten without changing a value (2026-11-04)

The rules are in `RULES.md`, written before any run. `e34` is the `SpikeTrack-e34` binary with two
changes:
- `exclusive_masks` in place of the per-pixel non-overlap loop in `Session::release`, with
  `Bitmask::from_logits` built a word at a time;
- the model input made once per frame under the pipelined trunk.

`base` is the `SpikeTrack-pipe` binary (the E32 default).

| gate | result | verdict |
|---|---|---|
| H-same (`job-run1.log`) | byte-identical predictions on video 36 in all 3 speed rounds, and on the first 32 frames of videos 21, 22, 47 and 82 | **pass** |
| H-speed, run 1 (`run1/`) | wall e34/base **0.993, 0.996, 1.024** (median 0.996); push 1.017, 0.963, 1.039. Every arm started below load 6, and every arm was read (load 5.3-6.2). base wall 1,021-1,038 ms | **fail**: round 3 is above 1.02 |

**E34 is not adopted.** The two changes are exact, but no gain shows in the frame: the median is
0.4% below base, inside the 2% spread between rounds. The profile booked 41.6 ms a frame to
"outputs + release". By the code, the overlap loop was about 12 M bit tests a frame and looked like
most of that. The measurement says it was not. That stage's time goes somewhere this change does not
touch. Nothing moves into the main tree.

`exclusive_masks_equal_the_dense_loop_they_replaced` stays in the `SpikeTrack-e34` copy with the
code. Its first run caught a real difference: a NaN score on the first object blocks the pixels it
covers in the old loop. The new code was then made to match.
