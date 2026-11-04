# E33: the interactive neck on the trunk's worker plus the Core ML segmentation head (2026-11-04)

The rules are in `RULES.md`, written before any run. The arm `is16` is the E32 default plus
`--inter-neck coreml-fp16 --seg-head coreml-fp16`. Both arms ran on one binary from the build copy
(`SpikeTrack-pipe`).

| gate | result | verdict |
|---|---|---|
| IS-speed, run 1 (`job-run1.log`, `run1/`) | wall is16/base **0.968, 0.945, 0.946** (median **0.946**); push 1.006, 0.962, 0.940. Every arm started below load 6; start and end load 3.6-4.8, every arm read | **fail**: round 1 is above 0.95, and the median is above 0.92 |
| IS-G2 | stopped after the speed gate failed, during video 21; its partial output (4 files, no evaluation) was deleted | **not read** |
| IS-G3 | not run | **not read** |

**E33 is not adopted.** As RULES.md decided before the result, `--inter-neck` stays in the build
copy only, and `--seg-head` keeps its Ojas default. The accuracy gates were not needed for the
verdict, so the job was stopped to free the machine.

Tracks against base in the speed rounds: 32/32 frames with equal id sets in every round, mean box IoU
1.0000, min 0.9919.

## Why the smoke run overstated it

The 8-frame smoke (`../e33-smoke/`) put the pair at 782 ms against 900-905 ms, which is 0.87x. Over
32 frames, base's steady wall was 994-1,010 ms and is16's 950-963 ms, only 40-55 ms apart. Two
inferences, neither measured on its own:
- The interactive neck matters only on frames with births or reconditions, and the first 8 frames
  of video 36 have more of those than later frames.
- The worker computes that neck on every frame, and the GPU it shares with the main thread then
  takes that time from the frame.

The segmentation head's program saves about 110 ms of Ojas work. The host copies at its boundary
(about 800 MB of memcpy a frame, counted from the code) take part of that back.

What would change the verdict: run the interactive neck on demand on a thread of its own, not
speculatively, and feed the segmentation head the worker's host levels without uploading them and
reading them back. Neither is built.

Later the same day, E36 (`../e36-combined/`) measured these two placements together with E35's
detector transformer, under its own rules. See it for where the two ended up.
