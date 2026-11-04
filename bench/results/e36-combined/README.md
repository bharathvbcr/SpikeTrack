# E36: the detector transformer, segmentation head and interactive neck on Core ML together (2026-11-04)

The rules are in `RULES.md`, saved at 12:37:45 CT, before the job took the lock at 12:38:01. The arm
`all16` is the E32 default plus:
- `--det-transformer coreml-fp16` (E35);
- `--seg-head coreml-fp16` (E25-E26, E33);
- `--inter-neck coreml-fp16` (E33).

Both arms ran on one binary from `SpikeTrack-e34`. That copy also carries E34's exact, unadopted
changes, in both arms.

| gate | result | verdict |
|---|---|---|
| A-speed, run 1 (`job-run1.log`, `run1/`) | wall all16/base **0.843, 0.862, 0.860** (median **0.860**); push 0.867, 0.881, 0.818. Every arm started below load 6; start and end load 3.8-4.5, every arm read | **pass** |
| A-G2, 32 frames x 5 videos (`../track/held32-coreml/t16m16n16me16all16/`, `g2.log`) | HOTA **0.6023** against 0.6025; per video -0.0003, +0.0015, -0.0001, -0.0004, -0.0011 | **pass** |
| A-G3, 1,500 frames (`../track/held300-coreml/t16m16n16me16all16/`, `g3.log`) | HOTA **0.56703** against 0.56952, which passes the 0.567 bar by 0.00003. Per video: 21 -0.0014, 22 +0.0001, 36 -0.0007, **47 -0.0233**, 82 -0.0005 | **fail**: video 47 is 0.0233 below t16m16n16me16's 0.4838, beyond the 0.01 allowed |

**E36 is not adopted.** Nothing moves into the main tree, which stays at the E32 default. The three
flags stay in `SpikeTrack-e34`. A port was staged in the scratchpad while G3 ran and was not applied.

All three programs ran in every accuracy run. Each video's `run.json` in G2 records
`det_transformer`, `seg_head` and `inter_neck` as `coreml-fp16`, with the package named.

## Video 47

Video 47 is the sparsest held-out video: 2,045 ground-truth boxes in 300 frames. all16 predicts
2,891 boxes there, against 2,754. The extra boxes lower detection (DetA 0.4153 -> 0.4012) and
association (AssA 0.5652 -> 0.5302). The other four videos change by at most 37 boxes.

Which of the three programs causes the drop is not measured. One candidate: the transformer's fp16
error. Over all 200 queries the boxes are at 2.9e-2 relative L2 (`../e35-dettr/README.md`), and that
error sits on the rejected queries. A query near the score threshold that moves can add a
detection. That is an inference from the parity numbers, not a measurement on video 47.

## What this does establish

- The three programs together take the frame to 0.84-0.86x, in three tight rounds. So the savings
  add, as E35's README inferred. Each one alone was inside the gate's noise.
- 32 frames are not enough for this change: G2 passed on every video, and the regression showed only
  over 300 frames on the sparsest video.

## Neural Engine probes (the same day, not a gate)

These ran on the export venv's `check --units CPU_AND_NE` (`../coreml-mattn/`, `../coreml-memenc/`,
`../coreml-necks/`):
- Memory attention: mac_heavy stopped the run at 31 GB of footprint while it compiled for the Neural
  Engine. No result.
- Memory encoder: 13.7 ms, against 7.2 ms on the GPU, worst cosine 0.99952.
- Detector neck: 12.0 ms, against 17.7 ms on the GPU, but relative L2 1.5e-2, against 3.6e-4.

None is used.
