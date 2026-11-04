# E35: the detector's encoders and DETR decoder as one fp16 Core ML program with adaptable weights as inputs (2026-11-04)

The rules are in `RULES.md`, written before any run. The arm `dt16` is the E32 default plus
`--det-transformer coreml-fp16` (`sam3.1_dettr_fp16.mlpackage`, `tools/baselines/coreml_dettr.py`).
Both arms ran on one binary from `SpikeTrack-e34`, which also carries E34's exact changes in both arms.

## Before the gates

- The fp32 torch mirror against upstream's fixture: worst relative L2 1.3e-5 (the decoder queries).
- The fp16 package (`../coreml-dettr/`): predict 19.5 ms median (n 10).
  - Against the fixture, with the base weights fed as inputs: memory rel L2 6.8e-4. Across all 200
    queries, the queries are at 5.8e-2 and the boxes at 2.9e-2, with a worst box coordinate 0.26 off.
  - The error sits on rejected queries. On the 47 the detector keeps, box coordinates are within
    4e-4, and last-layer query rel L2 is at most 2.0e-2 (median 6.8e-3). The worst query overall
    has logit -3.3.
  - With LoRA v3's folded weights fed as inputs, against the mirror with the same fold: boxes 2.2e-2
    and queries 4.2e-2 over all 200, the same pattern.
- In the frame, steady dispatches fall from about 2,850 to about 1,170 a frame.

## Gates

| gate | result | verdict |
|---|---|---|
| T-speed, run 1 (`job-run1.log`, `run1/`) | wall dt16/base **0.935, 0.907, 1.082** (median 0.935); push 1.012, 0.885, 0.943. Every arm started below load 6; start and end load 4.8-7.7, every arm read | **fail**: round 3 is above 0.95, and the median is above 0.92 |
| T-G2, T-G3 | not run, as RULES.md says after a failed T-speed | **not read** |

**E35 is not adopted.** The flag stays in `SpikeTrack-e34` only.

Round 3's dt16 arm stalled early: frames 1 and 2 took 2.18 and 1.65 s, against 0.92 and 0.67 s in
round 2, and its steady push median (825 ms) is in line with the other rounds. The rule judges wall
time with every frame counted, and every arm was read, so the verdict stands. The run was not
repeated to look for a pass. A later experiment that wants this lever needs its own rules, written
before its runs. They should fix the stall-robust measure (the steady push median, say) in advance,
not choose it after this result.

Tracks against base: 31/32 frames with equal id sets in every round, mean box IoU 0.9872, min 0.364.
So one track differs on one frame. The accuracy gates would have judged that.

E36 (`../e36-combined/`) then measured this program together with E33's two placements, under its
own rules written before its runs.

## What this says about the frame

The program is about 10x faster than the Ojas stages it replaces (19.5 ms against about 113 ms for the
encoders plus about 105 ms for the decoder and its host prep). The frame gained about 6% at the
median. Inferred, not measured: the frame is now bound by total GPU time, not by the main thread.
By the standalone timings, the fp16 trunk (260 ms) on the worker, memory attention (3 x 62 ms) and
the Ojas stages add up to roughly the whole frame, and they share one GPU. Moving work between Ojas
and Core ML on the same GPU saves only the GPU time it removes. The dispatch and wait time it also
removes buys little.
