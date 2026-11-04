# E31 rules, written before any run (2026-11-04)

**The package is adapter-specific.** `--det-encoder coreml-fp16` runs the detector's geometry encoder
and 6-layer fusion encoder as one Core ML program with the `--lora` adapter folded in at export
(`sam3.1_detenc_fp16_<sha256[:12] of the adapter>.mlpackage`, `tools/baselines/coreml_detenc.py`).
A different adapter needs its own export; a weight edit inside the encoders is refused.

## What was measured before these rules

- Parity of the exported package against the torch mirror with the same LoRA v3 fold
  (`../coreml-detenc/fp16-CPU_AND_GPU-db6a8308ccdb.json`): `mem` cosine 0.9999998, rel L2 6.7e-4;
  `geo` cosine 0.9999995. Predict 13.9 ms median (n 10), standalone.
- The same encoders on Ojas cost about 114 ms of the frame (the E28 profile). That is the lever:
  roughly 100 ms of a ~1,340 ms frame (t16m16) if the host copies stay small. They might not: the
  hook reads `text` and `fpn[2]` back to the host and uploads `mem` (5,184 x 256 f32, 5.3 MB) every
  detector call, on the main thread (inline, as the segmentation head's hook works).

## Arms

- `base`: today's default (fp16 trunk pipelined + fp16 memory attention + fp16 necks), no flags.
- `e16`: `base` + `--det-encoder coreml-fp16`.

Both arms are run by one binary, built from the build copy (`../../../../SpikeTrack-pipe`), which
differs from the main tree only by the `ExternalEncoder` hook, `CoremlDetEncoder` and the flag.
Shipped configuration as in E30: LoRA v3 + `lorav3_exact_it_3p00.json`, prompt sperm, `--boxes
detector`, cap 64.

## Gates, all required for adoption

- **E-G2:** held-out HOTA over the first 32 frames of the 5 held-out videos >= 0.6005, with no video
  more than 0.01 below t16m16n16's (`../track/held32-coreml/t16m16n16/`, 0.6025).
- **E-G3:** held-out HOTA over 1,500 frames >= 0.567, with no video more than 0.01 below t16m16n16's
  per-video values (`../track/held300-coreml/t16m16n16/`, 0.5697).
- **E-speed:** video 36, first 32 frames, 3 rounds with the order rotated, steady frames >= 8
  (`tools/analysis/e25_frame.py`). `e16`'s wall below 0.95x `base`'s in all 3 rounds, median of the
  3 ratios below 0.92. The 1-minute load is logged at each arm's start and end; an arm with either
  above 8 is "not read", not a pass or a fail. The noise floor is `base`'s spread across rounds.

The order: E-speed first (cheapest, and it decides whether the accuracy runs are worth their
time), then E-G2, then E-G3.

## What adoption changes (decided before the result)

If `e16` passes all three, the change moves from the build copy into the main tree, with tests, as
an **opt-in flag**, not a new default: the default cannot depend on which adapter file the user
passes, and a missing package for a new adapter would turn the default into an error. The README
and the speed table then say what the flag buys with LoRA v3 and what export it needs. The demo
video is not re-rendered (it shows the default).

If it fails any gate, the flag stays in the build copy only, and the README of this folder says
which gate failed and by how much.

Hard stop: 13:30 CT. Whatever has not been read by then is reported as not read.
