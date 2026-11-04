# E36 rules, written before any gate run (2026-11-04, 12:37 CT)

E33 and E35 each moved Ojas GPU work into Core ML programs. Each lowered the frame by about 5%, and
each failed a speed gate built to see 8%. E35's README infers that the frame is bound by total GPU
time. Under that reading, each program saves only the GPU time it removes, and the savings of
several programs add. E36 tests that, as a new experiment: all three placements together, against the
same gates. It is not a re-scoring of E33 or E35: their verdicts stand.

The three placements, with each source:
- `--det-transformer coreml-fp16` (E35): geometry + fusion encoders + DETR decoder as one program,
  adaptable weights as inputs;
- `--seg-head coreml-fp16` (E25-E26, E33): the segmentation head;
- `--inter-neck coreml-fp16` (E33): the interactive neck on the trunk's worker.

## What was measured before these rules

- E33, same gates (inter-neck + seg-head): wall 0.968, 0.945, 0.946.
- E35, same gates (det-transformer): wall 0.935, 0.907, 1.082. Round 3 stalled on frames 1-2.
- Program parity: `../coreml-dettr/`, `../coreml-necks/inter-fp16-CPU_AND_GPU.json` and
  `../coreml-seghead/`.
- Neural Engine probes the same day: memory attention, the memory encoder and the detector neck.
  None is usable as a speed path (`../coreml-mattn/`, `../coreml-memenc/`, `../coreml-necks/`).

## Arms

- `base`: the E32 default, no flags.
- `all16`: `base` + `--det-transformer coreml-fp16 --seg-head coreml-fp16 --inter-neck coreml-fp16`.

One binary from `SpikeTrack-e34`. That copy also carries E34's exact changes, in both arms. Shipped
configuration as in E30-E35.

## Gates, all required for adoption

These are E32's gates, against t16m16n16me16:
- **A-G2:** held-out HOTA, first 32 frames of the 5 held-out videos, >= 0.6005. No video may be more
  than 0.01 below 0.6748, 0.6801, 0.5756, 0.5599 and 0.5544, in that order.
- **A-G3:** held-out HOTA over 1,500 frames >= 0.567. No video may be more than 0.01 below 0.6695,
  0.6691, 0.5319, 0.4838 and 0.5291, in that order.
- **A-speed:** video 36, first 32 frames, 3 rounds, order rotated, steady frames >= 8. `all16`'s wall
  must be below 0.95x `base`'s in all 3 rounds, and the median of the 3 ratios below 0.92. Each arm
  starts only below load 6, and an arm whose start or end load is 8 or more is "not read".

Order: A-speed, then A-G2, then A-G3. If A-speed fails, the accuracy gates are not run.

## What adoption changes (decided before the result)

If `all16` passes all three gates, the three placements default to `coreml-fp16` together under the
guard of `Placement::default_necks` (Metal, the native input, the default fp16 trunk, pipelined). The
transformer has two more conditions, both recorded in run.json:
- every `--lora` and `--weight-edit` target under its prefixes is one of its inputs;
- `--det-encoder` is not given.

When the transformer's conditions do not hold, all three default to ojas. The measured configuration is
the three together, and E33 measured the other two alone, where they failed.
The change gets its own default checks. If `all16` fails, nothing moves, and this folder's README
says which gate failed and by how much.

Cutoff: 17:30 CT.
