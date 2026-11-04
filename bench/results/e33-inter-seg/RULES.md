# E33 rules, written before any gate run (2026-11-04, 12:03 CT)

Two placements on top of the E32 default, judged together:
- `--inter-neck coreml-fp16`: the interactive neck (`interactive_convs`, which births and reconditions
  use) as a third neck program on the trunk's worker (`sam3.1_neck_inter_fp16.mlpackage`,
  `tools/baselines/coreml_necks.py export --neck inter`). The worker computes it every frame, so a
  frame that needs it finds it ready.
- `--seg-head coreml-fp16`: the existing Core ML segmentation head (E25-E26), unchanged. E26 rejected
  it (1.00-1.06x) because the frame paid host copies at its boundary. Its README said it was worth
  measuring again once the neck levels come from the worker, which they have since E30.

## What was measured before these rules

- Parity of the interactive-neck package against the 5 backbone fixtures' `neck.interactive_convs`
  outputs (`../coreml-necks/inter-fp16-CPU_AND_GPU.json`): worst rel L2 3.7e-4; predict 17.2 ms
  median (n 10). On Ojas the stage is about 46 ms a frame on video 36 (`../profile-e32/stages/`), and
  its 288 x 288 3x3 convolution alone is 65-90 ms a call (`../profile-e32/ops/`).
- Profile of the E32 default (`../profile-e32/`): steady frames 918-952 ms (controls), the
  segmentation head 136 ms (14%), the interactive neck 46 ms (5%).
- Smoke run (`../e33-smoke/`, 8 frames, video 36, load 4.6-4.9; not a gate). Mean steady frame:

  | arm | mean steady frame |
  |---|---|
  | base | 905 ms (900 ms rerun) |
  | inter-neck | 845 ms |
  | seg-head | 812 ms |
  | both | 782 ms |

  All 4 arms bore 39 tracks.

## Arms

- `base`: the E32 default (fp16 trunk pipelined + fp16 memory attention + fp16 necks + fp16 memory
  encoder), no flags.
- `is16`: `base` + `--inter-neck coreml-fp16 --seg-head coreml-fp16`.

One binary from the build copy (`../../../../SpikeTrack-pipe`), which differs from the main tree by
the `--inter-neck` flag and the exporter's `inter` neck. Shipped configuration as in E30-E32: LoRA v3 +
`lorav3_exact_it_3p00.json`, prompt sperm, `--boxes detector`, cap 64.

## Gates, all required for adoption

These are the gates E32 used, now against t16m16n16me16 (the E32 default):
- **IS-G2:** held-out HOTA over the first 32 frames of the 5 held-out videos >= 0.6005, with no
  video more than 0.01 below t16m16n16me16's (0.6748, 0.6801, 0.5756, 0.5599, 0.5544).
- **IS-G3:** held-out HOTA over 1,500 frames >= 0.567, with no video more than 0.01 below
  t16m16n16me16's (0.6695, 0.6691, 0.5319, 0.4838, 0.5291).
- **IS-speed:** video 36, first 32 frames, 3 rounds with the order rotated, steady frames >= 8.
  `is16`'s wall below 0.95x `base`'s in all 3 rounds, median of the 3 ratios below 0.92. Each arm
  starts only below load 6 (`E31_START_MAX=6`), and an arm whose start or end load is 8 or more is
  "not read".

Order: IS-speed first while the machine is quiet, then IS-G2, then IS-G3.

## What adoption changes (decided before the result)

If `is16` passes all three, then `--inter-neck` and `--seg-head` default to `coreml-fp16` under the
guard of `Placement::default_necks`: Metal, the native input, the default fp16 trunk, pipelined.
That is the configuration measured, and the interactive neck is only on the worker when the trunk is
pipelined. The change gets its own default checks, as E30 and E32 did. If `is16` fails any gate, the
flags stay where they are (`--inter-neck` in the build copy only), and this folder's README says
which gate failed and by how much. The two levers are not split after the result. Splitting them
would be a new experiment with its own rules.

Cutoff: 17:00 CT. Whatever has not been read by then is reported as not read.
