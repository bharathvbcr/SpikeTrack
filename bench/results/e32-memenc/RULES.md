# E32 rules, written before any gate run (2026-11-04)

`--mem-encoder coreml-fp16` runs the tracker's multiplex memory encoder as one fp16 Core ML program
(`sam3.1_memenc_fp16.mlpackage`, `tools/baselines/coreml_memenc.py`). It takes the 16 slot mask
planes at 1152 and a 16-value condition vector, and builds the 16 condition planes itself, so the
host fills 16 planes, not 32. It replaces the encodes whose masks arrive at the downsampler's input
size (every propagated frame). A merge's encode, from masks at the image size, stays on Ojas, and
`run.json` counts both (`mem_encoder_calls`). No adapter or weight edit touches these weights.

## What was measured before these rules

- Parity of the package against upstream's 8 fixture calls, with the masks resized to 1152
  (`../coreml-memenc/fp16-CPU_AND_GPU.json`, copied from the build copy): worst cosine 0.9999985,
  rel L2 at most 1.7e-3. The features are bf16-rounded afterwards anyway. Predict 7.2 ms median
  (n 10), standalone.
- On Ojas, the stage profile (`../profile-fp16/stages/36.profile-stages.md`) puts each encode at
  about 58 ms plus a 17 ms mask upload, 3 encodes a frame on video 36.
- Smoke run (8 frames, video 36, load about 4; not a gate): steady frames 0.95-1.26 s for base,
  0.77-1.12 s for me16; 39 tracks born in both; 24 external and 7 Ojas encodes.

## Arms

- `base`: today's default (fp16 trunk pipelined + fp16 memory attention + fp16 necks), no flags.
- `me16`: `base` + `--mem-encoder coreml-fp16`.

One binary from the build copy (`../../../../SpikeTrack-pipe`), which differs from the main tree by
the `ExternalMemEncoder` hook, `CoremlMemEncoder` and the flag. Shipped configuration as in E30/E31:
LoRA v3 + `lorav3_exact_it_3p00.json`, prompt sperm, `--boxes detector`, cap 64.

## Gates, all required for adoption

The same three as E31, against t16m16n16 (today's default):
- **M-G2:** held-out HOTA over the first 32 frames of the 5 held-out videos >= 0.6005, with no
  video more than 0.01 below t16m16n16's (0.6025).
- **M-G3:** held-out HOTA over 1,500 frames >= 0.567, with no video more than 0.01 below
  t16m16n16's per-video values (0.5697).
- **M-speed:** video 36, first 32 frames, 3 rounds with the order rotated, steady frames >= 8.
  `me16`'s wall below 0.95x `base`'s in all 3 rounds, median of the 3 ratios below 0.92. Each arm
  starts only below load 6 (`E31_START_MAX=6` in the job), and an arm whose start or end load is 8
  or more is "not read".

Order: M-speed first while the machine is quiet, then M-G2, then M-G3.

## What adoption changes (decided before the result)

This package does not depend on the adapter, so unlike E31 it could become a default. If `me16`
passes all three, `--mem-encoder` defaults to `coreml-fp16` under the same guard as
`Placement::default_mem_attn` (Metal, the native input size, the default fp16 trunk). That change
needs its own default checks, as E30's did. The demo video is not re-rendered (it shows the
default as of E30); the README says which default the video shows. If it fails any gate, the flag
stays in the build copy only, and this folder's README says which gate failed and by how much.

Cutoff: 15:30 CT. Whatever has not been read by then is reported as not read.
