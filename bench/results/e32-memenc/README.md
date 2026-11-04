# E32: the tracker's memory encoder as one fp16 Core ML program (2026-11-04)

The rules are in `RULES.md`, written before any run. The arm `me16` is today's default (fp16 trunk,
pipelined, with fp16 memory attention and fp16 necks) plus `--mem-encoder coreml-fp16`
(`sam3.1_memenc_fp16.mlpackage`, `tools/baselines/coreml_memenc.py`). Both arms ran on one binary
from the build copy (`SpikeTrack-pipe`), in the shipped configuration: LoRA v3 +
`lorav3_exact_it_3p00.json`, prompt sperm, `--boxes detector`, cap 64.

| gate | result | verdict |
|---|---|---|
| M-speed, run 1 (`job-run1.log`, `run1/`) | wall me16/base **0.865, 0.841, 0.829** (median **0.841**); push 0.815, 0.781, 0.788. Every arm started below load 6; start and end load 3.9-4.6, every arm read. base wall 1,183-1,248 ms (5%) | **pass** |
| M-G2, 32 frames x 5 videos (`../track/held32-coreml/t16m16n16me16/`, `g2.log`) | HOTA **0.6025** against t16m16n16's 0.6025; every video equal to 4 places (0.6748, 0.6801, 0.5756, 0.5599, 0.5544) | **pass** |
| M-G3, 1,500 frames (`../track/held300-coreml/t16m16n16me16/`, `g3.log`) | HOTA **0.5695** against t16m16n16's 0.5697; per video +0.0000, +0.0000, -0.0005, -0.0002, +0.0002 | **pass** |

**E32 passes every gate, and `--mem-encoder` now defaults to `coreml-fp16`** under the guard of
`Placement::default_mem_attn` (Metal, the native 1008 input, the default fp16 trunk), as RULES.md
decided before the result. `--mem-encoder ojas` gives the E30 default back, and
`--trunk-precision coreml-fp32` still restores the fp32 reference (the memory encoder follows memory
attention back to Ojas). The demo video was not re-rendered: it shows the default as of E30.

The new path ran in every gate run, not only the speed arms. Each `run.json` records
`mem_encoder: coreml-fp16`, the package, and `mem_encoder_calls`:

| run | external (Core ML) encodes | Ojas encodes (merges, masks at image size) |
|---|---|---|
| G2, videos 21 / 22 / 36 / 47 / 82 | 64 / 32 / 104 / 32 / 64 | 5 / 2 / 14 / 3 / 2 |
| G3, videos 21 / 22 / 36 / 47 / 82 | 600 / 300 / 1,220 / 471 / 674 | 10 / 3 / 28 / 10 / 9 |

Tracks against base in the speed rounds: 32/32 frames with equal id sets in every round, mean box IoU
1.0000, min 0.9919. They are not byte-identical, as expected for an fp16 encoder; the accuracy gates
judge them.

On video 36 the default frame is now about 1.0 s (wall 981-1,080 ms), from about 1.2 s.

## What was measured before the gates

- Parity of the package against upstream's 8 fixture calls, with the masks resized to 1152
  (`../coreml-memenc/fp16-CPU_AND_GPU.json`): worst cosine 0.9999985, rel L2 at most 1.7e-3. Predict
  7.2 ms median (n 10), standalone. On Ojas each encode was about 58 ms plus a 17 ms mask upload.
- The program takes the 16 slot mask planes and a 16-value condition vector, and broadcasts the 16
  condition planes itself, so the host fills 16 planes of 1152 x 1152, not 32. The hook writes the
  planes straight into Core ML's input buffer.
- Smoke run (8 frames, video 36, load about 4; not a gate): steady frames 0.95-1.26 s for base,
  0.77-1.12 s for me16; 39 tracks born in both.

## Port into the main tree

One patch from the build copy, checked to apply before the gates read, plus `track_visem.sh`'s
`SPIKETRACK_MEM_ENCODER` line (its header now also states every default correctly). The default is
`Placement::default_mem_encoder`, which returns `default_mem_attn`'s choice; the test
`the_memory_encoder_defaults_as_memory_attention_does` checks the two agree over every trunk, backend
and input size. Main-tree checks: `main-checks.sh`, output in `main-checks/` and `main-checks.log`:
- fmt clean; clippy `-D warnings` clean (it checked st-model, st-coreml and spiketrack).
- spiketrack + st-model + st-coreml tests: 158 passed, 0 failed.
- 2-frame runs, shipped configuration:
  - the default records `mem_encoder: coreml-fp16` with the package sha256, and 6 Core ML and 4 Ojas
    encodes;
  - `--mem-encoder ojas` records 0 Core ML and 10 Ojas encodes;
  - `--trunk-precision coreml-fp32` puts the memory encoder back on Ojas along with memory attention;
  - `--input-size 672` with `--mem-encoder coreml-fp16` is refused.
- A run with `--mem-encoder coreml-fp32`, which has no export, is refused with the export command
  (`main-checks/fp32-missing-package.log`, run by hand after the script).

Every memory encode that `track` makes goes through `encode_mask_memory`, so `mem_encoder_calls`
counts all of them. The one other caller of the encoder, `Vos::run`'s driver in `vos.rs`, serves
only the fixture tests (`vos_stages`, `frame_profile`, `vos_gate`). It stays on Ojas.

## Known limits

- A LoRA or weight edit under `tracker.model.maskmem_backbone.` is refused by
  `check_placement_weights` on the default path, because the package bakes those weights in. No
  adapter in this repository touches them (LoRA v3 is a detector adapter). Memory attention's default
  already has the same property for `tracker.model.transformer.encoder.`; pass `--mem-encoder ojas`.
- A missing package is an error, not a fallback, as for every Core ML placement:
  `tools/baselines/coreml_memenc.py export --precision fp16` writes it.
- The pix input is read back from the propagation features once per encode. `FrameFeats::host()`
  already holds those features token-major when memory attention runs on Core ML, so one more
  readback could be saved. Not measured.
