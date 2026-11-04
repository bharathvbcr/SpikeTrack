# E35 rules, written before any gate run (2026-11-04, 12:18 CT)

`--det-transformer coreml-fp16` runs the detector's geometry encoder, 6-layer fusion encoder and
6-layer DETR decoder (box refinement, boxRPB, presence token) as one fp16 Core ML program
(`sam3.1_dettr_fp16.mlpackage`, `tools/baselines/coreml_dettr.py`). That replaces about 270
dispatches and the decoder's host prep per frame with one prediction.

The program differs from E31's encoder package in one way: the weights an adapter can change are
inputs, not constants. They are every attention projection of the fusion encoder and the decoder,
plus `bbox_embed` and `presence_token_head`, the modules LoRA v3 adapts. `track` feeds them from the
weights it loaded, with `--lora` folded and `--weight-edit` applied. So one package serves the base
model and every adapter or edit that changes only those weights. An adapter or edit of any other
weight under the three prefixes is refused (`check_transformer_weights`).

## What was measured before these rules

- The fp32 torch mirror against upstream's fixture (`fixtures/detector/visem22_cell`), worst
  relative L2 by output: mem 2.4e-7, geo 6.4e-8, every layer's queries 1.3e-5, presence 1.2e-7,
  final boxes 7.6e-6.
- The package's parity and time (base weights against the fixture, and LoRA v3's folded weights
  against the mirror with the same fold): `../coreml-dettr/`, filled in by the check before any
  gate runs.

## Arms

- `base`: the E32 default, no flags.
- `dt16`: `base` + `--det-transformer coreml-fp16`.

One binary from `SpikeTrack-e34`. Shipped configuration as in E30-E34: LoRA v3 +
`lorav3_exact_it_3p00.json`, prompt sperm, `--boxes detector`, cap 64.

## Gates, all required for adoption

These are E32's gates, against t16m16n16me16 (the E32 default):
- **T-G2:** held-out HOTA over the first 32 frames of the 5 held-out videos >= 0.6005, with no
  video more than 0.01 below t16m16n16me16's (0.6748, 0.6801, 0.5756, 0.5599, 0.5544).
- **T-G3:** held-out HOTA over 1,500 frames >= 0.567, with no video more than 0.01 below
  t16m16n16me16's (0.6695, 0.6691, 0.5319, 0.4838, 0.5291).
- **T-speed:** video 36, first 32 frames, 3 rounds with the order rotated, steady frames >= 8.
  `dt16`'s wall below 0.95x `base`'s in all 3 rounds, median of the 3 ratios below 0.92. Each arm
  starts only below load 6; an arm whose start or end load is 8 or more is "not read".

Order: T-speed first, then T-G2, then T-G3. If T-speed fails, the accuracy gates are not run.

## What adoption changes (decided before the result)

If `dt16` passes all three, `--det-transformer` defaults to `coreml-fp16` under the guard of
`Placement::default_mem_attn` (Metal, the native input, the default fp16 trunk). Two further
conditions apply, and run.json records which one held:
- every `--lora` and `--weight-edit` target under the three prefixes is one of the program's inputs.
  Otherwise the default is ojas, and the run is not refused.
- `--det-encoder` is not given. It and the transformer replace the same encoders.

Passing `--det-transformer coreml-fp16` explicitly still refuses an adapter that touches a baked
weight. The change gets its own default checks. If `dt16` fails any gate, the flag stays in
`SpikeTrack-e34` only, and this folder's README says which gate failed and by how much.

Cutoff: 17:30 CT.
