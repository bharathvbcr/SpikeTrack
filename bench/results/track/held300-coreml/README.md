# G3: held-out HOTA of the fp16 CoreML placements, 1,500 frames (2026-11-04)

This is gate G3 of `../../e25-coreml-frame/RULES.md`, the README claim for fp16 defaults. The gate,
stated before the runs: HOTA >= 0.567, and no video more than 0.01 below the baseline.
- Videos: the five held-out VISEM videos (21, 22, 36, 47, 82), first 300 frames each, which is 1,500
  frames and 32,949 labelled heads.
- Configuration: the shipped one (LoRA v3 + `lorav3_exact_it_3p00.json`, prompt sperm,
  `--boxes detector`, cap 64).
- Jobs: `job.sh` (`job.log` for default and fast16; `job-t16m16.log`).
- Table: `summary.md`, from `tools/analysis/g2_placements.py ... 300 default fast16 t16m16`.

| config | flags | HOTA | per video (21 / 22 / 36 / 47 / 82) | worst video vs default | verdict |
|---|---|---|---|---|---|
| default (fp32 reference) | `--trunk-precision coreml-fp32` | 0.5700 | 0.6694 / 0.6690 / 0.5330 / 0.4840 / 0.5291 | | reproduces the documented 0.570 on every video |
| fast16 | `--trunk-precision coreml-fp16 --mem-attn coreml-fp16 --seg-head coreml-fp16` | 0.5705 | +0.0001 / +0.0001 / +0.0013 / -0.0001 / +0.0000 | 47: -0.0001 | **PASS** |
| t16m16 | `--trunk-precision coreml-fp16 --mem-attn coreml-fp16` | 0.5694 | +0.0001 / +0.0001 / -0.0012 / -0.0001 / -0.0002 | 36: -0.0012 | **PASS** |

The trunk pipeline was on in every run, its default with a CoreML trunk. E26 showed it gives
byte-identical tracks.

Wall per frame, `(total - load) / frames` per video. This is one sequential pass, not paired, so it is
indicative only. The interleaved 3-round speed evidence is `../../e27-pipe-necks/` (t16m16 there is the
`fast16` arm, 0.57-0.60x the coreml-fp32 default):

| config | 21 | 22 | 36 | 47 | 82 | mean |
|---|---|---|---|---|---|---|
| default (fp32 reference) | 2.03 | 1.53 | 2.90 | 1.81 | 2.14 | 2.08 s |
| fast16 | 1.05 | 0.75 | 1.58 | 0.94 | 1.17 | 1.10 s |
| t16m16 | 1.07 | 0.96 | 1.78 | 0.91 | 1.11 | 1.16 s |

## Outcome, per RULES.md

A placement becomes the Metal default when it passes G1, G2, G4, and G3 for fp16.
- **t16m16 becomes the default.** It is the fp16 trunk plus fp16 memory attention, segmentation head on
  Ojas. Each of its two placements passes G4 in E27, and the pair passes G2 (32 frames: 0.6025) and G3
  (above).
- **The CoreML segmentation head does not go in.** It failed G4 in E26: no frame-time gain, because of
  host copies. fast16 stays an explicit set of flags, with its HOTA (0.5705) stated here.
- The fp32 reference is one flag: `--trunk-precision coreml-fp32`, which also puts memory attention back
  on Ojas. That reproduces every result recorded before 2026-11-04 (`tools/jobs/track_visem.sh`:
  `SPIKETRACK_TRUNK=coreml-fp32`).

The default change itself is checked in `../../default-fp16/`: clippy, tests, and a 6-frame run with no
flags that is byte-identical to the explicit flags.
