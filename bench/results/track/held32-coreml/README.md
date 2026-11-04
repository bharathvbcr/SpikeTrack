# G2: held-out HOTA of the CoreML placements, first 32 frames (2026-11-04)

This is gate G2 of `../../e25-coreml-frame/RULES.md`; the rules were stated before the runs.
- Videos: the five held-out VISEM videos (21, 22, 36, 47, 82), first 32 frames each.
- Configuration: the shipped one (LoRA v3 + `lorav3_exact_it_3p00.json`, prompt sperm, `--boxes detector`, cap 64).
- Job: `job.sh`, log in `job.log`.
- Table: `summary.md`, written by `tools/analysis/g2_placements.py`.

| config | flags on top of the default | HOTA | dHOTA | worst video | verdict |
|---|---|---|---|---|---|
| default | (CoreML fp32 trunk, everything else Ojas) | 0.6025 | | | baseline reproduces the documented 0.6025 and every per-video value |
| m16 | `--trunk-pipeline on --mem-attn coreml-fp16` | 0.6025 | +0.0000 | +0.0000 | **PASS** (fp16 rule: >= 0.6005, no video < -0.01) |
| seg32 | `--trunk-pipeline on --seg-head coreml-fp32` | 0.6024 | -0.0000 | 36: -0.0001 | **PASS** (fp32 rule: \|d\| <= 0.002, videos <= 0.005) |
| fast16 | `--trunk-precision coreml-fp16 --trunk-pipeline on --mem-attn coreml-fp16 --seg-head coreml-fp16` | 0.6025 | -0.0000 | 82: -0.0007 | **PASS** (fp16 rule) |
| t16m16 (added after E27; `job-t16m16.log`) | `--trunk-precision coreml-fp16 --trunk-pipeline on --mem-attn coreml-fp16` | 0.6025 | -0.0000 | 82: -0.0007 | **PASS** (fp16 rule) |

t16m16 is fast16 without the CoreML segmentation head. That head failed the speed gate G4 in E26
(1.00-1.06x). t16m16 is the combination whose placements each pass G4: the fp16 trunk gives 0.73-0.77x
against m16 and the fp16 memory attention 0.76-0.82x against default, both in E27. Since 2026-11-04 it is
track's Metal default, and the `default` config here is pinned to `--trunk-precision coreml-fp32`, the
fp32 reference these runs used.

**Each placement ran [V].** Every `run/<video>.run.json` names its CoreML package and sha256 (for
example, m16 shows `external_mem_attn: coreml sam3.1_mattn_fp16.mlpackage …` and
`trunk_pipeline: true`). The tracks are not copies of the default's:
- m16 differs from default on videos 36 and 47;
- seg32 differs on 36;
- fast16 differs on all five videos.

These changes did not move HOTA by more than 0.0007 on any video.

**Wall per frame [V, one sequential pass, not an A/B].** Computed as `(total - load) / frames`,
mean over the 5 videos:

| config | mean wall per frame |
|---|---|
| default | 2.71 s |
| m16 | 1.82 s |
| seg32 | 1.92 s |
| fast16 | 1.29 s |

The configs ran one after another, so these numbers are indicative only. The interleaved
3-round speed evidence is `../../e26-pipeline/`.

G3, the 1,500-frame held-out run, passed for fast16 and t16m16 (`../held300-coreml/`).
