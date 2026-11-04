# E26 rules (pre-stated 2026-11-04, before any run of the trunk pipeline or the CoreML segmentation head)

## Candidates

- `--trunk-pipeline on`: the CoreML trunk runs on its own thread, one frame ahead. Frame t + 1's
  trunk runs while frame t's detector and tracker run. It is the same program on the same input, so
  the predictions must not change at all.
- `--seg-head coreml-fp32|coreml-fp16`: the detector's segmentation head (cross-attention to the
  prompt, pixel decoder, instance/semantic heads, mask embedding, mask product) as one CoreML ML
  Program from `tools/baselines/coreml_seghead.py`. It is built from base weights; LoRA v3 and the
  shipped edit do not touch them, and `track` refuses an adapter or edit that does.

## Gates

**P1, pipeline exactness.** In every round, `--trunk-pipeline on` must write a `36.json`
byte-identical to the same round's default arm. Anything else is a bug, not a tolerance.

**P2, pipeline speed.** Wall per frame, `(total - load) / frames` from run.json, must be below 0.95x
the default arm in all 3 rounds. The wall time is the measure because the pipeline moves trunk work
outside `push_frame`.

**S1, segmentation-head parity.** Checked against the `visem22_cell` fixture's `seg.pred_masks` /
`seg.semantic_seg` (`bench/results/coreml-seghead/`). fp32 must reach rel L2 <= 2.7e-5 on both, as
G1 in `../e25-coreml-frame/RULES.md`. fp16 is recorded.

**S2, segmentation-head speed and accuracy.** Speed is G4 of `../e25-coreml-frame/RULES.md`, read
on wall time. Accuracy is G2, and G3 for fp16, on the held-out videos.

## Outcomes

- **The pipeline.** If P1 and P2 pass, it becomes the default whenever the trunk is a CoreML
  program. P1 makes it accuracy-neutral by construction, so it needs no held-out run. If P1 fails,
  the pipeline is a bug to fix before anything ships.
- **The CoreML segmentation head.** The adoption outcomes of `../e25-coreml-frame/RULES.md` apply.
