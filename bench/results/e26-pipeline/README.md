# E26: the trunk pipeline and the CoreML segmentation head (2026-11-04)

E26 measures whole-frame time for the trunk pipeline and the CoreML segmentation head. The setup:
- `job.sh`; gates in `RULES.md`, stated before the runs;
- VISEM 36 (held-out, dense), first 32 frames, shipped configuration;
- 5 arms, 3 rounds, order rotated per round; table in `summary.md`.

"wall" is `(total - load) / frames` from run.json. Ratios are each round's own default arm.

| arm | wall / default, rounds 1-3 | push_frame / default | tracks vs default |
|---|---|---|---|
| pipe (`--trunk-pipeline on`) | 0.778, 0.911, 0.892 | 0.812, 0.871, 0.847 | **byte-identical `36.json`, all 3 rounds** |
| pipe-m16 (+ `--mem-attn coreml-fp16`) | 0.618, 0.705, 0.649 | 0.635, 0.656, 0.577 | 32/32 id sets, min IoU 0.9919 |
| pipe-m16-seg32 (+ `--seg-head coreml-fp32`) | 0.656, 0.717, 0.648 | 0.654, 0.684, 0.580 | 32/32 id sets, min IoU 0.9919 |
| fast16 (fp16 trunk, pipelined, fp16 memory attention and segmentation head) | 0.511, 0.529, 0.428 | 0.488, 0.453, 0.377 | 32/32 id sets, mean IoU 0.998, min 0.68 |

Default wall per frame was 3.0-3.2 s. fast16 was 1.36-1.59 s.

## Verdicts

**The pipeline passes P1 and P2 [V].** Its predictions are byte-identical to the default's in every
round. Its wall time is below 0.95x the default's in all 3 rounds (0.78-0.91). It becomes the default
whenever the trunk is a CoreML program.

**Memory attention fp16 passes G4 here [V].** pipe-m16 is 0.62-0.71x the default on wall time, and
`pipe` alone is 0.78-0.91x. That gives E25's mattn16 (failed only in its contended round 2) a clean
3-round measurement. Its accuracy is G2 (`../track/held32-coreml/`).

**The CoreML segmentation head buys nothing in the frame [V].**
- Against pipe-m16, adding it costs 1.06x, 1.02x and 1.00x.
- The program alone is 25 ms against Ojas's ~108 ms (`../coreml-seghead/`; parity rel L2 6.5e-6).
- [I] The frame pays for host copies: the neck's two finest levels read back (106 MB) and copied into
  Core ML's inputs, and 66 MB of mask logits copied out.
- It is not offered as is. It becomes worth measuring again only if the neck's levels already sit on
  the host, as they would if the necks ran on the pipeline's worker.

**fast16 is about 2x on frame time [V]. Its accuracy is G2 + G3.**
