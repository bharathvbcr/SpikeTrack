# E11b: the detector and memory encode split into device, readback and host stages (2026-10-30)

This is the same setup as E11 (`../README.md`): Metal, the default CoreML fp32 trunk, prompt
"sperm", r16 LoRA, `--boxes detector`, `--max-objects 64`, 16 frames of VISEM 22 and 82, with
`job.sh` under mac_heavy. The binary came from 5156b76 plus this change, with the tree dirty (sha in
`versions.txt`). It ran the spiketrack and st-model unit tests: 75 and 43 passed.

The detector marks its phases (`DETECTOR_STAGES` in `crates/st-model/src/detector.rs`) and the
Session marks the `pred_masks` readback. Memory encode marks its host part (the mask upsample,
readback and the host area rule) apart from the encode. `StageClock` syncs at each mark, so GPU
work is charged to the stage that dispatched it.

**Validity, judged by the pre-stated 10% rule [V].**
- 82 holds. Its stages run is 2598 ms per steady frame against controls of 2701 and 2721 (-4%).
- 22 fails, 4207 against 2389 and 2153 (+85%), so it is not read. Machine load was 75-95 during
  that run and 20-40 for the 82 runs. The stage that stretched most on 22 is the host decoder prep
  (1291 ms against 264 ms on 82), which is what host-bound work does on a contended CPU.
- With the marker on, predictions are byte-identical to control-1 on both videos.

## Video 82, steady frames (1-15), ms per frame

| stage | ms | share | runs on |
|---|---|---|---|
| trunk (CoreML) | 903.9 | 34.6% | CoreML |
| detector neck | 106.5 | 4.1% | device |
| detector: encoders | 84.8 | 3.2% | device |
| detector: decoder host prep (sine, query MLP, box RPB, upload), 6 layers | 264.4 | 10.1% | host |
| detector: decoder layer, 6 layers | 23.3 | 0.9% | device |
| detector: decoder readback + box update, 6 layers | 47.7 | 1.8% | host |
| detector: scoring | 25.2 | 1.0% | host |
| detector: segmentation head | 115.5 | 4.4% | device |
| detector mask readback (`pred_masks`, ~66 MB) | 7.7 | 0.3% | transfer |
| propagation neck | 121.6 | 4.7% | device |
| tracker propagate, state 0 / state 1 | 313.2 / 259.5 | 21.9% | device + host (not split) |
| memory encode: upsample + readback + host area rule | 42.1 | 1.6% | mixed |
| memory encode: encode | 178.8 | 6.8% | device |
| births, interactive neck, association, outputs | 103.6 | 4.0% | mixed |
| decode + upload, progress line + collector | 13.3 | 0.5% | host |

The stage sum is 2611.0 ms per frame, and the in-Session sum is 2597.7 against the measured
push_frame time of 2597.7.

## Reading [V for the numbers, I where marked]

- **The pre-stated rule (docs/SPEED_PLAN.md E11) did not foresee this outcome.**
  - Its first branch needed the detector's device stage to stay near 780 ms. That fails, because the
    device-side detector (neck, encoders, decoder layers, segmentation head) is 330 ms.
  - Its second branch was keyed to readback of 200 ms or more. That also fails: the `pred_masks`
    readback is 7.7 ms.
  - The detector's missing time is host compute, not readback:
    - The decoder's host stages total 337 ms per frame (prep 264, readback with the box update 48,
      scoring 25). That is 13% of the frame and more than the device detector's 330 ms.
    - The code is SpikeTrack's (`Detector::forward_marked` and `rpb_bias` in
      `crates/st-model/src/detector.rs`), so by the rule's intent the gap is SpikeTrack's.
- **What host prep computes per layer [V, read from the code]:**
  - The `boxRPB_embed_x` and `boxRPB_embed_y` MLPs (2 -> 256 -> 8) over 200 x 72 rows each, about
    150 MFLOP.
  - `ref_point_head` (512 -> 256 -> 256) over 200 rows, about 80 MFLOP.
  - The broadcast of the two into an `[8, 201, 5184]` bias, 8.3 M f32 (33 MB), then uploaded.
  - Across the 6 layers that is about 1.4 GFLOP of scalar host work, about 5 GFLOP/s at 264 ms
    [I: the split between the MLPs, the broadcast and the upload is not measured].
- **The decoder layers themselves are cheap on the GPU:** 23 ms for all six.
- **Memory encode is mostly device:** the encode is 179 ms and its host part 42 ms. E11's ops run
  had 145 ms of op time in memory encode, including the 23 ms `[1, 32, 1152, 1152]` upload.
- **The tracker states are not split yet.** E11 put ~69 and ~125 ms of their wall time outside the
  ops.

Files: `v82-stages/82.profile-stages.md`, `v22-stages/22.profile-stages.md` (invalid, kept), the
`*.log` per-frame telemetry, and `job.sh`. Run JSON and predictions are in each run directory; masks
are not committed.
