# Box convention: why zero-shot HOTA on VISEM is low, and `track --boxes detector`

Measured 2026-10-27. Figure: `gt_vs_pred_22_0.png` (VISEM 22 frame 0, upscaled 2x): VISEM-Tracking
ground truth in green, `spiketrack track` boxes (`--boxes mask`, upstream's) in red, the zero-shot
`spiketrack image` detector boxes in blue. Prompt "cell", no LoRA, Metal.

## 1. The VISEM 22 "balloon" was the box, not the mask

In the 8-frame CLI run, id 3's box went from x 0.772-0.906, y 0.140-0.287 (frames 0-1) to
x 0.477-0.900, y 0.142-0.838 (frame 2), while its written mask stayed one compact blob
(x 494-576, y 68-140 px; 652, 689, 674, 700 px on frames 0-3).

Upstream computes `out_boxes_xywh` with `masks_to_boxes` on the per-object masks **before**
`_apply_object_wise_non_overlapping_constraints` (`external/sam3/sam3/model/sam3_multiplex_tracking.py:729-746`,
and the batched path at `:965-1035`); `Session::release` does the same. A few raw-mask pixels on
neighbouring cells, which the non-overlap rule then gives to those cells, stretch the box while the
track's own mask does not move. So this is upstream behaviour, reproduced, not a port defect:

- Session parity on upstream's frames, which checks boxes (`box dmax`) as well as id sets and masks,
  passes on CPU and Metal (`../../parity-tracker/{cpu,metal}-f32-detect-track-visem22.log`).
- The bisect (`../balloon-bisect.log`; upstream vs ImageIO frames x CPU vs Metal, 8 frames) shows no
  mask growing in any configuration (largest 958 px, 0.31% of the frame). It printed mask areas only,
  not boxes, so on its own it does not speak to the box.

The CLI stability test now checks each track's written-mask centroid instead of its box centre.

Side finding from the bisect: ImageIO's JPEG decode differs from upstream's (Pillow) frames on 7-9%
of input samples and changes the frame-0 birth set (13 ids either way, a different subset of
cells). See GAP-IMAGEIO-VS-PILLOW-JPEG.

## 2. Low DetA is a box-convention mismatch

VISEM-Tracking boxes are tight sperm-head boxes (median about 20 x 19 px at 640 x 480). The "cell"
prompt's tracker masks cover the whole sperm, tail included, plus round cells VISEM does not
annotate. On frame 0 of the five held-out videos (8-frame smoke run, `../smoke-8f`):

| video | predictions | GT | pred median w x h px | GT median w x h px | GT matched at IoU >= 0.5 |
|---|---|---|---|---|---|
| 21 | 15 | 24 | 47 x 38 | 21.5 x 20.5 | 0 |
| 22 | 13 | 12 | 33 x 31 | 21 x 18 | 0 |
| 36 | 13 | 46 | 33 x 27 | 19 x 20 | 3 |
| 47 | 11 | 9 | 24 x 39 | 27 x 28 | 0 |
| 82 | 8 | 27 | 36 x 57.5 | 17 x 16 | 0 |

Boxes recomputed from the written (non-overlapping) masks change 17 of 535 boxes and HOTA only from
0.1838 to 0.1855 on that run, so the pre-non-overlap rule is not what costs the score.

## 3. The detector's boxes are the ones that can match VISEM

The same frame (VISEM 22, held out from LoRA training), `spiketrack image` on Metal
(`../../lora/cli-check`):

| boxes | count | median w x h px | GT best IoU, quartiles | GT matched at IoU >= 0.5 | GT containing a box centre |
|---|---|---|---|---|---|
| track, mask boxes | 13 | 33 x 31 | 0, 0, 0.049, 0.082, 0.218 | 0 of 12 | 0 of 12 |
| detector, zero-shot | 28 | 33.6 x 23.6 | 0.034, 0.049, 0.082, 0.118, 0.29 | 0 of 12 | 0 of 12 |
| detector, LoRA step 200 | 27 | 28.5 x 24.8 | 0.461, 0.635, 0.70, 0.74, 0.791 | 11 of 12 | 12 of 12 |

One frame, 12 sperm: a strong signal, not a result. The adapter (`detector.transformer.*` only)
moves the detector's boxes onto VISEM's heads, but `track --boxes mask` reports the unadapted
tracker's mask extents, so the adapter cannot show in HOTA that way.

`spiketrack track --boxes detector` (not upstream; default stays `mask`) reports, per track and
frame, the box of the detection the association assigned to it (`tracker::associate::track_detections`:
the highest-scoring kept detection whose best-overlap track it is, at IoM >= 0.1 after the
ambiguity rule; a newborn's own detection), and the mask box when none was. Masks, ids and every
other output are unchanged; `<video>.run.json` records `boxes`, `boxes_reported` and
`boxes_from_detector`.

Scripts (throwaway analysis, not shipped): box statistics against VISEM labels and the figure were
produced with short Python scripts over the files above; the numbers in the tables are their output.
