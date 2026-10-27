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

Side finding from the bisect: ImageIO's JPEG decode differed from upstream's (Pillow) frames on 7-9%
of input samples and changed the frame-0 birth set (13 ids either way, a different subset of
cells). Fixed by the Pillow-exact JPEG decoder (`crates/st-video/src/jpeg.rs`, commit be0244b): the
same bisect re-run (`../balloon-bisect-native-jpeg.log`) has the CLI's decoded input equal to
upstream's on 0 differing samples of 3,048,192 on all 8 frames, and the per-frame, per-id mask
areas from CLI-decoded frames identical to those from upstream's frames, on CPU and on Metal.
(CPU and Metal differ from each other by a pixel on a few masks from frame 5, as before.)

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

## 4. Held-out HOTA by adapter and box source (8-frame smoke)

Same build (750eca6), Metal, prompt "cell", the 5 held-out videos x their first 8 frames (40 frames,
879 GT boxes). Folders `../smoke-8f-{mask,detector,lora200-mask,lora200-detector}`.

| adapter | boxes | HOTA | DetA | AssA | LocA | boxes reported |
|---|---|---|---|---|---|---|
| none | mask (upstream) | 0.1838 | 0.0546 | 0.7526 | 0.6910 | 535 |
| none | detector | 0.1779 | 0.0578 | 0.6392 | 0.6833 | 535 (520 from a detection) |
| step 200 | mask | 0.2019 | 0.0821 | 0.5661 | 0.6458 | 242 |
| step 200 | detector | 0.3003 | 0.1347 | 0.6926 | 0.7438 | 242 (233 from a detection) |

Neither change helps much alone (adapter +0.018, detector boxes -0.006); together +0.117. The
adapter's gain is in the detector's boxes, and only `--boxes detector` carries it to the output.

The ceiling now is recall: the adapter yields 242 boxes for 879 GT. Its scores are lower; on VISEM 22
frame 0 only 6 of its 27 detections above 0.5 reach upstream's 0.65 birth threshold (zero-shot 19 of
28), so most sperm are never born. The threshold is upstream's and stays; whether the step-600
adapter's scores recover is the next measurement.

One run per cell, 8 frames per video: a smoke comparison with no noise estimate, not the held-out
result.

### Step-600 adapter (`../smoke-8f-lora600-*`, same 40 frames)

| prompt | boxes | HOTA | DetA | AssA | LocA | boxes reported |
|---|---|---|---|---|---|---|
| sperm | detector | **0.4694** | 0.2997 | 0.8028 | 0.7305 | 468 |
| sperm | mask | 0.3299 | 0.1874 | 0.7442 | 0.6507 | 468 |
| cell | detector | 0.2497 | 0.0906 | 0.7054 | 0.7442 | 154 |

The prompt the adapter was trained with ("sperm") is the one that benefits: with "cell" the adapted
detector yields few boxes above the birth threshold (154), with "sperm" 468. Per video (sperm,
detector boxes) HOTA ranges from 0.31 (video 36, the densest, 353 GT boxes in 8 frames) to 0.63
(video 22).

Scripts (throwaway analysis, not shipped): box statistics against VISEM labels and the figure were
produced with short Python scripts over the files above; the numbers in the tables are their output.
