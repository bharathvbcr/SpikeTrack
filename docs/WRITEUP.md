# SpikeTrack: the whole of SAM 3.1 on a laptop, with no Python

Draft of the hackathon submission text, 2026-10-27. Every number is in
[bench/results/SUMMARY.md](../bench/results/SUMMARY.md) with the command that produced it. Still open:
the Official Rules' wording on length, prior work and required Meta models (unread,
GAP-OFFICIAL-RULES-UNREAD).

## The problem

Sperm-motility analysis means counting and following dozens of small, fast, similar-looking
cells through microscope video. The labs that do it often cannot send clinical video to a cloud
service. SAM 3.1 can be prompted with a word ("sperm") to find every instance, and its Object
Multiplex tracker follows up to 16 objects per state through a video. Today it needs Python and
PyTorch, and its reference code is written for CUDA.

## What we built

One Rust binary, `spiketrack`, that runs all of SAM 3.1 on Apple silicon, from a text prompt to
tracked masks: the PE ViT backbone, the CLIP text encoder, the DETR detector, the multiplex
tracker, and the detect -> associate -> track loop that starts and keeps tracks. It runs on
Ojas's Metal backend, built on tessl, or on Ojas's CPU backend. On Metal, `track` by default runs
the ViT trunk as a CoreML fp32 ML Program exported from the checkpoint. That makes the frame 0.80x
and leaves the tracks unchanged on all 1,500 held-out frames. `--trunk-precision f32` keeps every
module on Ojas/tessl.

- `spiketrack image <img> --prompt cell`: boxes, scores and masks for every instance.
- `spiketrack track <frames|mp4> --prompt sperm`: tracked masks and boxes per frame.
- `spiketrack eval-visem`: HOTA against VISEM-Tracking.

The binary is 8.4 MB, links only macOS system frameworks, and runs with no Python on the
machine. JPEG frames decode byte-identically to Pillow through our own decoder, so the model sees
exactly the pixels the official code sees.

Built during the hackathon: everything in this repository (CONTRIBUTIONS.md lists it by date).
Pre-existing and used as dependencies: Ojas, tessl and sparsl (our own open-source libraries; the
op work we added to Ojas during the window is listed separately in CONTRIBUTIONS.md).

## Why it is new

- A pure-Rust SAM 3 exists (`rlx-sam3`), but it states "SAM 3.1 multiplex is intentionally out of
  scope".
- The only other non-PyTorch SAM 3.1 port (CoreML/Swift) runs the tracker only: no text prompt, no
  detector, so it cannot find or start tracks on its own.
- Ours runs the full model, and every module is checked against the official code.

## Evidence that it is SAM 3.1

Parity is against the official SAM 3.1 code run in fp32 on CPU, not against our own idea of the
model:

- Every module matches on CPU and on Metal (worst cosine 1.0000000).
- Image mode on 20 VISEM frames: the same detections as upstream on every frame, mask IoU 1.0000.
- Video object segmentation on 3 DAVIS clips: the same J&F as upstream (0.9107 / 0.9456 / 0.9514).
- The full text-prompted tracking loop on 32 VISEM frames: the same tracked ids on every frame,
  births included, mask IoU 0.9985, on CPU and on Metal.

## Making it work on sperm

Zero-shot, SAM 3.1 does not ground "sperm" (recall 0.000), and "cell" finds whole cells and debris
(recall 0.212 at IoU 0.5).

- We trained a rank-16 LoRA on the detector's attention: 983,040 parameters, 600 steps, on this
  laptop. On 120 frames of 5 held-out videos, "sperm" recall goes from 0.000 to 0.761 (precision
  0.584).
- The adapter merges into the weights at load time in Rust, matching PyTorch to 1.5e-8.
- LoRA on SAM 3 is prior work (SAM3-LoRA, MedSAM3). The new part is applying it to
  sperm-motility video with the multiplex tracker, and running the result with no Python.

VISEM-Tracking annotates sperm heads, while SAM's masks, and so upstream's boxes, cover the whole
sperm, tail included. `track --boxes detector` reports each track's matched detection box, which
is where the adapter's learning shows.

- On the first 8 frames of each held-out video: HOTA 0.18 zero-shot -> 0.47 with the adapter and
  detector boxes.
- On the first 300 frames of each held-out video (1,500 frames, 32,949 ground-truth boxes):
  HOTA 0.155 zero-shot -> 0.470 with the adapter and detector boxes (DetA 0.051 -> 0.320, AssA
  0.594 -> 0.725). Every video improves. The two densest (27-46 sperm per frame) gain least,
  because upstream's builder tracks at most 16 objects at a time.
- `--max-objects 64`, `track`'s default since 2026-10-28, removes that cap. The five-video HOTA
  rises to 0.492. Video 21 goes 0.620 -> 0.642, 36 goes 0.326 -> 0.352, and 82 goes 0.441 -> 0.486.
  DetA rises to 0.373 and AssA falls to 0.690, so more sperm are detected but identities hold less
  well. An earlier 0.487 had rerun only 36 and 82.
- These are zero-shot or detector-only adaptation results. They are not comparable to supervised
  trackers (HOTA 74.3 on VISEM).

## Limitations, stated plainly

- **Speed.** About 2.7 s per frame on the M5 Pro with the CoreML fp32 trunk (3.4 s with the Ojas
  trunk), still several times slower than the CoreML port's 0.35 s, which runs only the tracker.
  - Since E12 (2026-10-30), the detector's host decoder runs blocked and threaded, with identical
    output, and the frame is 0.86-0.88x of that (`bench/results/e12-host-decoder`).
  - The trunk's f32 matrix multiplies already run at the GPU's f32 peak.
  - The CoreML model load adds 3-12 s at startup.
  - An fp16 trunk gives 0.55x the frame but changes close detections. On video 21 that cost the
    cap-16 HOTA 0.024, so fp16 is not offered as a fast mode.
- **Gating.** Pixel-event gating is exact when it skips nothing. On VISEM, moving sperm change
  most attention windows, so exact gating saves little; the approximate policy skips more, at a
  measured cost.
- **Adapter side effects.** Prompts for VISEM's other classes (clusters, pinheads) now fire on 16%
  of frames.
- **mp4 input** does not follow upstream's OpenCV decode path. Frame folders are the comparable
  input.

## Prior work

rlx-sam3 (pure-Rust SAM 3, no multiplex); the sdkv2 SAM 3.1 CoreML port (tracker only);
Efficient-SAM2 (saliency-based skipping, needs training); EdgeTAM and TinySAM2 (smaller retrained
models); SAM3-LoRA and MedSAM3 (LoRA on SAM 3); SAM-based cell tracking (arXiv 2509.09943).
