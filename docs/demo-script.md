# Demo video script

The built video is `bench/results/demo/spiketrack-demo.mp4`: 2:24, 1280x720, narrated with the
macOS voice Samantha. It is made by `tools/locked.sh demo-video bash bench/results/demo/video/make.sh`
from `bench/results/demo/video/segments/` (one card spec and one narration text per segment). It
follows this script, with two differences:
- the one-binary segment says "no Python at run time", because the Core ML programs are exported
  once with coremltools;
- "births included" is dropped from the evidence segment, because no result in this repository
  states it.
To record a voiced version instead, the narration texts are the lines to read.

Target length about 2:30. Adjust to the Official Rules' limit once read (GAP-OFFICIAL-RULES-UNREAD).
Everything on screen is a live run or a clip the binary wrote, not a slide. Every number spoken is
in [bench/results/SUMMARY.md](../bench/results/SUMMARY.md).

Before recording: `cargo build --release -p spiketrack`, export `SPIKETRACK_WEIGHTS` and
`SPIKETRACK_MODEL_DIR` (README, Running), close other heavy jobs, and set the terminal font large.

## 0:00-0:20 The problem

On screen: a raw VISEM-Tracking frame from video 36 (`data/visem/VISEM_Tracking_Train_v4/Train/36/images/36_frame_0.jpg`).

Say: the narration in `bench/results/demo/video/segments/02-problem.txt`. It names both
alternatives up front: Meta's code needs Python, PyTorch and a CUDA GPU, and the Core ML port for
the Mac runs only the tracker, with no text prompt.

## 0:20-0:45 One binary, no Python

```sh
ls -lh target/release/spiketrack
otool -L target/release/spiketrack
env -i PATH=/usr/bin:/bin target/release/spiketrack --help | head -5
```

Say: "One 12 MB Rust binary. It links only macOS system libraries: no Python, no PyTorch."

## 0:45-1:10 Live: find every sperm in a frame

```sh
target/release/spiketrack image data/visem/VISEM_Tracking_Train_v4/Train/36/images/36_frame_0.jpg \
  --prompt sperm --backend metal --threshold 0.65 --tile 2x2 \
  --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
  --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json \
  --out out/demo-image && open out/demo-image/overlay.png
```

Say: "The word 'sperm', SAM 3.1's detector, and a small adapter trained on this Mac. On the first
frames of five unseen videos it finds 95 of the 112 sperm heads, at 83.7% F1."

## 1:10-1:50 Tracking: before and after

On screen: `bench/results/demo/36-before-after-fp16.mp4`, already side by side:
- left (`36-cell-zeroshot-fp16.mp4`): zero-shot SAM 3.1, prompt "cell"
- right (`36-sperm-lora-edit-fp16.mp4`): adapter + 768-weight edit, prompt "sperm"

Both are 200 frames of held-out video 36 written at 25 fps (about half speed), on the shipped
default (fp16 Core ML trunk and memory attention), made by
`tools/locked.sh demo-fp16 bash bench/results/demo/fp16-job.sh`. The v3 clips beside them are the
same configuration on the fp32 trunk that was the default until 2026-11-04.

Say: "Out of the box SAM 3.1 does not know the word 'sperm', and 'cell' boxes whole cells with
their tails. With a LoRA adapter and an edit of 768 weights, each head gets a box and keeps its id.
Over 1,500 frames of five unseen videos, tracking accuracy goes from HOTA 0.15 zero-shot to
0.57." (`bench/results/track/held300-lorav3-it300-sperm-detector-max64/eval.txt`)

## 1:50-2:15 Evidence that it is SAM 3.1

On screen: the README's "What works" table, scrolled slowly.

Say: "Every module was checked against Meta's official code: worst cosine 1.0000000 on CPU and on
the GPU, and the same tracked ids as upstream on 32 of 32 frames, births included."

## 2:15-2:30 Speed and limits, said plainly

Say: the narration in `bench/results/demo/video/segments/07-speed.txt`. The point to land is that
speed comes from Core ML, not from the calling language. SpikeTrack calls the same kind of Core ML
programs from Rust, and adds what the port cannot do: text prompts, detection, new tracks.

The video makes no speed comparison with the port. The last readable one (2026-11-03, about 2x at
fp32) predates the fp16 default. The two re-runs on that default (runs 4 and 5) were not readable
under their pre-stated rule. Quoting either would contradict what the frame does today.

Sources: `bench/results/e27-pipe-necks/README.md` (0.57-0.60x),
`bench/results/track/held300-coreml/README.md` (HOTA), `bench/results/profile-fp16/` (1.31 s/frame,
32 frames of video 36, quiet machine), `bench/results/ab-coreml-tracker/README.md` (the port).

## Not in this video, and why

- A side-by-side with the CoreML port: it cannot run from a text prompt, so there is no common
  input to show.
- Event gating on and off: on VISEM, moving sperm change most attention windows (55%), so exact
  gating saves little (`bench/results/gating-change-rates`).
