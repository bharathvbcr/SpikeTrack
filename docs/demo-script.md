# Demo video script

Target length about 2:30. Adjust to the Official Rules' limit once read (GAP-OFFICIAL-RULES-UNREAD).
Everything on screen is a live run or a clip the binary wrote, not a slide. Every number spoken is
in [bench/results/SUMMARY.md](../bench/results/SUMMARY.md).

Before recording: `cargo build --release -p spiketrack`, export `SPIKETRACK_WEIGHTS` and
`SPIKETRACK_MODEL_DIR` (README, Running), close other heavy jobs, and set the terminal font large.

## 0:00-0:20 The problem

On screen: a raw VISEM-Tracking frame from video 36 (`data/visem/VISEM_Tracking_Train_v4/Train/36/images/36_frame_0.jpg`).

Say: "Fertility labs count and follow dozens of sperm cells through microscope video. They often
cannot upload clinical video to a cloud service, and Meta's SAM 3.1 needs Python, PyTorch and a
CUDA GPU. SpikeTrack runs all of SAM 3.1 on a laptop, from a text prompt to tracked cells."

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

On screen, side by side (any editor, or `ffmpeg -i a.mp4 -i b.mp4 -filter_complex hstack`):
- left `bench/results/demo/36-cell-zeroshot.mp4`: zero-shot SAM 3.1, prompt "cell"
- right `bench/results/demo/36-sperm-lora-edit-v3.mp4`: adapter + 768-weight edit, prompt "sperm"

Both are 200 frames of held-out video 36 written at 25 fps (about half speed), made by
`tools/locked.sh demo-v3 bash bench/results/demo/v3-job.sh`.

Say: "Out of the box SAM 3.1 does not know the word 'sperm', and 'cell' boxes whole cells with
their tails. With a LoRA adapter and an edit of 768 weights, each head gets a box and keeps its id.
Over 1,500 frames of five unseen videos, tracking accuracy goes from HOTA 0.15 zero-shot to
0.57." (`bench/results/track/held300-lorav3-it300-sperm-detector-max64/eval.txt`)

## 1:50-2:15 Evidence that it is SAM 3.1

On screen: the README's "What works" table, scrolled slowly.

Say: "Every module was checked against Meta's official code: worst cosine 1.0000000 on CPU and on
the GPU, and the same tracked ids as upstream on 32 of 32 frames, births included."

## 2:15-2:30 Limits, said plainly

Say: "It is not fast yet. Step for step at 16 objects our tracker is about twice as slow as the
existing CoreML port at the same precision. But that port runs only the tracker: it cannot take a
text prompt, detect, or start new tracks. SpikeTrack runs all of SAM 3.1."

## Not in this video, and why

- A side-by-side with the CoreML port: it cannot run from a text prompt, so there is no common
  input to show.
- Event gating on and off: on VISEM, moving sperm change most attention windows (55%), so exact
  gating saves little (`bench/results/gating-change-rates`).
