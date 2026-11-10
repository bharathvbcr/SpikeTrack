# Portal answers

Each answer is in its own box. Use the box's copy button, or select inside it, then paste into the field. Character counts are for the text inside the box.

## Project Description (1,999 / 3,000)

```text
SpikeTrack runs all of Meta's SAM 3.1, from a text prompt to tracked masks, as one 12 MB Rust program on a Mac. We adapted it to find and track sperm cells in clinical microscope video.

Why: sperm motility, a core part of male-fertility testing, is still judged largely by eye. Following dozens of small, fast, look-alike cells through video is hard, and fertility labs often cannot upload patient video to a cloud service. SAM 3.1's official code needs Python, PyTorch and a CUDA GPU. The existing Mac (Core ML) port runs only the tracker: it cannot take a text prompt, detect objects, or start new tracks.

What we built:
- All of SAM 3.1 in Rust: the CLIP text encoder, the DETR detector, the PE ViT backbone, the Object Multiplex tracker, and the detect, associate and track loop that starts and keeps tracks. It runs on our own Metal GPU engine (Ojas and tessl). Its heaviest parts (image trunk, necks, memory attention, memory encoder) run as half-precision Core ML programs exported from the checkpoint. It links only macOS system libraries and needs no Python at run time.
- Checked against Meta's official code: every module matches (worst cosine 1.0000000, on CPU and on Metal). Image mode keeps the same detections on 20/20 frames. On 3 DAVIS clips, J&F equals the official code's. The full text-prompted tracking loop keeps the same tracked IDs on 32/32 frames.
- Adapted to sperm: zero-shot, SAM 3.1 does not ground "sperm" (HOTA 0.000). We trained a LoRA adapter on the Mac (1.13M parameters, about 0.13% of the model) and applied a 768-scalar weight edit. Results on five held-out VISEM-Tracking videos:
  - Frame 0 detection: 95 of 112 heads, 83.7% F1.
  - Tracking over 1,500 frames: HOTA from 0.155 to 0.570.
  - Everyday negative prompts: 0 false positives out of 600.
- Speed: about 1.0 s per frame on a dense video (M5 Pro), down from 2.9 s. It is not real time yet.

Every number links to a saved run log in the repository. SpikeTrack is a research tool, not a medical device.
```

## Public GitHub Repository

```text
https://github.com/bharathvbcr/SpikeTrack
```

## 3-Minute Demo Video

Paste your YouTube link here after uploading `bench/results/demo/spiketrack-demo.mp4` (2:32).

## Describe your use of Meta products to build the submitted project (1,482 / 3,000)

```text
- SAM 3.1 (Segment Anything 3.1 with Object Multiplex) is the model. We re-implemented its full architecture in Rust and load Meta's released checkpoint directly from safetensors.
  - We use every part: the text encoder for the prompt "sperm", the detector to find heads, the image backbone, and the multiplex tracker with its memory to keep each cell's ID and start new tracks.
  - We follow the SAM License, and the weights are not redistributed.
- Meta's official SAM 3.1 PyTorch code is our reference oracle.
  - We dumped fixtures from it and wrote parity tests for every module, for image mode, for video object segmentation on DAVIS, and for the full detect-and-track loop.
  - We matched its preprocessing exactly, down to a JPEG decoder that is byte-identical to Pillow's and the same resize and rounding.
- PyTorch, created at Meta, is the training and export tool. We trained the sperm adapter on the Mac's GPU (MPS) using SAM 3.1's own detection losses (one-to-many matching and the presence loss). PyTorch was also the first step of the one-time Core ML export of the trunk, necks and memory modules, which the Rust program then calls.
- We also inspected the checkpoint in Rust (875.5M parameters) to see why zero-shot "sperm" fails. The word's token embedding sits in a clinical text cluster away from "cell", and the scoring head compresses true sperm scores around the birth threshold. Those findings guided the small token and scoring edits that the adapter makes.
```

## Why are Meta's models important to your solution? (1,149 / 3,000)

```text
- Open weights make on-device use possible. Clinical video is patient data; with SAM 3.1's released checkpoint, the whole pipeline runs on the lab's own laptop with no cloud call.
- One model covers the whole task. Lab work needs a word turned into detections, then tracks that keep IDs through crossings and pick up cells that enter later. SAM 3.1 has text-prompted detection, segmentation and multi-object tracking with births in one model, so we did not have to stitch a detector, a segmenter and a tracker together.
- Strong priors make adaptation cheap. Zero-shot, the model cannot ground "sperm", but its masks and association are already good. About 0.13% of the parameters in a LoRA adapter, plus a 768-scalar edit, took tracking HOTA from 0.155 to 0.570 on unseen videos, with association accuracy around 0.72. We trained it on a laptop, not a GPU cluster.
- Meta's open reference code makes the port trustworthy. Because the official implementation is public, we could prove the Rust version matches it module by module and end to end. For a medical-research tool, "it matches the original model" is a claim we can show rather than assert.
```

## Provide feedback from your experience using partner technologies (138 / 150)

```text
SAM 3.1's open weights and reference code made a verifiable Rust port possible. An official Apple-silicon path would help on-device teams.
```

## What was the biggest obstacle your team hit while building this project?

```text
Making SAM 3.1 work on sperm at all. Zero-shot it finds none for "sperm" (HOTA 0.000), and "cell" boxes whole cells with tails while the labels mark heads. A LoRA adapter, a small weight edit and detector boxes fixed it. Second came speed without CUDA: 2.9 s to about 1.0 s per frame through fp16 Core ML.
```

## If your team took this project to production, what would most likely stop you from using Meta's models?

```text
Clinical deployment needs clear licence terms for a modified SAM 3.1 in a medical product, regulatory validation, and real-time speed; we are at about 1 s per frame.
```

## Yours to answer

These are ratings or opinions, so no draft is given:
- How likely are you to build with Meta Model API?
- Overall opinion of Meta Model API
- Recommend the Meta Model API?
- Recommend Muse Code?
- Compared with other AI models, how did Meta's models perform?
- When choosing an AI model or coding tool, what matters most?
- What makes it hardest to switch?

SpikeTrack runs SAM 3.1 locally. Nothing in the repository shows it used the hosted Meta Model API, Llama or Muse Code. If you did not use them, a neutral rating or a plain "not used in this project" is the honest answer.
