# SpikeTrack

SAM 3.1, Meta's Segment Anything with its Object Multiplex video tracker, built as one Rust binary
with no Python at runtime. It runs on Apple Silicon through Ojas's Metal backend on tessl, or on
Ojas's CPU backend. The evaluation is many-sperm tracking in VISEM-Tracking. Built for the Meta
hackathon, Oct 26 to Nov 4, 2026.

![VISEM-Tracking video 22, frame 30: every sperm head tracked with its id](docs/figures/visem22-frame30-sperm-lora.png)

`spiketrack track` on held-out VISEM video 22 with the prompt "sperm", the detector LoRA and
`--boxes detector`. The clip is [bench/results/demo/22-sperm-lora.mp4](bench/results/demo/22-sperm-lora.mp4)
(49 frames, made by `tools/jobs/demo_clip.sh 22 49 sperm`).

Status and plan: [HANDOFF.md](HANDOFF.md) (schedule, gates, task index), [tasks/](tasks/), open
questions in [gaps.jsonl](gaps.jsonl). What existed before the hackathon and what was built during
it: [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## What works (measured, 2026-10-27)

Every number links to the log that produced it; parity is against the official SAM 3.1 code run in
fp32 on CPU (tools/reference).

| Check | Result | Evidence |
|---|---|---|
| Every module against upstream (text encoder, detector, PE ViT trunk + necks, mask decoder, memory encoder, memory attention, interactive decoder), CPU and Metal | worst cosine 1.0000000 on both | [module-gates-cbd5565](bench/results/module-gates-cbd5565) |
| Image mode end to end (decode, backbone, text, detector), 20 VISEM frames, Metal | kept sets identical on 20/20, mask IoU 1.0000, ~2.3 s/frame | [d3-gate/metal-f32.log](bench/results/d3-gate/metal-f32.log) |
| Semi-supervised VOS (D5), 3 DAVIS clips, Metal | J&F equal to upstream's (0.9107 / 0.9456 / 0.9514), mask IoU vs upstream 1.0000 | [vos-gate](bench/results/vos-gate) |
| Text-prompted detect -> associate -> track session, VISEM 22, 32 frames (upstream's frame tensors), CPU and Metal | identical tracked-id sets on all 32 frames (13-15 objects, births included) on both, mask IoU 0.9985 (CPU 0.99854, Metal 0.99855) | [CPU](bench/results/parity-tracker/cpu-f32-detect-track-visem22.log), [Metal](bench/results/parity-tracker/metal-f32-detect-track-visem22.log) |
| No Python | 8.4 MB binary linking only system frameworks; runs with no python on PATH | [packaging](bench/results/packaging) |
| Detector LoRA (rank 16, 983k params, trained on this Mac), 120 frames of 5 held-out VISEM videos | sperm recall at IoU 0.5: prompt "sperm" 0.000 -> 0.761 (precision 0.584), "cell" 0.212 -> 0.576 at 0.5; absent VISEM classes now fire on 16% of items (was 0) | [lora](bench/results/lora) |
| LoRA merged in Rust | merged tensors vs PyTorch max abs diff 1.5e-8 on the real checkpoint; `--lora` on `image` and `track` | [merge-parity.log](bench/results/lora/merge-parity.log) |
| Tracking, first 300 frames of the 5 held-out VISEM videos (1,500 frames, 32,949 GT boxes) | HOTA 0.1549 zero-shot (prompt "cell", upstream boxes) -> **0.4695** with the adapter, prompt "sperm" and `--boxes detector` (DetA 0.051 -> 0.320, AssA 0.594 -> 0.725) | [zero-shot](bench/results/track/held300-zeroshot-cell-mask/eval.txt), [adapter](bench/results/track/held300-lora600-sperm-detector/eval.txt), [box-convention](bench/results/track/box-convention) |
| CoreML port on the same M5 Pro (the speed reference) | 0.272 s backbone + 0.0775 s track step per frame | [baselines](bench/results/baselines) |

Not yet: the speed target (restated to parity-or-better with the CoreML port; we are ~6-8x behind
on the tracker path today, see [profile-cbd5565](bench/results/profile-cbd5565)). The dense held-out
videos (27-46 sperm per frame) are bounded by upstream's 16-object tracking cap; `--max-objects` is
being measured. Open issues are in gaps.jsonl.

VISEM's ground truth is sperm-head boxes, while upstream's track boxes are the extent of the
whole-sperm mask, tail included, so zero-shot HOTA with upstream's boxes mostly measures that
mismatch. `track --boxes detector` (opt-in, not upstream) reports each track's assigned detection
box instead, which is where a detector LoRA shows. Every VISEM number states its box source; see
[track/box-convention](bench/results/track/box-convention).

## Layout

| Path | What |
|---|---|
| `crates/spiketrack` | The binary: `image`, `track`, `eval-visem` |
| `crates/st-model` | SAM 3.1 on the Ojas `Backend`: PE ViT trunk and necks, text encoder, detector and mask head, the multiplex tracker (memory attention, mask decoders, memory encoder), the VOS driver and the text-prompted detect-track session; a per-op profiler |
| `crates/st-weights` | The checkpoint (safetensors) mapped to typed structs, every tensor claimed exactly once; LoRA adapter merge |
| `crates/st-video` | Native frame decode (ImageIO, AVFoundation), Pillow-exact bilinear resize |
| `crates/st-eval` | HOTA (TrackEval definition), DAVIS J&F, the VISEM-Tracking loader |
| `crates/st-gate` | Event gating: per-window change maps and recompute policies for the backbone |
| `tools/reference` | Offline Python only: the official SAM 3 code on CPU, fixtures, oracles, the LoRA trainer |
| `bench/results` | Every reported number, with the command that produced it |

## Running

Weights and data are not in the repository. Get `facebook/sam3.1` from Hugging Face (gated, SAM
License) and VISEM-Tracking from Zenodo 7293726. Convert the checkpoint once, offline:
`tools/reference/.venv/bin/python -s tools/reference/convert_checkpoint.py`
(the venv setup is in `tools/reference/README.md`). The workspace builds against local Ojas and
tessl checkouts (CONTRIBUTIONS.md, reproducibility).

```sh
cargo build --release -p spiketrack
export SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors
export SPIKETRACK_MODEL_DIR=~/Code/research/models/sam3.1   # tokenizer vocab.json + merges.txt

# One image: detections.json (score, box, mask) and overlay.png
target/release/spiketrack image data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg \
  --prompt cell --backend metal --out out/image

# Video: track every instance of the prompt (a JPEG frame folder or an .mp4)
target/release/spiketrack track data/visem/VISEM_Tracking_Train_v4/Train/22/images \
  --prompt cell --backend metal --max-frames 64 --out out/track

# The same, also writing the masks and boxes, coloured by track id, as a playable H.264 video (49 fps for a frame folder, the input's rate for an mp4; --overlay-fps overrides)
target/release/spiketrack track data/visem/VISEM_Tracking_Train_v4/Train/22/images \
  --prompt cell --backend metal --max-frames 64 --overlay-video out/track/22-overlay.mp4 --out out/track

# The same with a detector LoRA adapter (trained offline by tools/reference/train_lora.py)
target/release/spiketrack track data/visem/VISEM_Tracking_Train_v4/Train/22/images \
  --prompt cell --backend metal --max-frames 64 --lora fixtures/lora_runs/visem-lora-r16/adapter.safetensors \
  --boxes detector --out out/track-lora

# HOTA against VISEM-Tracking's boxes
target/release/spiketrack eval-visem --pred out/track --split tools/reference/visem_split.json --set held_out
```

On a shared machine, heavy jobs run under its lock: `tools/locked.sh <label> <command>`; the parity
gates are `tools/jobs/module_gates.sh`, `tools/jobs/vos_gate.sh` and
`tools/jobs/detect_track_parity.sh`.

Preprocessing: JPEG folders follow upstream's image-folder path. JPEGs are decoded by our own decoder
(`crates/st-video/src/jpeg.rs`, a port of libjpeg-turbo 3.2.0's default path), byte-identical to
Pillow 13 on the VISEM test frames and 37 edge-case files. Frames are then Pillow-exact bilinear
resized and rounded through float16 as upstream's loader stores them. Progressive or other JPEG
variants that decoder refuses, and PNGs, go through ImageIO, and every output file records which
decoder ran. mp4 input does not match upstream's OpenCV path (GAP-MP4-PREPROCESSING).

## Rules

Meta's Official Rules have not been read yet. They are on the hackathon portal, and only Bharath
can reach them (gaps.jsonl `GAP-OFFICIAL-RULES-UNREAD`). Their wording on prior work, required
Meta models, repository visibility and the demo video goes here once read.

## Licences

Our code is offered under MIT or Apache-2.0, pending GAP-SAM-LICENSE-PORT-TERMS. SAM 3.1 is under
the SAM License; see NOTICE. VISEM-Tracking is CC BY 4.0.
