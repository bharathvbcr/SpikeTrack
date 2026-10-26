"""gp-zero-shot-check: can zero-shot SAM 3.1 find sperm in VISEM-Tracking?

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/zero_shot_check.py [--frames N]

Runs the official SAM 3.1 detector (the multiplex checkpoint's `detector.*`,
built by sam3_ref.py) on CPU through upstream's own `Sam3Processor`, i.e.
upstream's image preprocessing (torchvision v2 Resize) and scoring
(sigmoid(logit) * presence). No Meta Model API key was available
(gaps.jsonl GAP-META-MODEL-API-KEY), so this is the local checkpoint, not the
hosted endpoint.

Scores each text prompt against VISEM class-0 (sperm) boxes: recall and
precision at IoU 0.5 after greedy one-to-one matching, at score thresholds
0.3 / 0.4 / 0.5. Absent-concept prompts measure how often the presence head
fires anyway. Writes bench/results/zero-shot/.
"""

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN = REPO / "data/visem/VISEM_Tracking_Train_v4/Train"
OUT = REPO / "bench/results/zero-shot"
PROMPTS = ["sperm", "sperm cell", "spermatozoon", "cell"]
ABSENT = ["car", "cell nucleus"]
THRESHOLDS = (0.3, 0.4, 0.5)
SEED = 20261026


def sample_frames(n_videos, per_video):
    rng = random.Random(SEED)
    vids = sorted(p.name for p in TRAIN.iterdir() if p.is_dir())
    vids = rng.sample(vids, n_videos)
    out = []
    for v in vids:
        labels = sorted((TRAIN / v / "labels").glob("*.txt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        for lab in rng.sample(labels, per_video):
            img = TRAIN / v / "images" / (lab.stem + ".jpg")
            if img.exists():
                out.append((v, img, lab))
    return out


def gt_boxes(label_path, w, h, cls=0):
    boxes = []
    for line in label_path.read_text().split("\n"):
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != cls:
            continue
        cx, cy, bw, bh = (float(p) for p in parts[1:])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(pred, scores, gt, thr=0.5):
    """Greedy by score, one-to-one at IoU >= thr. Returns true positives."""
    order = np.argsort(-scores)
    m = iou(pred[order], gt)
    used = np.zeros(len(gt), bool)
    tp = 0
    for i in range(len(order)):
        if len(gt) == 0:
            break
        j = np.argmax(np.where(used, -1, m[i]))
        if not used[j] and m[i, j] >= thr:
            used[j] = True
            tp += 1
    return tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--per-video", type=int, default=5)
    args = ap.parse_args()
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_ref.use_fp32_mlp()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not missing and not unexpected
    proc = Sam3Processor(model.detector, device=torch.device("cpu"), confidence_threshold=min(THRESHOLDS))
    frames = sample_frames(args.videos, args.per_video)
    OUT.mkdir(parents=True, exist_ok=True)
    per_frame = []
    t_all = time.time()
    for k, (vid, img_path, lab) in enumerate(frames):
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        gt = gt_boxes(lab, w, h)
        t0 = time.time()
        state = proc.set_image(im)
        rec = {"video": vid, "frame": img_path.stem, "gt": int(len(gt)), "prompts": {}}
        for p in PROMPTS + ABSENT:
            st = proc.set_text_prompt(p, dict(state))
            boxes = st["boxes"].cpu().numpy().astype(np.float64)
            scores = st["scores"].cpu().numpy().astype(np.float64)
            rec["prompts"][p] = {"boxes": boxes.round(1).tolist(), "scores": scores.round(4).tolist()}
        rec["seconds"] = round(time.time() - t0, 1)
        per_frame.append(rec)
        print(f"[{k + 1}/{len(frames)}] {vid} {img_path.stem}: gt {len(gt)}, "
              + ", ".join(f"{p}={len(rec['prompts'][p]['scores'])}" for p in PROMPTS + ABSENT)
              + f" ({rec['seconds']} s)", flush=True)

    summary = {"frames": len(per_frame), "videos": args.videos, "seed": SEED, "iou": 0.5,
               "seconds_total": round(time.time() - t_all, 1), "prompts": {}}
    for p in PROMPTS + ABSENT:
        summary["prompts"][p] = {}
        for thr in THRESHOLDS:
            tp = n_pred = n_gt = frames_with_det = 0
            for rec, (vid, img_path, lab) in zip(per_frame, frames):
                im_w, im_h = Image.open(img_path).size
                gt = gt_boxes(lab, im_w, im_h)
                s = np.array(rec["prompts"][p]["scores"])
                b = np.array(rec["prompts"][p]["boxes"]).reshape(-1, 4)
                keep = s >= thr
                b, s = b[keep], s[keep]
                tp += match(b, s, gt)
                n_pred += len(s)
                n_gt += len(gt)
                frames_with_det += int(len(s) > 0)
            summary["prompts"][p][str(thr)] = {
                "recall": round(tp / n_gt, 4) if n_gt else None,
                "precision": round(tp / n_pred, 4) if n_pred else None,
                "predictions": n_pred, "gt": n_gt,
                "frames_with_any_detection": frames_with_det,
            }
    (OUT / "per_frame.json").write_text(json.dumps(per_frame))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
