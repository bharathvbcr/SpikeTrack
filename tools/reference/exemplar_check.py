"""Do box exemplars fix zero-shot SAM 3.1 on VISEM? (gp-exemplar-prompts, reference side)

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/exemplar_check.py

On the same 50 frames as zero_shot_check.py, give the detector K=3 ground-truth
sperm boxes from that frame as positive exemplars (upstream's
`add_geometric_prompt`), alone ("visual") and with text, and score recall and
precision on the frame's other sperm at IoU 0.5 (exemplar boxes and the
detections matching them are removed first). No training.
"""

import json
import pathlib
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402
from zero_shot_check import SEED, gt_boxes, iou, match, sample_frames  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "bench/results/exemplar"
K = 3
CONDITIONS = [None, "sperm", "cell"]  # None = exemplars only
THRESHOLDS = (0.3, 0.5)


def main():
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_ref.use_fp32_mlp()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not missing and not unexpected
    proc = Sam3Processor(model.detector, device=torch.device("cpu"), confidence_threshold=min(THRESHOLDS))
    rng = random.Random(SEED + 1)
    frames = sample_frames(10, 5)
    OUT.mkdir(parents=True, exist_ok=True)
    tallies = {str(c): {str(t): [0, 0, 0] for t in THRESHOLDS} for c in CONDITIONS}  # tp, preds, gt
    used = 0
    for k, (vid, img_path, lab) in enumerate(frames):
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        gt = gt_boxes(lab, w, h)
        if len(gt) < K + 1:
            print(f"[{k + 1}] {vid} {img_path.stem}: {len(gt)} gt, skipped (needs {K + 1})", flush=True)
            continue
        used += 1
        ex_idx = rng.sample(range(len(gt)), K)
        rest = np.delete(gt, ex_idx, axis=0)
        exemplars = gt[ex_idx]
        base = proc.set_image(im)
        image_only = {key: v for key, v in base["backbone_out"].items()}
        line = []
        for cond in CONDITIONS:
            st = {"backbone_out": dict(image_only), "original_height": h, "original_width": w}
            if cond is not None:
                st = proc.set_text_prompt(cond, st)
            for b in exemplars:
                cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
                st = proc.add_geometric_prompt([cx, cy, (b[2] - b[0]) / w, (b[3] - b[1]) / h], True, st)
            boxes = st["boxes"].cpu().numpy().astype(np.float64)
            scores = st["scores"].cpu().numpy().astype(np.float64)
            # Drop detections that are the exemplars themselves.
            if len(boxes):
                keep = iou(boxes, exemplars).max(axis=1) < 0.5
                boxes, scores = boxes[keep], scores[keep]
            for t in THRESHOLDS:
                sel = scores >= t
                tp = match(boxes[sel], scores[sel], rest)
                c = tallies[str(cond)][str(t)]
                c[0] += tp
                c[1] += int(sel.sum())
                c[2] += len(rest)
            line.append(f"{cond or 'visual'}={int((scores >= 0.5).sum())}")
        print(f"[{k + 1}/{len(frames)}] {vid} {img_path.stem}: gt {len(gt)} ({len(rest)} held out), " + ", ".join(line), flush=True)
    summary = {"frames_used": used, "k_exemplars": K, "iou": 0.5, "conditions": {}}
    for cond, by_t in tallies.items():
        summary["conditions"][cond] = {
            t: {"recall": round(tp / g, 4) if g else None, "precision": round(tp / n, 4) if n else None,
                "predictions": n, "held_out_gt": g}
            for t, (tp, n, g) in by_t.items()
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
