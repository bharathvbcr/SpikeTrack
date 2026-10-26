"""Looser matching for the zero-shot result: is IoU 0.5 unfair to ~20 px boxes?

    tools/reference/.venv/bin/python -s tools/reference/zero_shot_diagnose.py

Reads bench/results/zero-shot/per_frame.json and re-scores each prompt at IoU
0.1/0.3/0.5 and by centre distance (prediction centre within 10 px of a GT
centre, one-to-one), and reports predicted box sizes against GT sizes.
"""

import json
import pathlib

import numpy as np
from PIL import Image

from zero_shot_check import TRAIN, gt_boxes, iou

REPO = pathlib.Path(__file__).resolve().parents[2]
RES = REPO / "bench/results/zero-shot"


def greedy(score, ok):
    """One-to-one greedy on a [pred, gt] score matrix where ok marks allowed pairs."""
    used_p, used_g, n = set(), set(), 0
    for flat in np.argsort(-score, axis=None):
        p, g = np.unravel_index(flat, score.shape)
        if not ok[p, g]:
            break
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        n += 1
    return n


def main():
    frames = json.loads((RES / "per_frame.json").read_text())
    prompts = list(frames[0]["prompts"])
    lines = ["# Zero-shot diagnosis: looser matching", "",
             "Score threshold 0.3; recall over all GT sperm (class 0) in the 50 frames.", "",
             "| prompt | preds | IoU>=0.5 | IoU>=0.3 | IoU>=0.1 | centre <=10 px | median pred w x h | median GT w x h |",
             "|---|---|---|---|---|---|---|---|"]
    for p in prompts:
        tot_gt, hits, sizes, gsz, npred = 0, {"0.5": 0, "0.3": 0, "0.1": 0, "c10": 0}, [], [], 0
        for f in frames:
            img = TRAIN / f["video"] / "images" / f"{f['frame']}.jpg"
            w, h = Image.open(img).size
            gt = gt_boxes(TRAIN / f["video"] / "labels" / f"{f['frame']}.txt", w, h)
            s = np.array(f["prompts"][p]["scores"])
            b = np.array(f["prompts"][p]["boxes"]).reshape(-1, 4)[s >= 0.3]
            tot_gt += len(gt)
            npred += len(b)
            gsz += (gt[:, 2:] - gt[:, :2]).tolist()
            sizes += (b[:, 2:] - b[:, :2]).tolist()
            if len(b) == 0 or len(gt) == 0:
                continue
            m = iou(b, gt)
            for t in ("0.5", "0.3", "0.1"):
                hits[t] += greedy(m, m >= float(t))
            pc = (b[:, :2] + b[:, 2:]) / 2
            gc = (gt[:, :2] + gt[:, 2:]) / 2
            d = np.linalg.norm(pc[:, None] - gc[None], axis=-1)
            hits["c10"] += greedy(-d, d <= 10)
        med = lambda xs: "x".join(f"{v:.0f}" for v in np.median(np.array(xs), 0)) if xs else "-"  # noqa: E731
        lines.append(f"| {p} | {npred} | " + " | ".join(f"{hits[k] / tot_gt:.3f}" for k in ("0.5", "0.3", "0.1", "c10"))
                     + f" | {med(sizes)} | {med(gsz)} |")
    (RES / "diagnosis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
