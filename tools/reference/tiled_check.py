"""Is zero-shot SAM 3.1's miss on VISEM a resolution problem? (training-free)

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/tiled_check.py

Same 50 frames and scoring as zero_shot_check.py, but each 640x480 frame is
cut into a 2x2 grid of overlapping tiles (360x280, stride 280x200). Each tile
goes through upstream's Sam3Processor, which resizes it to 1008x1008, so
sperm cover about 2.2x more patches per side than in the whole frame. Boxes are
mapped back to frame coordinates and merged with class-agnostic NMS at IoU 0.5.
Compare with bench/results/zero-shot/ (whole frame).
"""

import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402
from zero_shot_check import THRESHOLDS, gt_boxes, iou, match, sample_frames  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "bench/results/zero-shot-tiled"
PROMPTS = ["sperm", "sperm cell", "cell"]
TILE = (360, 280)
STRIDE = (280, 200)


def tiles(w, h):
    xs = list(range(0, max(1, w - TILE[0]) + 1, STRIDE[0]))
    ys = list(range(0, max(1, h - TILE[1]) + 1, STRIDE[1]))
    if xs[-1] + TILE[0] < w:
        xs.append(w - TILE[0])
    if ys[-1] + TILE[1] < h:
        ys.append(h - TILE[1])
    return [(x, y) for y in ys for x in xs]


def nms(boxes, scores, thr=0.5):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        o = iou(boxes[[i]], boxes[order[1:]])[0]
        order = order[1:][o < thr]
    return np.array(keep, dtype=int)


def main():
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_ref.use_fp32_mlp()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not missing and not unexpected
    proc = Sam3Processor(model.detector, device=torch.device("cpu"), confidence_threshold=min(THRESHOLDS))
    frames = sample_frames(10, 5)
    OUT.mkdir(parents=True, exist_ok=True)
    tallies = {p: {str(t): [0, 0, 0] for t in THRESHOLDS} for p in PROMPTS}
    per_frame = []
    t_all = time.time()
    for k, (vid, img_path, lab) in enumerate(frames):
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        gt = gt_boxes(lab, w, h)
        got = {p: ([], []) for p in PROMPTS}
        t0 = time.time()
        for x0, y0 in tiles(w, h):
            st = proc.set_image(im.crop((x0, y0, x0 + TILE[0], y0 + TILE[1])))
            for p in PROMPTS:
                o = proc.set_text_prompt(p, dict(st))
                b = o["boxes"].cpu().numpy().astype(np.float64) + np.array([x0, y0, x0, y0])
                got[p][0].append(b.reshape(-1, 4))
                got[p][1].append(o["scores"].cpu().numpy().astype(np.float64))
        rec = {"video": vid, "frame": img_path.stem, "gt": int(len(gt)), "prompts": {}}
        for p in PROMPTS:
            b, s = np.concatenate(got[p][0]), np.concatenate(got[p][1])
            if len(b):
                keep = nms(b, s)
                b, s = b[keep], s[keep]
            rec["prompts"][p] = {"boxes": b.round(1).tolist(), "scores": s.round(4).tolist()}
            for t in THRESHOLDS:
                sel = s >= t
                c = tallies[p][str(t)]
                c[0] += match(b[sel], s[sel], gt)
                c[1] += int(sel.sum())
                c[2] += len(gt)
        per_frame.append(rec)
        print(f"[{k + 1}/{len(frames)}] {vid} {img_path.stem}: gt {len(gt)}, "
              + ", ".join(f"{p}={len(rec['prompts'][p]['scores'])}" for p in PROMPTS) + f" ({time.time() - t0:.0f} s)", flush=True)
    summary = {"frames": len(per_frame), "tile": TILE, "stride": STRIDE, "nms_iou": 0.5, "iou": 0.5,
               "seconds_total": round(time.time() - t_all, 1), "prompts": {}}
    for p, by_t in tallies.items():
        summary["prompts"][p] = {t: {"recall": round(tp / g, 4) if g else None, "precision": round(tp / n, 4) if n else None,
                                     "predictions": n, "gt": g} for t, (tp, n, g) in by_t.items()}
    (OUT / "per_frame.json").write_text(json.dumps(per_frame))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
