"""DAVIS J and F on synthetic masks: the oracle for st-eval's jf.rs.

    tools/reference/.venv/bin/python -s tools/reference/jf_oracle.py

Calls davis2017-evaluation ac7c43f's own `db_eval_iou` and `f_measure`
(~/Code/research/external/davis2017-evaluation) on random blob masks,
including empty, full, border-touching and void cases. Masks are stored
run-length encoded (alternating run lengths, starting with a 0-run) in
tools/reference/jf_cases.json (synthetic, committed).
"""

import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path("~/Code/research/external/davis2017-evaluation").expanduser()))
from davis2017.metrics import db_eval_iou, f_measure  # noqa: E402

OUT = pathlib.Path(__file__).parent / "jf_cases.json"


def rle(m):
    flat = m.astype(np.uint8).ravel()
    runs, cur, n = [], 0, 0
    for v in flat:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    return runs


def blobs(rng, h, w, k):
    m = np.zeros((h, w), np.uint8)
    for _ in range(k):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(2, max(3, w // 4))), int(rng.integers(2, max(3, h // 4))))
        cv2.ellipse(m, c, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return m.astype(bool)


def perturb(rng, m):
    k = np.ones((3, 3), np.uint8)
    out = m.astype(np.uint8)
    op = rng.integers(0, 3)
    if op == 0:
        out = cv2.dilate(out, k, iterations=int(rng.integers(1, 4)))
    elif op == 1:
        out = cv2.erode(out, k, iterations=int(rng.integers(1, 3)))
    shift = rng.integers(-3, 4, 2)
    out = np.roll(out, tuple(int(s) for s in shift), axis=(0, 1))
    return out.astype(bool)


def main():
    rng = np.random.default_rng(20261026)
    cases = []
    sizes = [(480, 854), (100, 160), (37, 53), (8, 8), (1, 12)]
    for h, w in sizes:
        for i in range(8):
            gt = blobs(rng, h, w, int(rng.integers(1, 5)))
            seg = perturb(rng, gt) if i % 4 else blobs(rng, h, w, 2)
            void = blobs(rng, h, w, 1) if i % 3 == 0 else None
            cases.append((h, w, gt, seg, void))
        z, o = np.zeros((h, w), bool), np.ones((h, w), bool)
        cases += [(h, w, z, z, None), (h, w, z, blobs(rng, h, w, 1), None), (h, w, blobs(rng, h, w, 1), z, None), (h, w, o, o, None), (h, w, o, blobs(rng, h, w, 2), None)]
    out = []
    for h, w, gt, seg, void in cases:
        j = float(db_eval_iou(gt, seg, void))
        f = float(f_measure(seg, gt, void))
        out.append({"h": h, "w": w, "gt": rle(gt), "seg": rle(seg), "void": rle(void) if void is not None else None, "j": j, "f": f})
    OUT.write_text(json.dumps({"source": "davis2017-evaluation ac7c43f metrics.py (db_eval_iou, f_measure, bound_th 0.008)",
                               "rle": "alternating run lengths over the row-major mask, first run is of zeros",
                               "cases": out}) + "\n")
    print(len(out), "cases; mean J", np.mean([c["j"] for c in out]).round(4), "mean F", np.mean([c["f"] for c in out]).round(4), "->", OUT)


if __name__ == "__main__":
    main()
