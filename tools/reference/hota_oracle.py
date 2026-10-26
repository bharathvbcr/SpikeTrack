"""TrackEval HOTA on synthetic box-tracking cases: the oracle for st-eval's HOTA.

    tools/reference/.venv/bin/python -s tools/reference/hota_oracle.py

Uses TrackEval 12c8791 (~/Code/research/external/TrackEval) directly: its HOTA
metric class on data dicts built the way its datasets build them (dense ids,
per-frame box IoU from its own `_calculate_box_ious`). Writes
tools/reference/hota_cases.json (synthetic, committed).
"""

import json
import pathlib
import sys

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # TrackEval predates numpy 1.24; np.float was builtin float (float64).
sys.path.insert(0, str(pathlib.Path("~/Code/research/external/TrackEval").expanduser()))
from trackeval.datasets._base_dataset import _BaseDataset  # noqa: E402
from trackeval.metrics.hota import HOTA  # noqa: E402

OUT = pathlib.Path(__file__).parent / "hota_cases.json"
FIELDS = ["HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA", "AssRe", "AssPr", "LocA", "HOTA", "DetA", "DetRe", "DetPr"]


def make_case(rng, frames, objects, fp_rate, fn_rate, switch_rate, jitter):
    """Ground truth: objects moving linearly; tracker: jittered copies with
    misses, false positives and id switches. Continuous jitter keeps the
    assignment free of ties."""
    pos = rng.uniform(0, 600, (objects, 2))
    vel = rng.normal(0, 4, (objects, 2))
    size = rng.uniform(12, 30, (objects, 2))
    tr_id = list(range(objects))
    next_id = objects
    case = []
    for t in range(frames):
        p = pos + vel * t
        gt = [[i, *p[i], *size[i]] for i in range(objects) if rng.random() > 0.05]
        pred = []
        for i in range(objects):
            if rng.random() < switch_rate:
                tr_id[i] = next_id
                next_id += 1
            if rng.random() < fn_rate:
                continue
            q = p[i] + rng.normal(0, jitter, 2)
            s = size[i] * rng.uniform(0.8, 1.2, 2)
            pred.append([tr_id[i], *q, *s])
        for _ in range(rng.poisson(fp_rate * objects)):
            pred.append([next_id, *rng.uniform(0, 600, 2), *rng.uniform(10, 30, 2)])
            next_id += 1
        case.append({"gt": gt, "pred": pred})
    return case


def trackeval_hota(case):
    gt_map, tr_map = {}, {}
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for f in case:
        g = np.array([gt_map.setdefault(b[0], len(gt_map)) for b in f["gt"]], dtype=int)
        p = np.array([tr_map.setdefault(b[0], len(tr_map)) for b in f["pred"]], dtype=int)
        gb = np.array([b[1:] for b in f["gt"]], dtype=float).reshape(-1, 4)
        pb = np.array([b[1:] for b in f["pred"]], dtype=float).reshape(-1, 4)
        sim = _BaseDataset._calculate_box_ious(gb, pb, box_format="xywh")
        data["gt_ids"].append(g)
        data["tracker_ids"].append(p)
        data["similarity_scores"].append(sim)
    data["num_timesteps"] = len(case)
    data["num_gt_ids"] = len(gt_map)
    data["num_tracker_ids"] = len(tr_map)
    data["num_gt_dets"] = sum(len(f["gt"]) for f in case)
    data["num_tracker_dets"] = sum(len(f["pred"]) for f in case)
    return HOTA().eval_sequence(data)


def main():
    rng = np.random.default_rng(20261026)
    configs = [
        dict(frames=30, objects=5, fp_rate=0.0, fn_rate=0.0, switch_rate=0.0, jitter=0.5),
        dict(frames=40, objects=8, fp_rate=0.1, fn_rate=0.1, switch_rate=0.02, jitter=2.0),
        dict(frames=60, objects=16, fp_rate=0.2, fn_rate=0.2, switch_rate=0.05, jitter=4.0),
        dict(frames=50, objects=30, fp_rate=0.3, fn_rate=0.3, switch_rate=0.1, jitter=6.0),
        dict(frames=20, objects=3, fp_rate=1.0, fn_rate=0.5, switch_rate=0.3, jitter=8.0),
    ]
    cases, seq_res = [], []
    for cfg in configs:
        case = make_case(rng, **cfg)
        res = trackeval_hota(case)
        seq_res.append(res)
        cases.append({"config": cfg, "frames": case, "expected": {k: np.asarray(res[k]).tolist() for k in FIELDS}})
        print(cfg, "HOTA", round(float(np.mean(res["HOTA"])) * 100, 3))
    comb = HOTA().combine_sequences({str(i): r for i, r in enumerate(seq_res)})
    OUT.write_text(json.dumps({
        "source": "TrackEval 12c8791 trackeval/metrics/hota.py, numpy " + np.__version__,
        "alphas": np.arange(0.05, 0.99, 0.05).tolist(),
        "box_format": "xywh (top-left x, y, width, height); ids are per case",
        "cases": cases,
        "combined": {k: np.asarray(comb[k]).tolist() for k in FIELDS},
    }) + "\n")
    print("combined HOTA", round(float(np.mean(comb["HOTA"])) * 100, 3), "->", OUT)


if __name__ == "__main__":
    main()
