"""Bucket `spiketrack track` errors against VISEM-Tracking labels (offline analysis, never shipped).

    python3 bench/results/weight-space/diag-train5/bucket_errors.py RUN_DIR VIDEO...

Per frame, predictions are matched greedily (score order) to class-0 (sperm) label boxes at
IoU >= 0.5, as in the README's box Prec/Rec. Each unmatched prediction gets the first bucket it fits:
  duplicate      IoU >= 0.1 with a class-0 box that another prediction already matched
  localization   IoU in [0.1, 0.5) with an unmatched class-0 box
  concept        IoU >= 0.1 with a class-1 (cluster) or class-2 (small/pinhead) box
  background     none of these
Each unmatched class-0 box: `near-miss` if some prediction overlaps it at IoU in [0.1, 0.5), else
`missed`. Labels are YOLO (class, cx, cy, w, h), normalized; predictions are box_xyxy_norm.

Geometry: for every matched pair and every near-miss (its best-overlapping prediction), the log size
ratios ln(gt_w/pred_w), ln(gt_h/pred_h) and the center offset in units of the label's size. At head
sizes (w ~ 0.02) `--post-box-delta`'s logit shift on w is close to a log-ratio shift, so the median
log ratio is the dw/dh that would center the size error.
"""

import json
import math
import pathlib
import statistics
import sys
from collections import Counter

VISEM = pathlib.Path("data/visem/VISEM_Tracking_Train_v4/Train")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def labels(video, frame):
    path = VISEM / video / "labels" / f"{video}_frame_{frame}.txt"
    out = []
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def geom(p, g):
    pw, ph = p[2] - p[0], p[3] - p[1]
    gw, gh = g[2] - g[0], g[3] - g[1]
    if min(pw, ph, gw, gh) <= 0:
        return None
    return (math.log(gw / pw), math.log(gh / ph),
            ((g[0] + g[2]) - (p[0] + p[2])) / 2 / gw, ((g[1] + g[3]) - (p[1] + p[3])) / 2 / gh)


def bucket(run, video, min_frame, geoms=None):
    pred = json.loads((run / f"{video}.json").read_text())
    fp, fn = Counter(), Counter()
    tp = 0
    fp_scores = {}
    if geoms is None:
        geoms = {"matched": [], "near-miss": []}
    for fr in pred["frames"]:
        f = fr["frame"]
        if f < min_frame:
            continue
        gt = labels(video, f)
        if gt is None:
            continue
        heads = [b for c, b in gt if c == 0]
        others = [b for c, b in gt if c != 0]
        objs = sorted(fr["objects"], key=lambda o: -o.get("score", 0.0))
        matched = [False] * len(heads)
        unmatched = []
        for o in objs:
            b = o["box_xyxy_norm"]
            best, bi = 0.0, -1
            for i, h in enumerate(heads):
                if not matched[i]:
                    v = iou(b, h)
                    if v > best:
                        best, bi = v, i
            if best >= 0.5:
                matched[bi] = True
                tp += 1
                g = geom(b, heads[bi])
                if g:
                    geoms["matched"].append(g)
            else:
                unmatched.append(o)
        for o in unmatched:
            b = o["box_xyxy_norm"]
            if any(matched[i] and iou(b, h) >= 0.1 for i, h in enumerate(heads)):
                k = "duplicate"
            elif any(not matched[i] and iou(b, h) >= 0.1 for i, h in enumerate(heads)):
                k = "localization"
            elif any(iou(b, x) >= 0.1 for x in others):
                k = "concept"
            else:
                k = "background"
            fp[k] += 1
            fp_scores.setdefault(k, []).append(o.get("score", 0.0))
        for i, h in enumerate(heads):
            if not matched[i]:
                near = [o["box_xyxy_norm"] for o in objs if 0.1 <= iou(o["box_xyxy_norm"], h) < 0.5]
                fn["near-miss" if near else "missed"] += 1
                if near:
                    g = geom(max(near, key=lambda b: iou(b, h)), h)
                    if g:
                        geoms["near-miss"].append(g)
    return tp, fp, fn, fp_scores


def main():
    run = pathlib.Path(sys.argv[1])
    videos = sys.argv[2:]
    for min_frame in (0, 15):
        total_tp, total_fp, total_fn = 0, Counter(), Counter()
        geoms = {"matched": [], "near-miss": []}
        print(f"== frames >= {min_frame}")
        for v in videos:
            tp, fp, fn, _ = bucket(run, v, min_frame, geoms)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            print(f"video {v}: TP {tp}  FP {dict(fp)}  FN {dict(fn)}")
        nfp, nfn = sum(total_fp.values()), sum(total_fn.values())
        print(f"all: TP {total_tp}, FP {nfp}, FN {nfn}; precision {total_tp / max(1, total_tp + nfp):.3f}, "
              f"recall {total_tp / max(1, total_tp + nfn):.3f}")
        for k, n in total_fp.most_common():
            print(f"  FP {k:13s} {n:5d}  {n / max(1, nfp):.1%}")
        for k, n in total_fn.most_common():
            print(f"  FN {k:13s} {n:5d}  {n / max(1, nfn):.1%}")
        for k, rows in geoms.items():
            if len(rows) < 2:
                print(f"  geom {k}: n {len(rows)}")
                continue
            cols = list(zip(*rows))
            q = [statistics.quantiles(c, n=4) for c in cols]
            print(f"  geom {k:9s} n {len(rows):5d}  median [IQR]  ln(gt_w/pred_w) {q[0][1]:+.3f} "
                  f"[{q[0][0]:+.3f},{q[0][2]:+.3f}]  ln(gt_h/pred_h) {q[1][1]:+.3f} "
                  f"[{q[1][0]:+.3f},{q[1][2]:+.3f}]  dcx/gt_w {q[2][1]:+.3f}  dcy/gt_h {q[3][1]:+.3f}")


if __name__ == "__main__":
    main()
