"""Detection NMS, boundary suppression and detection-to-track association on
synthetic masks: the oracle for st-model's tracker/associate.rs.

    tools/reference/.venv/bin/python -s tools/reference/assoc_oracle.py

Calls upstream's own functions (external/sam3 f1647e8, CPU branch):
`sam3_multiplex_detector_utils._nms_masks_core_batched` (IoM, as the builder
sets `det_nms_use_iom=True`), `Sam3VideoBase._suppress_detections_close_to_boundary`,
and `sam3_video_base._associate_det_trk_compilable` with the builder's thresholds
(use_iom_recondition=True, iom_thresh_recondition=0.5, assoc 0.1, trk assoc 0.5,
new det 0.65). Track masks are zero-padded to 16 as `_associate_det_trk` does, and
outputs are cut back to the real tracks. Masks are random blobs (some empty, some
nested, some duplicated) stored run-length encoded on `logit > 0` (alternating
runs, starting with a 0-run); scores include values exactly at the thresholds.
Writes tools/reference/assoc_cases.json (synthetic, committed).
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cpu_shims  # noqa: E402

cpu_shims.install()
from sam3.model.sam3_multiplex_detector_utils import _nms_masks_core_batched  # noqa: E402
from sam3.model.sam3_video_base import Sam3VideoBase, _associate_det_trk_compilable  # noqa: E402

OUT = pathlib.Path(__file__).parent / "assoc_cases.json"
H = W = 32
MAX_OBJ = 16
THRESH_SCORES = np.array([0.4, 0.5, 0.65, 0.8], dtype=np.float32)


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


def blob(rng):
    """Mask logits: an ellipse (sometimes empty or full), positive inside."""
    kind = rng.random()
    if kind < 0.08:
        return np.full((H, W), -1.0, np.float32)
    if kind < 0.12:
        return np.full((H, W), 1.0, np.float32)
    cy, cx = rng.uniform(0, H), rng.uniform(0, W)
    ry, rx = rng.uniform(1, H / 2), rng.uniform(1, W / 2)
    y, x = np.mgrid[0:H, 0:W]
    d = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    return (1.0 - d).astype(np.float32)


def related(rng, base):
    """A blob nested in, overlapping or equal to `base`."""
    r = rng.random()
    if r < 0.3:
        return base.copy()
    m = base.copy()
    if r < 0.65:
        m[rng.integers(0, H // 2) :, :] = -1.0  # a part of it
    else:
        m = np.maximum(m, blob(rng))  # it plus another blob
    return m


def scores(rng, n):
    s = rng.uniform(0, 1, n).astype(np.float32)
    at = rng.random(n) < 0.25
    s[at] = rng.choice(THRESH_SCORES, at.sum())
    return s


def main():
    rng = np.random.default_rng(20261026)
    nms_cases, bnd_cases, assoc_cases = [], [], []
    for _ in range(60):
        n = int(rng.integers(1, 14))  # upstream's batched NMS cannot reshape zero detections
        masks = []
        for i in range(n):
            masks.append(related(rng, masks[int(rng.integers(0, i))]) if i and rng.random() < 0.5 else blob(rng))
        probs = scores(rng, n)
        m = torch.from_numpy(np.stack(masks) if n else np.zeros((0, H, W), np.float32))
        keep = _nms_masks_core_batched(torch.from_numpy(probs)[None], m[None], 0.4, 0.1, nms_use_iom=True)[0]
        nms_cases.append({"masks": [rle(x > 0) for x in masks], "probs": [float(v) for v in probs], "keep": keep.tolist()})

    for _ in range(20):
        n = int(rng.integers(1, 10))
        boxes = rng.uniform(-0.05, 1.05, (n, 4)).astype(np.float32)
        boxes[:, [0, 2]] = np.sort(boxes[:, [0, 2]], axis=1)
        boxes[:, [1, 3]] = np.sort(boxes[:, [1, 3]], axis=1)
        boxes[0] = [0.0, 0.0, 0.05, 0.05]  # center exactly on the margin
        keep = Sam3VideoBase._suppress_detections_close_to_boundary(None, torch.from_numpy(boxes), margin=0.025)
        bnd_cases.append({"boxes": boxes.tolist(), "keep": keep.tolist()})

    for _ in range(80):
        nt = int(rng.integers(1, 9))
        nd = int(rng.integers(1, 12))
        trk = [blob(rng) for _ in range(nt)]
        det = [related(rng, trk[int(rng.integers(0, nt))]) if rng.random() < 0.6 else blob(rng) for _ in range(nd)]
        ds = scores(rng, nd)
        dk = rng.random(nd) < 0.8
        trk_pad = np.concatenate([np.stack(trk), np.zeros((MAX_OBJ - nt, H, W), np.float32)])
        r = _associate_det_trk_compilable(
            torch.from_numpy(np.stack(det)), torch.from_numpy(ds), torch.from_numpy(dk), torch.from_numpy(trk_pad),
            0.65, 0.5, 0.1, 0.8, True, False, 0.5, 0.5,
        )
        unm, nonempty, new, argmax, hc, hi, _keep, im = r
        assoc_cases.append({
            "det_masks": [rle(x > 0) for x in det], "det_scores": [float(v) for v in ds], "det_keep": dk.tolist(),
            "trk_masks": [rle(x > 0) for x in trk],
            "trk_is_unmatched": unm[:nt].tolist(), "trk_is_nonempty": nonempty[:nt].tolist(), "is_new_det": new.tolist(),
            "det_to_max_trk": argmax.tolist(), "det_is_high_conf": hc.tolist(), "det_is_high_overlap": hi.tolist(),
            "im_mask": im[:, :nt].reshape(-1).tolist(),
        })
    OUT.write_text(json.dumps({"h": H, "w": W, "nms": nms_cases, "boundary": bnd_cases, "assoc": assoc_cases}))
    n_new = sum(sum(c["is_new_det"]) for c in assoc_cases)
    n_unm = sum(sum(c["trk_is_unmatched"]) for c in assoc_cases)
    n_amb = sum(sum(c["det_is_high_conf"]) for c in assoc_cases)
    n_sup = sum(len(c["keep"]) - sum(c["keep"]) for c in nms_cases)
    print(f"{len(nms_cases)} NMS cases ({n_sup} not kept), {len(bnd_cases)} boundary, {len(assoc_cases)} association ({n_new} new dets, {n_unm} unmatched tracks, {n_amb} high-conf) -> {OUT.name}")


if __name__ == "__main__":
    main()
