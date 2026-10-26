"""Upstream SAM 3.1 multiplex tracker on DAVIS 2017 val clips, semi-supervised:
ground-truth masks on frame 0, then propagation. Scored with the official
davis2017-evaluation J and F. This is the reference the Rust tracker's J&F gate
compares against.

    tools/locked.sh vos bash tools/jobs/ref.sh tools/reference/vos_reference.py [--frames N]

Frames go through upstream's own loader (io_utils.load_video_frames: PIL
bilinear resize to 1008, float16 storage, mean/std 0.5). Writes
bench/results/vos-reference/<clip>.json (per-frame, per-object J and F) and
summary.json, plus the predicted masks as PNGs under fixtures/vos/<clip>/ (git-ignored).
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

sys.path.insert(0, str(pathlib.Path("~/Code/research/external/davis2017-evaluation").expanduser()))
from davis2017.metrics import db_eval_iou, f_measure  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DAVIS = REPO / "data/davis/DAVIS"
OUT = REPO / "bench/results/vos-reference"
MASKS = REPO / "fixtures/vos"
CLIPS = [l.strip() for l in (pathlib.Path(__file__).parent / "davis_clips.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="limit frames per clip (0 = all)")
    args = ap.parse_args()
    from sam3.model import io_utils

    model, missing, unexpected = sam3_ref.build_tracker_vos()
    print(f"tracker built: missing {len(missing)} {missing[:5]}, unexpected {len(unexpected)} {unexpected[:5]}", flush=True)
    assert not unexpected, unexpected
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"clips": {}, "loader": "io_utils.load_video_frames (PIL bilinear, float16 storage)", "missing_keys": missing}
    for clip in CLIPS:
        img_dir = DAVIS / "JPEGImages/480p" / clip
        ann = sorted((DAVIS / "Annotations/480p" / clip).glob("*.png"))
        images, H, W = io_utils.load_video_frames(str(img_dir), 1008, offload_video_to_cpu=True)
        n = len(images) if not args.frames else min(args.frames, len(images))
        gt = [np.array(Image.open(p)) for p in ann[:n]]
        obj_ids = sorted(int(v) for v in np.unique(gt[0]) if v != 0)
        with torch.inference_mode():
            state = model.init_state(video_height=H, video_width=W, num_frames=n)
            state["images"] = images[:n]
            masks0 = torch.from_numpy(np.stack([gt[0] == o for o in obj_ids])).float()
            model.add_new_masks(state, frame_idx=0, obj_ids=obj_ids, masks=masks0)
            model.propagate_in_video_preflight(state, run_mem_encoder=True)
            preds = {}
            t0 = time.time()
            for fi, ids, _low, video_res in model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False, tqdm_disable=True):
                preds[fi] = {int(o): (video_res[i, 0] > 0).cpu().numpy() for i, o in enumerate(ids)}
                if fi % 10 == 0:
                    print(f"{clip} frame {fi}/{n} ({time.time() - t0:.0f} s)", flush=True)
            secs = time.time() - t0
        # DAVIS protocol: skip the first (given) and last frame.
        frames = list(range(1, n - 1))
        per_obj = {}
        for o in obj_ids:
            j = [float(db_eval_iou(gt[f] == o, preds[f][o])) for f in frames]
            fm = [float(f_measure(preds[f][o], gt[f] == o)) for f in frames]
            per_obj[o] = {"J": float(np.mean(j)), "F": float(np.mean(fm)), "J_frames": j, "F_frames": fm}
        jm = float(np.mean([v["J"] for v in per_obj.values()]))
        fmm = float(np.mean([v["F"] for v in per_obj.values()]))
        res = {"clip": clip, "frames": n, "objects": obj_ids, "J": jm, "F": fmm, "JF": (jm + fmm) / 2,
               "seconds_cpu": round(secs, 1), "per_object": per_obj}
        (OUT / f"{clip}.json").write_text(json.dumps(res))
        d = MASKS / clip
        d.mkdir(parents=True, exist_ok=True)
        for f, m in preds.items():
            lab = np.zeros((H, W), np.uint8)
            for o, mo in m.items():
                lab[mo] = o
            Image.fromarray(lab).save(d / f"{f:05d}.png")
        summary["clips"][clip] = {k: res[k] for k in ("frames", "objects", "J", "F", "JF", "seconds_cpu")}
        print(f"{clip}: J {jm:.4f} F {fmm:.4f} J&F {(jm + fmm) / 2:.4f} over {len(frames)} frames, {secs:.0f} s", flush=True)
    vals = list(summary["clips"].values())
    summary["mean_JF"] = float(np.mean([v["JF"] for v in vals]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("mean J&F", round(summary["mean_JF"], 4))


if __name__ == "__main__":
    main()
