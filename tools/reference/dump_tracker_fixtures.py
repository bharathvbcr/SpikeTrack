"""Tracker fixtures (gp-tracker-multiplex): module boundaries of upstream's
multiplex VOS path on the first frames of a DAVIS clip, fp32 except where
upstream itself rounds (memories are stored in bf16).

    tools/locked.sh trk-fixtures bash tools/jobs/ref.sh tools/reference/dump_tracker_fixtures.py [--clip judo --frames 8]

Records, per call, every tensor argument, keyword argument and output of:
the memory attention (`transformer.encoder`), the multiplex mask decoder
(`sam_mask_decoder`), the interactive mask decoder (frame 0's mask-as-output
path), the memory encoder (`maskmem_backbone`), plus each frame's stored
output (`maskmem_features` as stored, `maskmem_pos_enc`, `image_features`,
`obj_ptr`, `object_score_logits`, `pred_masks`) and the frame inputs and
backbone features the tracker consumed. Writes fixtures/tracker/<clip>.safetensors
and fixtures/TRACKER_MANIFEST.json.
"""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DAVIS = REPO / "data/davis/DAVIS"
OUT = REPO / "fixtures/tracker"


def keep(x):
    if x.dtype == torch.bfloat16:
        return x.detach().contiguous().clone()  # stored as bf16: keep the exact bits
    if x.is_floating_point():
        return x.detach().float().contiguous().clone()
    return x.detach().contiguous().clone()


def flatten(prefix, obj, rec):
    """Every tensor inside nested tuples/lists/dicts, by path."""
    if torch.is_tensor(obj):
        rec[prefix] = keep(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}", v, rec)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flatten(f"{prefix}.{i}", v, rec)
    elif hasattr(obj, "tensors") and torch.is_tensor(getattr(obj, "tensors")):
        rec[f"{prefix}.tensors"] = keep(obj.tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="judo")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    from sam3.model import io_utils

    model, missing, unexpected = sam3_ref.build_tracker_vos()
    assert not unexpected, unexpected
    rec, calls, frame_of = {}, {}, {"f": -1}

    def hook(name):
        def fn(_m, a, kw, out):
            i = calls.get(name, 0)
            calls[name] = i + 1
            p = f"f{frame_of['f']}.{name}.call{i}"
            flatten(f"{p}.arg", list(a), rec)
            flatten(f"{p}.kw", kw, rec)
            flatten(f"{p}.out", out, rec)
        return fn

    hooks = [
        model.transformer.encoder.register_forward_hook(hook("memattn"), with_kwargs=True),
        model.sam_mask_decoder.register_forward_hook(hook("mask_decoder"), with_kwargs=True),
        model.interactive_sam_mask_decoder.register_forward_hook(hook("interactive_mask_decoder"), with_kwargs=True),
        model.maskmem_backbone.register_forward_hook(hook("memenc"), with_kwargs=True),
    ]
    img_dir = DAVIS / "JPEGImages/480p" / args.clip
    ann = sorted((DAVIS / "Annotations/480p" / args.clip).glob("*.png"))
    images, H, W = io_utils.load_video_frames(str(img_dir), 1008, offload_video_to_cpu=True)
    n = min(args.frames, len(images))
    gt0 = np.array(Image.open(ann[0]))
    obj_ids = sorted(int(v) for v in np.unique(gt0) if v != 0)
    for f in range(n):
        rec[f"f{f}.image"] = keep(images[f].float())  # as the tracker sees it (float16 storage -> float)
    with torch.inference_mode():
        state = model.init_state(video_height=H, video_width=W, num_frames=n)
        state["images"] = images[:n]
        masks0 = torch.from_numpy(np.stack([gt0 == o for o in obj_ids])).float()
        rec["f0.input_masks"] = masks0.contiguous()
        frame_of["f"] = 0
        model.add_new_masks(state, frame_idx=0, obj_ids=obj_ids, masks=masks0)
        model.propagate_in_video_preflight(state, run_mem_encoder=True)
        # Hook calls so far (add_new_masks, preflight) belong to frame 0. Each loop
        # step computes frame fi (frame 0, the conditioning frame, computes nothing
        # here) and then yields it, so the next compute is frame fi + 1.
        for fi, ids, low, video_res in model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=n, reverse=False, tqdm_disable=True):
            rec[f"f{fi}.out.low_res_masks"] = keep(low)
            rec[f"f{fi}.out.video_res_masks"] = keep(video_res)
            frame_of["f"] = fi + 1
        for kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for fi, out in state["output_dict"][kind].items():
                flatten(f"f{fi}.stored.{kind}", {k: v for k, v in out.items() if k != "conditioning_objects"}, rec)
        ms = state["multiplex_state"]
        rec["multiplex.valid_object_mask"] = keep(ms.get_valid_object_mask())
    for h in hooks:
        h.remove()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.clip}.safetensors"
    save_file(rec, str(path))
    manifest = {"clip": args.clip, "frames": n, "objects": obj_ids, "video_hw": [H, W], "file": str(path.relative_to(REPO)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "calls_per_module": calls,
                "missing_keys": missing, "num_buckets": int(ms.num_buckets), "multiplex_count": int(ms.multiplex_count),
                "tensors": {k: [list(v.shape), str(v.dtype)] for k, v in rec.items()}}
    (REPO / "fixtures/TRACKER_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{args.clip}: {n} frames, {len(obj_ids)} objects, {len(rec)} tensors, {path.stat().st_size / 1e6:.0f} MB; calls {calls}")


if __name__ == "__main__":
    main()
