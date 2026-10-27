"""Cache what the frozen part of SAM 3.1's detector hands to the LoRA-trained part.

    tools/locked.sh lora-cache bash tools/jobs/lora_cache.sh        # the job script wraps this
    SAM3_DEVICE=mps tools/reference/.venv/bin/python -s tools/reference/cache_detector_inputs.py \
        --device mps --frames-per-video 64 --heldout-frames-per-video 24

Why this is enough to train from (read from upstream, f1647e8):
* The fusion encoder and decoder read only the LAST backbone level: `Sam3Image._get_img_feats` takes
  `backbone_fpn[-num_feature_levels:]` and `vision_pos_enc[-num_feature_levels:]`, and the detector is
  built with `num_feature_levels=1` (sam3_ref.py, model_builder.py:1156, encoder `num_feature_levels=1`).
  With the multiplex tri-neck's scales (4, 2, 1) that level is 72x72 x 256. The other two levels feed
  only the segmentation head, which a box + class + presence loss never reads.
* The text path gives `language_features` [32, 1, 256] and `language_mask` [1, 32] per prompt.
* The geometry encoder's output depends on the image only through frozen weights; the trainer
  recomputes it (a cls token over the cached features) instead of caching a second tensor.
So per frame: one [256, 72, 72] fp16 tensor (2.65 MB). Position encodings are identical for every
frame (asserted) and stored once. The text encoder runs on the CPU (once per prompt); the vision
backbone runs on `--device`.

Output (git-ignored, fixtures/lora_cache/): `meta.json`, `pos.safetensors`, `text.safetensors`,
`frames/<video>.safetensors` with `f<frame>` tensors. Videos come from tools/reference/visem_split.json
(train: --frames-per-video, held-out: --heldout-frames-per-video, evenly spaced with seeded jitter, by
video). `--verify N` re-runs upstream's full `forward_grounding` (segmentation head included) on N cached
frames and compares it with the trainer's head-only path on the fp16-rounded cache.
"""

import argparse
import gc
import json
import os
import pathlib
import random
import sys
import time


def _device_from_argv():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--device="):
            return a.split("=", 1)[1]
    return "mps"


os.environ["SAM3_DEVICE"] = _device_from_argv()  # upstream reads this at import time

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lora_common as lc  # noqa: E402

DEFAULT_OUT = lc.REPO / "fixtures/lora_cache"
FP16_MAX = 65504.0
FEAT_MB = 256 * 72 * 72 * 2 / 2**20


def pick_frames(video, n, seed):
    """n evenly spaced labelled frames: one per equal stratum, position jittered by a seeded RNG."""
    labels = sorted((lc.VISEM / video / "labels").glob("*.txt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    labels = [p for p in labels if (lc.VISEM / video / "images" / (p.stem + ".jpg")).exists()]
    rng = random.Random(f"{seed}:{video}")
    n = min(n, len(labels))
    picks = []
    for i in range(n):
        lo, hi = i * len(labels) // n, (i + 1) * len(labels) // n
        picks.append(labels[lo + int(rng.random() * (hi - lo))])
    return picks


# Tolerances for head_agreement, set from measurements on the cache (2026-10-27):
# fp16 rounding of the cached features alone (relative 2^-11, which is all the cache changes) moves the joint
# probability of ANY query by at most 0.027 (frame 11/35 "cell", 12 draws), the logit of queries above p = 0.1 by
# at most 0.11 and their box by 0.005. CPU and MPS heads agree to 2e-3 in logit on identical inputs, and on the
# fp32 fixture (fixtures/detector/visem22_cell) the head-only path matches upstream to 2.5e-4. Low-score queries
# (p < 0.05) can move 0.5 in logit and 0.1 in box: DETR's iterative box refinement amplifies a perturbation for a
# query nothing downstream uses, and a logit of -3 to -6 is 0.2% to 5% probability, so the logit says little there.
MAX_PROB_DIFF = 0.05       # any query, joint probability: a threshold at 0.3/0.4/0.5 can only flip inside this band
MAX_LOGIT_DIFF_HIGH = 0.3  # queries with probability > 0.1 on either side
MAX_BOX_DIFF_HIGH = 0.02   # same queries, normalized cxcywh
MAX_PRESENCE_LOGIT_DIFF = 0.01


def head_agreement(full, ours):
    """Compare upstream's full forward_grounding with the head-only path on what the pipeline uses."""
    lf, lo = full["pred_logits"][0, :, 0], ours["pred_logits"][0, :, 0]
    pf, po = lf.sigmoid(), lo.sigmoid()
    dl = (lf - lo).abs()
    db = (full["pred_boxes"][0] - ours["pred_boxes"][0]).abs().amax(-1)
    hi = (pf > 0.1) | (po > 0.1)
    d_pres = (full["presence_logit_dec"] - ours["presence_logit_dec"]).abs().max().item()
    row = {
        "max_abs_prob_all_queries": (pf - po).abs().max().item(),
        "queries_above_0.1": int(hi.sum()),
        "max_abs_logit_above_0.1": dl[hi].max().item() if hi.any() else 0.0,
        "max_abs_box_above_0.1": db[hi].max().item() if hi.any() else 0.0,
        "max_abs_presence_logit": d_pres,
        "info_max_abs_logit_all_queries": dl.max().item(),
        "info_max_abs_box_all_queries": db.max().item(),
        "kept_flips_at_0.3": int(((pf > 0.3) != (po > 0.3)).sum()),
        "kept_flips_at_0.5": int(((pf > 0.5) != (po > 0.5)).sum()),
        "n_above_0.5_full": int((pf * full["presence_logit_dec"].sigmoid()[0] > 0.5).sum()),
        "n_above_0.5_ours": int((po * ours["presence_logit_dec"].sigmoid()[0] > 0.5).sum()),
    }
    row["ok"] = bool(row["max_abs_prob_all_queries"] < MAX_PROB_DIFF and row["max_abs_logit_above_0.1"] < MAX_LOGIT_DIFF_HIGH
                     and row["max_abs_box_above_0.1"] < MAX_BOX_DIFF_HIGH and d_pres < MAX_PRESENCE_LOGIT_DIFF)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--frames-per-video", type=int, default=64, help="training videos")
    ap.add_argument("--heldout-frames-per-video", type=int, default=24)
    ap.add_argument("--videos", default="", help="comma-separated subset of the split (smoke runs)")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--verify", type=int, default=2, help="frames to compare against upstream's full forward_grounding")
    ap.add_argument("--max-gb", type=float, default=15.0)
    ap.add_argument("--force", action="store_true", help="recompute videos that are already cached")
    args = ap.parse_args()

    split = lc.load_split()
    wanted = [v for v in args.videos.split(",") if v] or split["train"] + split["held_out"]
    unknown = [v for v in wanted if v not in split["train"] + split["held_out"]]
    assert not unknown, f"not in visem_split.json: {unknown}"
    plan = {v: pick_frames(v, args.frames_per_video if v in split["train"] else args.heldout_frames_per_video, args.seed) for v in wanted}
    n_frames = sum(len(p) for p in plan.values())
    est = n_frames * FEAT_MB / 1024
    print(f"{len(plan)} videos, {n_frames} frames, about {est:.2f} GB of fp16 features", flush=True)
    assert est <= args.max_gb, f"{est:.1f} GB exceeds --max-gb {args.max_gb}"

    from torchvision.transforms import v2

    import sam3_ref
    from sam3.model.sam3_image_processor import Sam3Processor

    dev = torch.device(args.device)
    t0 = time.time()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not missing and not unexpected, (missing[:3], unexpected[:3])
    det = model.detector
    del model
    gc.collect()
    det.eval()
    for p in det.parameters():
        p.requires_grad_(False)
    print(f"reference built in {time.time() - t0:.0f}s", flush=True)

    # Text encoder: CPU, once per prompt, exactly the fp32 reference path.
    text = {}
    with torch.no_grad():
        for p in lc.ALL_PROMPTS:
            o = det.backbone.forward_text([p], device=torch.device("cpu"))
            feat, mask = o["language_features"], o["language_mask"]
            assert feat.shape == (32, 1, lc.DIM) and mask.shape == (1, 32), (feat.shape, mask.shape)
            text[p] = (feat[:, 0].float().contiguous(), mask[0].to(torch.uint8).contiguous())
    print(f"text features for {len(text)} prompts", flush=True)

    lang = det.backbone.language_backbone  # stays on the CPU: it is not used again
    det.backbone.language_backbone = torch.nn.Identity()
    det.to(dev)
    det.backbone.language_backbone = lang
    proc = Sam3Processor(det, device=dev, confidence_threshold=0.5)
    out = args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    tensors = {}
    for i, p in enumerate(lc.ALL_PROMPTS):
        tensors[f"p{i}.feat"], tensors[f"p{i}.mask"] = text[p]
    save_file(tensors, str(out / "text.safetensors"))

    pos_ref = None
    old_meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    if (out / "pos.safetensors").exists():
        with safe_open(str(out / "pos.safetensors"), "pt") as pf:
            pos_ref = pf.get_tensor("pos")[None]
    frames_meta, stats, skipped = [], {"max_abs": 0.0, "fp16_roundtrip_max_abs_err": 0.0}, []
    verify_pool = []
    t_all = time.time()
    computed = False
    for v, picks in plan.items():
        shard = out / "frames" / f"{v}.safetensors"
        split_name = "train" if v in split["train"] else "held_out"
        rec_v = []
        tensors = {}
        for k, lab in enumerate(picks):
            frame = int(lab.stem.rsplit("_", 1)[1])
            img_path = lc.VISEM / v / "images" / (lab.stem + ".jpg")
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                rec_v.append({"video": v, "frame": frame, "split": split_name, "key": f"f{frame}",
                              "image": str(img_path.relative_to(lc.REPO)), "label": str(lab.relative_to(lc.REPO)), "w": w, "h": h})
                if shard.exists() and not args.force:
                    if v not in skipped:
                        skipped.append(v)
                    if len(verify_pool) < args.verify:  # a resumed run still verifies, from the cached fp16 features
                        with safe_open(str(shard), "pt") as sf:
                            verify_pool.append((rec_v[-1], sf.get_tensor(f"f{frame}").float()[None]))
                    continue
                t0 = time.time()
                x = proc.transform(v2.functional.to_image(im)).unsqueeze(0).to(dev)  # resize on the CPU: the reference numerics
            with torch.no_grad():  # not inference_mode: the cache is saved and later fed to autograd
                fo = det.backbone.forward_image(x, need_sam3_out=True, need_interactive_out=False, need_propagation_out=False)
                fpn, pos = fo["backbone_fpn"], fo["vision_pos_enc"]
                assert len(fpn) == 3 and len(pos) == 3, (len(fpn), len(pos))
                feat = fpn[-1].tensors
                assert tuple(feat.shape) == (1, lc.DIM, 72, 72) and tuple(pos[-1].shape) == (1, lc.DIM, 72, 72), (feat.shape, pos[-1].shape)
                feat_c, pos_c = feat.float().cpu(), pos[-1].float().cpu()
            peak = feat_c.abs().max().item()
            assert torch.isfinite(feat_c).all() and peak < FP16_MAX, f"{v} frame {frame}: max |feature| {peak}"
            half = feat_c.half()
            stats["max_abs"] = max(stats["max_abs"], peak)
            stats["fp16_roundtrip_max_abs_err"] = max(stats["fp16_roundtrip_max_abs_err"], (half.float() - feat_c).abs().max().item())
            if args.force or pos_ref is None:
                pos_ref = pos_c
            else:
                assert torch.equal(pos_ref, pos_c), "vision_pos_enc differs between frames"
            computed = True
            tensors[f"f{frame}"] = half[0].contiguous()
            if len(verify_pool) < args.verify:
                verify_pool.append((rec_v[-1], half.float()))
            print(f"[{v}] frame {frame} ({k + 1}/{len(picks)}) {time.time() - t0:.1f}s", flush=True)
        if tensors:
            tmp = shard.with_suffix(".tmp")
            save_file(tensors, str(tmp))
            tmp.rename(shard)  # a killed run never leaves a half-written shard that a resume would trust
        frames_meta += rec_v
    if computed:
        save_file({"pos": pos_ref[0].contiguous()}, str(out / "pos.safetensors"))
    elif old_meta.get("feature_stats"):
        stats = old_meta["feature_stats"]  # a pure resume keeps the statistics of the run that computed the features

    def write_meta(verify):
        meta = {
            "command": " ".join(sys.argv), "device": args.device, "seed": args.seed, "dtype": "float16",
            "frames_per_video": args.frames_per_video, "heldout_frames_per_video": args.heldout_frames_per_video,
            "videos": list(plan), "split_file": "tools/reference/visem_split.json",
            "prompts": lc.ALL_PROMPTS, "feature_shape": [lc.DIM, 72, 72], "text_shape": [32, lc.DIM],
            "levels_cached": "last of 3 (72x72); encoder/decoder use backbone_fpn[-1] only (num_feature_levels=1)",
            "frames": frames_meta, "feature_stats": stats, "videos_already_cached_and_skipped": skipped,
            "verify_vs_upstream_forward_grounding": verify, "seconds": round(time.time() - t_all, 1),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

    write_meta([])
    verify = []
    if verify_pool and pos_ref is not None:
        heads = lc.build_heads(device=args.device)
        from sam3.model.data_misc import FindStage

        for rec, feat in verify_pool:
            image = proc.transform(v2.functional.to_image(Image.open(lc.REPO / rec["image"]).convert("RGB"))).unsqueeze(0).to(dev)
            row = {"video": rec["video"], "frame": rec["frame"]}
            with torch.no_grad():
                bo = det.backbone.forward_image(image)
                for prompt in ("cell", "sperm"):
                    tfeat, tmask = text[prompt][0][:, None].to(dev), text[prompt][1][None].bool().to(dev)
                    fs = FindStage(img_ids=torch.zeros(1, dtype=torch.long, device=dev), text_ids=torch.zeros(1, dtype=torch.long, device=dev),
                                   input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
                    full = det.forward_grounding(backbone_out={**bo, "language_features": tfeat, "language_mask": tmask}, find_input=fs,
                                                 geometric_prompt=det._get_dummy_prompt(), find_target=None)
                    ours = lc.grounding_forward(heads, feat.to(dev), pos_ref.to(dev), tfeat, tmask, aux=False)
                    row[prompt] = head_agreement(full, ours)
                    assert row[prompt]["ok"], f"head-only path disagrees with upstream: {row[prompt]}"
            verify.append(row)
            print("verify", json.dumps(row), flush=True)
    write_meta(verify)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**30
    print(f"done: {len(frames_meta)} frames, {size:.2f} GB on disk, max |feature| {stats['max_abs']:.1f}, "
          f"fp16 round-trip err {stats['fp16_roundtrip_max_abs_err']:.2e}", flush=True)


if __name__ == "__main__":
    main()
