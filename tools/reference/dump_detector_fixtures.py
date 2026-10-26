"""Detector fixtures (gp-detector-and-mask-head): module boundaries of SAM 3.1's
image grounding path, fp32, from upstream's own Sam3Processor.

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_detector_fixtures.py

Detail cases (one file each, fixtures/detector/<case>.safetensors): the
detector's inputs as upstream computed them (FPN levels and position
encodings from the backbone, the resized text features and mask), then the
geometry encoder output, the prompt, every fusion-encoder layer, every
decoder layer (queries, reference boxes, presence), the scores, the
segmentation head (prompt cross-attention, pixel embedding) and the final
outputs. Gate cases (fixtures/detector/gate.safetensors): final outputs only
for 20 VISEM frames, for the D3 mask-IoU gate.
"""

import hashlib
import json
import pathlib
import random
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "fixtures/detector"
SAM3 = pathlib.Path("~/Code/research/external/sam3/assets/images").expanduser()
TRAIN = REPO / "data/visem/VISEM_Tracking_Train_v4/Train"
DETAIL = [
    ("visem22_cell", TRAIN / "22/images/22_frame_0.jpg", "cell"),
    ("testimage_shoe", SAM3 / "test_image.jpg", "shoe"),
    ("truck_wheel", SAM3 / "truck.jpg", "wheel"),
]
GATE_PROMPT = "cell"
GATE_FRAMES = 20
KEEP_MASKS = 0.3  # masks stored for queries whose final probability passes this


def f32(x):
    return x.detach().float().contiguous().clone()


def instrument(det, rec):
    hooks = []
    tr = det.transformer

    def put(name):
        def hook(_m, _inp, out):
            if isinstance(out, tuple):
                for i, o in enumerate(out):
                    if torch.is_tensor(o):
                        rec[f"{name}.{i}"] = f32(o)
            elif isinstance(out, dict):
                for k, v in out.items():
                    if torch.is_tensor(v):
                        rec[f"{name}.{k}"] = f32(v)
            elif torch.is_tensor(out):
                rec[name] = f32(out)
        return hook

    hooks.append(det.geometry_encoder.register_forward_hook(put("geo")))
    for i, lay in enumerate(tr.encoder.layers):
        hooks.append(lay.register_forward_hook(put(f"enc.layer{i}")))
    hooks.append(tr.encoder.register_forward_hook(put("enc")))
    for i, lay in enumerate(tr.decoder.layers):
        hooks.append(lay.register_forward_hook(put(f"dec.layer{i}")))
    hooks.append(tr.decoder.register_forward_hook(put("dec")))
    hooks.append(det.dot_prod_scoring.register_forward_hook(put("scoring")))
    seg = det.segmentation_head
    hooks.append(seg.cross_attend_prompt.register_forward_hook(put("seg.cross_attend_prompt")))
    hooks.append(seg.pixel_decoder.register_forward_hook(put("seg.pixel_embed")))
    hooks.append(seg.register_forward_hook(put("seg")))
    return lambda: [h.remove() for h in hooks]


def run(proc, det, img, prompt, detail):
    seen = {}
    pre = det.backbone.vision_backbone.register_forward_pre_hook(
        lambda _m, args: seen.__setitem__("image", f32(args[0]))
    )
    state = proc.set_image(img)
    pre.remove()
    bo = state["backbone_out"]
    rec = {"in.image": seen["image"]}  # upstream's preprocessed [1, 3, 1008, 1008]
    if detail:
        for i, (f, p) in enumerate(zip(bo["backbone_fpn"], bo["vision_pos_enc"])):
            rec[f"in.fpn{i}"] = f32(getattr(f, "tensors", f))
            rec[f"in.pos{i}"] = f32(p)
    restore = instrument(det, rec) if detail else (lambda: None)
    st = proc.set_text_prompt(prompt, dict(state))
    restore()
    if detail:
        rec["in.text"] = f32(st["backbone_out"]["language_features"])  # [32, 1, 256] seq first
        rec["in.text_pad"] = st["backbone_out"]["language_mask"].to(torch.uint8).contiguous()  # [1, 32], 1 = pad
    return st, rec


def final_outputs(det, proc, st, prefix, rec, keep_all_masks):
    """Recompute the raw outputs the processor thresholds, from the stored state."""
    out = det.forward_grounding(
        backbone_out=st["backbone_out"], find_input=proc.find_stage,
        geometric_prompt=st["geometric_prompt"], find_target=None,
    )
    logits = out["pred_logits"][0, :, 0]
    presence = out["presence_logit_dec"][0]
    prob = logits.sigmoid() * presence.sigmoid()
    rec[f"{prefix}.pred_logits"] = f32(logits)
    rec[f"{prefix}.presence_logit_dec"] = f32(presence.reshape(1))
    rec[f"{prefix}.pred_boxes"] = f32(out["pred_boxes"][0])
    keep = torch.nonzero(prob > KEEP_MASKS).flatten()
    if keep_all_masks:
        rec[f"{prefix}.pred_masks"] = f32(out["pred_masks"][0])
    rec[f"{prefix}.kept_queries"] = keep.to(torch.int64).contiguous()
    rec[f"{prefix}.kept_masks"] = f32(out["pred_masks"][0, keep])
    return prob, keep


def main():
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_ref.use_fp32_mlp()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not missing and not unexpected
    det = model.detector
    proc = Sam3Processor(det, device=torch.device("cpu"), confidence_threshold=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"command": "SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_detector_fixtures.py",
                "dtype_policy": "fp32; ViT MLP in fp32 (sam3_ref.use_fp32_mlp)", "files": []}
    with torch.inference_mode():
        for name, path, prompt in DETAIL:
            st, rec = run(proc, det, Image.open(path).convert("RGB"), prompt, detail=True)
            prob, keep = final_outputs(det, proc, st, "out", rec, keep_all_masks=True)
            f = OUT / f"{name}.safetensors"
            save_file(rec, str(f))
            manifest["files"].append({"file": str(f.relative_to(REPO)), "image": str(path), "prompt": prompt,
                                      "detections_at_0.5": int((prob > 0.5).sum()), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                      "tensors": {k: list(v.shape) for k, v in rec.items()}})
            print(f"{name}: {len(rec)} tensors, {f.stat().st_size / 1e6:.0f} MB, {(prob > 0.5).sum().item()} detections at 0.5", flush=True)
        rng = random.Random(20261027)
        vids = sorted(p.name for p in TRAIN.iterdir() if p.is_dir())
        gate, frames = {}, []
        for i in range(GATE_FRAMES):
            v = vids[i % len(vids)]
            imgs = sorted((TRAIN / v / "images").glob("*.jpg"))
            path = rng.choice(imgs)
            st, rec = run(proc, det, Image.open(path).convert("RGB"), GATE_PROMPT, detail=False)
            # End to end: the Rust side runs backbone, text and detector from the image.
            gate[f"f{i}.in.image"] = rec["in.image"]
            prob, keep = final_outputs(det, proc, st, f"f{i}.out", gate, keep_all_masks=False)
            frames.append({"index": i, "image": str(path.relative_to(REPO)), "kept": int(len(keep))})
            print(f"gate {i}: {path.name}, {len(keep)} queries above {KEEP_MASKS}", flush=True)
        f = OUT / "gate.safetensors"
        save_file(gate, str(f))
        manifest["gate"] = {"file": str(f.relative_to(REPO)), "prompt": GATE_PROMPT, "keep_masks_above": KEEP_MASKS,
                            "frames": frames, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (REPO / "fixtures/DETECTOR_MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
