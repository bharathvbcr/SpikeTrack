"""Positive control for the CPU reference detector: upstream's own test images
with prompts that must fire, plus a box-size dump for one VISEM frame.

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/positive_control.py
"""

import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402
from zero_shot_check import TRAIN, gt_boxes, iou  # noqa: E402

SAM3 = pathlib.Path("~/Code/research/external/sam3/assets/images").expanduser()


def main():
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_ref.use_fp32_mlp()
    model, _, _ = sam3_ref.build_reference()
    proc = Sam3Processor(model.detector, device=torch.device("cpu"), confidence_threshold=0.3)
    for img, prompts in (("truck.jpg", ["truck", "wheel", "zebra"]), ("groceries.jpg", ["banana", "bottle", "airplane"])):
        im = Image.open(SAM3 / img).convert("RGB")
        st = proc.set_image(im)
        for p in prompts:
            o = proc.set_text_prompt(p, dict(st))
            s = o["scores"].numpy()
            print(f"{img} '{p}': {len(s)} detections, top scores {np.round(np.sort(s)[::-1][:5], 3).tolist()}", flush=True)
    vid, frame = "14", "14_frame_466"
    im = Image.open(TRAIN / vid / "images" / f"{frame}.jpg").convert("RGB")
    gt = gt_boxes(TRAIN / vid / "labels" / f"{frame}.txt", *im.size)
    st = proc.set_image(im)
    print(f"VISEM {frame} {im.size}: gt boxes w,h = {np.round(gt[:, 2:] - gt[:, :2], 1).tolist()}")
    for p in ("sperm", "sperm cell", "cell", "sperm head", "small particle"):
        o = proc.set_text_prompt(p, dict(st))
        b = o["boxes"].numpy()
        s = o["scores"].numpy()
        best = iou(b.astype(np.float64), gt).max(axis=1).round(3).tolist() if len(b) and len(gt) else []
        print(f"  '{p}': {len(s)} dets; w,h {np.round(b[:, 2:] - b[:, :2], 1).tolist()[:8]}; best IoU vs gt {best[:8]}", flush=True)


if __name__ == "__main__":
    main()
