"""Pillow BILINEAR resize cases: the oracle for st-video's resize.

    tools/reference/.venv/bin/python -s tools/reference/resize_oracle.py

Small random RGB images (up- and downscaling, odd sizes, 1-pixel edges) are
stored in full in tools/reference/resize_cases.json (committed). Real frames
(a VISEM and a DAVIS frame, decoded by PIL) are stored as raw RGB bytes in
fixtures/resize/ (git-ignored), with the SHA-256 of PIL's 1008x1008 output.
"""

import base64
import hashlib
import json
import pathlib

import numpy as np
import PIL
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).parent / "resize_cases.json"
FIX = REPO / "fixtures/resize"


def main():
    rng = np.random.default_rng(20261026)
    small = []
    for (w, h), (ow, oh) in [((13, 7), (40, 21)), ((64, 48), (17, 9)), ((5, 5), (5, 11)), ((9, 1), (30, 3)),
                             ((1, 1), (4, 4)), ((31, 29), (1008 // 16, 1008 // 16)), ((100, 60), (33, 77)), ((50, 50), (50, 50))]:
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        r = np.asarray(Image.fromarray(a).resize((ow, oh), Image.BILINEAR))
        small.append({"w": w, "h": h, "ow": ow, "oh": oh,
                      "src": base64.b64encode(a.tobytes()).decode(), "out": base64.b64encode(r.tobytes()).decode()})
    FIX.mkdir(parents=True, exist_ok=True)
    real = []
    for name, p in [("visem22_f0", REPO / "data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg"),
                    ("davis_judo_0", REPO / "data/davis/DAVIS/JPEGImages/480p/judo/00000.jpg")]:
        im = Image.open(p).convert("RGB")
        a = np.asarray(im)
        (FIX / f"{name}.rgb").write_bytes(a.tobytes())
        r = np.asarray(im.resize((1008, 1008), Image.BILINEAR))
        real.append({"name": name, "w": im.width, "h": im.height, "file": f"fixtures/resize/{name}.rgb",
                     "src_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                     "out_sha256": hashlib.sha256(r.tobytes()).hexdigest()})
    OUT.write_text(json.dumps({"pillow": PIL.__version__, "filter": "BILINEAR", "small": small, "real": real}) + "\n")
    print(len(small), "small and", len(real), "real cases ->", OUT)


if __name__ == "__main__":
    main()
