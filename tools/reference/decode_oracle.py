"""Pillow's decode of a few VISEM-Tracking JPEG frames: the oracle for st-video's ImageIO decode.

Writes fixtures/decode/<video>_frame_<n>.rgb (raw 8-bit RGB, row-major, no header) and
fixtures/decode/manifest.json (size, SHA-256 of the source JPEG and of the RGB, the JPEG's chroma
subsampling, whether it carries an ICC profile, and the Pillow / libjpeg versions). fixtures/ is
git-ignored: nothing here is data to commit.

    tools/reference/.venv/bin/python -s tools/reference/decode_oracle.py

Read by crates/st-video/tests/decode.rs. Light: one small JPEG decode per frame, no lock needed.
"""

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, JpegImagePlugin, features

ROOT = Path(__file__).resolve().parents[2]
VISEM = ROOT / "data" / "visem" / "VISEM_Tracking_Train_v4" / "Train"
OUT = ROOT / "fixtures" / "decode"

# (video, frame): spread over videos and over time within a video.
FRAMES = [("11", 0), ("11", 700), ("22", 0), ("22", 300), ("47", 300), ("52", 1000)]

SAMPLING = {-1: "unknown", 0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for video, n in FRAMES:
        src = VISEM / video / "images" / f"{video}_frame_{n}.jpg"
        if not src.is_file():
            print(f"missing {src}", file=sys.stderr)
            return 1
        raw = src.read_bytes()
        with Image.open(src) as im:
            if im.format != "JPEG":
                print(f"{src}: {im.format}, expected JPEG", file=sys.stderr)
                return 1
            sampling = SAMPLING.get(JpegImagePlugin.get_sampling(im), "unknown")
            icc = im.info.get("icc_profile") is not None
            mode = im.mode
            rgb = im.convert("RGB")
            w, h = rgb.size
            data = rgb.tobytes()
        name = f"{video}_frame_{n}.rgb"
        (OUT / name).write_bytes(data)
        entries.append(
            {
                "file": name,
                "jpeg": str(src.relative_to(ROOT)),
                "width": w,
                "height": h,
                "jpeg_mode": mode,
                "subsampling": sampling,
                "icc_profile": icc,
                "jpeg_sha256": hashlib.sha256(raw).hexdigest(),
                "rgb_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        print(f"{name}: {w}x{h} {mode} {sampling} icc={icc}")
    manifest = {
        "pillow": Image.__version__,
        "libjpeg": features.version("jpg"),
        "libjpeg_turbo": features.check_feature("libjpeg_turbo"),
        "command": "tools/reference/.venv/bin/python -s tools/reference/decode_oracle.py",
        "frames": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(entries)} frames and manifest.json to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
