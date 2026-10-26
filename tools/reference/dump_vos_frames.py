"""Inputs and references for the Rust VOS gate (gp-tracker-multiplex, D5).

    tools/locked.sh vos-frames bash tools/jobs/ref.sh tools/reference/dump_vos_frames.py

Per DAVIS clip in davis_clips.txt, fixtures/vos/<clip>.safetensors (git-ignored) holds:
- `frames`: upstream's preprocessed frames exactly as its tracker stores them
  (io_utils.load_video_frames: PIL bilinear to 1008, float16, mean/std 0.5), F16 [T, 3, 1008, 1008];
- `gt`: DAVIS ground-truth labels, U8 [T, H, W] (0 = background, k = object k);
- `upstream`: upstream's predicted labels from vos_reference.py (fixtures/vos/<clip>/*.png), U8 [T, H, W].
So the Rust side decodes no image. Run after vos_reference.py.
"""

import json
import pathlib

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

REPO = pathlib.Path(__file__).resolve().parents[2]
DAVIS = REPO / "data/davis/DAVIS"
VOS = REPO / "fixtures/vos"
CLIPS = [l.strip() for l in (pathlib.Path(__file__).parent / "davis_clips.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]


def main():
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import sam3_ref  # noqa: F401  (installs the CPU shims before sam3 imports)
    from sam3.model import io_utils

    manifest = {}
    for clip in CLIPS:
        images, H, W = io_utils.load_video_frames(str(DAVIS / "JPEGImages/480p" / clip), 1008, offload_video_to_cpu=True)
        assert images.dtype == torch.float16, images.dtype
        gt = np.stack([np.array(Image.open(p)) for p in sorted((DAVIS / "Annotations/480p" / clip).glob("*.png"))])
        ups = sorted((VOS / clip).glob("*.png"))
        assert len(ups) == len(images), f"{clip}: {len(ups)} upstream masks for {len(images)} frames (run vos_reference.py)"
        upstream = np.stack([np.array(Image.open(p)) for p in ups])
        out = VOS / f"{clip}.safetensors"
        save_file({"frames": images.contiguous(), "gt": torch.from_numpy(gt.astype(np.uint8)), "upstream": torch.from_numpy(upstream.astype(np.uint8))}, str(out))
        manifest[clip] = {"frames": len(images), "hw": [H, W], "objects": sorted(int(v) for v in np.unique(gt[0]) if v), "file": str(out.relative_to(REPO))}
        print(clip, manifest[clip], f"{out.stat().st_size / 1e6:.0f} MB", flush=True)
    (REPO / "fixtures/VOS_MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
