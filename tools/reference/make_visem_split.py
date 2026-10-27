"""Write tools/reference/visem_split.json: the 15 train / 5 held-out VISEM-Tracking videos.

    python3 tools/reference/make_visem_split.py

The split is by video, never by frame (consecutive 50 fps frames are near duplicates, so a frame
split would leak). The rule is deterministic and uses no RNG:

1. For each of the 20 annotated videos, count class-0 (sperm) boxes per labelled frame
   (`labels/*.txt`, YOLO format) -- the density that decides how hard a video is.
2. Sort ascending by that density (ties by numeric id) and cut into 5 consecutive groups of 4.
3. In each group, hold out the video whose sha256("spiketrack-lora-split-v1:<id>") hex digest is
   smallest. The other 15 are train.

Stratifying by density keeps sparse (3 boxes/frame) and crowded (60 boxes/frame) videos on both
sides. Leakage check (see the JSON's `fixture_videos`): the existing fixtures and gates run the
base model and only compare Rust against PyTorch; none trains anything, so no video is excluded
from the held-out set.
"""

import hashlib
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN = REPO / "data/visem/VISEM_Tracking_Train_v4/Train"
OUT = pathlib.Path(__file__).resolve().parent / "visem_split.json"
SALT = "spiketrack-lora-split-v1:"
GROUPS = 5


def density(video):
    labels = sorted((TRAIN / video / "labels").glob("*.txt"))
    boxes = 0
    for lab in labels:
        for line in lab.read_text().split("\n"):
            parts = line.split()
            if len(parts) == 5 and int(parts[0]) == 0:
                boxes += 1
    return boxes / len(labels), len(labels)


def main():
    vids = sorted((p.name for p in TRAIN.iterdir() if p.is_dir()), key=int)
    assert len(vids) == 20, vids
    stats = {v: density(v) for v in vids}
    order = sorted(vids, key=lambda v: (stats[v][0], int(v)))
    size = len(order) // GROUPS
    held_out, groups = [], []
    for g in range(GROUPS):
        group = order[g * size:(g + 1) * size]
        pick = min(group, key=lambda v: hashlib.sha256((SALT + v).encode()).hexdigest())
        held_out.append(pick)
        groups.append({"videos": group, "held_out": pick})
    train = [v for v in vids if v not in held_out]
    assert len(train) == 15 and len(held_out) == 5 and not set(train) & set(held_out)
    doc = {
        "rule": (
            "By video, never by frame. Sort the 20 VISEM-Tracking videos ascending by class-0 boxes "
            "per labelled frame (ties by numeric id), cut into 5 consecutive groups of 4, and hold out "
            "from each group the video whose sha256('spiketrack-lora-split-v1:<id>') hex digest is "
            "smallest. The other 15 train. No RNG: tools/reference/make_visem_split.py regenerates this file."
        ),
        "salt": SALT,
        "train": sorted(train, key=int),
        "held_out": sorted(held_out, key=int),
        "groups": groups,
        "class0_boxes_per_labelled_frame": {v: round(stats[v][0], 2) for v in order},
        "labelled_frames": {v: stats[v][1] for v in vids},
        "fixture_videos": {
            "videos": ["11", "22", "47", "82"],
            "leakage": (
                "None, so no video is excluded from held-out. Backbone fixtures (22, 47, 82), the detector "
                "detail fixture (22), the D3 gate (20 frames cycling through all 20 videos) and the "
                "detect_track reference (11) all run the BASE model and check Rust against PyTorch parity; they "
                "never train. The LoRA adapter is evaluated only on the held-out videos' frames."
            ),
        },
        "command": "python3 tools/reference/make_visem_split.py",
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print("train   ", doc["train"])
    print("held_out", doc["held_out"])


if __name__ == "__main__":
    main()
