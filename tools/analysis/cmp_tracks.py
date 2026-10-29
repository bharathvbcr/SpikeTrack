"""Two `spiketrack track` prediction files side by side: per-frame id sets, and the largest box and
score difference on the ids both have (throwaway analysis; nothing ships).

    python3 tools/analysis/cmp_tracks.py A/22.json B/22.json
"""
import json
import sys


def load(p):
    return {fr["frame"]: {o["id"]: o for o in fr["objects"]} for fr in json.load(open(p))["frames"]}


a, b = load(sys.argv[1]), load(sys.argv[2])
id_diff = [f for f in a if set(a[f]) != set(b.get(f, {}))]
box = score = 0.0
for f in a:
    for i in set(a[f]) & set(b.get(f, {})):
        box = max(box, max(abs(x - y) for x, y in zip(a[f][i]["box_xyxy_norm"], b[f][i]["box_xyxy_norm"])))
        score = max(score, abs(a[f][i]["score"] - b[f][i]["score"]))
print(f"{sys.argv[2]}: frames {len(a)}/{len(b)}, frames with different ids {len(id_diff)} {id_diff[:10]}, "
      f"max |box| {box:.2e}, max |score| {score:.2e}")
