"""E24 summary: steady tracker step per --mem-attn arm and round, plus how far each CoreML arm's
tracks are from the Ojas arm's in the same round (id sets per frame, box IoU of shared ids).

    python3 -I tools/analysis/e24_mattn.py bench/results/e24-coreml-mattn
"""

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ab_coreml_tracker import spiketrack_round  # noqa: E402

ARMS = ("ojas", "coreml-fp32", "coreml-fp16")
ROUNDS = (1, 2, 3)


def boxes(d):
    """{frame: {id: [x0, y0, x1, y1]}} from a track run's 82.json."""
    data = json.loads((d / "82.json").read_text())
    out = {}
    for fr in data["frames"]:
        out[fr["frame"]] = {o["id"]: o["box_xyxy_norm"] for o in fr["objects"]}
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 1.0


def main():
    out = pathlib.Path(sys.argv[1])
    lines = ["# E24 summary: --mem-attn arms, VISEM 82, 26 frames, 16 objects (steady frames >= 8)", ""]
    lines += ["| round | " + " | ".join(f"{a} tracker ms (median)" for a in ARMS) + " | fp32 / ojas | fp16 / ojas |",
              "|---|" + "---|" * (len(ARMS) + 2)]
    med = {}
    for r in ROUNDS:
        row = []
        for a in ARMS:
            st = spiketrack_round(out / f"{a}-r{r}")
            med[(a, r)] = statistics.median(st["tracker_ms"])
            row.append(f"{med[(a, r)]:.1f}")
        lines.append(f"| {r} | " + " | ".join(row)
                     + f" | {med[('coreml-fp32', r)] / med[('ojas', r)]:.3f} | {med[('coreml-fp16', r)] / med[('ojas', r)]:.3f} |")
    lines += ["", "## Tracks against the Ojas arm (same round)", "",
              "| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU |", "|---|---|---|---|---|"]
    for r in ROUNDS:
        ref = boxes(out / f"ojas-r{r}")
        for a in ARMS[1:]:
            got = boxes(out / f"{a}-r{r}")
            eq = sum(set(ref[f]) == set(got.get(f, {})) for f in ref)
            ious = [iou(ref[f][i], got[f][i]) for f in ref for i in ref[f] if f in got and i in got[f]]
            lines.append(f"| {r} | {a} | {eq}/{len(ref)} | {statistics.mean(ious):.4f} | {min(ious):.4f} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
