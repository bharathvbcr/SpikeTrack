"""E25/E26 summary: steady whole-frame time per CoreML placement arm and round, and each arm's tracks
against the default arm's in the same round.

    python3 -I tools/analysis/e25_frame.py bench/results/e25-coreml-frame 36 [ARM ...]

The first arm is the reference (default: E25's arms). Two times per run:
- push_frame: the median steady (frame >= 8) `push_frame` ms from the log;
- wall: (total_seconds - model_load_seconds) / frames from run.json, which also counts what runs
  outside push_frame (decode, the next frame's trunk queued by --trunk-pipeline, the collector).
An arm whose run failed is listed as failed and left out of the ratios.
"""

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ab_coreml_tracker import log_frames  # noqa: E402
from e24_mattn import iou  # noqa: E402

E25_ARMS = ("default", "necks32", "mattn16", "fast16")
ROUNDS = (1, 2, 3)


def boxes(d, vid):
    data = json.loads((d / f"{vid}.json").read_text())
    return {fr["frame"]: {o["id"]: o["box_xyxy_norm"] for o in fr["objects"]} for fr in data["frames"]}


def wall_ms(d, vid):
    r = json.loads((d / f"{vid}.run.json").read_text())
    return 1e3 * (r["total_seconds"] - r["model_load_seconds"]) / r["frames"]


def run_times(out, arm, r, vid):
    """(push_frame median ms, wall ms) or None when the run failed."""
    d = out / f"{arm}-r{r}"
    if not (d / f"{vid}.run.json").is_file():
        return None
    return statistics.median(ms for ms, _ in log_frames(out / f"{arm}-r{r}.log")), wall_ms(d, vid)


def main():
    out, vid = pathlib.Path(sys.argv[1]), sys.argv[2]
    arms = tuple(sys.argv[3:]) or E25_ARMS
    ref = arms[0]
    t = {(a, r): run_times(out, a, r, vid) for r in ROUNDS for a in arms}
    if any(t[(ref, r)] is None for r in ROUNDS):
        raise SystemExit(f"the reference arm {ref} failed in a round")
    fmt = lambda v, i: "failed" if v is None else f"{v[i]:.0f}"
    ratio = lambda a, r, i: "-" if t[(a, r)] is None else f"{t[(a, r)][i] / t[(ref, r)][i]:.3f}"
    lines = [f"# Whole-frame ms, VISEM {vid}, steady frames >= 8: push_frame median / wall per frame", "",
             "| round | " + " | ".join(arms) + " | " + " | ".join(f"{a} / {ref} (push, wall)" for a in arms[1:]) + " |",
             "|---|" + "---|" * (2 * len(arms) - 1)]
    for r in ROUNDS:
        lines.append(f"| {r} | " + " | ".join(f"{fmt(t[(a, r)], 0)} / {fmt(t[(a, r)], 1)}" for a in arms) + " | "
                     + " | ".join(f"{ratio(a, r, 0)}, {ratio(a, r, 1)}" for a in arms[1:]) + " |")
    summary = []
    for a in arms[1:]:
        ok = [r for r in ROUNDS if t[(a, r)] is not None]
        if not ok:
            summary.append(f"{a} failed")
            continue
        push = [t[(a, r)][0] / t[(ref, r)][0] for r in ok]
        wall = [t[(a, r)][1] / t[(ref, r)][1] for r in ok]
        summary.append(f"{a} push {statistics.median(push):.3f} (max {max(push):.3f}), wall {statistics.median(wall):.3f} "
                       f"(max {max(wall):.3f}) over {len(ok)} round(s)")
    lines += ["", "Median (and worst) over rounds of the per-round ratio: " + "; ".join(summary), ""]
    lines += [f"## Tracks against {ref} (same round)", "",
              "| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |",
              "|---|---|---|---|---|---|"]
    for r in ROUNDS:
        ref_dir = out / f"{ref}-r{r}"
        refb = boxes(ref_dir, vid)
        for a in arms[1:]:
            d = out / f"{a}-r{r}"
            if t[(a, r)] is None:
                lines.append(f"| {r} | {a} | failed | | | |")
                continue
            got = boxes(d, vid)
            eq = sum(set(refb[f]) == set(got.get(f, {})) for f in refb)
            ious = [iou(refb[f][i], got[f][i]) for f in refb for i in refb[f] if f in got and i in got[f]]
            same = (d / f"{vid}.json").read_bytes() == (ref_dir / f"{vid}.json").read_bytes()
            lines.append(f"| {r} | {a} | {eq}/{len(refb)} | {statistics.mean(ious):.4f} | {min(ious):.4f} | {'yes' if same else 'no'} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
