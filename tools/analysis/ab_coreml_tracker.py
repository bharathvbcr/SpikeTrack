"""Summarise tools/jobs/ab_coreml_tracker.sh (throwaway analysis; nothing ships): the tracker
step of SpikeTrack against the CoreML port's `track_step` at 16 objects, per round, steady
frames only, and the decision pre-stated in bench/results/ab-coreml-tracker/README.md.

    python3 -I tools/analysis/ab_coreml_tracker.py bench/results/ab-coreml-tracker

SpikeTrack's tracker step per frame = the stages "tracker propagate, state 0[...]" + "memory
encode[...]" (wall clock, device synced at each stage boundary); its backbone = "trunk" +
"propagation neck". The port's step is one `track_step` call on one state; timed call i is the
state's step i + 2 (two untimed calls first). Steady = frame / step >= STEADY_FROM.
"""
import json
import pathlib
import statistics
import sys

STEADY_FROM = 8
ROUNDS = (1, 2, 3)
STAGE_RUN_TOLERANCE = 0.10


def stage_rows(md):
    rows, inside = [], False
    for line in md.read_text().splitlines():
        if line == "```tsv":
            inside = True
            continue
        if inside and line == "```":
            break
        if inside and line != "frame\tstage\tms":
            f, stage, ms = line.split("\t")
            rows.append((int(f), stage, float(ms)))
    if not rows:
        raise SystemExit(f"{md}: no per-frame tsv block (built before track_profile wrote one?)")
    return rows


def spiketrack_round(d):
    rows = stage_rows(d / "82.profile-stages.md")
    if any("state 1" in s for _, s, _ in rows):
        raise SystemExit(f"{d}: a second tracker state ran; the 16-object comparison needs one")
    frames = sorted({f for f, _, _ in rows if f >= STEADY_FROM})
    if not frames:
        raise SystemExit(f"{d}: no frame >= {STEADY_FROM}")
    per = {f: {"tracker": 0.0, "backbone": 0.0, "in_session": 0.0, "births": 0.0} for f in frames}
    seen = {f: set() for f in frames}
    for f, s, ms in rows:
        if f not in per:
            continue
        p = per[f]
        for part in ("tracker propagate", "memory encode"):
            if s.startswith(part):
                p["tracker"] += ms
                seen[f].add(part)
        if s in ("trunk", "propagation neck"):
            p["backbone"] += ms
            seen[f].add(s)
        if s.startswith("births"):
            p["births"] += ms
        if s not in ("decode + upload", "progress line + collector"):
            p["in_session"] += ms
    need = {"tracker propagate", "memory encode", "trunk", "propagation neck"}
    for f in frames:
        if seen[f] != need:
            # A renamed stage would otherwise drop out of the sum and fail open.
            raise SystemExit(f"{d}: frame {f} lacks stage(s) {sorted(need - seen[f])}")
    run = json.loads((d / "82.run.json").read_text())
    return {
        "frames": frames,
        "tracker_ms": [per[f]["tracker"] for f in frames],
        "backbone_ms": [per[f]["backbone"] for f in frames],
        "in_session_ms": [per[f]["in_session"] for f in frames],
        "frames_with_births": sum(per[f]["births"] > 1.0 for f in frames),
        "tracks": run.get("tracks"),
        "max_objects": run.get("max_objects"),
        "trunk_precision": run.get("trunk_precision"),
    }


def log_frames(log):
    """Steady (push_frame ms, active tracks) per frame from a `track` log
    ("frame N (i/T): X s, K active track(s), ...")."""
    out = {}
    for line in log.read_text().splitlines():
        if line.startswith("frame "):
            head, rest = line.split(": ", 1)
            secs, active = rest.split(", ")[:2]
            out[int(head.split()[1])] = (1e3 * float(secs.removesuffix(" s")), int(active.split()[0]))
    steady = [v for f, v in sorted(out.items()) if f >= STEADY_FROM]
    if not steady:
        raise SystemExit(f"{log}: no frame line >= {STEADY_FROM}")
    return steady


def coreml_steady(path, model):
    r = json.loads(path.read_text())[model]
    if "error" in r:
        raise SystemExit(f"{path}: {model} failed: {r['error']}")
    skip = max(0, STEADY_FROM - r.get("untimed_calls_before", 2)) if model == "tracker" else 0
    return [1e3 * s for s in r["all_s"][skip:]]


def stats(xs):
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def spread(medians):
    return max(medians) / min(medians) - 1.0


def main():
    out = pathlib.Path(sys.argv[1])
    lines = [f"# Tracker step A/B summary (steady frames / steps >= {STEADY_FROM})", ""]
    st = {r: spiketrack_round(out / f"st-r{r}") for r in ROUNDS}
    fp16 = {r: coreml_steady(out / f"coreml-fp16-r{r}.json", "tracker") for r in ROUNDS}
    bb16 = {r: coreml_steady(out / f"coreml-fp16-r{r}.json", "backbone") for r in ROUNDS}
    fp32_paths = {r: out / f"coreml-fp32-r{r}.json" for r in ROUNDS}
    fp32 = {r: coreml_steady(p, "tracker") for r, p in fp32_paths.items() if p.exists()}

    # Precondition: the stage runs' in-Session time matches the unprofiled controls.
    ctrl = [ms for c in (1, 2) for ms, _ in log_frames(out / f"st-control-{c}.log")]
    ctrl_mean = statistics.mean(ctrl)
    # The port is dense 16; removed objects' slots are never reused, so check ours stayed at 16.
    active = {r: [k for _, k in log_frames(out / f"st-r{r}.log")] for r in ROUNDS}
    full = all(min(active[r]) >= 16 for r in ROUNDS)
    valid = True
    lines += ["## Precondition: stage runs against unprofiled controls", "",
              f"Controls' steady push_frame mean {ctrl_mean:.1f} ms over {len(ctrl)} frames.", ""]
    for r in ROUNDS:
        m = statistics.mean(st[r]["in_session_ms"])
        dev = m / ctrl_mean - 1.0
        ok = abs(dev) <= STAGE_RUN_TOLERANCE
        valid &= ok
        lines.append(f"- round {r}: in-Session {m:.1f} ms/frame ({dev:+.1%}) {'ok' if ok else 'OUTSIDE 10%'}")
    lines.append("")

    lines += ["## Per round, ms", "",
              "| round | SpikeTrack tracker median (min) | port fp16 step | port fp32 step | ratio vs fp16 | ratio vs fp32 | SpikeTrack backbone | port backbone fp16 |",
              "|---|---|---|---|---|---|---|---|"]
    med = lambda xs: statistics.median(xs)  # noqa: E731
    for r in ROUNDS:
        s = stats(st[r]["tracker_ms"])
        f16 = med(fp16[r])
        f32 = med(fp32[r]) if r in fp32 else None
        f32_cell, ratio32 = ("not run", "—") if f32 is None else (f"{f32:.1f}", f"{s['median'] / f32:.2f}")
        lines.append(
            f"| {r} | {s['median']:.1f} ({s['min']:.1f}) | {f16:.1f} | {f32_cell} | "
            f"{s['median'] / f16:.2f} | {ratio32} | "
            f"{med(st[r]['backbone_ms']):.1f} | {med(bb16[r]):.1f} |"
        )
    lines.append("")
    for r in ROUNDS:
        lines.append(f"- round {r}: SpikeTrack {len(st[r]['frames'])} frames, {st[r]['tracks']} tracks, cap {st[r]['max_objects']}, "
                     f"active {min(active[r])}-{max(active[r])} on steady frames, "
                     f"trunk {st[r]['trunk_precision']}, births on {st[r]['frames_with_births']} steady frames")
    lines.append("")

    st_spread = spread([med(st[r]["tracker_ms"]) for r in ROUNDS])
    f16_spread = spread([med(fp16[r]) for r in ROUNDS])
    lines += ["## Noise", "",
              f"- round-to-round spread of the medians: SpikeTrack {st_spread:.1%}, port fp16 {f16_spread:.1%}"]
    arms = [("fp16", fp16, f16_spread)]
    if len(fp32) == len(ROUNDS):
        f32_spread = spread([med(fp32[r]) for r in ROUNDS])
        lines.append(f"- port fp32 {f32_spread:.1%}")
        arms.insert(0, ("fp32", fp32, f32_spread))
    every = json.loads((out / "coreml-fp16-r1.json").read_text())["tracker"]["all_s"]
    lines += ["- step cost against bank depth (port fp16, round 1, every timed call from step 2): "
              + ", ".join(f"{1e3 * s:.1f}" for s in every), ""]

    lines += ["## Decision (pre-stated)", ""]
    if not valid:
        lines.append("NOT READ: a stage run is outside 10% of the controls, so its stage times are not taken literally.")
    if not full:
        lines.append("NOT READ: a steady frame held fewer than 16 active objects, so it is not the port's 16-object step.")
    valid &= full
    if len(fp32) != len(ROUNDS):
        lines.append("Precision-matched arm (port fp32) NOT RUN: the precision-matched verdict is missing, not passed.")
    for name, arm, sp in arms:
        noise = max(st_spread, sp)
        ratios = [med(st[r]["tracker_ms"]) / med(arm[r]) for r in ROUNDS]
        if all(x <= 1.0 + noise for x in ratios):
            verdict = "MET (parity or better)"
        elif all(x > 1.0 + noise for x in ratios):
            verdict = "NOT MET"
        else:
            verdict = "INCONCLUSIVE (rounds disagree beyond the noise)"
        lines.append(f"- vs port {name}: ratios {', '.join(f'{x:.2f}' for x in ratios)}; noise {noise:.1%}; "
                     f"{verdict}{'' if valid else ' (not read: precondition failed)'}")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    # A verdict that was not read, or an arm that did not run, must not exit like a read one.
    if not valid or len(fp32) != len(ROUNDS):
        raise SystemExit("verdict not read or precision-matched arm missing; see summary.md")


if __name__ == "__main__":
    main()
