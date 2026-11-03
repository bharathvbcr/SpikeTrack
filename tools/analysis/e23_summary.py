"""Summarise bench/results/e23-mattn-kv-cache/job.sh (throwaway analysis; nothing ships): the
tracker step of the old and new binaries per round, the pre-stated speed rule, whether the cached
path ran (gate 7), and memory at frames 8 and 33.

    python3 -I tools/analysis/e23_summary.py bench/results/e23-mattn-kv-cache [--control | --one-tree]

`--control` reads the same-tree control rounds instead (`ctl-r*` against `cnew-r*`, written to
`control-summary.md`), and `--one-tree` the one-tree confirmation (`1t-ctl-r*` against
`1t-new-r*`, written to `one-tree-summary.md`): the tracker step only.

    python3 -I tools/analysis/e23_summary.py bench/results/e23-mattn-kv-cache/paired --paired 19

`--paired OPS_FRAMES` reads the pre-stated paired run (`ops-*` and `pNN-{1,2}-{ctl,new}`, with
`.load` samples and `pNN.flagged` markers) and writes its `summary.md`.
"""
import math
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ab_coreml_tracker import log_frames, spiketrack_round  # noqa: E402

ROUNDS = (1, 2, 3)
FRAME_ROWS = 5184
# flag: (control arm, new arm, title, summary file)
SPEED_ONLY = {
    "--control": ("ctl", "cnew", "E23 same-tree control", "control-summary.md"),
    "--one-tree": ("1t-ctl", "1t-new", "E23 one-tree confirmation", "one-tree-summary.md"),
}


def memory_rows(log):
    """(metal allocated MiB, footprint MiB) at frames 8 and 33 of a `track` log."""
    out = {}
    for line in log.read_text().splitlines():
        m = re.match(r"frame (\d+) \(.*metal allocated (\d+) MiB.*footprint (\d+) MiB", line)
        if m and int(m.group(1)) in (8, 33):
            out[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return out


def big_memory_linears(md):
    """Linear calls inside the tracker's stages over more rows than one frame has."""
    hits = []
    for line in md.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != 9 or not cells[0].startswith("tracker propagate") or cells[1] != "linear_bias_forward":
            continue
        rows = int(re.match(r"\[1, (\d+), ", cells[2]).group(1))
        if rows > FRAME_ROWS:
            hits.append(f"{cells[0]}: {cells[2]} x{cells[3]}")
    return hits


def speed(out, old, new, title):
    """The per-round tracker-step table and the pre-stated speed rule, `new` against `old`."""
    lines = [f"# {title} (steady frames >= 8)", ""]
    med = lambda xs: statistics.median(xs)  # noqa: E731
    arms = {a: {r: spiketrack_round(out / f"{a}-r{r}") for r in ROUNDS} for a in (old, new)}
    lines += [f"| round | {old} tracker step ms | {new} | {new} / {old} |", "|---|---|---|---|"]
    ratios = []
    for r in ROUNDS:
        o, n = med(arms[old][r]["tracker_ms"]), med(arms[new][r]["tracker_ms"])
        ratios.append(n / o)
        lines.append(f"| {r} | {o:.1f} | {n:.1f} | {n / o:.3f} |")
    spread = {a: max(med(arms[a][r]["tracker_ms"]) for r in ROUNDS) / min(med(arms[a][r]["tracker_ms"]) for r in ROUNDS) - 1 for a in arms}
    noise = max(spread.values())
    lines += ["", f"Same-binary spread: {old} {spread[old]:.1%}, {new} {spread[new]:.1%}; noise {noise:.1%}."]
    verdict = "SPEEDUP" if all(x < 1 - noise for x in ratios) else "no speedup claimed"
    lines += [f"Speed (pre-stated: every ratio < 1 - noise): {verdict}.", ""]
    return lines


# Two-sided 95% Student t quantiles by degrees of freedom (the paired run has 3, 5 or 9).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
OPS_STAGE = "tracker propagate, state 0: memory attention (device)"


def t_interval(xs):
    m = statistics.mean(xs)
    h = T975[len(xs) - 1] * statistics.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h


def spearman(xs, ys):
    rank = lambda v: [sorted(v).index(x) + (v.count(x) - 1) / 2 for x in v]  # noqa: E731
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


def ops_reading(md):
    """(memory-attention stage ms, summed ms of its linears over more than one frame's rows) from a
    `--profile ops` file, frames >= 1 aggregated (frame 0's rows are labelled apart)."""
    stage = big = 0.0
    for line in md.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == 4 and cells[0] == OPS_STAGE:
            stage = float(cells[2])
        if len(cells) == 9 and cells[0] == OPS_STAGE and cells[1] == "linear_bias_forward":
            if int(re.match(r"\[1, (\d+), ", cells[2]).group(1)) > FRAME_ROWS:
                big += float(cells[4])
    return stage, big


def paired(out, ops_frames):
    """The pre-stated paired run in `out` (README, "Paired speed run")."""
    lines = ["# E23 paired speed run", "", "## Mechanism (`--profile ops`, each op synced)", ""]
    lines += ["| run | memory attention ms / frame | linears over 5,184 rows ms / frame |", "|---|---|---|"]
    ops = {}
    for d in sorted(out.glob("ops-*")):
        if d.is_dir():
            stage, big = ops_reading(d / "82.profile-ops.md")
            ops[d.name] = (stage / ops_frames, big / ops_frames)
            lines.append(f"| {d.name} | {stage / ops_frames:.1f} | {big / ops_frames:.1f} |")
    arm = lambda a, i: statistics.mean(v[i] for k, v in ops.items() if k.endswith(a))  # noqa: E731
    lines += ["", f"Mean A − B, memory attention: {arm('ctl', 0) - arm('new', 0):.1f} ms / frame; "
              f"A's big linears: {arm('ctl', 1):.1f} ms / frame; B's: {arm('new', 1):.1f}.", ""]

    pairs = []
    for p in range(1, 17):
        (first,) = out.glob(f"p{p:02d}-1-*.log")
        (second,) = out.glob(f"p{p:02d}-2-*.log")
        a, b = first.stem.split("-")[2], second.stem.split("-")[2]
        t1 = statistics.median(spiketrack_round(out / first.stem)["tracker_ms"])
        t2 = statistics.median(spiketrack_round(out / second.stem)["tracker_ms"])
        loads = [float(x.split()[1]) for f in (first, second) for x in f.with_suffix(".load").read_text().splitlines() if x.strip()]
        flagged = (out / f"p{p:02d}.flagged").exists()
        if a == b:
            kind, x = f"{a}/{a}", math.log(t2 / t1)
        else:
            kind = "A/B"
            ta, tb = (t1, t2) if a == "ctl" else (t2, t1)
            x = math.log(tb / ta)
        pairs.append(dict(p=p, kind=kind, order=f"{a},{b}", t1=t1, t2=t2, x=x, load=statistics.mean(loads), flagged=flagged))

    lines += ["## Wall clock (median tracker step, frames >= 8)", "",
              "| pair | kind | order | first ms | second ms | value | mean load | flagged |", "|---|---|---|---|---|---|---|---|"]
    for q in pairs:
        lines.append(f"| {q['p']} | {q['kind']} | {q['order']} | {q['t1']:.1f} | {q['t2']:.1f} | {q['x']:+.4f} | {q['load']:.1f} | {'yes' if q['flagged'] else ''} |")
    lines.append("")
    verdicts = {}
    for label, keep in (("all pairs (decides)", lambda q: True), ("unflagged pairs only", lambda q: not q["flagged"])):
        ab = [q["x"] for q in pairs if q["kind"] == "A/B" and keep(q)]
        null = [q["x"] for q in pairs if q["kind"] != "A/B" and keep(q)]
        lines.append(f"### {label}: {len(ab)} A/B pairs, {len(null)} same-binary pairs")
        if len(ab) < 2 or len(null) < 2:
            lines += ["Too few pairs to read.", ""]
            verdicts[label] = "too few pairs"
            continue
        nm, nlo, nhi = t_interval(null)
        m, lo, hi = t_interval(ab)
        below = sum(x < 0 for x in ab)
        need = round(0.8 * len(ab))
        if not nlo <= 0 <= nhi:
            v = "INVALID (same-binary pairs biased)"
        elif hi < 0 and below >= need:
            v = "SPEEDUP"
        elif lo > 0 and len(ab) - below >= need:
            v = "SLOWDOWN"
        else:
            v = "no difference resolved"
        verdicts[label] = v
        pct = lambda y: f"{math.expm1(y):+.1%}"  # noqa: E731
        lines += [f"- Same-binary mean ln ratio {nm:+.4f}, 95% CI [{nlo:+.4f}, {nhi:+.4f}] ({pct(nlo)} to {pct(nhi)}).",
                  f"- A/B mean ln(B/A) {m:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]: B/A {pct(m)} ({pct(lo)} to {pct(hi)}); "
                  f"{below} of {len(ab)} pairs below 0 (needed {need}).",
                  f"- Verdict: **{v}**.", ""]
    ab = [q for q in pairs if q["kind"] == "A/B"]
    lines.append(f"Descriptive: Spearman rho of A/B ln ratio against mean load = {spearman([q['load'] for q in ab], [q['x'] for q in ab]):+.2f} (n = {len(ab)}).")
    return lines, verdicts


def main():
    out = pathlib.Path(sys.argv[1])
    if sys.argv[2:3] == ["--paired"]:
        lines, _ = paired(out, int(sys.argv[3]))
        text = "\n".join(lines) + "\n"
        (out / "summary.md").write_text(text)
        print(text)
        return
    if len(sys.argv) == 3 and sys.argv[2] in SPEED_ONLY:
        old, new, title, name = SPEED_ONLY[sys.argv[2]]
        text = "\n".join(speed(out, old, new, title)) + "\n"
        (out / name).write_text(text)
        print(text)
        return
    lines = speed(out, "old", "new", "E23 summary")

    ctrl = {a: log_frames(out / f"{a}-control.log") for a in ("old", "new")}
    lines.append("Unprofiled controls, steady push_frame mean: " + ", ".join(
        f"{a} {statistics.mean(ms for ms, _ in ctrl[a]):.1f} ms" for a in ctrl))
    for a in ("old", "new"):
        mem = memory_rows(out / f"{a}-control.log")
        lines.append(f"- {a}: " + "; ".join(f"frame {f}: metal allocated {m[0]} MiB, footprint {m[1]} MiB" for f, m in sorted(mem.items())))
    lines.append("")

    hits = big_memory_linears(out / "new-ops" / "82.profile-ops.md")
    lines.append("Gate 7 (cached path ran): " + ("PASS, no tracker linear over more than one frame's rows" if not hits
                                                 else "FAIL: " + "; ".join(hits)))
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    if hits:
        raise SystemExit("gate 7 failed")


if __name__ == "__main__":
    main()
