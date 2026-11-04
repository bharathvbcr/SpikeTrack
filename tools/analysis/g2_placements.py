"""G2/G3 of bench/results/e25-coreml-frame/RULES.md: each config's held-out HOTA against the shipped
baseline, per video, with the pre-stated verdict.

    python3 -I tools/analysis/g2_placements.py bench/results/track/held32-coreml 32 default m16 seg32 fast16

The baseline per video is the reference run in the same folder (`default/`, today's binary). It must
reproduce the documented numbers (32 frames: 0.6025; 300 frames: 0.570) to within 1e-4, or every
verdict is void. fp32 configs: |dHOTA| <= 0.002 and every video within 0.005. fp16 configs: HOTA >=
documented - 0.002 (32) / 0.003 (300) and no video more than 0.01 below the baseline.
"""

import json
import pathlib
import sys

DOCUMENTED = {
    32: (0.6025, {"21": 0.6744, "22": 0.6799, "36": 0.5752, "47": 0.5596, "82": 0.5552}),
    300: (0.570, {"21": 0.669, "22": 0.669, "36": 0.533, "47": 0.484, "82": 0.529}),
}
FP16 = {"m16", "fast16", "seg16", "t16m16"}


def load(d):
    e = json.loads((d / "eval.json").read_text())
    return e["combined"]["hota"], {v["id"]: v["hota"] for v in e["videos"]}


def main():
    base, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
    ref, *cfgs = sys.argv[3:]
    doc_total, doc_vid = DOCUMENTED[n]
    r_total, r_vid = load(base / ref)
    tol = 1e-4 if n == 32 else 5e-4
    repro = abs(r_total - doc_total) <= tol and all(abs(r_vid[k] - doc_vid[k]) <= tol for k in doc_vid)
    lines = [f"# Held-out HOTA, first {n} frames, against `{ref}` on today's binary", "",
             f"Baseline reproduces the documented {doc_total} (per video {doc_vid}): "
             f"{'yes' if repro else 'NO, verdicts void'} ({r_total:.4f}; " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(r_vid.items())) + ")", "",
             "| config | HOTA | dHOTA | " + " | ".join(f"{k} d" for k in sorted(r_vid)) + " | verdict |",
             "|---|---|---|" + "---|" * len(r_vid) + "---|"]
    for c in cfgs:
        d = base / c
        if not (d / "eval.json").is_file():
            lines.append(f"| {c} | missing | | " + " | " * len(r_vid) + " |")
            continue
        total, vid = load(d)
        dv = {k: vid[k] - r_vid[k] for k in r_vid}
        if c in FP16:
            floor = doc_total - (0.002 if n == 32 else 0.003)
            ok = total >= floor and min(dv.values()) >= -0.01
            rule = f"fp16: >= {floor:.4f}, no video < -0.01"
        else:
            ok = abs(total - r_total) <= 0.002 and max(abs(x) for x in dv.values()) <= 0.005
            rule = "fp32: |d| <= 0.002, videos <= 0.005"
        verdict = ("PASS" if ok else "FAIL") + f" ({rule})" + ("" if repro else ", void")
        lines.append(f"| {c} | {total:.4f} | {total - r_total:+.4f} | " + " | ".join(f"{dv[k]:+.4f}" for k in sorted(dv)) + f" | {verdict} |")
    text = "\n".join(lines) + "\n"
    (base / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
