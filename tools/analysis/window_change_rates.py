"""How much of each VISEM frame changes between consecutive frames, at the
granularity event gating would skip (throwaway analysis; nothing ships).

    tools/reference/.venv/bin/python -s tools/analysis/window_change_rates.py

Frames are resized to the model input (1008x1008, bilinear, grey). A 14x14
patch "changed" when its mean |frame_t - frame_t-1| exceeds tau (0-255 scale);
a window (one of the 3x3 attention windows, 24x24 patches) changed when any of
its patches did. Gating can skip a window's 28 windowed layers only if no patch
in it changed, so the window fraction is the one that bounds the speedup.
"""

import json
import pathlib

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
VIDEOS = sorted((REPO / "data/visem/VISEM_Tracking_Train_v4/Train").glob("*/*.mp4"))
OUT = REPO / "bench/results/gating-change-rates"
FRAMES = 300  # consecutive frames per video (~6 s at 50 fps), from frame 0
TAUS = (2, 4, 8, 16, 32)
STRIDES = (1, 2, 5)  # compare frame t with t - stride (the gate need not run every frame)


def frames(path, n):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, (1008, 1008), interpolation=cv2.INTER_LINEAR).astype(np.float32))
    cap.release()
    return np.stack(out), fps, f.shape[:2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in VIDEOS:
        x, fps, hw = frames(v, FRAMES + max(STRIDES))
        for s in STRIDES:
            d = np.abs(x[s:] - x[:-s])  # [T, 1008, 1008]
            patch = d.reshape(len(d), 72, 14, 72, 14).mean(axis=(2, 4))  # [T, 72, 72]
            for tau in TAUS:
                pch = patch > tau
                win = pch.reshape(len(d), 3, 24, 3, 24).any(axis=(2, 4))  # [T, 3, 3]
                rows.append({
                    "video": v.parent.name, "fps": round(fps, 2), "src_hw": list(hw), "stride": s, "tau": tau,
                    "frames": int(len(d)),
                    "patch_changed_mean": float(pch.mean()),
                    "window_changed_mean": float(win.mean()),
                    "frames_all_windows_changed": float(win.all(axis=(1, 2)).mean()),
                    "frames_no_window_changed": float((~win).all(axis=(1, 2)).mean()),
                })
        print(v.parent.name, "done", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    lines = ["# VISEM per-window change rates (no model)", "",
             f"{len(VIDEOS)} videos, {FRAMES} consecutive frames each from frame 0, resized to 1008x1008 grey.",
             "Mean over videos of the per-frame fraction changed; window = one of the 3x3 attention windows.", "",
             "| stride | tau | patches changed | windows changed | frames with all 9 changed | frames with none changed |",
             "|---|---|---|---|---|---|"]
    for s in STRIDES:
        for tau in TAUS:
            sel = [r for r in rows if r["stride"] == s and r["tau"] == tau]
            m = lambda k: np.mean([r[k] for r in sel])  # noqa: E731
            lines.append(f"| {s} | {tau} | {m('patch_changed_mean'):.3f} | {m('window_changed_mean'):.3f} | "
                         f"{m('frames_all_windows_changed'):.3f} | {m('frames_no_window_changed'):.3f} |")
    lines += ["", "Per video, stride 1, tau 8 (windows changed):", ""]
    for r in rows:
        if r["stride"] == 1 and r["tau"] == 8:
            lines.append(f"- {r['video']}: windows {r['window_changed_mean']:.3f}, patches {r['patch_changed_mean']:.3f}")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
