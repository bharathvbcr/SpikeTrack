"""SAM 3.1's necks as CoreML ML Programs: the detector and propagation necks in one, or any neck alone.

    PY=tools/baselines/.venv-export/bin/python   (torch 2.7.0, coremltools 9.0, safetensors, numpy<2.3)
    $PY -s tools/baselines/coreml_necks.py torch-check
    tools/locked.sh cml-necks $PY -s tools/baselines/coreml_necks.py export --precision fp32
    tools/locked.sh cml-necks $PY -s tools/baselines/coreml_necks.py check --precision fp32

The module mirrors st-model's `Backbone::neck` (crates/st-model/src/backbone.rs) for the two necks a
steady frame runs (`convs`, the detector's, and `propagation_convs`), and for `interactive_convs`, the
neck births and reconditions use (`--neck inter`, a one-neck package only). Per neck, levels at 288,
144 and 72:
    Up4:  dconv_2x2_0 (stride 2) -> GELU -> dconv_2x2_1 (stride 2) -> conv_1x1 -> conv_3x3 (pad 1)
    Up2:  dconv_2x2 (stride 2) -> conv_1x1 -> conv_3x3
    Same: conv_1x1 -> conv_3x3

    input   trunk   [1, 72, 72, 1024] f32, NHWC (what the trunk package returns)
    outputs det0 [1,256,288,288], det1 [1,256,144,144], det2 [1,256,72,72], prop0, prop1, prop2 (NCHW f32)

check compares each output with the backbone fixtures' `neck.convs.{i}.out` / `neck.propagation_convs.{i}.out`
fed the fixture's own `trunk.blocks.31.out`. The package is derived from gated weights and lives next
to them, never in the repo.
"""

import argparse
import json
import os
import pathlib
import statistics
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = pathlib.Path(os.environ.get("SPIKETRACK_MODEL_DIR", "~/Code/research/models/sam3.1")).expanduser()
CKPT = MODEL_DIR / "sam3.1_multiplex.safetensors"
PREFIX = "detector.backbone.vision_backbone."
FIXTURES = ["visem_22_f0", "visem_47_f300", "visem_82_f600", "sam3_truck", "sam3_groceries"]
OUT = REPO / "bench/results/coreml-necks"
NECKS = {"det": "convs", "prop": "propagation_convs", "inter": "interactive_convs"}
# The two-neck package (no --neck) holds these.
PAIR = ("det", "prop")
NAMES = [f"{n}{i}" for n in PAIR for i in range(3)]


def package(precision, neck=None):
    if neck is None:
        return MODEL_DIR / "coreml" / f"sam3.1_necks_{precision}.mlpackage"
    return MODEL_DIR / "coreml" / f"sam3.1_neck_{neck}_{precision}.mlpackage"


def build(only=None):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    w = {}
    with safe_open(str(CKPT), "pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX) and any(k.startswith(PREFIX + v + ".") for v in NECKS.values()):
                w[k[len(PREFIX):]] = f.get_tensor(k).float()

    class Conv(torch.nn.Module):
        def __init__(self, p, transpose=False, padding=0):
            super().__init__()
            self.w = torch.nn.Parameter(w[p + ".weight"], requires_grad=False)
            self.b = torch.nn.Parameter(w[p + ".bias"], requires_grad=False)
            self.transpose, self.padding = transpose, padding

        def forward(self, x):
            if self.transpose:
                return F.conv_transpose2d(x, self.w, self.b, stride=2)
            return F.conv2d(x, self.w, self.b, padding=self.padding)

    class Level(torch.nn.Module):
        def __init__(self, p, kind):
            super().__init__()
            self.kind = kind
            if kind == "up4":
                self.d0, self.d1 = Conv(f"{p}.dconv_2x2_0", True), Conv(f"{p}.dconv_2x2_1", True)
            elif kind == "up2":
                self.d = Conv(f"{p}.dconv_2x2", True)
            self.c1, self.c3 = Conv(f"{p}.conv_1x1"), Conv(f"{p}.conv_3x3", padding=1)

        def forward(self, x):
            if self.kind == "up4":
                x = self.d1(F.gelu(self.d0(x)))
            elif self.kind == "up2":
                x = self.d(x)
            return self.c3(self.c1(x))

    class Necks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.necks = torch.nn.ModuleList(
                torch.nn.ModuleList(Level(f"{v}.{i}", k) for i, k in enumerate(("up4", "up2", "same")))
                for n, v in NECKS.items() if (n in PAIR if only is None else n == only)
            )

        def forward(self, trunk):
            x = trunk.permute(0, 3, 1, 2)
            return tuple(lvl(x) for neck in self.necks for lvl in neck)

    return Necks().eval()


def stats(got, want):
    g, w = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    d = g - w
    with np.errstate(all="ignore"):
        dot = float(g @ w)
    return {
        "cosine": dot / float(np.linalg.norm(g) * np.linalg.norm(w)),
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(w)),
        "max_abs": float(np.abs(d).max()),
        "finite": bool(np.isfinite(g).all()),
    }


def fixture(name):
    from safetensors.numpy import load_file

    fx = load_file(str(REPO / f"fixtures/backbone/{name}.safetensors"))
    want = {f"{n}{i}": fx[f"neck.{v}.{i}.out"] for n, v in NECKS.items() for i in range(3)}
    return fx["trunk.blocks.31.out"], want


def torch_check():
    import torch

    m = build()
    worst = 0.0
    with torch.no_grad():
        for name in FIXTURES:
            x, want = fixture(name)
            outs = m(torch.from_numpy(x))
            for k, o in zip(NAMES, outs):
                st = stats(o.numpy(), want[k])
                worst = max(worst, st["rel_l2"])
            print(name, "ok")
    print(f"worst rel_l2 {worst:.3e}")


def names(neck):
    """Output names: det0..prop2 for the two-neck package, l0..l2 for a one-neck package."""
    return NAMES if neck is None else ["l0", "l1", "l2"]


def wanted(want, neck):
    return want if neck is None else {f"l{i}": want[f"{neck}{i}"] for i in range(3)}


def export(precision, neck=None):
    import coremltools as ct
    import torch

    m = build(neck)
    x, _ = fixture(FIXTURES[0])
    with torch.no_grad():
        traced = torch.jit.trace(m, torch.from_numpy(x))
    cp = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="trunk", shape=x.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in names(neck)],
        convert_to="mlprogram",
        compute_precision=cp,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_necks.py"
    what = "detector + propagation necks" if neck is None else f"{NECKS[neck]} neck"
    mlmodel.short_description = f"SAM 3.1 {what} ({precision}), from {CKPT.name}"
    dst = package(precision, neck)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    print(f"saved {dst}")


def check(precision, units, n, neck=None):
    import coremltools as ct

    model = ct.models.MLModel(str(package(precision, neck)), compute_units=getattr(ct.ComputeUnit, units))
    parity, worst = {}, 0.0
    x0 = None
    for name in FIXTURES:
        x, want = fixture(name)
        want = wanted(want, neck)
        x0 = x if x0 is None else x0
        out = model.predict({"trunk": x})
        parity[name] = {k: stats(np.asarray(out[k]), want[k]) for k in names(neck)}
        worst = max(worst, max(v["rel_l2"] for v in parity[name].values()))
        print(name, {k: round(v["rel_l2"], 8) for k, v in parity[name].items()})
    for _ in range(2):
        model.predict({"trunk": x0})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict({"trunk": x0})
        times.append((time.perf_counter() - t0) * 1e3)
    res = {"precision": precision, "units": units, "worst_rel_l2": worst, "parity": parity,
           "predict_ms": {"n": n, "median": statistics.median(times), "min": min(times)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{neck or 'both'}-{precision}-{units}.json").write_text(json.dumps(res, indent=1))
    print(f"worst rel_l2 {worst:.3e}; predict median {statistics.median(times):.1f} ms (min {min(times):.1f}, n {n})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("torch-check")
    e = sub.add_parser("export")
    e.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    e.add_argument("--neck", choices=list(NECKS), help="one neck per package (default: both in one)")
    c = sub.add_parser("check")
    c.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    c.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_ONLY", "CPU_AND_NE"])
    c.add_argument("-n", type=int, default=10)
    c.add_argument("--neck", choices=list(NECKS))
    a = ap.parse_args()
    {"torch-check": lambda: torch_check(), "export": lambda: export(a.precision, a.neck),
     "check": lambda: check(a.precision, a.units, a.n, a.neck)}[a.cmd]()


if __name__ == "__main__":
    main()
