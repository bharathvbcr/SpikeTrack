"""SAM 3.1's multiplex memory encoder (the tracker's `maskmem_backbone`) as one CoreML ML Program.

    PY=tools/baselines/.venv-export/bin/python   (torch 2.7.0, coremltools 9.0, safetensors, numpy<2.3)
    $PY -s tools/baselines/coreml_memenc.py torch-check
    tools/locked.sh cml-memenc $PY -s tools/baselines/coreml_memenc.py export --precision fp16
    tools/locked.sh cml-memenc $PY -s tools/baselines/coreml_memenc.py check --precision fp16

Mirrors st-model's `MemoryEncoder::fwd` (crates/st-model/src/tracker/memory_encoder.rs) for masks already
at the downsampler's input size (16 x the grid, 1152 at 1008), the size `Vos::encode_mask_memory`
receives from `TrackerState::encode_memory`:

    x    = cat(masks [1,16,S,S], cond planes [1,16,S,S])         slot masks, then condition flags
    x    = 4 x (conv3x3 s2 p1 -> LayerNorm2d(1e-6) -> GELU)       32 -> 16 -> 64 -> 256 -> 1024
    x    = conv1x1(x) (1024 -> 256) + pix_feat_proj(pix_feat)
    x    = 2 x CXBlock: x + gamma * pw2(GELU(pw1(LN2d(dwconv7x7(x)))))

The 16 condition planes are constant (0 or 1) per slot, so the program takes them as `cond` [1,16,1,1]
and broadcasts them itself: the host builds and the program reads 16 planes, not 32.

    inputs  pix_feat [1,256,72,72], masks [1,16,1152,1152], cond [1,16,1,1]
    output  features [1,256,72,72]

The fixture (fixtures/tracker/judo.safetensors, upstream's calls) holds masks at 1008. The checks resize
them to 1152 with bilinear interpolation (align_corners False), as the Ojas encoder does before its
downsampler, then compare against upstream's outputs. No adapter or weight edit touches these weights.
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
FIXTURE = REPO / "fixtures/tracker/judo.safetensors"
OUT = REPO / "bench/results/coreml-memenc"
P = "tracker.model.maskmem_backbone."
D, GRID, MUX, S = 256, 72, 16, 1152
FRAMES = 8


def package(precision):
    return MODEL_DIR / "coreml" / f"sam3.1_memenc_{precision}.mlpackage"


def build():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    w = {}
    with safe_open(str(CKPT), "pt") as f:
        for k in f.keys():
            if k.startswith(P):
                w[k[len(P):]] = f.get_tensor(k).float().clone()

    def ln2d(x, p):
        u = x.mean(1, keepdim=True)
        v = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(v + 1e-6)
        return w[p + ".weight"][:, None, None] * x + w[p + ".bias"][:, None, None]

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.params = torch.nn.ParameterList([torch.nn.Parameter(w[k], requires_grad=False) for k in sorted(w)])

        def forward(self, pix_feat, masks, cond):
            x = torch.cat([masks, cond.expand(1, MUX, masks.shape[2], masks.shape[3])], dim=1)
            for i in range(4):
                c = f"mask_downsampler.encoder.{3 * i}"
                x = F.conv2d(x, w[c + ".weight"], w[c + ".bias"], stride=2, padding=1)
                x = F.gelu(ln2d(x, f"mask_downsampler.encoder.{3 * i + 1}"))
            x = F.conv2d(x, w["mask_downsampler.encoder.12.weight"], w["mask_downsampler.encoder.12.bias"])
            x = x + F.conv2d(pix_feat, w["pix_feat_proj.weight"], w["pix_feat_proj.bias"])
            for i in range(2):
                f = f"fuser.layers.{i}"
                y = F.conv2d(x, w[f + ".dwconv.weight"], w[f + ".dwconv.bias"], padding=3, groups=D)
                y = ln2d(y, f + ".norm").permute(0, 2, 3, 1)
                y = F.linear(F.gelu(F.linear(y, w[f + ".pwconv1.weight"], w[f + ".pwconv1.bias"])),
                             w[f + ".pwconv2.weight"], w[f + ".pwconv2.bias"])
                x = x + (w[f + ".gamma"] * y).permute(0, 3, 1, 2)
            return x

    return Enc().eval()


def fixture_call(f):
    """Upstream's call f: inputs resized to the program's mask size, and its features."""
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    s = safe_open(str(FIXTURE), "np")
    p = f"f{f}.memenc.call{f}"
    pix = s.get_tensor(p + ".arg.0").astype(np.float32)
    m32 = torch.from_numpy(s.get_tensor(p + ".arg.1").astype(np.float32))
    m32 = F.interpolate(m32, size=(S, S), mode="bilinear", align_corners=False).numpy()
    cond = m32[:, MUX:, :1, :1].copy()
    if not np.all(m32[:, MUX:] == cond):
        raise SystemExit(f"call {f}: a condition plane is not constant")
    inputs = {"pix_feat": pix, "masks": np.ascontiguousarray(m32[:, :MUX]), "cond": cond}
    return inputs, s.get_tensor(p + ".out.vision_features")


def stats(got, want):
    gg, ww = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    d = gg - ww
    return {"cosine": float(gg @ ww) / float(np.linalg.norm(gg) * np.linalg.norm(ww)),
            "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ww)),
            "max_abs": float(np.abs(d).max()), "finite": bool(np.isfinite(gg).all())}


ORDER = ["pix_feat", "masks", "cond"]


def torch_check():
    import torch

    m = build()
    for f in range(FRAMES):
        inputs, want = fixture_call(f)
        with torch.no_grad():
            got = m(*[torch.from_numpy(inputs[k]) for k in ORDER]).numpy()
        print(f"call {f}: {stats(got, want)}")


def export(precision):
    import coremltools as ct
    import torch

    m = build()
    inputs, _ = fixture_call(0)
    with torch.no_grad():
        traced = torch.jit.trace(m, tuple(torch.from_numpy(inputs[k]) for k in ORDER))
    cp = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=k, shape=inputs[k].shape, dtype=np.float32) for k in ORDER],
        outputs=[ct.TensorType(name="features", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=cp,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_memenc.py"
    mlmodel.short_description = f"SAM 3.1 multiplex memory encoder ({precision}), from {CKPT.name}"
    dst = package(precision)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    print(f"saved {dst}")


def check(precision, units, n):
    import coremltools as ct

    model = ct.models.MLModel(str(package(precision)), compute_units=getattr(ct.ComputeUnit, units))
    parity = []
    for f in range(FRAMES):
        inputs, want = fixture_call(f)
        parity.append(stats(np.asarray(model.predict(inputs)["features"]), want))
        print(f"call {f}: {parity[-1]}")
    inputs, _ = fixture_call(0)
    for _ in range(2):
        model.predict(inputs)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(inputs)
        times.append((time.perf_counter() - t0) * 1e3)
    res = {"precision": precision, "units": units, "reference": "upstream fixture, masks resized to 1152",
           "parity": parity, "worst_cosine": min(p["cosine"] for p in parity),
           "predict_ms": {"n": n, "median": statistics.median(times), "min": min(times)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{precision}-{units}.json").write_text(json.dumps(res, indent=1))
    print(f"worst cosine {res['worst_cosine']:.7f}; predict median {statistics.median(times):.1f} ms (min {min(times):.1f}, n {n})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("torch-check")
    for name in ("export", "check"):
        p = sub.add_parser(name)
        p.add_argument("--precision", choices=["fp32", "fp16"], required=True)
        if name == "check":
            p.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_ONLY", "CPU_AND_NE"])
            p.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "torch-check":
        torch_check()
    elif a.cmd == "export":
        export(a.precision)
    else:
        check(a.precision, a.units, a.n)


if __name__ == "__main__":
    main()
