"""SAM 3.1's tracker memory attention as one CoreML ML Program, padded to fixed shapes.

    PY=tools/baselines/.venv-export/bin/python   (torch 2.7.0, coremltools 9.0, safetensors, numpy<2.3)
    $PY -s tools/baselines/coreml_mattn.py torch-check            # the torch module vs the judo fixture
    tools/locked.sh cml-mattn $PY -s tools/baselines/coreml_mattn.py export --precision fp32
    tools/locked.sh cml-mattn $PY -s tools/baselines/coreml_mattn.py check --precision fp32 [--units CPU_AND_GPU]

The module mirrors st-model's `MemoryAttention::layers_fwd` (crates/st-model/src/tracker/memory_attention.rs)
line by line, for one bucket, with the memory padded to MAX_FRAMES frames and MAX_PTRS pointer tokens:

    inputs (all f32)  image     [1, T, 256]            current frame features (also the source)
                      src_pos   [1, T, 256]            their positions
                      mem_image [1, MAX_FRAMES*T, 256] memory frames' image features, frame slots in order
                      mem       [1, MAX_FRAMES*T, 256] their mask memories
                      mem_pos   [1, MAX_FRAMES*T, 256] their positions (with temporal encodings)
                      ptr       [1, MAX_PTRS, 256]     object pointer tokens
                      ptr_pos   [1, MAX_PTRS, 256]     their positions
                      key_bias  [1, 1, 1, MAX_FRAMES*T + MAX_PTRS]  0 on real keys, -1e4 on padding
    output            out       [1, T, 256]            after the final LayerNorm

Unused frame and pointer slots hold zeros and are masked by key_bias (a large finite negative, never -inf,
so an fp16 build cannot NaN). The RoPE tables are constants: query rows the 72 x 72 axial table, frame key
slots the same table repeated, pointer slots unrotated (cos 1, sin 0). Pointer keys are
img_k(zeros) + ca_k(ptr), i.e. img_k's bias plus ca_k(ptr), as st-model pads them.

The package is derived from gated weights: it lives next to them, never in the repo.
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = pathlib.Path(os.environ.get("SPIKETRACK_MODEL_DIR", "~/Code/research/models/sam3.1")).expanduser()
CKPT = MODEL_DIR / "sam3.1_multiplex.safetensors"
FIXTURE = REPO / "fixtures/tracker/judo.safetensors"
OUT = REPO / "bench/results/coreml-mattn"
PREFIX = "tracker.model.transformer.encoder"
D, HEADS, HD, GRID = 256, 8, 32, 72
T = GRID * GRID
MAX_FRAMES, MAX_PTRS = 7, 256
NKEYS = MAX_FRAMES * T + MAX_PTRS
LN_EPS = 1e-5
MASK = -1e4


def package(precision):
    return MODEL_DIR / "coreml" / f"sam3.1_mattn_{precision}.mlpackage"


def axial_tables():
    """ojas_core::axial_rope_2d_tables(32, 72, 72, 10000, 1): freqs and angles in f32, cos/sin in f64."""
    quarter = HD // 4
    freqs = np.array([np.float32(1.0) / np.float32(10000.0) ** np.float32(4 * j / HD) for j in range(quarter)], np.float32)
    t = np.arange(T)
    col, row = (t % GRID).astype(np.float32), (t // GRID).astype(np.float32)
    ang = np.concatenate([col[:, None] * freqs[None, :], row[:, None] * freqs[None, :]], axis=1).astype(np.float32)
    a = ang.astype(np.float64)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)  # [T, 16] each


def build():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    w = {}
    with safe_open(str(CKPT), "pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX + "."):
                w[k[len(PREFIX) + 1:]] = f.get_tensor(k).float()

    cos, sin = axial_tables()
    kcos = np.concatenate([np.tile(cos, (MAX_FRAMES, 1)), np.ones((MAX_PTRS, HD // 2), np.float32)])
    ksin = np.concatenate([np.tile(sin, (MAX_FRAMES, 1)), np.zeros((MAX_PTRS, HD // 2), np.float32)])

    class Lin(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.w = torch.nn.Parameter(w[p + ".weight"], requires_grad=False)
            self.b = torch.nn.Parameter(w[p + ".bias"], requires_grad=False)

        def forward(self, x):
            return F.linear(x, self.w, self.b)

    class Norm(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.w = torch.nn.Parameter(w[p + ".weight"], requires_grad=False)
            self.b = torch.nn.Parameter(w[p + ".bias"], requires_grad=False)

        def forward(self, x):
            return F.layer_norm(x, (D,), self.w, self.b, LN_EPS)

    def rope(x, c, s):
        # x [1, N, H, HD]; c, s [N, HD/2]. out[2i] = x0 c - x1 s, out[2i+1] = x0 s + x1 c.
        n = x.shape[1]
        xr = x.reshape(1, n, HEADS, HD // 2, 2)
        x0, x1 = xr[..., 0], xr[..., 1]
        c, s = c[None, :, None, :], s[None, :, None, :]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).reshape(1, n, HEADS, HD)

    def bhtd(x):
        return x.permute(0, 2, 1, 3)

    def heads(x):
        return x.reshape(1, x.shape[1], HEADS, HD)

    class Layer(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            p = f"layers.{i}"
            self.sa = torch.nn.ModuleList([Lin(f"{p}.self_attn_{n}_proj") for n in ("q", "k", "v", "out")])
            self.ca_q, self.ca_k, self.ca_v, self.ca_out = (Lin(f"{p}.cross_attn_{n}_proj") for n in ("q", "k", "v", "out"))
            self.img_q, self.img_k = Lin(f"{p}.image_cross_attn_q_proj"), Lin(f"{p}.image_cross_attn_k_proj")
            self.l1, self.l2 = Lin(f"{p}.linear1"), Lin(f"{p}.linear2")
            self.n1, self.n2, self.n3 = Norm(f"{p}.norm1"), Norm(f"{p}.norm2"), Norm(f"{p}.norm3")

    class MemAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(i) for i in range(4)])
            self.norm = Norm("norm")
            self.register_buffer("q_cos", torch.from_numpy(cos))
            self.register_buffer("q_sin", torch.from_numpy(sin))
            self.register_buffer("k_cos", torch.from_numpy(kcos))
            self.register_buffer("k_sin", torch.from_numpy(ksin))

        def forward(self, image, src_pos, mem_image, mem, mem_pos, ptr, ptr_pos, key_bias):
            tgt = image + src_pos * 0.1
            key_pos = torch.cat([mem_pos, ptr_pos], dim=1)
            for l in self.layers:
                t2 = l.n1(tgt)
                q = rope(heads(l.sa[0](t2)), self.q_cos, self.q_sin)
                k = rope(heads(l.sa[1](t2)), self.q_cos, self.q_sin)
                v = heads(l.sa[2](t2))
                a = F.scaled_dot_product_attention(bhtd(q), bhtd(k), bhtd(v))
                tgt = tgt + l.sa[3](bhtd(a).reshape(1, T, D))
                t2 = l.n2(tgt)
                q = l.img_q(image) + l.ca_q(t2)
                k_frames = l.img_k(mem_image) + l.ca_k(mem)
                k_ptrs = l.img_k.b + l.ca_k(ptr)  # img_k(zeros) + ca_k(ptr)
                k = torch.cat([k_frames, k_ptrs], dim=1) + key_pos
                v = torch.cat([l.ca_v(mem), l.ca_v(ptr)], dim=1)
                q = rope(heads(q), self.q_cos, self.q_sin)
                k = rope(heads(k), self.k_cos, self.k_sin)
                a = F.scaled_dot_product_attention(bhtd(q), bhtd(k), bhtd(heads(v)), attn_mask=key_bias)
                tgt = tgt + l.ca_out(bhtd(a).reshape(1, T, D))
                t2 = l.n3(tgt)
                tgt = tgt + l.l2(F.gelu(l.l1(t2)))
            return self.norm(tgt)

    m = MemAttn().eval()
    missing = [k for k in w if not any(k == n or k.startswith(n) for n in _names(m))]
    if missing:
        raise SystemExit(f"checkpoint tensors not used by the module: {missing[:5]} ...")
    return m


def _names(m):
    # The checkpoint names the module reads, rebuilt from its structure (Lin/Norm hold .w/.b).
    out = ["norm"]
    for i in range(4):
        p = f"layers.{i}"
        out += [f"{p}.self_attn_{n}_proj" for n in ("q", "k", "v", "out")]
        out += [f"{p}.cross_attn_{n}_proj" for n in ("q", "k", "v", "out")]
        out += [f"{p}.image_cross_attn_q_proj", f"{p}.image_cross_attn_k_proj", f"{p}.linear1", f"{p}.linear2"]
        out += [f"{p}.norm{j}" for j in (1, 2, 3)]
    return out


def padded_inputs(frames_image, frames_mem, frames_pos, ptr, ptr_pos, image, src_pos):
    """Pack F <= MAX_FRAMES frames ([F*T, D] each) and P <= MAX_PTRS pointers into the fixed contract."""
    f = frames_image.shape[0] // T
    p = ptr.shape[0]
    if f > MAX_FRAMES or p > MAX_PTRS:
        raise SystemExit(f"{f} frames / {p} pointers exceed {MAX_FRAMES} / {MAX_PTRS}")
    z = lambda n: np.zeros((n, D), np.float32)
    pad = lambda x, n: np.concatenate([x, z(n - x.shape[0])])[None]
    bias = np.full((1, 1, 1, NKEYS), MASK, np.float32)
    bias[..., : f * T] = 0
    bias[..., MAX_FRAMES * T : MAX_FRAMES * T + p] = 0
    return {
        "image": image[None].astype(np.float32),
        "src_pos": src_pos[None].astype(np.float32),
        "mem_image": pad(frames_image, MAX_FRAMES * T),
        "mem": pad(frames_mem, MAX_FRAMES * T),
        "mem_pos": pad(frames_pos, MAX_FRAMES * T),
        "ptr": pad(ptr, MAX_PTRS),
        "ptr_pos": pad(ptr_pos, MAX_PTRS),
        "key_bias": bias,
    }


def fixture_cases():
    from safetensors import safe_open

    with safe_open(str(FIXTURE), "np") as s:
        for f in range(1, 8):
            p = f"f{f}.memattn.call{f - 1}"
            g = lambda k: s.get_tensor(f"{p}.kw.{k}")[:, 0, :]
            image, src, src_pos = g("image"), g("src"), g("src_pos")
            if not np.array_equal(image, src):
                raise SystemExit(f"{p}: src != image; the module takes the image as its source")
            mi, mem, mpos, mipos = g("memory_image"), g("memory"), g("memory_pos"), g("memory_image_pos")
            n = mi.shape[0]
            inputs = padded_inputs(mi, mem[:n], mipos, mem[n:], mpos[n:], image, src_pos)
            want = s.get_tensor(f"{p}.out.memory")[:, 0, :]
            yield f, n // T, mem.shape[0] - n, inputs, want


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


def torch_check():
    import torch

    m = build()
    worst = 0.0
    with torch.no_grad():
        for f, frames, ptrs, inp, want in fixture_cases():
            got = m(**{k: torch.from_numpy(v) for k, v in inp.items()})[0].numpy()
            st = stats(got, want)
            worst = max(worst, st["rel_l2"])
            print(f"f{f}: {frames} frames, {ptrs} pointers: {st}")
    print(f"worst rel_l2 {worst:.3e}")
    return worst


def export(precision):
    import coremltools as ct
    import torch

    m = build()
    cases = list(fixture_cases())
    _, _, _, inp, want = cases[-1]
    ex = {k: torch.from_numpy(v) for k, v in inp.items()}
    order = ["image", "src_pos", "mem_image", "mem", "mem_pos", "ptr", "ptr_pos", "key_bias"]
    with torch.no_grad():
        print("torch vs fixture f7:", stats(m(*[ex[k] for k in order])[0].numpy(), want))
        traced = torch.jit.trace(m, tuple(ex[k] for k in order))
    cp = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=k, shape=inp[k].shape, dtype=np.float32) for k in order],
        outputs=[ct.TensorType(name="out", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=cp,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_mattn.py"
    mlmodel.short_description = (
        f"SAM 3.1 tracker memory attention ({precision}), {MAX_FRAMES} frame slots x {T} + {MAX_PTRS} pointer slots, from {CKPT.name}"
    )
    dst = package(precision)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    ops = {}
    for fn in mlmodel._mil_program.functions.values():
        for op in fn.operations:
            ops[op.op_type] = ops.get(op.op_type, 0) + 1
    print(f"saved {dst}; ops: {json.dumps(dict(sorted(ops.items())))}")


def check(precision, units, n):
    import coremltools as ct

    t0 = time.perf_counter()
    model = ct.models.MLModel(str(package(precision)), compute_units=getattr(ct.ComputeUnit, units))
    load_s = time.perf_counter() - t0
    rows, worst = [], 0.0
    full = None
    for f, frames, ptrs, inp, want in fixture_cases():
        got = np.asarray(model.predict(inp)["out"])[0]
        st = stats(got, want)
        worst = max(worst, st["rel_l2"])
        rows.append({"frame": f, "memory_frames": frames, "pointer_tokens": ptrs, **st})
        print(f"f{f}: {frames} frames, {ptrs} pointers: {st}")
        full = inp
    for _ in range(2):
        model.predict(full)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(full)
        times.append((time.perf_counter() - t0) * 1e3)
    again = np.asarray(model.predict(full)["out"])
    same = bool(np.array_equal(again, np.asarray(model.predict(full)["out"])))
    res = {
        "precision": precision,
        "units": units,
        "package": str(package(precision)),
        "load_s": load_s,
        "worst_rel_l2": worst,
        "rows": rows,
        "predict_ms": {"n": n, "median": statistics.median(times), "min": min(times), "max": max(times)},
        "deterministic": same,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{precision}-{units}.json").write_text(json.dumps(res, indent=1))
    print(f"worst rel_l2 {worst:.3e}; predict median {res['predict_ms']['median']:.1f} ms (min {min(times):.1f}, n {n}); deterministic {same}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("torch-check")
    e = sub.add_parser("export")
    e.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    c = sub.add_parser("check")
    c.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    c.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_ONLY", "CPU_AND_NE"])
    c.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "torch-check":
        torch_check()
    elif a.cmd == "export":
        export(a.precision)
    else:
        check(a.precision, a.units, a.n)


if __name__ == "__main__":
    main()
