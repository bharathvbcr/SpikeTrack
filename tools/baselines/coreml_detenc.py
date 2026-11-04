"""SAM 3.1's detector encoders (geometry encoder + 6-layer fusion encoder) as one CoreML ML Program.

    PY=tools/baselines/.venv-export/bin/python   (torch 2.7.0, coremltools 9.0, safetensors, numpy<2.3)
    $PY -s tools/baselines/coreml_detenc.py torch-check
    tools/locked.sh cml-enc $PY -s tools/baselines/coreml_detenc.py export --precision fp16 [--lora FILE]
    tools/locked.sh cml-enc $PY -s tools/baselines/coreml_detenc.py check --precision fp16 [--lora FILE]

Mirrors st-model's `Detector::forward_impl` up to "DETR decoder." (crates/st-model/src/detector.rs):

    img  = fpn2 as tokens [1, 5184, 256]
    geo  = encode_norm(3 x EncLayer(norm(final_proj(cls)), memory img, keys img + pos))
    prompt = [text tokens; geo]
    mem  = 6 x EncLayer(mem, q/k + pos, memory prompt)        pre-norm, ReLU FFN, 8 heads
    EncLayer: x += SA(n1 x (+pos on q, k)); x += CA(n2 x, memory (+pos on keys)); x += FFN(n3 x)

Fixed shapes: the text is 32 slots, and `text_bias` [1, 1, 1, 33] masks padded slots (-1e4) in the
fusion encoder's cross-attention. The last key is the geometry token, always 0. st-model packs only the
valid tokens instead. Masking the padding gives the same softmax up to the key order of a sum, so the
program matches it to rounding, not bit for bit.

    inputs  fpn2 [1,256,72,72], pos [1,5184,256], text [1,32,256], text_bias [1,1,1,33]
    outputs mem [1,5184,256], geo [1,1,256]

LoRA: v3 targets every fusion-encoder layer (`detector.transformer.encoder.layers.*`), so a program
built with `--lora` bakes that adapter in. It is the program for that adapter only, and its package
name carries the adapter's sha256 prefix. The fold is st-weights' (`crates/st-weights/src/lora.rs`):
W' = W + (alpha / r) * B A in f32, per module row range. The shipped weight edit touches only
`detector.dot_prod_scoring`, not these weights. The package is derived from gated weights and lives
next to them.
"""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import struct
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = pathlib.Path(os.environ.get("SPIKETRACK_MODEL_DIR", "~/Code/research/models/sam3.1")).expanduser()
CKPT = MODEL_DIR / "sam3.1_multiplex.safetensors"
FIXTURE = REPO / "fixtures/detector/visem22_cell.safetensors"
OUT = REPO / "bench/results/coreml-detenc"
GEO = "detector.geometry_encoder."
ENC = "detector.transformer.encoder."
D, HEADS, TEXT, GRID = 256, 8, 32, 72
MASK = -1e4


def adapter_tag(lora):
    if not lora:
        return "base"
    return hashlib.sha256(pathlib.Path(lora).read_bytes()).hexdigest()[:12]


def package(precision, lora):
    return MODEL_DIR / "coreml" / f"sam3.1_detenc_{precision}_{adapter_tag(lora)}.mlpackage"


def lora_fold(w, lora):
    """Fold the adapter's modules that target these weights into `w` (name -> f32 torch tensor)."""
    import torch
    from safetensors import safe_open

    raw = pathlib.Path(lora).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    meta = json.loads(json.loads(raw[8:8 + n])["__metadata__"]["spiketrack_lora"])
    assert meta["format"] == "spiketrack-lora-v1", meta["format"]
    folded = 0
    with safe_open(str(lora), "pt") as f:
        for m in meta["modules"]:
            p = m["param"]
            if p not in w:
                continue
            a = f.get_tensor(f"{m['module']}.lora_A.weight").float()
            b = f.get_tensor(f"{m['module']}.lora_B.weight").float()
            s = np.float32(np.float64(m["alpha"]) / np.float64(m["rank"]))
            r0, r1 = m["rows"]
            w[p][r0:r1] = w[p][r0:r1] + float(s) * (b @ a)
            folded += 1
    return folded


def build(lora=None):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    w = {}
    with safe_open(str(CKPT), "pt") as f:
        for k in f.keys():
            if k.startswith((GEO, ENC)):
                w[k] = f.get_tensor(k).float().clone()
    folded = lora_fold(w, lora) if lora else 0
    P = lambda n: torch.nn.Parameter(w[n], requires_grad=False)

    def ln(x, p):
        return F.layer_norm(x, (D,), w[p + ".weight"], w[p + ".bias"], 1e-5)

    def mha(p, q_in, k_in, v_in, bias=None):
        wq, wk, wv = w[p + ".in_proj_weight"].split(D)
        bq, bk, bv = w[p + ".in_proj_bias"].split(D)
        hd = D // HEADS
        q = F.linear(q_in, wq, bq).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
        k = F.linear(k_in, wk, bk).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
        v = F.linear(v_in, wv, bv).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return F.linear(a.permute(0, 2, 1, 3).reshape(1, -1, D), w[p + ".out_proj.weight"], w[p + ".out_proj.bias"])

    def enc_layer(p, x, self_pos, memory, mem_pos, bias=None):
        t = ln(x, p + ".norm1")
        qk = t + self_pos if self_pos is not None else t
        x = x + mha(p + ".self_attn", qk, qk, t)
        t = ln(x, p + ".norm2")
        keys = memory + mem_pos if mem_pos is not None else memory
        x = x + mha(p + ".cross_attn_image", t, keys, memory, bias)
        t = ln(x, p + ".norm3")
        f = F.linear(F.relu(F.linear(t, w[p + ".linear1.weight"], w[p + ".linear1.bias"])),
                     w[p + ".linear2.weight"], w[p + ".linear2.bias"])
        return x + f

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Registered so that tracing keeps them as constants of the program.
            self.params = torch.nn.ParameterList([P(k) for k in sorted(w)])

        def forward(self, fpn2, pos, text, text_bias):
            img = fpn2.reshape(1, D, GRID * GRID).permute(0, 2, 1)
            cls = w[GEO + "cls_embed.weight"].reshape(1, 1, D)
            geo = ln(F.linear(cls, w[GEO + "final_proj.weight"], w[GEO + "final_proj.bias"]), GEO + "norm")
            for i in range(3):
                geo = enc_layer(f"{GEO}encode.{i}", geo, None, img, pos)
            geo = ln(geo, GEO + "encode_norm")
            prompt = torch.cat([text, geo], dim=1)
            mem = img
            for i in range(6):
                mem = enc_layer(f"{ENC}layers.{i}", mem, pos, prompt, None, text_bias)
            return mem, geo

    return Enc().eval(), folded


def fixture_inputs():
    from safetensors import safe_open

    s = safe_open(str(FIXTURE), "np")
    g = s.get_tensor
    pad = g("in.text_pad")[0].astype(bool)               # [32], True on padding
    text = g("in.text")[:, 0, :][None].astype(np.float32)  # [1, 32, 256]
    bias = np.zeros((1, 1, 1, TEXT + 1), np.float32)
    bias[0, 0, 0, :TEXT][pad] = MASK
    inputs = {"fpn2": g("in.fpn2").astype(np.float32), "pos": g("enc.pos_embed")[:, 0, :][None].astype(np.float32),
              "text": text, "text_bias": bias}
    want = {"mem": g("enc.memory")[:, 0, :][None], "geo": g("geo.0")}
    return inputs, want, int((~pad).sum())


def stats(got, want):
    gg, ww = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    d = gg - ww
    with np.errstate(all="ignore"):
        dot = float(gg @ ww)
    return {"cosine": dot / float(np.linalg.norm(gg) * np.linalg.norm(ww)), "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ww)),
            "max_abs": float(np.abs(d).max()), "finite": bool(np.isfinite(gg).all())}


ORDER = ["fpn2", "pos", "text", "text_bias"]


def torch_check():
    import torch

    m, _ = build()
    inputs, want, n = fixture_inputs()
    with torch.no_grad():
        mem, geo = m(*[torch.from_numpy(inputs[k]) for k in ORDER])
    print(f"{n} valid text tokens; mem {stats(mem.numpy(), want['mem'])}; geo {stats(geo.numpy(), want['geo'])}")


def export(precision, lora):
    import coremltools as ct
    import torch

    m, folded = build(lora)
    inputs, _, _ = fixture_inputs()
    with torch.no_grad():
        traced = torch.jit.trace(m, tuple(torch.from_numpy(inputs[k]) for k in ORDER))
    cp = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=k, shape=inputs[k].shape, dtype=np.float32) for k in ORDER],
        outputs=[ct.TensorType(name="mem", dtype=np.float32), ct.TensorType(name="geo", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=cp,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_detenc.py"
    mlmodel.short_description = (f"SAM 3.1 detector geometry + fusion encoders ({precision}), from {CKPT.name}, "
                                 f"adapter {adapter_tag(lora)} ({folded} LoRA modules folded)")
    dst = package(precision, lora)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    print(f"saved {dst} ({folded} LoRA modules folded)")


def check(precision, units, n, lora):
    import coremltools as ct
    import torch

    model = ct.models.MLModel(str(package(precision, lora)), compute_units=getattr(ct.ComputeUnit, units))
    inputs, want, _ = fixture_inputs()
    if lora:
        # The fixture is upstream's base model; with an adapter the reference is this script's own
        # fp32 torch mirror with the same fold (torch-check ties the mirror to the fixture).
        m, _ = build(lora)
        with torch.no_grad():
            mem, geo = m(*[torch.from_numpy(inputs[k]) for k in ORDER])
        want = {"mem": mem.numpy(), "geo": geo.numpy()}
    out = model.predict(inputs)
    parity = {k: stats(np.asarray(out[k]), want[k]) for k in want}
    print(parity)
    for _ in range(2):
        model.predict(inputs)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(inputs)
        times.append((time.perf_counter() - t0) * 1e3)
    res = {"precision": precision, "units": units, "adapter": adapter_tag(lora), "parity": parity,
           "reference": "torch mirror with the same fold" if lora else "upstream fixture",
           "predict_ms": {"n": n, "median": statistics.median(times), "min": min(times)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{precision}-{units}-{adapter_tag(lora)}.json").write_text(json.dumps(res, indent=1))
    print(f"worst rel_l2 {max(v['rel_l2'] for v in parity.values()):.3e}; predict median {statistics.median(times):.1f} ms (min {min(times):.1f}, n {n})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("torch-check")
    for name in ("export", "check"):
        p = sub.add_parser(name)
        p.add_argument("--precision", choices=["fp32", "fp16"], required=True)
        p.add_argument("--lora", default=None)
        if name == "check":
            p.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_ONLY", "CPU_AND_NE"])
            p.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "torch-check":
        torch_check()
    elif a.cmd == "export":
        export(a.precision, a.lora)
    else:
        check(a.precision, a.units, a.n, a.lora)


if __name__ == "__main__":
    main()
