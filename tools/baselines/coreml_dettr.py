"""SAM 3.1's detector transformer (geometry encoder, 6-layer fusion encoder and 6-layer DETR decoder with
box refinement) as one CoreML ML Program whose adaptable weights are inputs.

    PY=tools/baselines/.venv-export/bin/python   (torch 2.7.0, coremltools 9.0, safetensors, numpy<2.3)
    $PY -s tools/baselines/coreml_dettr.py torch-check
    tools/locked.sh cml-dettr $PY -s tools/baselines/coreml_dettr.py export --precision fp16
    tools/locked.sh cml-dettr $PY -s tools/baselines/coreml_dettr.py check --precision fp16 [--lora FILE]

Mirrors st-model's `Detector::forward_impl` from the top neck level to the last decoder layer
(crates/st-model/src/detector.rs); the scoring, the segmentation head and the box delta stay outside.
The encoders are tools/baselines/coreml_detenc.py's. The decoder, per layer:

    sine   = gen_sineembed_for_position(refs)                 (y, x, w, h), 128 interleaved sin/cos each
    pos    = [0; ref_point_head(sine)]                        presence row first
    rpb    = [0; boxRPB_y(dy) + boxRPB_x(dx)]                 log-spaced box offsets, per head
    x      = norm2(x + SA(x + pos, x + pos, x))
    x      = catext_norm(x + CA_text(x + pos, prompt, prompt))  padded prompt slots masked
    x      = norm1(x + CA(x + pos, mem + pos_mem, mem, bias rpb))
    x      = norm3(x + linear2(relu(linear1(x))))
    hs     = norm(x[1:]); presence = presence_token_head(presence_token_out_norm(x[0]))
    refs   = sigmoid(inverse_sigmoid(refs) + bbox_embed(hs))

The weights an adapter can change are inputs, not constants (`DYNAMIC`: every attention projection of
the fusion encoder and the decoder, bbox_embed and presence_token_head, the modules LoRA v3 adapts).
The caller passes them already folded, so one package serves the base model and every adapter that
touches only those weights. Everything else is a constant of the program.

    inputs  fpn2 [1,256,72,72], pos [1,5184,256], text [1,32,256], text_bias [1,1,1,33], one per DYNAMIC
            (named with '.' as '_')
    outputs mem [1,5184,256], geo [1,1,256], hs [6,200,256], presence [6], boxes [200,4]

Precisions: fp32; fp16; and fp16q32, fp16 except every op whose output has a dimension of 200 or 201
(the decoder's per-query path: queries, sine embeddings, reference-point and box heads, boxRPB, the
box refinement), which stays fp32. In pure fp16 the refinement drifts on rejected queries (boxes 2.9e-2
rel L2 over all 200), and E36 traced a held-out regression on video 47 to that program
(bench/results/e36-combined/README.md).

`boxes` are the refined references after the last layer (cxcywh, sigmoid space): `pred_boxes` before
`--post-box-delta`. The check feeds the base weights against upstream's fixture, and with `--lora` the
folded weights against this script's own fp32 mirror with the same fold. The package is derived from
gated weights and lives next to them.
"""

import argparse
import json
import math
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coreml_detenc as cd  # noqa: E402

DEC = "detector.transformer.decoder."
OUT = cd.REPO / "bench/results/coreml-dettr"
D, HEADS, TEXT, GRID, Q, LAYERS = cd.D, cd.HEADS, cd.TEXT, cd.GRID, 200, 6
DYNAMIC = (
    [f"{cd.ENC}layers.{i}.{a}.{p}" for i in range(6) for a in ("self_attn", "cross_attn_image")
     for p in ("in_proj_weight", "out_proj.weight")]
    + [f"{DEC}layers.{i}.{a}.{p}" for i in range(6) for a in ("self_attn", "ca_text", "cross_attn")
       for p in ("in_proj_weight", "out_proj.weight")]
    + [f"{DEC}bbox_embed.layers.{i}.weight" for i in range(3)]
    + [f"{DEC}presence_token_head.layers.{i}.weight" for i in range(3)]
)


def input_name(param):
    return param.replace(".", "_")


def package(precision):
    return cd.MODEL_DIR / "coreml" / f"sam3.1_dettr_{precision}.mlpackage"


def weights(lora=None):
    """Every parameter the program reads (name -> f32 torch tensor), with `lora` folded."""
    from safetensors import safe_open

    w = {}
    with safe_open(str(cd.CKPT), "pt") as f:
        for k in f.keys():
            if k.startswith((cd.GEO, cd.ENC, DEC)):
                w[k] = f.get_tensor(k).float().clone()
    folded = cd.lora_fold(w, lora) if lora else 0
    return w, folded


def build(w):
    import torch
    import torch.nn.functional as F

    def ln(x, p):
        return F.layer_norm(x, (D,), w[p + ".weight"], w[p + ".bias"], 1e-5)

    def mha(W, p, q_in, k_in, v_in, bias=None):
        wq, wk, wv = W(p + ".in_proj_weight").split(D)
        bq, bk, bv = w[p + ".in_proj_bias"].split(D)
        hd = D // HEADS
        q = F.linear(q_in, wq, bq).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
        k = F.linear(k_in, wk, bk).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
        v = F.linear(v_in, wv, bv).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return F.linear(a.permute(0, 2, 1, 3).reshape(1, -1, D), W(p + ".out_proj.weight"), w[p + ".out_proj.bias"])

    def mlp(W, p, x, n):
        for i in range(n):
            x = F.linear(x, W(f"{p}.layers.{i}.weight"), w[f"{p}.layers.{i}.bias"])
            if i + 1 < n:
                x = F.relu(x)
        return x

    def enc_layer(W, p, x, self_pos, memory, mem_pos, bias=None):
        t = ln(x, p + ".norm1")
        qk = t + self_pos if self_pos is not None else t
        x = x + mha(W, p + ".self_attn", qk, qk, t)
        t = ln(x, p + ".norm2")
        keys = memory + mem_pos if mem_pos is not None else memory
        x = x + mha(W, p + ".cross_attn_image", t, keys, memory, bias)
        t = ln(x, p + ".norm3")
        f = F.linear(F.relu(F.linear(t, w[p + ".linear1.weight"], w[p + ".linear1.bias"])),
                     w[p + ".linear2.weight"], w[p + ".linear2.bias"])
        return x + f

    dim_t = torch.tensor([10000.0 ** ((2 * (i // 2)) / 128) for i in range(128)], dtype=torch.float32)
    even = (torch.arange(128) % 2 == 0)
    coords = torch.arange(GRID, dtype=torch.float32) / GRID
    log8 = math.log2(8.0)

    def sine(refs):  # [Q, 4] -> [Q, 512], (y, x, w, h)
        parts = []
        for c in (1, 0, 2, 3):
            p = refs[:, c:c + 1] * (2 * math.pi) / dim_t
            parts.append(torch.where(even, torch.sin(p), torch.cos(p)))
        return torch.cat(parts, dim=1)

    def enc_log(d):
        v = d * 8.0
        return torch.sign(v) * torch.log2(v.abs() + 1.0) / log8

    def rpb(refs):  # [1, HEADS, 1 + Q, GRID * GRID]
        x0, x1 = refs[:, 0:1] - 0.5 * refs[:, 2:3], refs[:, 0:1] + 0.5 * refs[:, 2:3]
        y0, y1 = refs[:, 1:2] - 0.5 * refs[:, 3:4], refs[:, 1:2] + 0.5 * refs[:, 3:4]
        dx = torch.stack([enc_log(coords - x0), enc_log(coords - x1)], dim=-1)  # [Q, GRID, 2]
        dy = torch.stack([enc_log(coords - y0), enc_log(coords - y1)], dim=-1)
        ex = mlp(lambda n: w[n], DEC + "boxRPB_embed_x", dx, 2)  # [Q, GRID, HEADS]
        ey = mlp(lambda n: w[n], DEC + "boxRPB_embed_y", dy, 2)
        b = ey.permute(2, 0, 1)[:, :, :, None] + ex.permute(2, 0, 1)[:, :, None, :]  # [HEADS, Q, y, x]
        b = b.reshape(HEADS, Q, GRID * GRID)
        b = torch.cat([torch.zeros(HEADS, 1, GRID * GRID), b], dim=1)
        return b[None]

    def inverse_sigmoid(x):
        x = x.clamp(0.0, 1.0)
        return torch.log(x.clamp(min=1e-3) / (1.0 - x).clamp(min=1e-3))

    class DetTr(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Registered so that tracing keeps them as constants of the program.
            self.params = torch.nn.ParameterList(
                [torch.nn.Parameter(w[k], requires_grad=False) for k in sorted(w) if k not in DYNAMIC])

        def forward(self, fpn2, pos, text, text_bias, *dyn):
            given = dict(zip(DYNAMIC, dyn))

            def W(n):
                return given[n] if n in given else w[n]

            img = fpn2.reshape(1, D, GRID * GRID).permute(0, 2, 1)
            cls = w[cd.GEO + "cls_embed.weight"].reshape(1, 1, D)
            geo = ln(F.linear(cls, w[cd.GEO + "final_proj.weight"], w[cd.GEO + "final_proj.bias"]), cd.GEO + "norm")
            for i in range(3):
                geo = enc_layer(W, f"{cd.GEO}encode.{i}", geo, None, img, pos)
            geo = ln(geo, cd.GEO + "encode_norm")
            prompt = torch.cat([text, geo], dim=1)
            mem = img
            for i in range(6):
                mem = enc_layer(W, f"{cd.ENC}layers.{i}", mem, pos, prompt, None, text_bias)

            mem_keys = mem + pos
            refs = torch.sigmoid(w[DEC + "reference_points.weight"])
            x = torch.cat([w[DEC + "presence_token.weight"].reshape(1, 1, D),
                           w[DEC + "query_embed.weight"].reshape(1, Q, D)], dim=1)
            hs_all, pres_all = [], []
            for li in range(LAYERS):
                p = f"{DEC}layers.{li}"
                qpos = mlp(W, DEC + "ref_point_head", sine(refs), 2)
                qp = torch.cat([torch.zeros(1, D), qpos], dim=0)[None]
                bias = rpb(refs)
                qk = x + qp
                x = ln(x + mha(W, p + ".self_attn", qk, qk, x), p + ".norm2")
                x = ln(x + mha(W, p + ".ca_text", x + qp, prompt, prompt, text_bias), p + ".catext_norm")
                x = ln(x + mha(W, p + ".cross_attn", x + qp, mem_keys, mem, bias), p + ".norm1")
                f = F.linear(F.relu(F.linear(x, w[p + ".linear1.weight"], w[p + ".linear1.bias"])),
                             w[p + ".linear2.weight"], w[p + ".linear2.bias"])
                x = ln(x + f, p + ".norm3")
                hs = ln(x[0, 1:], DEC + "norm")
                pres = mlp(W, DEC + "presence_token_head", ln(x[0, :1], DEC + "presence_token_out_norm"), 3)
                refs = torch.sigmoid(inverse_sigmoid(refs) + mlp(W, DEC + "bbox_embed", hs, 3))
                hs_all.append(hs)
                pres_all.append(pres.reshape(1))
            return mem, geo, torch.stack(hs_all), torch.cat(pres_all), refs

    return DetTr().eval()


def fixture_inputs():
    inputs, want, n = cd.fixture_inputs()
    from safetensors import safe_open

    g = safe_open(str(cd.FIXTURE), "np").get_tensor
    want["hs"] = g("dec.0")[:, :, 0, :]
    want["presence"] = g("dec.2").reshape(LAYERS)
    want["boxes"] = g("out.pred_boxes")
    return inputs, want, n


def feed(inputs, w):
    return dict(inputs, **{input_name(k): w[k].numpy() for k in DYNAMIC})


ORDER = cd.ORDER
OUTS = ["mem", "geo", "hs", "presence", "boxes"]


def run_torch(m, inputs, w):
    import torch

    args = [torch.from_numpy(inputs[k]) for k in ORDER] + [w[k] for k in DYNAMIC]
    with torch.no_grad():
        return {k: v.numpy() for k, v in zip(OUTS, m(*args))}


def torch_check():
    w, _ = weights()
    m = build(w)
    inputs, want, n = fixture_inputs()
    got = run_torch(m, inputs, w)
    print(f"{n} valid text tokens")
    for k in want:
        print(k, cd.stats(got[k], want[k]))


def per_query(op):
    """Whether an op's output has a dimension of Q or Q + 1: the decoder's per-query path."""
    for o in op.outputs:
        shape = getattr(o, "shape", None) or ()
        if any(isinstance(d, int) and d in (Q, Q + 1) for d in shape):
            return True
    return False


def export(precision):
    import coremltools as ct
    import torch

    w, _ = weights()
    m = build(w)
    inputs, _, _ = fixture_inputs()
    args = [torch.from_numpy(inputs[k]) for k in ORDER] + [w[k] for k in DYNAMIC]
    with torch.no_grad():
        traced = torch.jit.trace(m, tuple(args))
    if precision == "fp16":
        cp = ct.precision.FLOAT16
    elif precision == "fp16q32":
        cp = ct.transform.FP16ComputePrecision(op_selector=lambda op: not per_query(op))
    else:
        cp = ct.precision.FLOAT32
    tensors = [ct.TensorType(name=k, shape=inputs[k].shape, dtype=np.float32) for k in ORDER]
    tensors += [ct.TensorType(name=input_name(k), shape=tuple(w[k].shape), dtype=np.float32) for k in DYNAMIC]
    mlmodel = ct.convert(
        traced,
        inputs=tensors,
        outputs=[ct.TensorType(name=k, dtype=np.float32) for k in OUTS],
        convert_to="mlprogram",
        compute_precision=cp,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_dettr.py"
    mlmodel.short_description = (f"SAM 3.1 detector encoders + DETR decoder ({precision}), from {cd.CKPT.name}; "
                                 f"{len(DYNAMIC)} adaptable weights are inputs")
    dst = package(precision)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    print(f"saved {dst}")


def check(precision, units, n, lora):
    import coremltools as ct

    model = ct.models.MLModel(str(package(precision)), compute_units=getattr(ct.ComputeUnit, units))
    inputs, want, _ = fixture_inputs()
    w, folded = weights(lora)
    if lora:
        want = run_torch(build(w), inputs, w)
    fed = feed(inputs, w)
    out = model.predict(fed)
    parity = {k: cd.stats(np.asarray(out[k]).reshape(np.shape(want[k])), want[k]) for k in want}
    for k, v in parity.items():
        print(k, v)
    for _ in range(2):
        model.predict(fed)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(fed)
        times.append((time.perf_counter() - t0) * 1e3)
    tag = cd.adapter_tag(lora)
    res = {"precision": precision, "units": units, "adapter": tag, "lora_modules_folded": folded,
           "reference": "this script's fp32 mirror with the same fold" if lora else "upstream fixture",
           "parity": parity, "predict_ms": {"n": n, "median": statistics.median(times), "min": min(times)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{precision}-{units}-{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"worst rel_l2 {max(v['rel_l2'] for v in parity.values()):.3e}; "
          f"predict median {statistics.median(times):.1f} ms (min {min(times):.1f}, n {n})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("torch-check")
    e = sub.add_parser("export")
    e.add_argument("--precision", choices=["fp32", "fp16", "fp16q32"], required=True)
    c = sub.add_parser("check")
    c.add_argument("--precision", choices=["fp32", "fp16", "fp16q32"], required=True)
    c.add_argument("--lora", default=None)
    c.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_ONLY", "CPU_AND_NE"])
    c.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "torch-check":
        torch_check()
    elif a.cmd == "export":
        export(a.precision)
    else:
        check(a.precision, a.units, a.n, a.lora)


if __name__ == "__main__":
    main()
