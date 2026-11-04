"""SAM 3.1's detector segmentation head as one CoreML ML Program.

    PY=tools/baselines/.venv-export/bin/python   (torch 2.7.0, coremltools 9.0, safetensors, numpy<2.3)
    $PY -s tools/baselines/coreml_seghead.py torch-check
    tools/locked.sh cml-seg $PY -s tools/baselines/coreml_seghead.py export --precision fp32
    tools/locked.sh cml-seg $PY -s tools/baselines/coreml_seghead.py check --precision fp32

Mirrors st-model's `Detector::forward_impl` from "Segmentation head." on (crates/st-model/src/detector.rs):

    t   = LayerNorm(mem)                                   cross_attn_norm
    ca  = MHA(t, prompt, prompt), 8 heads, padded prompt keys masked by prompt_bias
    x   = (mem + ca) as [1, 256, 72, 72]
    for skip in (fpn1, fpn0): x = relu(GroupNorm8(conv3x3(skip + nearest2x(x))))   pixel_decoder
    instance = conv1x1(x); semantic = conv1x1 -> 1 channel
    pred_masks = MLP3(hs) @ instance                       mask_predictor.mask_embed (ReLU between)

    inputs  mem [1,5184,256], prompt [1,33,256], prompt_bias [1,1,1,33] (0 real, -1e4 padding),
            fpn0 [1,256,288,288], fpn1 [1,256,144,144], hs [1,200,256] (the last decoder layer, normed)
    outputs pred_masks [1,200,288,288], semantic_seg [1,1,288,288]

None of these weights is touched by LoRA v3 or the shipped edit (they target the transformer and
dot_prod_scoring); spiketrack refuses `--seg-head coreml-*` with an adapter or edit that touches
`detector.segmentation_head.`. The package is derived from gated weights and lives next to them.
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
FIXTURE = REPO / "fixtures/detector/visem22_cell.safetensors"
OUT = REPO / "bench/results/coreml-seghead"
PREFIX = "detector.segmentation_head."
D, HEADS, Q, PMAX, GRID = 256, 8, 200, 33, 72
MASK = -1e4


def package(precision):
    return MODEL_DIR / "coreml" / f"sam3.1_seghead_{precision}.mlpackage"


def build():
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    w = {}
    with safe_open(str(CKPT), "pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                w[k[len(PREFIX):]] = f.get_tensor(k).float()
    P = lambda n: torch.nn.Parameter(w[n], requires_grad=False)

    class Seg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_w, self.ln_b = P("cross_attn_norm.weight"), P("cross_attn_norm.bias")
            self.in_w, self.in_b = P("cross_attend_prompt.in_proj_weight"), P("cross_attend_prompt.in_proj_bias")
            self.out_w, self.out_b = P("cross_attend_prompt.out_proj.weight"), P("cross_attend_prompt.out_proj.bias")
            self.pd_w = torch.nn.ParameterList([P(f"pixel_decoder.conv_layers.{i}.weight") for i in range(2)])
            self.pd_b = torch.nn.ParameterList([P(f"pixel_decoder.conv_layers.{i}.bias") for i in range(2)])
            self.gn_w = torch.nn.ParameterList([P(f"pixel_decoder.norms.{i}.weight") for i in range(2)])
            self.gn_b = torch.nn.ParameterList([P(f"pixel_decoder.norms.{i}.bias") for i in range(2)])
            self.inst_w, self.inst_b = P("instance_seg_head.weight"), P("instance_seg_head.bias")
            self.sem_w, self.sem_b = P("semantic_seg_head.weight"), P("semantic_seg_head.bias")
            self.me_w = torch.nn.ParameterList([P(f"mask_predictor.mask_embed.layers.{i}.weight") for i in range(3)])
            self.me_b = torch.nn.ParameterList([P(f"mask_predictor.mask_embed.layers.{i}.bias") for i in range(3)])

        def forward(self, mem, prompt, prompt_bias, fpn0, fpn1, hs):
            t = F.layer_norm(mem, (D,), self.ln_w, self.ln_b, 1e-5)
            wq, wk, wv = self.in_w.split(D)
            bq, bk, bv = self.in_b.split(D)
            hd = D // HEADS
            q = F.linear(t, wq, bq).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
            k = F.linear(prompt, wk, bk).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
            v = F.linear(prompt, wv, bv).reshape(1, -1, HEADS, hd).permute(0, 2, 1, 3)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=prompt_bias)
            ca = F.linear(a.permute(0, 2, 1, 3).reshape(1, -1, D), self.out_w, self.out_b)
            g = GRID
            x = (mem + ca).permute(0, 2, 1).reshape(1, D, g, g)
            for i, skip in enumerate((fpn1, fpn0)):
                x = skip + F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = F.conv2d(x, self.pd_w[i], self.pd_b[i], padding=1)
                x = F.relu(F.group_norm(x, 8, self.gn_w[i], self.gn_b[i], 1e-5))
            inst = F.conv2d(x, self.inst_w, self.inst_b)
            sem = F.conv2d(x, self.sem_w, self.sem_b)
            e = hs
            for i in range(3):
                e = F.linear(e, self.me_w[i], self.me_b[i])
                if i < 2:
                    e = F.relu(e)
            m = inst.shape[-1]
            masks = torch.matmul(e, inst.reshape(1, D, m * m)).reshape(1, Q, m, m)
            return masks, sem

    return Seg().eval()


def fixture_inputs():
    from safetensors import safe_open

    s = safe_open(str(FIXTURE), "np")
    g = s.get_tensor
    mem = g("enc.memory")[:, 0, :][None]
    prompt_full = g("enc.memory_text")[:, 0, :]          # [33, 256]: 32 text slots then the geometry CLS
    pad = g("in.text_pad")[0].astype(bool)               # [32], True on padding
    valid = np.concatenate([~pad, [True]])
    prompt = np.zeros((1, PMAX, D), np.float32)
    n = int(valid.sum())
    prompt[0, :n] = prompt_full[valid]                    # st-model packs the valid tokens then the CLS
    bias = np.full((1, 1, 1, PMAX), MASK, np.float32)
    bias[..., :n] = 0
    hs = g("dec.0")[5, :, 0, :][None]
    inputs = {"mem": mem.astype(np.float32), "prompt": prompt, "prompt_bias": bias,
              "fpn0": g("in.fpn0"), "fpn1": g("in.fpn1"), "hs": hs.astype(np.float32)}
    want = {"pred_masks": g("seg.pred_masks"), "semantic_seg": g("seg.semantic_seg")}
    return inputs, want, n


def stats(got, want):
    gg, ww = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    d = gg - ww
    with np.errstate(all="ignore"):
        dot = float(gg @ ww)
    return {"cosine": dot / float(np.linalg.norm(gg) * np.linalg.norm(ww)), "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ww)),
            "max_abs": float(np.abs(d).max()), "finite": bool(np.isfinite(gg).all())}


ORDER = ["mem", "prompt", "prompt_bias", "fpn0", "fpn1", "hs"]


def torch_check():
    import torch

    m = build()
    inputs, want, n = fixture_inputs()
    with torch.no_grad():
        masks, sem = m(*[torch.from_numpy(inputs[k]) for k in ORDER])
    print(f"{n} prompt tokens; pred_masks {stats(masks.numpy(), want['pred_masks'])}; semantic {stats(sem.numpy(), want['semantic_seg'])}")


def export(precision):
    import coremltools as ct
    import torch

    m = build()
    inputs, _, _ = fixture_inputs()
    with torch.no_grad():
        traced = torch.jit.trace(m, tuple(torch.from_numpy(inputs[k]) for k in ORDER))
    cp = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=k, shape=inputs[k].shape, dtype=np.float32) for k in ORDER],
        outputs=[ct.TensorType(name="pred_masks", dtype=np.float32), ct.TensorType(name="semantic_seg", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=cp,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_seghead.py"
    mlmodel.short_description = f"SAM 3.1 detector segmentation head ({precision}), from {CKPT.name}"
    dst = package(precision)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    print(f"saved {dst}")


def check(precision, units, n):
    import coremltools as ct

    model = ct.models.MLModel(str(package(precision)), compute_units=getattr(ct.ComputeUnit, units))
    inputs, want, _ = fixture_inputs()
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
    res = {"precision": precision, "units": units, "parity": parity,
           "predict_ms": {"n": n, "median": statistics.median(times), "min": min(times)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{precision}-{units}.json").write_text(json.dumps(res, indent=1))
    print(f"worst rel_l2 {max(v['rel_l2'] for v in parity.values()):.3e}; predict median {statistics.median(times):.1f} ms (min {min(times):.1f}, n {n})")


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
