"""SAM 3.1's ViT trunk alone as a CoreML ML Program, to test whether it can replace the ojas trunk.

    PY=<venv with torch 2.7.0, coremltools 9.0, safetensors and upstream sam3 installed --no-deps>
    tools/locked.sh cml-export $PY -s tools/baselines/coreml_trunk.py export --precision fp16
    tools/locked.sh cml-export $PY -s tools/baselines/coreml_trunk.py export --precision fp32
    tools/locked.sh cml-check  $PY -s tools/baselines/coreml_trunk.py check --precision fp16 [-n 10]

export: builds upstream's ViT (sam3.model_builder._create_vit_backbone, real-valued RoPE so the graph
has no complex ops), loads `detector.backbone.vision_backbone.trunk.*` from the converted checkpoint,
and converts one forward to `$SPIKETRACK_MODEL_DIR/coreml/sam3.1_trunk_<precision>.mlpackage`. Precisions: fp32, fp16 (every op),
mixed (only linear, attention and the patch conv in fp16) and mixed-res (fp16 except the residual adds
and layer norms).
Input `image` [1, 3, 1008, 1008] f32, normalized (mean/std 0.5), as the ojas trunk takes it; output
`trunk` [1, 72, 72, 1024] f32 NHWC, the ojas trunk's layout (`Backbone::trunk`). The forward is
upstream's ViT.forward up to the last block, without its final NCHW permute. The package is derived
from gated weights: it lives next to them and is never committed.

check: feeds the upstream input of each backbone fixture to the package and compares the output with
the fixture's `trunk.blocks.31.out` (upstream fp32 on CPU), then times n calls after 2 warm-ups.
It also says whether two calls on the same input give byte-identical output. Results go to
bench/results/coreml-trunk/<precision>-<units>.json.

The venv (sdkv2's port calls coremltools 9.0 + torch 2.7.0 the ceiling of its export path):
    uv venv --python 3.12 tools/baselines/.venv-export
    uv pip install --python tools/baselines/.venv-export/bin/python torch==2.7.0 torchvision==0.22.0 \\
      coremltools==9.0 'numpy<2.3' timm tqdm ftfy==6.1.1 regex iopath huggingface_hub safetensors pillow \\
      opencv-python-headless pycocotools einops scipy scikit-image psutil 'setuptools<81'
    uv pip install --python tools/baselines/.venv-export/bin/python --no-deps -e ~/Code/research/external/sam3
(upstream at f1647e8 with tools/reference/sam3-cpu.patch applied, as in tools/reference/README.md).
"""

import argparse
import json
import os
import pathlib
import platform
import statistics
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/reference"))
MODEL_DIR = pathlib.Path(os.environ.get("SPIKETRACK_MODEL_DIR", "~/Code/research/models/sam3.1")).expanduser()
CKPT = MODEL_DIR / "sam3.1_multiplex.safetensors"
PREFIX = "detector.backbone.vision_backbone.trunk."
FIXTURES = ["visem_22_f0", "visem_47_f300", "visem_82_f600", "sam3_truck", "sam3_groceries"]
OUT = REPO / "bench/results/coreml-trunk"


def package(precision):
    return MODEL_DIR / "coreml" / f"sam3.1_trunk_{precision}.mlpackage"


def build_trunk():
    import cpu_shims

    cpu_shims.install()
    import torch
    from safetensors import safe_open
    from sam3 import model_builder as mb

    import sam3_ref

    sam3_ref.use_fp32_mlp()
    vit = mb._create_vit_backbone(compile_mode=None, use_fa3=False, use_rope_real=True)
    sd = {}
    with safe_open(str(CKPT), "pt") as f:
        complex_keys = set(json.loads(f.metadata()["complex64_as_f32_pairs"]))
        for k in f.keys():
            if k.startswith(PREFIX):
                t = f.get_tensor(k)
                sd[k[len(PREFIX):]] = torch.view_as_complex(t.contiguous()) if k in complex_keys else t
    missing, unexpected = vit.load_state_dict(sd, strict=False)
    # The real-valued RoPE buffers are derived at construction; take them from the checkpoint's tables.
    for blk in vit.blocks:
        a = blk.attn
        if getattr(a, "freqs_cis", None) is not None and a.use_rope_real:
            a.freqs_cis_real.copy_(a.freqs_cis.real)
            a.freqs_cis_imag.copy_(a.freqs_cis.imag)
    missing = [k for k in missing if not k.endswith(("freqs_cis_real", "freqs_cis_imag"))]
    if missing or unexpected:
        raise SystemExit(f"trunk weights: missing {missing}, unexpected {unexpected}")
    vit.eval()
    return vit


class TrunkNHWC:
    """ViT.forward without the NCHW permute of its output (no cls token, ln_post is Identity)."""

    def __new__(cls, vit):
        import torch
        from sam3.model.vitdet import get_abs_pos

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vit = vit

            def forward(self, image):
                v = self.vit
                x = v.patch_embed(image)
                h, w = x.shape[1], x.shape[2]
                x = x + get_abs_pos(v.pos_embed, v.pretrain_use_cls_token, (h, w), v.retain_cls_token, tiling=v.tile_abs_pos)
                x = v.ln_pre(x)
                for blk in v.blocks:
                    x = blk(x)
                return v.ln_post(x)

        return M().eval()


# The ops each precision runs in fp16; everything else stays fp32 (coremltools casts at the seams).
FP16_OPS = {
    # matmul-class ops only: the 128 linears, 32 attentions and the patch-embed conv.
    "mixed": {"linear", "scaled_dot_product_attention", "conv"},
    # the linears and the patch conv only; attention stays fp32.
    "mixed-lin": {"linear", "conv"},
}
# fp16 everywhere except the residual stream (its values reach |234|, where fp16's spacing is 0.125)
# and the norms that read it.
FP32_OPS_MIXED_RES = {"add", "layer_norm"}


def compute_precision(ct, precision):
    if precision == "fp16":
        return ct.precision.FLOAT16
    if precision == "fp32":
        return ct.precision.FLOAT32
    if precision == "mixed-res":
        return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in FP32_OPS_MIXED_RES)
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type in FP16_OPS[precision])


def export(precision):
    import coremltools as ct
    import torch

    vit = build_trunk()
    m = TrunkNHWC(vit)
    from safetensors.numpy import load_file

    fx = load_file(str(REPO / "fixtures/backbone/visem_22_f0.safetensors"))
    x = torch.from_numpy(fx["input.image"])
    with torch.no_grad():
        ref = m(x).numpy()
        want = fx["trunk.blocks.31.out"]
        print(f"torch wrapper vs fixture trunk.blocks.31.out: max abs {np.abs(ref - want).max():.3e}")
        traced = torch.jit.trace(m, x)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, 1008, 1008), dtype=np.float32)],
        outputs=[ct.TensorType(name="trunk", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=compute_precision(ct, precision),
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "SpikeTrack tools/baselines/coreml_trunk.py"
    mlmodel.short_description = f"SAM 3.1 ViT trunk ({precision}), from {CKPT.name}"
    dst = package(precision)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(dst))
    print(f"saved {dst}")


def stats(got, want):
    g, w = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    d = g - w
    # numpy's Accelerate matmul raises spurious divide/overflow flags on finite inputs; the values are checked below.
    with np.errstate(all="ignore"):
        dot = float(g @ w)
    return {
        "cosine": dot / float(np.linalg.norm(g) * np.linalg.norm(w)),
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(w)),
        "max_abs": float(np.abs(d).max()),
        "ref_max_abs": float(np.abs(w).max()),
        "finite": bool(np.isfinite(g).all()),
    }


def check(precision, n, units):
    import coremltools as ct
    from safetensors.numpy import load_file

    t0 = time.perf_counter()
    model = ct.models.MLModel(str(package(precision)), compute_units=getattr(ct.ComputeUnit, units))
    load_s = time.perf_counter() - t0
    parity = {}
    first_s = None
    for name in FIXTURES:
        fx = load_file(str(REPO / f"fixtures/backbone/{name}.safetensors"))
        if "trunk.blocks.31.out" not in fx:
            parity[name] = "no trunk.blocks.31.out in fixture"
            continue
        t0 = time.perf_counter()
        got = np.asarray(model.predict({"image": fx["input.image"]})["trunk"])
        first_s = first_s if first_s is not None else time.perf_counter() - t0
        parity[name] = stats(got.reshape(fx["trunk.blocks.31.out"].shape), fx["trunk.blocks.31.out"])
        print(name, parity[name])
    x = {"image": load_file(str(REPO / "fixtures/backbone/visem_22_f0.safetensors"))["input.image"]}
    a = np.asarray(model.predict(x)["trunk"])
    b = np.asarray(model.predict(x)["trunk"])
    deterministic = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    print(f"two calls on the same input byte-identical: {deterministic}")
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(x)
        times.append(time.perf_counter() - t0)
    res = {
        "package": str(package(precision)),
        "precision": precision,
        "compute_units": units,
        "macos": platform.mac_ver()[0],
        "load_s": round(load_s, 2),
        "first_call_s": round(first_s, 3) if first_s is not None else None,
        "n": n,
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "all_s": [round(t, 4) for t in times],
        "load_avg": os.getloadavg(),
        "repeat_call_byte_identical": deterministic,
        "parity_vs_upstream_fp32_trunk_blocks_31": parity,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{precision}-{units}.json"
    dst.write_text(json.dumps(res, indent=1) + "\n")
    print(f"median {res['median_s']} s, min {res['min_s']}, max {res['max_s']} over {n}; wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "check"])
    ap.add_argument("--precision", choices=["fp16", "fp32", "mixed", "mixed-lin", "mixed-res"], required=True)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_ONLY"])
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.precision)
    else:
        check(a.precision, a.n, a.units)


if __name__ == "__main__":
    main()
