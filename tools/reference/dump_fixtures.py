"""Dump fp32 backbone fixtures from the official SAM 3.1 code on CPU.

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_fixtures.py

Needs external/sam3 on branch spiketrack-cpu (tools/reference/sam3-cpu.patch).
Writes fixtures/backbone/<image>.safetensors and fixtures/MANIFEST.json (both
git-ignored). Image 0 also gets every block output and the internals of block 0
(windowed) and block 7 (global) for per-op parity tests.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "fixtures"
VISEM = REPO / "data/visem/VISEM_Tracking_Train_v4/Train"
SAM3 = pathlib.Path("~/Code/research/external/sam3").expanduser()

IMAGES = [
    ("visem_22_f0", VISEM / "22/images/22_frame_0.jpg"),
    ("visem_47_f300", VISEM / "47/images/47_frame_300.jpg"),
    ("visem_82_f600", VISEM / "82/images/82_frame_600.jpg"),
    ("sam3_truck", SAM3 / "assets/images/truck.jpg"),
    ("sam3_groceries", SAM3 / "assets/images/groceries.jpg"),
]
STAGE_BLOCKS = (7, 15, 23, 31)
DETAIL_BLOCKS = (0, 7)


class Recorder:
    def __init__(self):
        self.t = {}
        self.ops = {}

    def put(self, name, x, op):
        self.t[name] = x.detach().float().contiguous().clone()
        self.ops[name] = op


def instrument(model, rec, detail):
    """Hooks that record into `rec`. `detail` adds per-op tensors."""
    from sam3.model import vitdet

    trunk = model.detector.backbone.vision_backbone.trunk
    neck = model.detector.backbone.vision_backbone
    hooks = []
    current = {"block": None}

    def fwd(name, op, inputs=False):
        def hook(_m, inp, out):
            if inputs:
                rec.put(name + ".in", inp[0], op + " input")
            rec.put(name, out, op)

        return hook

    hooks.append(trunk.patch_embed.register_forward_hook(
        fwd("trunk.patch_embed.out", "Conv2d(3,1024,k=14,s=14,bias=False) -> NHWC", inputs=detail)))
    hooks.append(trunk.ln_pre.register_forward_hook(
        fwd("trunk.ln_pre.out", f"{trunk.ln_pre} over patch_embed + tiled pos_embed (input recorded)", inputs=True)))
    for i, blk in enumerate(trunk.blocks):
        # Every block sets the index, so the RoPE/SDPA/MLP wrappers below never
        # attribute a later block's call to a detail block.
        hooks.append(blk.register_forward_pre_hook(
            lambda _m, inp, i=i: current.__setitem__("block", i)))
        if detail or i in STAGE_BLOCKS:
            hooks.append(blk.register_forward_hook(fwd(f"trunk.blocks.{i}.out", f"Block {i} output [B,72,72,1024]")))
        if detail and i in DETAIL_BLOCKS:
            p = f"trunk.blocks.{i}"
            kind = "global 1x5184" if blk.window_size == 0 else "windowed 9x576"
            hooks.append(blk.norm1.register_forward_hook(fwd(p + ".norm1", "LayerNorm(1024) over last axis", inputs=True)))
            hooks.append(blk.attn.qkv.register_forward_hook(fwd(p + ".attn.qkv", f"Linear(1024,3072,bias) [{kind}]")))
            hooks.append(blk.attn.proj.register_forward_hook(fwd(p + ".attn.proj", "Linear(1024,1024,bias)", inputs=True)))
            hooks.append(blk.norm2.register_forward_hook(fwd(p + ".norm2", "LayerNorm(1024)", inputs=True)))
            hooks.append(blk.mlp.fc2.register_forward_hook(fwd(p + ".mlp.fc2", "Linear(4736,1024,bias)", inputs=True)))

    real_rope = vitdet.apply_rotary_enc
    real_sdpa = F.scaled_dot_product_attention

    def rope(q, k, freqs_cis, repeat_freqs_k=False):
        qo, ko = real_rope(q, k, freqs_cis=freqs_cis, repeat_freqs_k=repeat_freqs_k)
        b = current["block"]
        if detail and b in DETAIL_BLOCKS:
            p = f"trunk.blocks.{b}.attn"
            op = "2-D axial RoPE, interleaved pairs (2i,2i+1) times freqs_cis[T,32]"
            rec.put(p + ".rope.q_in", q, op + " input q [B,H,T,64]")
            rec.put(p + ".rope.k_in", k, op + " input k")
            rec.put(p + ".rope.q_out", qo, op)
            rec.put(p + ".rope.k_out", ko, op)
            rec.put(p + ".rope.freqs_cis", torch.view_as_real(freqs_cis), "freqs_cis complex64 as view_as_real [T,32,2] (re,im)")
        return qo, ko

    def sdpa(q, k, v, *a, **kw):
        o = real_sdpa(q, k, v, *a, **kw)
        b = current["block"]
        if detail and b in DETAIL_BLOCKS and q.shape[-1] == 64 and current.get("sdpa_done") != b:
            p = f"trunk.blocks.{b}.attn"
            op = f"F.scaled_dot_product_attention(q,k,v) non-causal, no mask, scale 1/sqrt(64); args {a} {kw}"
            rec.put(p + ".sdpa.q", q, op + " q (after RoPE)")
            rec.put(p + ".sdpa.k", k, op + " k (after RoPE)")
            rec.put(p + ".sdpa.v", v, op + " v")
            rec.put(p + ".sdpa.out", o, op + " output [B,H,T,64], before head merge and proj")
            current["sdpa_done"] = b
        return o

    def addmm_act(activation, linear, mat1):
        y = F.linear(mat1, linear.weight, linear.bias)
        g = F.gelu(y)
        b = current["block"]
        if detail and b in DETAIL_BLOCKS and current.get("mlp_done") != b:
            p = f"trunk.blocks.{b}.mlp"
            rec.put(p + ".fc1.in", mat1, "Linear(1024,4736,bias) input")
            rec.put(p + ".fc1.out", y, "Linear(1024,4736,bias) output, before GELU")
            rec.put(p + ".gelu.out", g, "F.gelu exact (erf) of fc1.out")
            current["mlp_done"] = b
        assert activation in (F.gelu, torch.nn.GELU), activation
        return g

    vitdet.apply_rotary_enc = rope
    vitdet.addmm_act = addmm_act
    F.scaled_dot_product_attention = sdpa

    for neck_name in ("convs", "interactive_convs", "propagation_convs"):
        for lvl, seq in enumerate(getattr(neck, neck_name)):
            p = f"neck.{neck_name}.{lvl}"
            hooks.append(seq.register_forward_hook(fwd(p + ".out", f"FPN {neck_name}[{lvl}] output NCHW")))
            if detail and neck_name == "convs":
                for sub_name, sub in seq.named_children():
                    hooks.append(sub.register_forward_hook(fwd(f"{p}.{sub_name}", f"{type(sub).__name__} {sub}", inputs=True)))

    def restore():
        for h in hooks:
            h.remove()
        vitdet.apply_rotary_enc = real_rope
        F.scaled_dot_product_attention = real_sdpa

    return restore


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    sam3_ref.use_fp32_mlp()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not missing and not unexpected, (missing, unexpected)
    neck = model.detector.backbone.vision_backbone
    sam3_rev = subprocess.check_output(["git", "-C", str(SAM3), "rev-parse", "HEAD"], text=True).strip()
    (OUT / "backbone").mkdir(parents=True, exist_ok=True)
    manifest = {
        "generator": "tools/reference/dump_fixtures.py",
        "command": "SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_fixtures.py",
        "sam3_commit": sam3_rev,
        "checkpoint": str(sam3_ref.CKPT),
        "checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "torch": torch.__version__,
        "dtype_policy": "fp32 everywhere; upstream's bf16 addmm_act in the ViT MLP is replaced by fp32 linear + exact GELU",
        "preprocess": "PIL RGB, resize to 1008x1008 BILINEAR, /255, (x-0.5)/0.5",
        "files": [],
    }
    for idx, (name, path) in enumerate(IMAGES):
        detail = idx == 0
        rec = Recorder()
        _raw, x = sam3_ref.load_image(path)
        rec.put("input.image", x, "normalized input [1,3,1008,1008]")
        restore = instrument(model, rec, detail)
        t0 = time.time()
        with torch.inference_mode():
            out = neck(x)
        restore()
        dt = time.time() - t0
        sam3_pos = out[1] if isinstance(out, tuple) else None
        if idx == 0 and sam3_pos is not None:
            for lvl, pe in enumerate(sam3_pos):
                rec.put(f"neck.pos_enc.{lvl}", pe, "PositionEmbeddingSine for the level's feature map (same for all necks)")
        fpath = OUT / "backbone" / f"{name}.safetensors"
        save_file(rec.t, str(fpath))
        manifest["files"].append({
            "file": str(fpath.relative_to(REPO)),
            "image": str(path),
            "sha256": sha256(fpath),
            "forward_seconds_cpu": round(dt, 2),
            "tensors": {k: {"shape": list(v.shape), "dtype": "f32", "op": rec.ops[k]} for k, v in rec.t.items()},
        })
        print(f"{name}: {len(rec.t)} tensors, {fpath.stat().st_size / 1e6:.0f} MB, neck forward {dt:.1f}s", flush=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
