"""Dump an fp32 trunk + three-neck fixture from the official SAM 3.1 code at a smaller input size.

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_backbone_size.py \
        --size 672 --image data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_0.jpg \
        --out fixtures/backbone672/visem_22_f0.safetensors

Needs external/sam3 on branch spiketrack-cpu (tools/reference/sam3-cpu.patch). Runs the same
module classes as `dump_fixtures.py` (the `Sam3TriViTDetNeck` the multiplex builder makes), but
builds the PE ViT for `img_size=SIZE`: upstream's `_create_vit_backbone` hard-codes 1008, so
`model_builder.ViT` is wrapped to override that one argument. Nothing else changes; the
checkpoint loads as it is, except for the four global blocks' `attn.freqs_cis` buffers, which
upstream registers as persistent state sized for the 72 x 72 grid. They are left out here, so
those blocks keep the table upstream's own `_setup_rope_freqs` computes for the new grid
(`compute_axial_cis(..., scale_pos = 24 / grid)`); the other 28 blocks' 576-row tables and every
other tensor load unchanged.

Writes (all fp32): `input.image`, `trunk.ln_pre.out`, `trunk.blocks.{7,15,23,31}.out` (NHWC),
`neck.{convs,interactive_convs,propagation_convs}.{0,1,2}.out` (NCHW), `neck.pos_enc.{0,1,2}`,
and the two RoPE tables `trunk.blocks.{0,7}.attn.freqs_cis` as `view_as_real` `[T, 32, 2]`.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402  (installs the CPU shims on import)

STAGE_BLOCKS = (7, 15, 23, 31)
VIT_PREFIX = "detector.backbone.vision_backbone."


def build_neck(size):
    """The multiplex tri-neck with a PE ViT of `size` x `size` input (patch 14, window 24)."""
    from sam3 import model_builder as mb

    if size % 336:
        raise SystemExit(f"--size {size}: the 24-token windows need a multiple of 336")
    real_vit = mb.ViT

    def vit_at_size(**kwargs):
        assert kwargs["img_size"] == 1008, "the builder's ViT call changed; re-read _create_vit_backbone"
        kwargs["img_size"] = size
        return real_vit(**kwargs)

    mb.ViT = vit_at_size
    try:
        return mb._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=False)
    finally:
        mb.ViT = real_vit


def load_checkpoint(neck):
    """Load the converted checkpoint's vision backbone into `neck`, minus the grid-sized tables."""
    sd = sam3_ref.load_state_dict()
    own = neck.state_dict()
    state, skipped = {}, []
    for key, value in sd.items():
        if not key.startswith(VIT_PREFIX):
            continue
        name = key[len(VIT_PREFIX):]
        if name in own and own[name].shape != value.shape:
            skipped.append((name, tuple(value.shape), tuple(own[name].shape)))
            continue
        state[name] = value
    missing, unexpected = neck.load_state_dict(state, strict=False)
    return missing, unexpected, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sam3_ref.use_fp32_mlp()
    neck = build_neck(args.size)
    missing, unexpected, skipped = load_checkpoint(neck)
    expected = {f"trunk.blocks.{i}.attn.freqs_cis" for i in STAGE_BLOCKS}
    # Missing is exactly the four tables left out on purpose (they keep upstream's own, for this grid).
    assert set(missing) == expected and not unexpected, (missing, unexpected)
    assert {name for name, _, _ in skipped} == expected, skipped
    print(f"loaded; kept upstream's own freqs_cis for blocks {STAGE_BLOCKS}: {skipped[0][1]} in the checkpoint, {skipped[0][2]} here")
    neck.eval()

    rec = {}
    ops = {}

    def put(name, tensor, op):
        rec[name] = tensor.detach().float().contiguous().clone()
        ops[name] = op

    trunk = neck.trunk
    hooks = [trunk.ln_pre.register_forward_hook(lambda _m, _i, out: put("trunk.ln_pre.out", out, "ln_pre over patch_embed + tiled pos_embed"))]
    for i in STAGE_BLOCKS:
        hooks.append(trunk.blocks[i].register_forward_hook(
            lambda _m, _i, out, i=i: put(f"trunk.blocks.{i}.out", out, f"Block {i} output NHWC")))
    for name in ("convs", "interactive_convs", "propagation_convs"):
        for lvl, seq in enumerate(getattr(neck, name)):
            hooks.append(seq.register_forward_hook(
                lambda _m, _i, out, name=name, lvl=lvl: put(f"neck.{name}.{lvl}.out", out, f"FPN {name}[{lvl}] output NCHW")))
    for i in (0, 7):
        put(f"trunk.blocks.{i}.attn.freqs_cis", torch.view_as_real(trunk.blocks[i].attn.freqs_cis), "freqs_cis complex64 as view_as_real [T,32,2] (re,im)")

    _raw, x = sam3_ref.load_image(args.image, size=args.size)
    put("input.image", x, f"normalized input [1,3,{args.size},{args.size}] (PIL bilinear resize, /255, (x-0.5)/0.5)")
    t0 = time.time()
    with torch.inference_mode():
        out = neck(x)
    seconds = time.time() - t0
    for h in hooks:
        h.remove()
    for lvl, pe in enumerate(out[1]):
        put(f"neck.pos_enc.{lvl}", pe, "PositionEmbeddingSine for the level's feature map")

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(rec, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "generator": "tools/reference/dump_backbone_size.py",
        "size": args.size,
        "image": str(args.image),
        "sam3_commit": "f1647e8 + tools/reference/sam3-cpu.patch",
        "torch": torch.__version__,
        "dtype_policy": "fp32 everywhere; upstream's bf16 addmm_act in the ViT MLP is replaced by fp32 linear + exact GELU",
        "forward_seconds_cpu": round(seconds, 2),
        "sha256": digest,
        "tensors": {k: {"shape": list(v.shape), "op": ops[k]} for k, v in rec.items()},
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    print(f"{path}: {len(rec)} tensors, {path.stat().st_size / 1e6:.0f} MB, neck forward {seconds:.1f}s, sha256 {digest[:12]}")


if __name__ == "__main__":
    main()
