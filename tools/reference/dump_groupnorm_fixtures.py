"""GroupNorm fixtures for Ojas's group_norm_forward (requested by ojas-78, lane B2): the
two nn.GroupNorm(8, 256) of SAM 3.1's pixel decoder, recorded with real activations.

    tools/reference/.venv/bin/python -s tools/reference/dump_groupnorm_fixtures.py

Rebuilds upstream's PixelDecoder (maskformer_segmentation.py, 3 stages, nearest) with the
checkpoint's weights and feeds it what upstream's segmentation head feeds it on the
detector fixture fixtures/detector/visem22_cell.safetensors: [fpn0 (288), fpn1 (144),
encoder memory + its prompt cross-attention, reshaped to 72x72] (maskformer_segmentation.py:131-147). Forward hooks
record each GroupNorm's input and output. As a check that the hooks see upstream's path,
the decoder's output must match the fixture's recorded `seg.pixel_embed`.

Writes fixtures/groupnorm/pixel_decoder.safetensors (git-ignored):
  gn0.{input,weight,bias,output}  [1,256,144,144] (norms.0)
  gn1.{input,weight,bias,output}  [1,256,288,288] (norms.1)
metadata: groups=8, eps=1e-5, source, pixel_embed max |diff| vs the fixture.
"""

import json
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402  (installs the CPU shims before sam3 imports)

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = REPO / "fixtures/detector/visem22_cell.safetensors"
OUT = REPO / "fixtures/groupnorm/pixel_decoder.safetensors"
PREFIX = "detector.segmentation_head.pixel_decoder."


def main():
    from sam3.model.maskformer_segmentation import PixelDecoder

    dec = PixelDecoder(hidden_dim=256, num_upsampling_stages=3, interpolation_mode="nearest").eval()
    with safe_open(sam3_ref.CKPT, "pt") as ck:
        sd = {k[len(PREFIX):]: ck.get_tensor(k) for k in ck.keys() if k.startswith(PREFIX)}
    dec.load_state_dict(sd, strict=True)  # raises on any missing or unexpected key
    with safe_open(str(FIXTURE), "pt") as fx:
        fpn0, fpn1 = fx.get_tensor("in.fpn0"), fx.get_tensor("in.fpn1")
        memory = fx.get_tensor("enc.memory")  # [5184, 1, 256], the encoder's image tokens
        # The head first cross-attends the image tokens to the prompt and adds the result
        # (maskformer_segmentation.py cross_attend_prompt; crates/st-model/src/detector.rs).
        ca = fx.get_tensor("seg.cross_attend_prompt.0")
        want = fx.get_tensor("seg.pixel_embed")
    enc72 = (memory + ca).permute(1, 2, 0).reshape(1, 256, 72, 72)
    rec = {}

    def hook(i):
        def fn(m, a, out):
            rec[f"gn{i}.input"] = a[0].detach().contiguous().clone()
            rec[f"gn{i}.output"] = out.detach().contiguous().clone()
            rec[f"gn{i}.weight"] = m.weight.detach().contiguous().clone()
            rec[f"gn{i}.bias"] = m.bias.detach().contiguous().clone()
            assert m.num_groups == 8 and m.eps == 1e-5, (m.num_groups, m.eps)
        return fn

    hooks = [dec.norms[i].register_forward_hook(hook(i)) for i in range(2)]
    with torch.inference_mode():
        got = dec([fpn0, fpn1, enc72])
    for h in hooks:
        h.remove()
    diff = (got - want).abs().max().item()
    assert diff < 1e-4, f"pixel decoder output differs from the fixture's seg.pixel_embed by {diff}"
    assert set(rec) == {f"gn{i}.{k}" for i in range(2) for k in ("input", "output", "weight", "bias")}, sorted(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    meta = {"groups": "8", "eps": "1e-5", "source": str(FIXTURE.relative_to(REPO)), "pixel_embed_max_abs_diff": f"{diff:.3e}"}
    save_file(rec, str(OUT), metadata=meta)
    print(json.dumps({k: list(v.shape) for k, v in rec.items()}), meta, f"{OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
