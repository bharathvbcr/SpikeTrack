"""Recompute fixture tensors from their recorded inputs with independent torch
code, so the layout facts the Rust port relies on are checked, not assumed.

    tools/reference/.venv/bin/python -s tools/reference/check_fixtures.py
"""

import math
import pathlib
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

P = "detector.backbone.vision_backbone."


def axial_table(dim, side, scale):
    """freqs_cis as SAM 3 builds it: x then y frequencies, 1/theta^(4i/dim)."""
    f = 1.0 / (10000.0 ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    t = torch.arange(side * side, dtype=torch.float32)
    tx, ty = (t % side) * scale, torch.div(t, side, rounding_mode="floor") * scale
    ang = torch.cat([torch.outer(tx, f), torch.outer(ty, f)], dim=-1)
    return torch.stack([ang.cos(), ang.sin()], dim=-1)  # [T, dim/2, 2]


def rope_interleaved(x, tab):
    """Rotate pairs (2i, 2i+1) of the last axis by tab[t, i] = (cos, sin)."""
    xr = x.reshape(*x.shape[:-1], -1, 2)
    c, s = tab[..., 0], tab[..., 1]
    a, b = xr[..., 0], xr[..., 1]
    return torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)


def report(name, got, want):
    got, want = got.double().flatten(), want.double().flatten()
    err = (got - want).abs().max().item()
    cos = F.cosine_similarity(got, want, dim=0).item()
    ok = err <= 1e-4 * max(1.0, want.abs().max().item())
    print(f"{'ok  ' if ok else 'FAIL'} {name:55s} max|d|={err:.3e} cos={cos:.9f}")
    return ok


def main():
    ck = sam3_ref.load_state_dict()
    f = safe_open(str(REPO / "fixtures/backbone/visem_22_f0.safetensors"), "pt")
    g = f.get_tensor
    ok = True

    # RoPE tables: stored == formula; windowed = 24x24 integer grid, global = 72x72 scaled by 1/3.
    for b, side, scale in ((0, 24, 1.0), (7, 72, 24 / 72)):
        stored = torch.view_as_real(ck[f"{P}trunk.blocks.{b}.attn.freqs_cis"])
        ok &= report(f"blocks.{b} freqs_cis == axial formula (side {side}, scale {scale:.4f})", stored, axial_table(64, side, scale))
        ok &= report(f"blocks.{b} fixture freqs == checkpoint", g(f"trunk.blocks.{b}.attn.rope.freqs_cis"), stored)
    for b in range(32):
        side = 72 if b in (7, 15, 23, 31) else 24
        ref = ck[f"{P}trunk.blocks.{0 if side == 24 else 7}.attn.freqs_cis"]
        assert torch.equal(ck[f"{P}trunk.blocks.{b}.attn.freqs_cis"], ref), b
    print("ok   every windowed block shares block 0's table, every global block block 7's")

    # Patch embed, tiled pos embed, ln_pre.
    x = g("input.image")
    pe = F.conv2d(x, ck[f"{P}trunk.patch_embed.proj.weight"], stride=14).permute(0, 2, 3, 1)
    ok &= report("patch_embed = conv2d k14 s14 no bias, NHWC", g("trunk.patch_embed.out"), pe)
    pos = ck[f"{P}trunk.pos_embed"][:, 1:].reshape(1, 24, 24, 1024)
    pos = pos.repeat(1, 3, 3, 1)  # tile 3x3, cls dropped
    ok &= report("ln_pre input = patch_embed + pos_embed tiled 3x3", g("trunk.ln_pre.out.in"), pe + pos)
    ln = F.layer_norm(pe + pos, (1024,), ck[f"{P}trunk.ln_pre.weight"], ck[f"{P}trunk.ln_pre.bias"], 1e-5)
    ok &= report("ln_pre = LayerNorm eps 1e-5", g("trunk.ln_pre.out"), ln)

    # Block internals.
    for b in (0, 7):
        p = f"trunk.blocks.{b}"
        w = lambda n: ck[f"{P}{p}.{n}"]  # noqa: E731
        xin = g(f"{p}.norm1.in")
        n1 = F.layer_norm(xin, (1024,), w("norm1.weight"), w("norm1.bias"), 1e-6)
        ok &= report(f"{p}.norm1 = LayerNorm eps 1e-6", g(f"{p}.norm1"), n1)
        if b == 0:
            t = n1.reshape(1, 3, 24, 3, 24, 1024).permute(0, 1, 3, 2, 4, 5).reshape(9, 576, 1024)
        else:
            t = n1.reshape(1, 5184, 1024)
        qkv = F.linear(t, w("attn.qkv.weight"), w("attn.qkv.bias"))
        ok &= report(f"{p}.attn.qkv on window-partitioned tokens", g(f"{p}.attn.qkv").reshape(qkv.shape), qkv)
        q, k, v = qkv.reshape(t.shape[0], t.shape[1], 3, 16, 64).permute(2, 0, 3, 1, 4).unbind(0)
        ok &= report(f"{p} rope.q_in = q heads", g(f"{p}.attn.rope.q_in"), q)
        tab = torch.view_as_real(w("attn.freqs_cis"))
        ok &= report(f"{p} rope.q_out = interleaved RoPE(q_in)", g(f"{p}.attn.rope.q_out"), rope_interleaved(q, tab))
        ok &= report(f"{p} rope.k_out = interleaved RoPE(k_in)", g(f"{p}.attn.rope.k_out"), rope_interleaved(k, tab))
        qr, kr = g(f"{p}.attn.sdpa.q"), g(f"{p}.attn.sdpa.k")
        att = torch.softmax((qr.double() @ kr.double().transpose(-1, -2)) / math.sqrt(64), -1) @ g(f"{p}.attn.sdpa.v").double()
        ok &= report(f"{p} sdpa.out = softmax(qk^T/8) v (f64)", g(f"{p}.attn.sdpa.out"), att)
        h = g(f"{p}.mlp.fc1.in")
        y = F.linear(h, w("mlp.fc1.weight"), w("mlp.fc1.bias"))
        ok &= report(f"{p}.mlp.fc1", g(f"{p}.mlp.fc1.out"), y)
        ok &= report(f"{p}.mlp GELU is erf", g(f"{p}.mlp.gelu.out"), 0.5 * y * (1 + torch.erf(y / math.sqrt(2))))

    # Neck level 0 chain (scale 4).
    c = "neck.convs.0"
    nw = lambda n: ck[f"{P}convs.0.{n}"]  # noqa: E731
    x0 = g(f"{c}.dconv_2x2_0.in")
    ok &= report("neck input = block 31 output NCHW", x0, g("trunk.blocks.31.out").permute(0, 3, 1, 2))
    d0 = F.conv_transpose2d(x0, nw("dconv_2x2_0.weight"), nw("dconv_2x2_0.bias"), stride=2)
    ok &= report("neck convT k2 s2 1024->512", g(f"{c}.dconv_2x2_0"), d0)
    d1 = F.conv_transpose2d(F.gelu(d0), nw("dconv_2x2_1.weight"), nw("dconv_2x2_1.bias"), stride=2)
    ok &= report("neck GELU then convT k2 s2 512->256", g(f"{c}.dconv_2x2_1"), d1)
    c1 = F.conv2d(d1, nw("conv_1x1.weight"), nw("conv_1x1.bias"))
    c3 = F.conv2d(c1, nw("conv_3x3.weight"), nw("conv_3x3.bias"), padding=1)
    ok &= report("neck conv1x1 then conv3x3 pad 1 -> [1,256,288,288]", g(f"{c}.out"), c3)

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
