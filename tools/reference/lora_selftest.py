"""Checks for the LoRA pipeline that need no cache and no backbone (CPU, < 4 GB).

    SAM3_DEVICE=cpu PYTHONWARNINGS=ignore tools/reference/.venv/bin/python -s tools/reference/lora_selftest.py [--device cpu|mps]

1. upstream's loss imports through the shims, and the shimmed focal loss equals upstream's own pure-torch branch;
2. `LoRAMultiheadAttention` with B = 0 equals upstream's `MultiheadAttention` (eval) for every mask variant the
   detector passes (none, key-padding, 2-D bool, 3-D float + key-padding, box-RPB bias), batch-first and not;
3. with random A, B the wrapper equals upstream's module holding the merged weights `W + (alpha/r) B A`
   (the claim crates/st-weights/src/lora.rs relies on);
4. the head-only detector (real checkpoint weights) gives the same outputs with aux on/off, and with LoRA B = 0
   equals the un-adapted head;
5. one training step on random features: finite loss, gradients reach the LoRA parameters, adapter file round-trips.
"""

import argparse
import copy
import pathlib
import resource
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lora_common as lc  # noqa: E402


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes


def mem(device):
    if device == "mps":
        return f", mps allocated {torch.mps.current_allocated_memory() / 2**30:.2f} GB, driver {torch.mps.driver_allocated_memory() / 2**30:.2f} GB"
    return ""


def probe_mps():
    """What MPS can do that the pipeline needs; each line is a fact about this torch build, none is fatal."""
    d = torch.device("mps")
    print(f"torch {torch.__version__}, mps built {torch.backends.mps.is_built()}, available {torch.backends.mps.is_available()}, "
          f"recommended max {torch.mps.recommended_max_memory() / 2**30:.1f} GB", flush=True)

    def attempt(name, fn):
        try:
            t0 = time.time()
            r = fn()
            torch.mps.synchronize()
            print(f"MPS ok   {name} {r if r is not None else ''} ({time.time() - t0:.2f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 (probe: report anything)
            print(f"MPS FAIL {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)

    x = torch.randn(2, 4, 8, device=d)
    # the vision trunk's RoPE (only needed by cache_detector_inputs.py): complex tables and products
    attempt("complex RoPE ops (view_as_complex, polar, complex mul)", lambda: tuple(torch.view_as_real(
        torch.view_as_complex(x.view(2, 4, 4, 2).contiguous()) * torch.polar(torch.ones(4, 4, device=d), torch.randn(4, 4, device=d))).shape))
    attempt("float64 tensor", lambda: torch.zeros(2, device=d, dtype=torch.float64).dtype)
    attempt("uint8 antialiased bilinear resize", lambda: tuple(torch.nn.functional.interpolate(torch.zeros(1, 3, 64, 64, device=d, dtype=torch.uint8), size=(32, 32), mode="bilinear", antialias=True).shape))

    def sdpa():
        q = torch.randn(1, 8, 5184, 32, device=d, requires_grad=True)
        torch.nn.functional.scaled_dot_product_attention(q, q, q).sum().backward()
        return f"{torch.mps.driver_allocated_memory() / 2**30:.2f} GB driver"
    attempt("SDPA 5184 tokens x 8 heads, forward + backward", sdpa)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        raise SystemExit(1)


def test_loss_shims():
    lc.install_loss_shims()
    from sam3.train.loss import loss_fns

    x = torch.randn(5, 1)
    t = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0]])
    a = loss_fns.sigmoid_focal_loss(x, t, num_boxes=5, alpha=0.5, gamma=0.0, triton=True)
    b = loss_fns.sigmoid_focal_loss(x, t, num_boxes=5, alpha=0.5, gamma=0.0, triton=False)
    check("focal shim == upstream pure-torch branch", torch.allclose(a, b, atol=1e-7), f"{a.item():.6f} vs {b.item():.6f}")
    lc.build_loss()
    check("loss wrapper builds", True)


def test_attention_parity():
    from sam3.model.model_misc import MultiheadAttention

    torch.manual_seed(0)
    e, h = 256, 8
    # (upstream class, batch_first): the encoder uses upstream's batch-first one, the decoder cross_attn
    # its sequence-first one, and the decoder's self_attn and ca_text are torch.nn.MultiheadAttention.
    for cls, batch_first in ((MultiheadAttention, True), (MultiheadAttention, False), (torch.nn.MultiheadAttention, False)):
        base = cls(embed_dim=e, num_heads=h, dropout=0.1, batch_first=batch_first).eval()
        with torch.no_grad():
            base.in_proj_bias.normal_(0, 0.1)
            base.out_proj.bias.normal_(0, 0.1)
        for p in base.parameters():
            p.requires_grad_(False)
        wrapped = lc.LoRAMultiheadAttention(base, "detector.test", 4, 8.0, 0.1).eval()
        bsz, tl, sl = 2, 7, 11
        shape = (lambda n: (bsz, n, e)) if batch_first else (lambda n: (n, bsz, e))
        q, kv = torch.randn(shape(tl)), torch.randn(shape(sl))
        kpm = torch.zeros(bsz, sl, dtype=torch.bool)
        kpm[:, -3:] = True
        cases = {
            "none": {},
            "key_padding": {"key_padding_mask": kpm},
            "bool2d": {"attn_mask": torch.rand(tl, sl) < 0.2},
            "float3d+kpm": {"attn_mask": torch.randn(bsz * h, tl, sl), "key_padding_mask": kpm},
            "rpb_bias": {"attn_bias": torch.randn(bsz, h, tl, sl)},
        }
        if cls is torch.nn.MultiheadAttention:
            cases.pop("rpb_bias")  # only upstream's own cross_attn gets the box-RPB bias
        for name, kw in cases.items():
            if name == "bool2d":
                kw["attn_mask"][:, 0] = False  # keep every row attendable
            with torch.no_grad():
                ref = base(q, kv, kv, **kw)[0]
                got = wrapped(q, kv, kv, **kw)[0]
            err = (ref - got).abs().max().item()
            check(f"attention parity B=0 [{name}, {cls.__module__.split(".")[0]} batch_first={batch_first}]", err < 2e-5, f"max|d|={err:.2e}")

        # random adapter == merged weights
        with torch.no_grad():
            for p in list(wrapped.lora_A) + list(wrapped.lora_B):
                p.normal_(0, 0.05)
        entries = lc.adapter_entries([wrapped])
        merged = copy.deepcopy(base)
        state = {"detector.test.in_proj_weight": merged.in_proj_weight.data, "detector.test.out_proj.weight": merged.out_proj.weight.data}
        lc.merge_into_state(state, entries)
        for name in [n for n in ("none", "float3d+kpm", "rpb_bias") if n in cases]:
            kw = cases[name]
            with torch.no_grad():
                ref = merged(q, kv, kv, **kw)[0]
                got = wrapped(q, kv, kv, **kw)[0]
            err = (ref - got).abs().max().item()
            check(f"adapter == merged weights [{name}, {cls.__module__.split(".")[0]} batch_first={batch_first}]", err < 5e-5, f"max|d|={err:.2e}")
        assert (wrapped(q, kv, kv)[0] - base(q, kv, kv)[0]).abs().max() > 1e-3, "random adapter had no effect"


def synthetic_inputs(device, bsz):
    g = torch.Generator().manual_seed(1)
    feats = torch.randn(bsz, lc.DIM, 72, 72, generator=g).to(device)
    pos = torch.randn(1, lc.DIM, 72, 72, generator=g).to(device)
    txt = torch.randn(32, bsz, lc.DIM, generator=g).to(device)
    mask = torch.zeros(bsz, 32, dtype=torch.bool)
    mask[:, 6:] = True
    return feats, pos, txt, mask.to(device)


def test_heads_and_training(device, bsz):
    t0 = time.time()
    det = lc.build_heads(device=device)
    print(f"heads built+loaded in {time.time() - t0:.1f}s, rss {rss_gb():.2f} GB", flush=True)
    feats, pos, txt, mask = synthetic_inputs(device, bsz)
    with torch.no_grad():
        a = lc.grounding_forward(det, feats, pos, txt, mask, aux=False)
        b = lc.grounding_forward(det, feats, pos, txt, mask, aux=True)
    check("aux on == aux off (final layer)", torch.allclose(a["pred_logits"], b["pred_logits"], atol=1e-4) and torch.allclose(a["pred_boxes"], b["pred_boxes"], atol=1e-4))
    check("aux outputs: 5 earlier decoder layers", len(b["aux_outputs"]) == 5 and "presence_logit_dec" in b["aux_outputs"][0])
    check("200 queries, presence per prompt", a["pred_logits"].shape == (bsz, 200, 1) and a["presence_logit_dec"].shape[0] == bsz, str(tuple(a["pred_logits"].shape)))

    wrappers = lc.attach_lora(det, seed=0)
    n_params = sum(p.numel() for p in lc.lora_parameters(det))
    check("30 attention modules, ~1M trainable (4 projections each)", len(wrappers) == 30 and n_params == 30 * 4 * (16 * 256 * 2), f"{n_params} params")
    with torch.no_grad():
        c = lc.grounding_forward(det, feats, pos, txt, mask, aux=False)
    err = (a["pred_logits"] - c["pred_logits"]).abs().max().item()
    check("LoRA with B=0 == un-adapted heads", err < 2e-3, f"max|d logit|={err:.2e}")

    wrapper, matcher = lc.build_loss()
    gts = [torch.tensor([[0.2, 0.2, 0.25, 0.26], [0.5, 0.5, 0.56, 0.55]]).numpy(), torch.zeros(0, 4).numpy()][:bsz]
    targets = lc.make_targets(gts, device)
    params = lc.lora_parameters(det)
    opt = torch.optim.AdamW(params, lr=1e-3)
    lc.set_lora_train(det, True)
    for step in range(2):
        torch.manual_seed(step)
        t0 = time.time()
        out = lc.grounding_forward(det, feats, pos, txt, mask, aux=True, act_ckpt=True)
        losses = lc.compute_loss(wrapper, matcher, out, targets)
        loss = losses["core_loss"]
        check(f"step {step} loss finite", bool(torch.isfinite(loss)), f"{loss.item():.3f} " + str({k: round(float(v), 3) for k, v in losses.items() if "_aux" not in k and "_f1" not in k and "acc" not in k}))
        loss.backward()
        gb = [w.lora_B[i].grad for w in wrappers for i in range(4)]
        ga = [w.lora_A[i].grad for w in wrappers for i in range(4)]
        nz_b = sum(g is not None and g.abs().sum() > 0 for g in gb)
        nz_a = sum(g is not None and g.abs().sum() > 0 for g in ga)
        print(f"  step {step}: grad nonzero B {nz_b}/120, A {nz_a}/120, {time.time() - t0:.1f}s, rss {rss_gb():.2f} GB{mem(device)}", flush=True)
        if step == 0:
            check("every B receives gradient at step 0 (A cannot: B = 0)", nz_b == 120 and nz_a == 0)
        else:
            check("A receives gradient once B != 0", nz_a == 120 and nz_b == 120)
        opt.step()
        opt.zero_grad()
    lc.set_lora_train(det, False)

    path = pathlib.Path(tempfile.mkdtemp(prefix="st_lora_selftest_")) / "adapter.safetensors"
    lc.save_adapter(path, lc.adapter_entries(wrappers), {"selftest": True})
    meta, entries = lc.load_adapter(path)
    check("adapter file: 120 modules, per-module rank/alpha/param", len(entries) == 120 and meta["modules"][0]["param"].endswith("in_proj_weight") and meta["modules"][3]["param"].endswith("out_proj.weight"))
    with torch.no_grad():
        d = lc.grounding_forward(det, feats, pos, txt, mask, aux=False)
    det3 = lc.build_heads(device=device)
    lc.attach_lora(det3, seed=9)
    lc.load_adapter_into(det3, path)
    with torch.no_grad():
        e = lc.grounding_forward(det3, feats, pos, txt, mask, aux=False)
    err = (d["pred_logits"] - e["pred_logits"]).abs().max().item()
    check("adapter round trip reproduces the adapted outputs", err < 1e-5, f"max|d logit|={err:.2e}")
    path.unlink()
    path.parent.rmdir()


def test_v2_heads_and_o2m(device, bsz):
    det = lc.build_heads(device=device)
    feats, pos, txt, mask = synthetic_inputs(device, bsz)
    g = torch.Generator().manual_seed(7)
    txt_pre = torch.randn(32, bsz, 1024, generator=g).to(device)
    with torch.no_grad():
        txt_ref = det.backbone.language_backbone.resizer(txt_pre)
        ref = lc.grounding_forward(det, feats, pos, txt_ref, mask, aux=False)

    wrappers = lc.attach_lora(det, seed=0, include_heads=True)
    entries0 = lc.adapter_entries(wrappers)
    check("41 modules (30 attn + 11 head linears) = 131 entries", len(wrappers) == 41 and len(entries0) == 131, f"{len(wrappers)}/{len(entries0)}")
    with torch.no_grad():
        got0 = lc.grounding_forward(det, feats, pos, txt, mask, aux=False, txt_pre_resizer=txt_pre)
    err0 = (ref["pred_logits"] - got0["pred_logits"]).abs().max().item()
    check("v2 LoRA B=0 == un-adapted heads (with resizer)", err0 < 2e-3, f"max|d logit|={err0:.2e}")

    wrapper, matcher = lc.build_loss(o2m=True, o2m_weight=2.0)
    gts = [torch.tensor([[0.2, 0.2, 0.25, 0.26], [0.5, 0.5, 0.56, 0.55]]).numpy(), torch.zeros(0, 4).numpy()][:bsz]
    targets = lc.make_targets(gts, device)
    params = lc.lora_parameters(det)
    opt = torch.optim.AdamW(params, lr=1e-3)
    lc.set_lora_train(det, True)
    for step in range(2):
        torch.manual_seed(10 + step)
        out = lc.grounding_forward(det, feats, pos, txt, mask, aux=True, act_ckpt=True, dac=True, txt_pre_resizer=txt_pre)
        check("o2m outputs present when dac=True", "pred_logits_o2m" in out and "pred_boxes_o2m" in out)
        losses = lc.compute_loss(wrapper, matcher, out, targets)
        loss = losses["core_loss"]
        check(f"v2+o2m step {step} loss finite", bool(torch.isfinite(loss)), f"{loss.item():.3f}")
        loss.backward()
        gb = [b.grad for w in wrappers for b in w.lora_B]
        ga = [a.grad for w in wrappers for a in w.lora_A]
        nz_b = sum(g is not None and g.abs().sum() > 0 for g in gb)
        nz_a = sum(g is not None and g.abs().sum() > 0 for g in ga)
        if step == 0:
            check("every v2 B (all 131 incl. resizer/bbox/scoring/presence) gets gradient at step 0", nz_b == 131 and nz_a == 0, f"B={nz_b}/131 A={nz_a}/131")
        else:
            check("every v2 A gets gradient at step 1", nz_a == 131 and nz_b == 131, f"B={nz_b}/131 A={nz_a}/131")
        opt.step()
        opt.zero_grad()
    lc.set_lora_train(det, False)

    entries = lc.adapter_entries(wrappers)
    fresh = lc.build_heads(device=device)
    state = {f"detector.{n}": p.data for n, p in fresh.named_parameters()}
    lc.merge_into_state(state, [(m, p, r, k, a, A.to(device), B.to(device)) for m, p, r, k, a, A, B in entries])
    with torch.no_grad():
        a_out = lc.grounding_forward(det, feats, pos, txt, mask, aux=False, txt_pre_resizer=txt_pre)
        b_out = lc.grounding_forward(fresh, feats, pos, txt, mask, aux=False, txt_pre_resizer=txt_pre)
    dl = (a_out["pred_logits"] - b_out["pred_logits"]).abs().max().item()
    db = (a_out["pred_boxes"] - b_out["pred_boxes"]).abs().max().item()
    dp = (a_out["presence_logit_dec"] - b_out["presence_logit_dec"]).abs().max().item()
    check("v2 merged weights == LoRA-wrapped forward", dl < 2e-3 and db < 1e-4 and dp < 1e-4, f"dl={dl:.2e} db={db:.2e} dp={dp:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--skip-heads", action="store_true", help="only the attention and loss checks")
    args = ap.parse_args()
    if args.device == "mps":
        probe_mps()
    test_loss_shims()
    test_attention_parity()
    if not args.skip_heads:
        test_heads_and_training(args.device, args.batch)
        test_v2_heads_and_o2m(args.device, args.batch)
    print("ALL PASS", f"peak rss {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
