"""LoRA domain adaptation of SAM 3.1's detector on VISEM-Tracking (gp-domain-lora). Offline PyTorch.

    tools/locked.sh lora-train bash tools/jobs/lora_train.sh          # the job script wraps this

What is trained
---------------
Rank-16 / alpha-32 / dropout-0.1 LoRA on the q, k, v and output projections of 30 attention modules:
the fusion encoder's `self_attn` and `cross_attn_image` (6 layers) and the DETR decoder's `self_attn`,
`cross_attn` (image) and `ca_text` (text) (6 layers), 983,040 parameters. The segmentation head's
prompt cross-attention (`segmentation_head.cross_attend_prompt`) is NOT adapted: a box + class +
presence loss never reads `pred_masks`, so it would get no gradient. The text encoder, the vision
backbone and every other weight stay frozen; their outputs are cached (cache_detector_inputs.py).

Supervision and loss: VISEM YOLO boxes, one prompt per class (`sperm`, `sperm cluster`, `pinhead
sperm`); a frame's classes that are present are positive prompts with that class's boxes, absent
classes and a few everyday objects are negative prompts (no boxes). The loss is upstream's own recipe
for box-only fine-tuning (train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml): the
Hungarian matcher `BinaryHungarianMatcherV2` and `Sam3LossWrapper` with `Boxes` (L1 5, GIoU 2) and
`IABCEMdetr` (ce 20, presence 20, pos_weight 10), on the final and the five auxiliary decoder
layers. For a negative prompt `IABCEMdetr` zeroes the per-query class loss and trains the presence
head to say "not here" (focal, alpha 0.5, gamma 0): the calibration the brief asks for. Simplification
vs the full recipe: no one-to-many (DAC) queries, so no o2m matcher.

Evaluation (end of run, held-out VISEM videos, same cached features, LoRA off vs on): recall and
precision at IoU 0.5 with zero_shot_check.py's greedy matching, scored two ways (`processor`:
sigmoid(logit) * sigmoid(presence), what Sam3Processor and so zero_shot_check.py report; `joint`:
sigmoid(logit), which already contains the presence and is what the video path thresholds), false
positives on absent-concept prompts (seen and unseen), and 10-bin ECE of the detection scores and of
the presence head. HOTA needs the tracker and the Rust runtime and is not computed here.
"""

import argparse
import json
import math
import os
import pathlib
import random
import resource
import sys
import time


def _arg(name, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


os.environ["SAM3_DEVICE"] = _arg("--device", "mps")  # upstream reads this at import time

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lora_common as lc  # noqa: E402

REPO = lc.REPO
THRESHOLDS = (0.3, 0.4, 0.5)
ECE_FLOOR = 0.05  # detection-level ECE over queries scoring at least this
N_BINS = 10


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def mem_gb(device):
    if device == "mps":
        return {"mps_allocated_gb": round(torch.mps.current_allocated_memory() / 2**30, 2),
                "mps_driver_gb": round(torch.mps.driver_allocated_memory() / 2**30, 2)}
    return {}


# ------------------------------------------------------------------ cache access

class Cache:
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.frames = self.meta["frames"]
        assert self.meta["prompts"] == lc.ALL_PROMPTS, "cache was built with a different prompt list"
        with safe_open(str(self.root / "text.safetensors"), "pt") as f:
            self.text = {p: (f.get_tensor(f"p{i}.feat"), f.get_tensor(f"p{i}.mask").bool()) for i, p in enumerate(lc.ALL_PROMPTS)}
        with safe_open(str(self.root / "pos.safetensors"), "pt") as f:
            self.pos = f.get_tensor("pos")[None]
        self._shards, self._gt = {}, {}

    def split(self, name):
        return [r for r in self.frames if r["split"] == name]

    def feat(self, rec):
        sh = self._shards.get(rec["video"])
        if sh is None:
            sh = self._shards[rec["video"]] = safe_open(str(self.root / "frames" / f"{rec['video']}.safetensors"), "pt")
        return sh.get_tensor(rec["key"]).float()

    def gt(self, rec, cls):
        k = (rec["video"], rec["frame"], cls)
        if k not in self._gt:
            self._gt[k] = lc.label_boxes(REPO / rec["label"], cls)  # normalized xyxy
        return self._gt[k]

    def present(self, rec):
        return [c for c in lc.CLASS_PROMPTS if len(self.gt(rec, c))]

    def batch(self, items, device):
        """items: [(frame record, prompt, class or None)] -> model inputs and normalized GT lists."""
        feats = torch.stack([self.feat(r) for r, _, _ in items]).to(device)
        txt = torch.stack([self.text[p][0] for _, p, _ in items], 1).to(device)
        mask = torch.stack([self.text[p][1] for _, p, _ in items]).to(device)
        gts = [self.gt(r, c) if c is not None else np.zeros((0, 4), np.float32) for r, _, c in items]
        return feats, self.pos.to(device), txt, mask, gts


def sample_items(cache, train_frames, rng, n, pos_frac):
    """n (frame, prompt, class) items: a positive (present class, its boxes) or a negative prompt."""
    items = []
    for _ in range(n):
        rec = rng.choice(train_frames)
        present = cache.present(rec)
        if present and rng.random() < pos_frac:
            c = rng.choice(present)
            items.append((rec, lc.CLASS_PROMPTS[c], c))
        else:
            absent = [lc.CLASS_PROMPTS[c] for c in lc.CLASS_PROMPTS if c not in present]
            items.append((rec, rng.choice(absent + lc.EVERYDAY_NEGATIVES), None))
    return items


# ------------------------------------------------------------------ evaluation

def ece(conf, correct, bins=N_BINS):
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, np.float64)
    if len(conf) == 0:
        return None
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.sum() * abs(correct[sel].mean() - conf[sel].mean())
    return round(float(total / len(conf)), 4)


def eval_plan(cache, frames):
    """Every (frame, prompt, class, role) the evaluation runs. role: pos / absent_class / seen_negative / unseen_negative."""
    plan = []
    for rec in frames:
        present = cache.present(rec)
        for c, p in lc.CLASS_PROMPTS.items():
            plan.append((rec, p, c, "pos" if c in present else "absent_class"))
        plan.append((rec, lc.PROBE_PROMPT, 0, "pos" if 0 in present else "absent_class"))
        plan += [(rec, p, None, "seen_negative") for p in lc.EVAL_SEEN_NEGATIVES]
        plan += [(rec, p, None, "unseen_negative") for p in lc.EVAL_UNSEEN_NEGATIVES]
    return plan


@torch.no_grad()
def run_eval_plan(det, cache, plan, device, batch):
    rows = []
    for i in range(0, len(plan), batch):
        chunk = plan[i:i + batch]
        feats, pos, txt, mask, _ = cache.batch([(r, p, None) for r, p, _, _ in chunk], device)
        out = lc.grounding_forward(det, feats, pos, txt, mask, aux=False)
        joint, pres, proc = lc.scores(out)
        boxes = out["pred_boxes_xyxy"].float().cpu().numpy()
        for j, (rec, prompt, cls, role) in enumerate(chunk):
            rows.append({"rec": rec, "prompt": prompt, "cls": cls, "role": role, "boxes": boxes[j],
                         "joint": joint[j].float().cpu().numpy(), "proc": proc[j].float().cpu().numpy(), "presence": float(pres[j])})
    return rows


def summarize(cache, rows):
    from zero_shot_check import match, iou as box_iou

    def px(r):
        return r["boxes"] * np.array([r["rec"]["w"], r["rec"]["h"]] * 2, dtype=np.float64)

    def gt_px(r, cls):
        return cache.gt(r["rec"], cls).astype(np.float64) * np.array([r["rec"]["w"], r["rec"]["h"]] * 2)

    out = {"detection": {}, "negatives": {}, "ece": {}}
    # --- detection per positive-class prompt (class 0 prompts over every held-out frame, as zero_shot_check.py)
    for prompt, cls in [(lc.CLASS_PROMPTS[0], 0), (lc.PROBE_PROMPT, 0), (lc.CLASS_PROMPTS[1], 1), (lc.CLASS_PROMPTS[2], 2)]:
        sel = [r for r in rows if r["prompt"] == prompt and r["cls"] == cls and (cls == 0 or r["role"] == "pos")]
        d = {}
        for kind in ("proc", "joint"):
            d[kind] = {}
            for t in THRESHOLDS:
                tp = npred = ngt = 0
                for r in sel:
                    g = gt_px(r, cls)
                    keep = r[kind] >= t
                    tp += match(px(r)[keep], r[kind][keep], g) if keep.any() else 0
                    npred += int(keep.sum())
                    ngt += len(g)
                d[kind][str(t)] = {"recall": round(tp / ngt, 4) if ngt else None, "precision": round(tp / npred, 4) if npred else None,
                                   "tp": tp, "predictions": npred, "gt": ngt, "frames": len(sel)}
        out["detection"][prompt] = d
        # detection-level calibration: queries scoring >= ECE_FLOOR, correct = greedy IoU 0.5 match to GT
        for kind in ("proc", "joint"):
            conf, ok = [], []
            for r in sel:
                keep = np.nonzero(r[kind] >= ECE_FLOOR)[0]
                order = keep[np.argsort(-r[kind][keep])]
                g = gt_px(r, cls)
                used = np.zeros(len(g), bool)
                m = box_iou(px(r)[order], g) if len(order) and len(g) else np.zeros((len(order), len(g)))
                for qi, q in enumerate(order):
                    hit = False
                    if len(g):
                        jx = int(np.argmax(np.where(used, -1, m[qi])))
                        if not used[jx] and m[qi, jx] >= 0.5:
                            used[jx], hit = True, True
                    conf.append(float(r[kind][q]))
                    ok.append(hit)
            out["ece"].setdefault("detection", {}).setdefault(prompt, {})[kind] = {"ece": ece(conf, ok), "queries": len(conf)}

    # --- absent-concept prompts: how often does anything fire?
    for role in ("absent_class", "seen_negative", "unseen_negative"):
        sel = [r for r in rows if r["role"] == role]
        d = {"items": len(sel), "presence_above_0.5": round(float(np.mean([r["presence"] > 0.5 for r in sel])), 4) if sel else None,
             "mean_presence": round(float(np.mean([r["presence"] for r in sel])), 4) if sel else None}
        for kind in ("proc", "joint"):
            for t in THRESHOLDS:
                d[f"{kind}_any_detection_at_{t}"] = round(float(np.mean([(r[kind] >= t).any() for r in sel])), 4) if sel else None
        out["negatives"][role] = d
    # --- presence head calibration over every (frame, prompt): present concept = 1, absent = 0
    conf = [r["presence"] for r in rows]
    label = [r["role"] == "pos" for r in rows]
    out["ece"]["presence"] = {"ece": ece(conf, label), "items": len(rows), "positives": int(sum(label))}
    return out


def evaluate(det, cache, device, batch, frames):
    plan = eval_plan(cache, frames)
    results = {}
    for name, enabled in (("zero_shot", False), ("lora", True)):
        lc.set_lora_enabled(det, enabled)
        t0 = time.time()
        rows = run_eval_plan(det, cache, plan, device, batch)
        results[name] = summarize(cache, rows)
        results[name]["seconds"] = round(time.time() - t0, 1)
        print(f"eval {name}: {len(rows)} (frame, prompt) items in {results[name]['seconds']}s", flush=True)
    lc.set_lora_enabled(det, True)
    results["frames"] = len(frames)
    results["videos"] = sorted({r["video"] for r in frames})
    results["items_per_frame"] = len(plan) // max(1, len(frames))
    return results


def table(results):
    lines = []
    s = lc.CLASS_PROMPTS[0]
    for kind, label in (("proc", "processor score (comparable to zero_shot_check.py)"), ("joint", "joint score (video path)")):
        lines += [f"class-0 detection at IoU 0.5, {label}, {results['frames']} held-out frames", "",
                  "| prompt | thr | zero-shot recall | zero-shot precision | LoRA recall | LoRA precision |", "|---|---|---|---|---|---|"]
        for prompt in (s, lc.PROBE_PROMPT):
            for t in THRESHOLDS:
                z = results["zero_shot"]["detection"][prompt][kind][str(t)]
                l = results["lora"]["detection"][prompt][kind][str(t)]
                f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
                lines.append(f"| {prompt} | {t} | {f(z['recall'])} | {f(z['precision'])} | {f(l['recall'])} | {f(l['precision'])} |")
        lines.append("")
    lines += ["absent-concept prompts: fraction of (frame, prompt) with any detection >= 0.5 (processor score) / presence head > 0.5", "",
              "| role | items | zero-shot any-det | LoRA any-det | zero-shot presence>0.5 | LoRA presence>0.5 |", "|---|---|---|---|---|---|"]
    for role in ("absent_class", "seen_negative", "unseen_negative"):
        z, l = results["zero_shot"]["negatives"][role], results["lora"]["negatives"][role]
        lines.append(f"| {role} | {z['items']} | {z['proc_any_detection_at_0.5']} | {l['proc_any_detection_at_0.5']} | {z['presence_above_0.5']} | {l['presence_above_0.5']} |")
    lines += ["", f"ECE (10 bins): presence head zero-shot {results['zero_shot']['ece']['presence']['ece']}, LoRA {results['lora']['ece']['presence']['ece']}; "
              f"class-0 detections (joint) zero-shot {results['zero_shot']['ece']['detection'][s]['joint']['ece']}, LoRA {results['lora']['ece']['detection'][s]['joint']['ece']}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ training

def lr_at(step, total, warmup, base, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def check_merged_equivalence(det, cache, device, adapter_path, rec):
    """The adapter merged into a fresh head's weights (merge_into_state, the formula lora.rs implements)
    must reproduce the LoRA-wrapped head on a real cached input."""
    meta, entries = lc.load_adapter(adapter_path)
    fresh = lc.build_heads(device=device)
    state = {f"detector.{n}": p.data for n, p in fresh.named_parameters()}
    lc.merge_into_state(state, [(m, p, r, k, a, A.to(device), B.to(device)) for m, p, r, k, a, A, B in entries])
    items = [(rec, lc.CLASS_PROMPTS[0], 0), (rec, lc.EVERYDAY_NEGATIVES[0], None)]
    feats, pos, txt, mask, _ = cache.batch(items, device)
    with torch.no_grad():
        lc.set_lora_enabled(det, True)
        a = lc.grounding_forward(det, feats, pos, txt, mask, aux=False)
        b = lc.grounding_forward(fresh, feats, pos, txt, mask, aux=False)
    return {"max_abs_logit": (a["pred_logits"] - b["pred_logits"]).abs().max().item(),
            "max_abs_box": (a["pred_boxes"] - b["pred_boxes"]).abs().max().item(),
            "max_abs_presence_logit": (a["presence_logit_dec"] - b["presence_logit_dec"]).abs().max().item()}


def save_checkpoint(run_dir, det, opt, step, cfg, name):
    wrappers = lc.lora_wrappers(det)
    lc.save_adapter(run_dir / name, lc.adapter_entries(wrappers),
                    {"dropout": cfg["dropout"], "trainer": "tools/reference/train_lora.py", "step": step, "seed": cfg["seed"], "run": cfg["run"]})
    torch.save({"step": step, "opt": opt.state_dict(), "params": [p.detach().cpu() for p in lc.lora_parameters(det)], "cfg": cfg}, run_dir / "checkpoint.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--cache", type=pathlib.Path, default=REPO / "fixtures/lora_cache")
    ap.add_argument("--run", default="visem-lora-r16")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="adapters + checkpoint (git-ignored); default fixtures/lora_runs/<run>")
    ap.add_argument("--results", type=pathlib.Path, default=None, help="log + eval (small); default bench/results/lora/<run>")
    ap.add_argument("--steps", type=int, default=600, help="optimizer steps")
    ap.add_argument("--batch", type=int, default=2, help="items per forward")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rank", type=int, default=lc.RANK)
    ap.add_argument("--alpha", type=float, default=lc.ALPHA)
    ap.add_argument("--dropout", type=float, default=lc.DROPOUT)
    ap.add_argument("--pos-frac", type=float, default=0.5, help="share of items that are positive prompts")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--no-act-ckpt", action="store_true", help="keep encoder activations (needs far more memory)")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="run at most this many more steps of the --steps schedule, checkpoint and exit (continue with --resume); bounds one locked job")
    ap.add_argument("--eval-only", type=pathlib.Path, default=None, help="skip training; evaluate this adapter")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=4)
    args = ap.parse_args()

    run_dir = args.out or REPO / "fixtures/lora_runs" / args.run
    res_dir = args.results or REPO / "bench/results/lora" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    cfg["command"] = " ".join(sys.argv)
    cfg["torch"] = torch.__version__

    cache = Cache(args.cache)
    train_frames, held_frames = cache.split("train"), cache.split("held_out")
    split = lc.load_split()
    assert not {r["video"] for r in train_frames} & set(split["held_out"]), "held-out video in the training cache"
    assert not {r["video"] for r in held_frames} & set(split["train"]), "training video in the held-out cache"
    print(f"cache: {len(train_frames)} train frames ({len({r['video'] for r in train_frames})} videos), "
          f"{len(held_frames)} held-out frames ({len({r['video'] for r in held_frames})} videos); device {args.device}", flush=True)

    device = args.device
    det = lc.build_heads(device=device)
    wrappers = lc.attach_lora(det, rank=args.rank, alpha=args.alpha, dropout=args.dropout, seed=args.seed)
    params = lc.lora_parameters(det)
    n_params = sum(p.numel() for p in params)
    print(f"{len(wrappers)} attention modules, {n_params:,} trainable parameters", flush=True)
    cfg["trainable_params"] = n_params

    if args.eval_only:
        lc.load_adapter_into(det, args.eval_only)
        adapter_path = args.eval_only
        log = {"config": cfg, "steps": []}
    else:
        assert train_frames, "no training frames in the cache"
        loss_wrapper, matcher = lc.build_loss()
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        start, log = 0, {"config": cfg, "steps": []}
        if args.resume and (run_dir / "checkpoint.pt").exists():
            ck = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
            with torch.no_grad():
                for p, v in zip(params, ck["params"]):
                    p.copy_(v)
            opt.load_state_dict(ck["opt"])
            start = ck["step"]
            if (res_dir / "train_log.json").exists():
                log = json.loads((res_dir / "train_log.json").read_text())
                log["steps"] = [s for s in log["steps"] if s["step"] <= start]
            print(f"resumed at step {start}", flush=True)
        lc.set_lora_train(det, True)
        t_start = time.time()
        end = args.steps if args.stop_after is None else min(args.steps, start + args.stop_after)
        if args.stop_after is not None and args.stop_after < 1:
            raise SystemExit("--stop-after must be at least 1")
        for step in range(start, end):
            t0 = time.time()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.warmup, args.lr)
            torch.manual_seed(args.seed * 1_000_003 + step)  # dropout stream: per step, so a resume replays it
            rng = random.Random(f"{args.seed}:{step}")
            agg, n_pos = {}, 0
            for _ in range(args.accum):
                items = sample_items(cache, train_frames, rng, args.batch, args.pos_frac)
                n_pos += sum(c is not None for _, _, c in items)
                feats, pos, txt, mask, gts = cache.batch(items, device)
                out = lc.grounding_forward(det, feats, pos, txt, mask, aux=True, act_ckpt=not args.no_act_ckpt)
                losses = lc.compute_loss(loss_wrapper, matcher, out, lc.make_targets(gts, device))
                loss = losses["core_loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"step {step}: non-finite loss {loss.item()} for {[(r['video'], r['frame'], p) for r, p, _ in items]}")
                (loss / args.accum).backward()
                for k in ("core_loss", "loss_bbox", "loss_giou", "loss_ce", "presence_loss"):
                    agg[k] = agg.get(k, 0.0) + float(losses[k]) / args.accum
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec = {"step": step + 1, **{k: round(v, 5) for k, v in agg.items()}, "grad_norm": round(gnorm, 4),
                   "lr": opt.param_groups[0]["lr"], "positives": n_pos, "items": args.batch * args.accum,
                   "sec": round(time.time() - t0, 2), "rss_gb": round(rss_gb(), 2), **mem_gb(device)}
            log["steps"].append(rec)
            if (step + 1) % args.log_every == 0 or step == start:
                print(json.dumps(rec), flush=True)
                (res_dir / "train_log.json").write_text(json.dumps(log, indent=1))
            if (step + 1) % args.save_every == 0 and step + 1 < args.steps:
                save_checkpoint(run_dir, det, opt, step + 1, cfg, f"adapter_step{step + 1}.safetensors")
        lc.set_lora_train(det, False)
        if end < args.steps:
            save_checkpoint(run_dir, det, opt, end, cfg, f"adapter_step{end}.safetensors")
            log["train_seconds_last_chunk"] = round(time.time() - t_start, 1)
            (res_dir / "train_log.json").write_text(json.dumps(log, indent=1))
            print(f"stopped at step {end} of {args.steps} (--stop-after {args.stop_after}); continue with --resume", flush=True)
            return
        save_checkpoint(run_dir, det, opt, args.steps, cfg, "adapter.safetensors")
        log["train_seconds"] = round(time.time() - t_start, 1)
        (res_dir / "train_log.json").write_text(json.dumps(log, indent=1))
        adapter_path = run_dir / "adapter.safetensors"
        print(f"adapter: {adapter_path}", flush=True)

    if not args.no_eval:
        assert held_frames, "no held-out frames in the cache"
        results = evaluate(det, cache, device, args.eval_batch, held_frames)
        results["merged_vs_wrapped"] = check_merged_equivalence(det, cache, device, adapter_path, held_frames[0])
        results["adapter"] = str(adapter_path)
        results["split_file"] = "tools/reference/visem_split.json"
        (res_dir / "eval.json").write_text(json.dumps(results, indent=1))
        md = table(results)
        (res_dir / "eval.md").write_text(md + "\n")
        print(md, flush=True)
        print("merged-weights vs LoRA-wrapped forward:", json.dumps(results["merged_vs_wrapped"]), flush=True)
    log["peak_rss_gb"] = round(rss_gb(), 2)
    (res_dir / "train_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
