"""Oracle for the text-prompted detect -> associate -> track loop (gp-tracker-multiplex,
DETECT_TRACK_SPEC.md): upstream's builder-path model (sam3_ref.build_reference, CPU, fp32
policy) on the first frames of a VISEM video with a text prompt on frame 0, forward only.

    tools/locked.sh dt-ref bash tools/jobs/ref.sh tools/reference/detect_track_reference.py [--video 11 --frames 32 --prompt cell]

Records per visit of `_det_track_one_frame_impl` (the prompt frame is visited twice; key
prefix `f{t}.v{k}`), into fixtures/detect_track/<video>.safetensors:
- `det.*`: the detector's per-query scores and boxes, the keep mask, and the masks of kept
  queries only (`det.kept_index`, `det.kept_masks` at 288x288 logits);
- `prop.*`: tracker propagation output (low-res mask logits per track, object scores);
- `assoc.*`: every tensor field of `_associate_det_trk`'s result (IoM-derived matches);
- `mem.*`: the low-res masks and scores `_tracker_update_memories` encodes, and the memory
  features each tracker state stores for frame t after the visit (bf16 bits kept);
- `out.*`: `build_outputs` (per obj id mask at video resolution, packed u8).
Then every yielded frame of `propagate_in_video`: `y{t}.obj_ids`, `y{t}.probs`,
`y{t}.masks` (u8 [N, H, W]), `y{t}.boxes_xywh`. Non-tensor state (update plans, metadata,
association bookkeeping) goes to fixtures/detect_track/<video>.json. Also `frames`: the
model's input frames, F16 [T, 3, 1008, 1008], so the Rust side decodes no image.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
VISEM = REPO / "data/visem/VISEM_Tracking_Train_v4/Train"
OUT = REPO / "fixtures/detect_track"


def keep(x):
    # Upstream's postprocessed yields are numpy arrays (sam3_multiplex_tracking.py:789-795).
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    x = x.detach().cpu()
    if x.dtype == torch.bfloat16:
        return x.contiguous().clone()
    if x.dtype == torch.bool:
        return x.to(torch.uint8).contiguous().clone()
    if x.is_floating_point():
        return x.float().contiguous().clone()
    return x.contiguous().clone()


def jsonable(obj, depth=0):
    """Non-tensor state as JSON; tensors only by shape/dtype unless small."""
    if depth > 6:
        return repr(type(obj))
    if torch.is_tensor(obj):
        if obj.numel() <= 64:
            return {"tensor": obj.detach().cpu().float().tolist() if obj.is_floating_point() else obj.detach().cpu().tolist()}
        return {"tensor_shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        return jsonable(torch.from_numpy(obj), depth)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(jsonable(v, depth + 1) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, depth + 1) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dict__"):
        return {"__type__": type(obj).__name__, **{k: jsonable(v, depth + 1) for k, v in vars(obj).items()}}
    return repr(obj)


def tensors_of(prefix, obj, rec):
    if torch.is_tensor(obj):
        rec[prefix] = keep(obj)
    elif isinstance(obj, np.ndarray):
        rec[prefix] = keep(torch.from_numpy(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            tensors_of(f"{prefix}.{k}", v, rec)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            tensors_of(f"{prefix}.{i}", v, rec)
    elif hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        for k, v in vars(obj).items():
            tensors_of(f"{prefix}.{k}", v, rec)


def round_tracker_features_through_bf16():
    """Upstream's video detector casts the tracker's FPN features (interactive and
    propagation, all three levels) to bfloat16 before handing them to the tracker
    (sam3_multiplex_detector.py:563-586, for its all-gather; under CUDA autocast the
    tracker then computes on them in bf16). That cast is the only `NestedTensor.to(bf16)`
    on this path. On CPU the tracker's f32 convs refuse bf16 inputs, so the fp32 reference
    policy keeps upstream's rounding and computes in f32: a NestedTensor cast to bf16 is
    rounded through bf16 and handed back in its original dtype. Plain-tensor casts (the
    stored memory features) are untouched."""
    from sam3.model.data_misc import NestedTensor

    orig = NestedTensor.to

    def to(self, *args, **kwargs):
        if torch.bfloat16 in args or kwargs.get("dtype") is torch.bfloat16:
            src = self.tensors.dtype
            out = orig(self, *args, **kwargs)
            return type(self)(out.tensors.to(src), out.mask)
        return orig(self, *args, **kwargs)

    NestedTensor.to = to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="11")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--prompt", default="cell")
    args = ap.parse_args()

    # Upstream's folder loader wants "<index>.jpg" names: link the first frames.
    src = VISEM / args.video / "images"
    clip = OUT / f"{args.video}_frames"
    clip.mkdir(parents=True, exist_ok=True)
    for i in range(args.frames):
        f = src / f"{args.video}_frame_{i}.jpg"
        assert f.exists(), f
        link = clip / f"{i:05d}.jpg"
        if not link.exists():
            link.symlink_to(f)

    round_tracker_features_through_bf16()
    model, missing, unexpected = sam3_ref.build_reference()
    assert not unexpected, unexpected[:10]
    # The builder batches detection over 16 frames (batched_grounding_batch_size=16). On CPU in
    # fp32 that batch alone exceeded the lock's 28 GiB footprint cap (2026-10-27, killed at the
    # second batch). Batch composition changes only accumulation order, not the per-frame
    # algorithm (DETECT_TRACK_SPEC.md, detector cadence), and the Rust session detects one frame
    # at a time; the oracle uses 4.
    model.batched_grounding_batch_size = 4
    rec, meta = {}, {"video": args.video, "prompt": args.prompt, "frames": args.frames, "missing_keys": len(missing), "visits": [], "batched_grounding_batch_size": 4}
    cur = {"tag": None}
    visits = {}

    def wrap(name, fn):
        def inner(*a, **kw):
            out = fn(*a, **kw)
            tag = cur["tag"]
            if tag is None:
                return out
            if name == "run_backbone_and_detection":
                det_out, pos = out
                d = {k: v[0] for k, v in det_out.items()}
                p = pos[0]
                rec[f"{tag}.det.keep"] = keep(p)
                for k, v in d.items():
                    if k == "mask":
                        idx = torch.nonzero(p, as_tuple=False).flatten()
                        rec[f"{tag}.det.kept_index"] = keep(idx)
                        rec[f"{tag}.det.kept_masks"] = keep(v.index_select(0, idx.to(v.device)))
                    else:
                        rec[f"{tag}.det.{k}"] = keep(v)
            elif name == "run_tracker_propagation":
                masks, scores = out
                rec[f"{tag}.prop.low_res_masks"] = keep(masks)
                rec[f"{tag}.prop.obj_scores"] = keep(scores)
            elif name == "_associate_det_trk":
                tensors_of(f"{tag}.assoc", out, rec)
                meta.setdefault("assoc", {})[tag] = jsonable(out)
            elif name == "run_tracker_update_planning_phase":
                plan, md = out
                meta.setdefault("plan", {})[tag] = jsonable(plan)
                meta.setdefault("metadata", {})[tag] = jsonable(md)
            elif name == "_tracker_update_memories":
                for k, v in kw.items():
                    if torch.is_tensor(v):
                        rec[f"{tag}.mem.kw.{k}"] = keep(v)
                meta.setdefault("mem_kw", {})[tag] = jsonable({k: v for k, v in kw.items() if not torch.is_tensor(v) and k != "tracker_states_local"})
            elif name == "build_outputs":
                for oid, m in out.items():
                    rec[f"{tag}.out.{oid}"] = keep(m > 0 if m.dtype != torch.bool else m)
            return out

        return inner

    for name in ["run_backbone_and_detection", "run_tracker_propagation", "_associate_det_trk", "run_tracker_update_planning_phase", "_tracker_update_memories", "build_outputs"]:
        setattr(model, name, wrap(name, getattr(model, name)))

    impl = model._det_track_one_frame_impl

    def impl_wrapped(*a, **kw):
        t = kw["frame_idx"] if "frame_idx" in kw else a[0]
        k = visits.get(t, 0)
        visits[t] = k + 1
        cur["tag"] = f"f{t}.v{k}"
        meta["visits"].append(cur["tag"])
        t0 = time.time()
        out = impl(*a, **kw)
        # Memory each tracker state stored for frame t after this visit (the returned
        # state list, which includes states created for this frame's births).
        for si, st in enumerate(out[2]):
            od = st.get("output_dict", {}) if isinstance(st, dict) else {}
            for kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
                o = od.get(kind, {}).get(t)
                if o is not None and o.get("maskmem_features") is not None:
                    rec[f"{cur['tag']}.mem.state{si}.{kind}.maskmem_features"] = keep(o["maskmem_features"])
                    if o.get("obj_ptr") is not None:
                        rec[f"{cur['tag']}.mem.state{si}.{kind}.obj_ptr"] = keep(o["obj_ptr"])
        print(f"{cur['tag']}: {time.time() - t0:.1f} s", flush=True)
        cur["tag"] = None
        return out

    model._det_track_one_frame_impl = impl_wrapped

    with torch.inference_mode():
        state = model.init_state(resource_path=str(clip), offload_video_to_cpu=True)
        imgs = state["input_batch"].img_batch.tensors
        assert imgs.shape[0] == args.frames and imgs.shape[-1] == 1008, imgs.shape
        rec["frames"] = imgs.detach().cpu().to(torch.float16).contiguous()
        H, W = state["orig_height"], state["orig_width"]
        meta["hw"] = [H, W]
        model.add_prompt(state, frame_idx=0, text_str=args.prompt)
        for t, o in model.propagate_in_video(state, start_frame_idx=0, reverse=False):
            rec[f"y{t}.obj_ids"] = keep(o["out_obj_ids"])
            rec[f"y{t}.probs"] = keep(o["out_probs"])
            rec[f"y{t}.masks"] = keep(o["out_binary_masks"])
            rec[f"y{t}.boxes_xywh"] = keep(o["out_boxes_xywh"])
            print(f"yield {t}: {len(o['out_obj_ids'])} objects {o['out_obj_ids'].tolist()}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    save_file(rec, str(OUT / f"{args.video}.safetensors"))
    (OUT / f"{args.video}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rec)} tensors, {sum(v.numel() * v.element_size() for v in rec.values()) / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
