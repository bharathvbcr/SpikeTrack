"""Per-frame track bookkeeping on synthetic multi-frame sequences: the oracle for
st-model's tracker/bookkeeping.rs.

    SAM3_DEVICE=cpu PYTHONWARNINGS=ignore tools/reference/.venv/bin/python -s tools/reference/bookkeeping_oracle.py

What runs is upstream's own code (external/sam3, CPU-patched branch), on a stand-in
`self` and no model weights:

* `Sam3MultiplexBase.run_tracker_update_planning_phase` itself, unmodified, called as a
  bound method of a `Sam3MultiplexTrackingWithInteractivity` created with
  `object.__new__` (no `__init__`, so no tracker/detector). It therefore executes the real
  `_associate_det_trk`, `_process_hotstart_gpu`, `_recondition_masklets`,
  `_suppress_overlapping_based_on_recent_occlusion`, `realize_adt_result`,
  `get_new_det_gpu_ids` / `_drop_new_det_with_obj_limit`, `update_masklet_confirmation_status`
  and the GPU-metadata compaction / extension. The instance carries exactly the builder's
  values (model_builder.py:1170-1190, class defaults for the rest) and
  `world_size=1, rank=0, is_multiplex=True, bucket_capacity=16`.
* The stand-ins: `self.tracker` (an object with `input_mask_size`, a recording
  `add_new_masks`, and a no-op `propagate_in_video_preflight`; the memory encoder and mask
  decoder are the model calls this oracle leaves out), `self._tracker_update_memories`
  (no-op) and the "execution phase" (`tracker_states_local` is updated by a few lines here:
  newborn ids appended, removed ids dropped; upstream's execution phase only touches the
  tracker, never the metadata).
* `Sam3MultiplexTracking.propagate_in_video` (sam3_multiplex_tracking.py:302), the real generator, driven with a stub
  `_run_single_frame_inference` (replays the planning phase's output for the frame, with the
  `unconfirmed_obj_ids` expression of sam3_multiplex_tracking.py:657-661, transcribed) and a
  stub `_postprocess_output(_batched)`, to record which ids are hidden in which yielded frame
  and when a frame is released (postprocess_batch_size 1) and to check batch size 16 hides
  the same ids.

The synthetic world: ellipse entities with lifetimes, tracker masks that go empty for a
few frames, detections with scores at the thresholds, duplicate/false-positive/straddling/
boundary detections, crowds of more than 16 candidates. Association inputs and outputs are
recorded, so the Rust side can be fed the recorded `Association` (and also re-derives it
from the masks with associate.rs). Masks are 20x20, stored run-length encoded on
`logit > 0` (alternating runs, starting with a 0-run).

Tied scores among capped new detections are generated only for up to 16 candidates: upstream
orders those with `np.argsort`, which on this machine (numpy 2.5.4, arm64) returns the stable
order up to 16 elements and a scrambled one from 17 on (4000/4000 tied random inputs differ
from the stable order at 17-39 elements, 0/8000 below). Sequences that would hit a tie
among more than 16 capped candidates are regenerated with another seed.

Writes tools/reference/bookkeeping_cases.json (synthetic, committed).
"""

import contextlib
import io
import json
import logging
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cpu_shims  # noqa: E402

cpu_shims.install()
logging.disable(logging.CRITICAL)  # upstream warns on every capped frame
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking, Sam3MultiplexTrackingWithInteractivity  # noqa: E402
from sam3.model.sam3_video_base import MaskletConfirmationStatus, RealizedAssociateDetTrkresult  # noqa: E402

OUT = pathlib.Path(__file__).parent / "bookkeeping_cases.json"
H = W = 20
N_SEQUENCES = 30
TRUNCATED = {5: 2, 17: 6}  # sequence index -> frames processed before propagation stops


# ----------------------------------------------------------------------------- upstream stand-in


class TrackerStub:
    """What `_recondition_masklets` and the planning phase read off `self.tracker`."""

    input_mask_size = 64  # upstream: 1152; only the (unused here) binary mask size

    def __init__(self):
        self.calls = []

    def add_new_masks(self, inference_state, frame_idx, obj_ids, masks, reconditioning=False):
        assert reconditioning, "only reconditioning calls reach the tracker in the planning phase"
        self.calls.append([int(o) for o in obj_ids])

    def propagate_in_video_preflight(self, state, run_mem_encoder=True):
        pass


def make_model():
    model = object.__new__(Sam3MultiplexTrackingWithInteractivity)
    nn.Module.__init__(model)
    cfg = dict(
        rank=0, world_size=1, is_multiplex=True, bucket_capacity=16,
        score_threshold_detection=0.4, image_only_det_thresh=0.5, det_nms_thresh=0.1, det_nms_use_iom=True,
        assoc_iou_thresh=0.1, trk_assoc_iou_thresh=0.5, new_det_thresh=0.65,
        hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        init_trk_keep_alive=0, max_trk_keep_alive=8, min_trk_keep_alive=-4,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7, allow_unoccluded_to_suppress=False,
        suppress_det_close_to_boundary=True, decrease_trk_keep_alive_for_empty_masklets=False,
        o2o_matching_masklets_enable=False, fill_hole_area=0, sprinkle_removal_area=0,
        max_num_objects=16, max_num_kboxes=0, recondition_every_nth_frame=16,
        use_iom_recondition=True, iom_thresh_recondition=0.5, iou_thresh_recondition=0.8,
        masklet_confirmation_enable=True, masklet_confirmation_consecutive_det_thresh=3,
        reconstruction_bbox_iou_thresh=-1, reconstruction_bbox_det_score=0.8,
        postprocess_batch_size=16,
    )
    for k, v in cfg.items():
        setattr(model, k, v)
    model.tracker = TrackerStub()
    model._tracker_update_memories = lambda *a, **k: None
    # capture what the real code computes inside the planning phase
    model.captured = {}
    real_assoc = model._associate_det_trk
    real_supp = model._get_objects_to_suppress_based_on_most_recently_occluded

    def assoc(*a, **k):
        r = real_assoc(*a, **k)
        model.captured["adt"] = {
            "trk_is_unmatched": r.trk_is_unmatched.clone().tolist(), "trk_is_nonempty": r.trk_is_nonempty.clone().tolist(),
            "is_new_det": r.is_new_det.clone().tolist(), "det_to_max_trk": r.det_to_max_iou_trk_idx.clone().tolist(),
            "det_is_high_conf": r.det_is_high_conf.clone().tolist(), "det_is_high_overlap": r.det_is_high_iou.clone().tolist(),
            "im_mask": r.im_mask.clone().reshape(-1).tolist(),
        }
        return r

    def supp(*a, **k):
        r = real_supp(*a, **k)
        model.captured["suppress"] = [i for i, v in enumerate(r.tolist()) if v]
        return r

    model._associate_det_trk = assoc
    model._get_objects_to_suppress_based_on_most_recently_occluded = supp
    return model


# ----------------------------------------------------------------------------- synthetic world


def rle(m):
    flat = m.astype(np.uint8).ravel()
    runs, cur, n = [], 0, 0
    for v in flat:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    return runs


YY, XX = np.mgrid[0:H, 0:W]


def ellipse(cy, cx, ry, rx):
    return (1.0 - (((YY - cy) / ry) ** 2 + ((XX - cx) / rx) ** 2)).astype(np.float32)


EMPTY = np.full((H, W), -3.0, np.float32)
LN4 = np.float32(np.log(4.0))  # sigmoid(LN4) is 0.8 up to rounding: the recondition gate
F32 = np.float32


def f32_neighbours(x, k):
    out = [F32(x)]
    for i in range(k):
        out.append(np.nextafter(out[-1], F32(np.inf)))
    lo = [F32(x)]
    for i in range(k):
        lo.append(np.nextafter(lo[-1], F32(-np.inf)))
    return sorted(set(out + lo))


THRESH = [F32(0.4), F32(0.5), F32(0.65), F32(0.8)]
THRESH_NEAR = sorted({F32(v) for t in THRESH for v in (t, np.nextafter(t, F32(0)), np.nextafter(t, F32(1)))})
GATE_LOGITS = [F32(5.0), F32(3.0), F32(2.0), F32(1.0), F32(0.5)] + f32_neighbours(LN4, 3)


class Entity:
    def __init__(self, rng, t0, t1, detectable=True, near=None):
        self.t0, self.t1, self.detectable = t0, t1, detectable
        self.det_until = 10**9  # last frame the detector sees it
        if near is None:
            self.cy, self.cx = rng.uniform(4, H - 4), rng.uniform(4, W - 4)
        else:
            self.cy, self.cx = near
        self.ry, self.rx = rng.uniform(1.8, 4.0), rng.uniform(1.8, 4.0)
        self.vy, self.vx = rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)

    def alive(self, t):
        return self.t0 <= t < self.t1

    def mask(self, t, rng, jitter=0.5):
        dt = t - self.t0
        return ellipse(self.cy + self.vy * dt + rng.uniform(-jitter, jitter), self.cx + self.vx * dt + rng.uniform(-jitter, jitter),
                       self.ry * rng.uniform(0.92, 1.08), self.rx * rng.uniform(0.92, 1.08))


PROFILES = {
    # name: (initial entities, spawn prob/frame, p_det, fp prob, loss prob, dup det prob, boundary prob, thresholds prob)
    "plain": dict(n0=3, spawn=0.05, p_det=0.9, fp=0.05, loss=0.02, dup=0.05, boundary=0.05, thr=0.2),
    "lossy": dict(n0=4, spawn=0.04, p_det=0.8, fp=0.04, loss=0.18, dup=0.05, boundary=0.05, thr=0.3),
    "ghosts": dict(n0=2, spawn=0.02, p_det=0.85, fp=0.5, loss=0.03, dup=0.05, boundary=0.1, thr=0.3),
    "crowd": dict(lo=0.6, n0=30, spawn=0.3, p_det=0.9, fp=0.2, loss=0.03, dup=0.1, boundary=0.1, thr=0.0),
    "straddle": dict(n0=0, spawn=0.0, p_det=0.9, fp=0.0, loss=0.0, dup=0.0, boundary=0.0, thr=0.0, straddle=True),
    "thresholds": dict(n0=6, spawn=0.1, p_det=0.9, fp=0.1, loss=0.05, dup=0.1, boundary=0.1, thr=0.9),
    "recond": dict(lo=0.85, n0=4, spawn=0.0, p_det=1.0, fp=0.0, loss=0.0, dup=0.0, boundary=0.0, thr=0.0),
    # detected for the first 7 / 6 frames after birth, then never: the 8th unmatched frame lands exactly on / just before
    # the last frame of the 15-frame hot-start window
    "win7": dict(n0=2, spawn=0.0, p_det=1.0, fp=0.0, loss=0.0, dup=0.0, boundary=0.0, thr=0.0, window=7),
    "win6": dict(n0=2, spawn=0.0, p_det=1.0, fp=0.0, loss=0.0, dup=0.0, boundary=0.0, thr=0.0, window=6),
    "sparse": dict(n0=1, spawn=0.15, p_det=0.5, fp=0.1, loss=0.05, dup=0.0, boundary=0.3, thr=0.3),
}
PROFILE_ORDER = ["plain", "lossy", "ghosts", "crowd", "straddle", "thresholds", "sparse", "lossy", "recond", "crowd",
                 "ghosts", "straddle", "thresholds", "win7", "win6", "lossy"]


class TieCap(Exception):
    pass


def det_score(rng, p, distinct):
    if not distinct and rng.random() < p["thr"]:
        return F32(rng.choice(THRESH_NEAR))
    return F32(rng.uniform(p.get("lo", 0.3), 1.0))


def gen_detections(rng, p, t, entities, ghosts_out, live_pairs, distinct):
    """Detections for frame t: list of (logits, score, boundary, source entity or None)."""
    dets = []
    for e in entities:
        if e.alive(t) and e.detectable and t <= e.det_until and rng.random() < p["p_det"]:
            dets.append((e.mask(t, rng), det_score(rng, p, distinct), False, e))
            if rng.random() < p["dup"]:
                m = e.mask(t, rng)
                m[rng.integers(H // 3, H):, :] = -1.0  # a piece of it
                dets.append((m, det_score(rng, p, distinct), False, e))
    if rng.random() < p["fp"]:
        g = Entity(rng, t, t + int(rng.integers(2, 14)), detectable=False)
        ghosts_out.append(g)
        dets.append((g.mask(t, rng), F32(rng.uniform(0.5, 1.0)), False, g))
    if p.get("straddle") and live_pairs is not None:
        a, b = live_pairs
        if a.alive(t) and b.alive(t):
            ma, mb = a.mask(t, rng, 0.0), b.mask(t, rng, 0.0)
            part = (ma > 0) & (XX > a.cx + 0.5) | (mb > 0) & (XX < b.cx - 0.5)
            straddle = np.where(part | (YY >= H - 4), 1.0, -1.0).astype(np.float32)
            dets.append((straddle, F32(0.5), False, None))
    for _ in range(int(rng.random() < p["boundary"]) + int(rng.random() < p["boundary"])):
        dets.append((ellipse(rng.uniform(0, 3), rng.uniform(2, W - 2), 2.5, 2.5), F32(rng.uniform(0.5, 1.0)), True, None))
    if rng.random() < 0.15:  # a detection below the keep threshold
        dets.append((ellipse(rng.uniform(4, H - 4), rng.uniform(4, W - 4), 3, 3), F32(rng.uniform(0.0, 0.4)), False, None))
    kept = [d for d in dets if d[1] > F32(0.4) and not d[2]]
    unkept = [d for d in dets if not (d[1] > F32(0.4) and not d[2])]
    order = rng.permutation(len(kept))
    return [kept[i] for i in order] + unkept


def run_sequence(profile, seed, n_process=None):
    """One sequence through upstream's planning phase. Returns the recorded case. `n_process` cuts
    propagation short (`max_frame_num_to_track`), the only way a hidden-id look-ahead can miss."""
    p = PROFILES[profile]
    rng = np.random.default_rng(seed)
    n_frames = int(rng.integers(20, 37))
    start = int(rng.choice([0, 0, 0, 5, 11]))
    last = n_frames - 1 if n_process is None else min(n_frames - 1, start + n_process - 1)
    distinct = profile == "crowd"
    entities = [Entity(rng, start, n_frames if rng.random() < 0.8 else int(rng.integers(start + 3, n_frames + 1))) for _ in range(p["n0"])]
    live_pairs = None
    if p.get("straddle"):
        a = Entity(rng, start, n_frames, near=(8.0, 5.5))
        b = Entity(rng, start + 1, n_frames, near=(8.0, 12.5))
        a.ry = a.rx = b.ry = b.rx = 3.2
        a.vy = a.vx = b.vy = b.vx = 0.0
        entities = [a, b]
        live_pairs = (a, b)
    if "window" in p:
        n_frames = max(n_frames, start + 18)
        last = n_frames - 1 if n_process is None else min(n_frames - 1, start + n_process - 1)
        for e in entities:
            e.t1, e.det_until = n_frames, e.t0 + p["window"]
    ghosts = []
    bind = {}  # obj id -> {"ent", "lost_until"}
    model = make_model()
    meta = model._initialize_metadata()
    states = []
    frames = []
    outs_by_frame = {}

    for t in range(start, last + 1):
        if rng.random() < p["spawn"] and t > start:
            entities.append(Entity(rng, t, n_frames if rng.random() < 0.7 else min(n_frames, t + int(rng.integers(2, 15)))))
        dets = gen_detections(rng, p, t, entities, ghosts, live_pairs, distinct)
        if distinct and dets:  # crowd: all scores distinct
            used = set()
            fixed = []
            for m, s, b, src in dets:
                while float(s) in used:
                    s = np.nextafter(s, F32(2.0))
                used.add(float(s))
                fixed.append((m, s, b, src))
            dets = fixed
        prev_ids = meta["obj_ids_all_gpu"].copy()
        trk_logits, trk_scores = [], []
        for oid in prev_ids.tolist():
            tr = bind[oid]
            if t >= tr["lost_until"] and rng.random() < p["loss"]:
                tr["lost_until"] = t + int(rng.integers(1, 7))
            present = tr["ent"].alive(t) and t >= tr["lost_until"]
            trk_logits.append(tr["ent"].mask(t, rng) if present else EMPTY.copy())
            trk_scores.append(F32(rng.choice(GATE_LOGITS)) if present else F32(-2.0))
        n, nd = len(prev_ids), len(dets)
        trk_t = torch.from_numpy(np.stack(trk_logits)) if n else torch.zeros(0, H, W)
        trk_scores_t = torch.tensor(trk_scores, dtype=torch.float32) if n else torch.zeros(0)
        det_t = torch.from_numpy(np.stack([d[0] for d in dets])) if nd else torch.zeros(0, H, W)
        det_scores_t = torch.tensor([float(d[1]) for d in dets], dtype=torch.float32) if nd else torch.zeros(0)
        det_keep_t = torch.tensor([bool(d[1] > F32(0.4) and not d[2]) for d in dets], dtype=torch.bool) if nd else torch.zeros(0, dtype=torch.bool)
        trk_before = trk_t.clone()

        model.tracker.calls.clear()
        model.captured.clear()
        plan, meta_new = model.run_tracker_update_planning_phase(
            frame_idx=t, num_frames=n_frames, reverse=False,
            det_out={"mask": det_t, "scores": det_scores_t},
            det_keep=det_keep_t, tracker_low_res_masks_global=trk_t, tracker_obj_scores_global=trk_scores_t,
            tracker_metadata_prev=meta, tracker_states_local=states,
        )
        adt = model.captured["adt"]
        suppress = model.captured.get("suppress", [])

        # What the in-place mask surgery did, checked against the claim the Rust port relies on.
        det_bin = (det_t > 0).numpy()
        before_bin, after_bin = (trk_before > 0).numpy(), (trk_t > 0).numpy()
        pairs = dict(plan["trk_id_to_max_iou_high_conf_det"])
        recond = sorted([[o, int(pairs[o])] for call in model.tracker.calls for o in call])
        ids_list = prev_ids.tolist()
        for row, oid in enumerate(ids_list):
            hit = [d for o, d in recond if o == oid]
            if row in suppress:  # suppression runs after reconditioning and writes -10
                assert not after_bin[row].any(), "suppressed rows are overwritten with -10"
            elif hit:
                assert (after_bin[row] == det_bin[hit[0]]).all(), "reconditioned mask is the detection's binary mask"
            else:
                assert (after_bin[row] == before_bin[row]).all()

        new_inds = [int(i) for i in plan["new_det_fa_inds"]]
        new_ids = [int(i) for i in plan["new_det_obj_ids"]]
        dropped = int(plan["num_obj_dropped_due_to_limit"])
        cand = [i for i, v in enumerate(adt["is_new_det"]) if v]
        if dropped > 0 and len(cand) > 16:  # np.argsort is stable up to 16 elements here, not beyond
            sc = [float(det_scores_t[i]) for i in cand]
            if len(set(sc)) != len(sc):
                raise TieCap()
        removed = sorted(int(o) for o in plan["obj_ids_newly_removed"])
        prev_arr = prev_ids
        ids_all = meta_new["obj_ids_all_gpu"].tolist()
        gm = meta_new["gpu_metadata"]
        conf = meta_new["rank0_metadata"]["masklet_confirmation"]
        status = conf["status"]
        # sam3_multiplex_tracking.py:657-661, transcribed:
        unconfirmed = meta_new["obj_ids_all_gpu"][status == MaskletConfirmationStatus.UNCONFIRMED.value].tolist()
        assert not meta_new["rank0_metadata"]["removed_obj_ids"] and not meta_new["rank0_metadata"]["suppressed_obj_ids"][t]

        frames.append({
            "f": t,
            "prev_ids": ids_list,
            "trk_masks": [rle(m) for m in before_bin],
            "trk_score_logits": [float(x) for x in trk_scores],
            "det_masks": [rle(m) for m in det_bin],
            "det_scores": [float(d[1]) for d in dets],
            "det_keep": det_keep_t.tolist(),
            "assoc": adt,
            "out": {
                "new_det_fa_inds_precap": cand,
                "births": [[d, i] for d, i in zip(new_inds, new_ids)],
                "dropped": dropped,
                "removed_ids": removed,
                "unmatched_ids": [int(o) for o in plan["unmatched_trk_obj_ids"]],
                "empty_ids": [int(o) for o, ne in zip(prev_arr.tolist(), adt["trk_is_nonempty"]) if not ne],
                "det_to_matched": [[int(d), [int(o) for o in ids]] for d, ids in sorted(plan["det_to_matched_trk_obj_ids"].items())],
                "trk_to_det": [[int(o), int(d)] for o, d in plan["trk_id_to_max_iou_high_conf_det"].items()],
                "recondition": recond,
                "suppressed": suppress,
                "obj_ids": ids_all,
                "unconfirmed": [int(o) for o in unconfirmed],
            },
            "meta": {
                "max_obj_id": int(meta_new["max_obj_id"]),
                "scores": [[int(k), float(v)] for k, v in sorted(meta_new["obj_id_to_score"].items())],
                "first_frame": gm["obj_first_frame"].tolist() if len(ids_all) else [],
                "unmatch_count": gm["consecutive_unmatch_count"].tolist() if len(ids_all) else [],
                "overlap": gm["overlap_pair_counts"].reshape(-1).tolist() if len(ids_all) else [],
                "last_occluded": gm["last_occluded_tensor"].tolist() if len(ids_all) else [],
                "consecutive_det": conf["consecutive_det_num"].tolist(),
                "confirmed": [bool(s == MaskletConfirmationStatus.CONFIRMED.value) for s in status],
            },
        })
        outs_by_frame[t] = {"ids": ids_all, "unconfirmed": [int(o) for o in unconfirmed], "scores": dict(meta_new["obj_id_to_score"])}

        # execution phase stand-in: bind newborns to their source, drop removed ids from the states
        for d, oid in zip(new_inds, new_ids):
            src = dets[d][3]
            if src is None:
                src = Entity(rng, t, min(n_frames, t + int(rng.integers(2, 20))), detectable=False)
            bind[oid] = {"ent": src, "lost_until": -1}
            states_by_parity(states, oid)
        for oid in removed:
            for s in states:
                if oid in s["obj_ids"]:
                    s["obj_ids"].remove(oid)
                    s["obj_idx_to_id"] = {i: o for i, o in enumerate(s["obj_ids"])}
        states[:] = [s for s in states if s["obj_ids"]]
        meta = meta_new

    yields = replay_yields(model, n_frames, start, last, outs_by_frame)
    return {"name": f"{profile}-{seed}", "start": start, "last": last, "num_frames": n_frames, "frames": frames, "yields": yields}


def states_by_parity(states, oid):
    """Two tracker states, by id parity, so reconditioning exercises the per-state loop."""
    for s in states:
        if s["parity"] == oid % 2:
            s["obj_ids"].append(oid)
            s["obj_idx_to_id"] = {i: o for i, o in enumerate(s["obj_ids"])}
            return
    states.append({"parity": oid % 2, "obj_ids": [oid], "obj_idx_to_id": {0: oid}})


# ----------------------------------------------------------------------------- yield replay


def replay_yields(model, n_frames, start, last, outs_by_frame):
    """Drive the real `propagate_in_video` generator with the planning phase's outputs."""
    results = {}
    for batch in (1, 16):
        model.postprocess_batch_size = batch
        processed = [0]
        frame_of_out = {}
        record = []

        def run_single(inference_state, frame_idx, reverse, is_instance_processing=False):
            processed[0] += 1
            o = outs_by_frame[frame_idx]
            out = {"obj_id_to_mask": {i: None for i in o["ids"]}, "obj_id_to_score": o["scores"], "obj_id_to_sam2_score": {},
                   "removed_obj_ids": set(), "suppressed_obj_ids": set(), "unconfirmed_obj_ids": o["unconfirmed"], "frame_stats": {}}
            frame_of_out[id(out)] = frame_idx
            return out

        def note(out, removed, suppressed, unconfirmed):
            assert not removed and not suppressed
            record.append({"frame": frame_of_out[id(out)], "after": processed[0],
                           "hide": None if unconfirmed is None else [int(o) for o in unconfirmed]})

        def post_single(inference_state, out, removed=None, suppressed=None, unconfirmed=None):
            note(out, removed, suppressed, unconfirmed)
            return {}

        def post_batched(H_video, W_video, batched_outs):
            for out, removed, suppressed, unconfirmed in batched_outs:
                note(out, removed, suppressed, unconfirmed)
            return [{} for _ in batched_outs]

        model._compile_model = lambda: None
        model._run_single_frame_inference = run_single
        model._cache_frame_outputs = lambda *a, **k: None
        model._postprocess_output = post_single
        model._postprocess_output_batched = post_batched
        state = {"feature_cache": {}, "orig_height": H, "orig_width": W, "num_frames": n_frames,
                 "previous_stages_out": [("x" if i == start else None) for i in range(n_frames)],
                 "sam2_inference_states": [], "cached_frame_outputs": {}}
        # the base-class generator: the interactive subclass's wrapper only parses the action history
        # (always a full propagation on the text path) and delegates here (sam3_multiplex_tracking.py:2297-2329)
        with contextlib.redirect_stderr(io.StringIO()):  # tqdm
            yielded = [y for y, _ in Sam3MultiplexTracking.propagate_in_video(model, state, start_frame_idx=start, max_frame_num_to_track=last - start)]
        assert [r["frame"] for r in record] == yielded == list(range(start, last + 1))
        results[batch] = record
    # batching only changes when a frame is postprocessed, never which ids are hidden
    assert [(r["frame"], r["hide"]) for r in results[1]] == [(r["frame"], r["hide"]) for r in results[16]]
    return results[1]


# ----------------------------------------------------------------------------- sigmoid probe


def sigmoid_probe():
    """`sigmoid(x) > 0.8` as `_recondition_masklets` evaluates it (float32 tensor vs python 0.8),
    for logits around ln 4, in tensors of several lengths (torch's vectorized and tail paths)."""
    xs = f32_neighbours(LN4, 40)
    contexts = {}
    for length in (1, 3, 8, 64):
        got = []
        for i, x in enumerate(xs):
            buf = torch.full((length,), 5.0, dtype=torch.float32)
            buf[i % length] = float(x)
            got.append(bool((buf.sigmoid() > 0.8)[i % length]))
        contexts[str(length)] = got
    return {"logits": [float(x) for x in xs], "gt": contexts}


# ----------------------------------------------------------------------------- birth selection


def birth_cases(rng):
    """`RealizedAssociateDetTrkresult.get_new_det_gpu_ids` (cap, order, ids) on random candidates,
    with tied scores for up to 16 candidates and distinct scores above that."""
    model = make_model()
    cases = []
    for _ in range(120):
        n_det = int(rng.integers(1, 30))
        n_new = int(rng.integers(0, n_det + 1))
        new_inds = np.sort(rng.choice(n_det, n_new, replace=False)).astype(np.int64)
        if n_new > 16:
            scores = rng.permutation(n_det).astype(np.float32) / F32(n_det)
        else:
            scores = (rng.integers(0, 4, n_det).astype(np.float32) / F32(4.0)) + F32(0.5)
            scores[rng.random(n_det) < 0.3] = F32(0.9)
        prev = int(rng.integers(0, 17))
        max_id = int(rng.integers(-1, 60))
        res = RealizedAssociateDetTrkresult(new_det_fa_inds=new_inds, unmatched_trk_obj_ids=None, det_to_matched_trk_obj_ids=None,
                                            trk_id_to_max_iou_high_conf_det=None, empty_trk_obj_ids=None)
        ids, _gpu, dropped = res.get_new_det_gpu_ids(
            {"num_obj_per_gpu": np.array([prev]), "max_obj_id": max_id, "num_buc_per_gpu": np.array([1])},
            False, torch.from_numpy(scores), model)
        cases.append({"new_inds": new_inds.tolist(), "scores": [float(x) for x in scores], "prev": prev, "max_obj_id": max_id,
                      "births": [[int(d), int(i)] for d, i in zip(res.new_det_fa_inds, ids)], "dropped": int(dropped)})
    return cases


# ----------------------------------------------------------------------------- IoU threshold


def iou_cases():
    """The occlusion helper (the real method) on mask pairs whose IoU is 0.7 exactly in rationals
    (7k/10k) and one pixel either side: `iou >= 0.7` is a float32 tensor against a python float."""
    model = make_model()
    cases = []
    for k in range(1, 41):
        for delta in (-1, 0, 1):
            union, inter = 10 * k, 7 * k + delta
            a, b = np.zeros(H * W, bool), np.zeros(H * W, bool)
            a[:union], b[:inter] = True, True
            masks = torch.from_numpy(np.stack([a, b]).reshape(2, H, W))
            r = model._get_objects_to_suppress_based_on_most_recently_occluded(masks, torch.tensor([3, 1]), np.array([0, 1]), frame_idx=None)
            cases.append({"a": union, "b": inter, "last": [3, 1], "suppress": r.tolist()})
    assert any(c["suppress"][0] for c in cases) and not all(c["suppress"][0] for c in cases)
    return cases


# ----------------------------------------------------------------------------- main


def summarize(seqs):
    c = dict(frames=0, births=0, dropped_frames=0, removed=0, suppressed=0, recondition=0, none_hides=0, hides=0, confirmed_events=0)
    for s in seqs:
        prev_conf = set()
        for fr in s["frames"]:
            o = fr["out"]
            c["frames"] += 1
            c["births"] += len(o["births"])
            c["dropped_frames"] += o["dropped"] > 0
            c["removed"] += len(o["removed_ids"])
            c["suppressed"] += len(o["suppressed"])
            c["recondition"] += len(o["recondition"])
            conf = {i for i, v in zip(o["obj_ids"], fr["meta"]["confirmed"]) if v}
            c["confirmed_events"] += len(conf - prev_conf)
            prev_conf = conf
        for y in s["yields"]:
            c["none_hides"] += y["hide"] is None
            c["hides"] += bool(y["hide"])
    return c


def main():
    seqs = []
    for k in range(N_SEQUENCES):
        profile = PROFILE_ORDER[k % len(PROFILE_ORDER)]
        for attempt in range(50):
            try:
                seqs.append(run_sequence(profile, 20261026 + 1000 * k + attempt, n_process=TRUNCATED.get(k)))
                break
            except TieCap:
                continue
        else:
            raise RuntimeError(f"no tie-free sequence for {profile} #{k}")
    probe = sigmoid_probe()
    births = birth_cases(np.random.default_rng(20261026))
    assert any(c["dropped"] > 0 and len(c["new_inds"]) > 1 and len(set(c["scores"][i] for i in c["new_inds"])) < len(c["new_inds"]) for c in births)
    for ctx, got in probe["gt"].items():
        assert got == probe["gt"]["1"], f"torch sigmoid>0.8 differs between tensor lengths ({ctx})"
    OUT.write_text(json.dumps({"h": H, "w": W, "sigmoid_probe": probe, "birth_cases": births, "iou_cases": iou_cases(), "sequences": seqs}, separators=(",", ":")))
    stats = summarize(seqs)
    print(f"{len(seqs)} sequences -> {OUT.name} ({OUT.stat().st_size / 1e6:.2f} MB): {stats}")


if __name__ == "__main__":
    main()
