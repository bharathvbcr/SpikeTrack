# SAM 3.1 Object Multiplex: text-prompted video tracking glue (detector + multiplex tracker)

Read-only study of upstream `external/sam3` (branch `spiketrack-cpu`, base f1647e8). Nothing was run;
every statement is from reading code on the path `build_sam3_multiplex_video_predictor` constructs.
Labels: **[V]** read in code on the builder's path, **[I]** inferred from code, **[U]** unverified / could not be determined.

Path abbreviations (all under `sam3/model/` unless noted):
`M` = `sam3/model_builder.py`, `BP` = `sam3_base_predictor.py`, `B` = `sam3_multiplex_base.py`,
`T` = `sam3_multiplex_tracking.py`, `V` = `sam3_video_base.py`, `D` = `sam3_multiplex_detector.py`,
`DU` = `sam3_multiplex_detector_utils.py`, `I` = `sam3_image.py`, `MU` = `multiplex_utils.py`,
`VM` = `video_tracking_multiplex.py`, `VD` = `video_tracking_multiplex_demo.py`.

## 0. Headline findings (things a port must not get wrong)

1. **Association uses mask IoM (intersection over the smaller area), not IoU, for every threshold.** `use_iom_recondition=True`
   routes `assoc_iou_thresh` (0.1), `trk_assoc_iou_thresh` (0.5) and the recondition test (0.5) through `mask_iom` on 288x288
   binarized masks. No Hungarian, no greedy one-to-one: thresholded many-to-many. [V `V:177-184,203,211-213`; `B:2034-2047`; `M:1176,1186-1187`]
2. **Birth is immediate; visibility is delayed.** A detection with score >= 0.65 and IoM < 0.1 to every live track becomes a tracker
   object on the same frame. Output for a frame is hidden for tracks still "unconfirmed" two frames later (3 consecutive matched
   frames confirm). [V `V:210-213`, `T:343,417-427`, `V:2017-2023`]
3. **Hot-start "removed" set is dead on this path.** `rank0_metadata["removed_obj_ids"]` and `suppressed_obj_ids` are only written by
   `Sam3MultiplexBase._process_hotstart` (`B:2368-2492`), which is only called from `V:824` inside
   `Sam3VideoBase.run_tracker_update_planning_phase`, which `Sam3MultiplexBase` overrides (`B:1018`). The GPU path
   (`_process_hotstart_gpu`) returns `to_remove_mask` that actually removes the track from the tracker, but the retroactive hide of
   buffered frames is therefore done only by the unconfirmed set. `to_suppress_mask` from `_process_hotstart_gpu` is computed and
   never used (`B:1062` assigns; no later read). [V: `rg -uu` over the checkout finds `removed_obj_ids` written only in the two
   `_process_hotstart` twins (`B:2491`, `sam3_video_base.py:1742`); everything else reads or clears it.]
4. **Tracks are never removed after their first 15 frames.** Removal rules require `obj_first_frame > frame - 15`. After that a track
   lives to the end of the video; "occlusion" is just the tracker's own object score going <= 0 (mask forced to -1024). A re-detected
   object whose old track is empty is born again under a **new id**. [V `B:2275-2293,2339-2343`; `VM:839-848`]
5. **The prompt frame is processed twice.** `add_prompt` runs the full detect+track step on frame f (births objects), then forward
   `propagate_in_video` starts at `range(f, ...)` (includes f) with no reset. Second visit reuses the stored birth output, so the
   detections match their own tracks: confirmation counter reaches 2 on f, overlap counters double count. Also on the second visit
   step 10 re-encodes frame f's memory from the stored `+-10` birth masks (area-proxy scores, shrinkage suppression instead of hard
   non-overlap, `conditioning_objects` read from the stored cond output = all newborns), **overwriting the birth-time (preflight)
   encode**: frame f's final memory is the step-10 version. For the usual prompt on frame 0 (0 % 16 == 0) reconditioning also fires
   on that second visit for every newborn whose detection score is >= 0.8 (tracker score is +10, so its 0.8 gate passes, and the
   detection matches its own track with IoM ~1). A port that encodes frame f only once at birth will not match.
   [V `T:1696-1737,273-299,2729-2745`; `VD:3434-3441`; `B:1132-1147,1198-1204`]
6. **Max 16 live objects** (builder `max_num_objects=16`), so with this builder every state is a single 16-slot bucket and the
   "past 16 objects, multiple buckets" code is unreachable (`M:1075,1191`; `V:63-76`). Several tracker *states* (each one bucket)
   can coexist.

## 1. Which class runs per frame, and what is shared

| Role | Class actually constructed [V] |
|---|---|
| Predictor | `Sam3MultiplexVideoPredictor(Sam3BasePredictor)` (`M:1237-1243`, `sam3_multiplex_video_predictor.py:22`) |
| Model | `Sam3MultiplexTrackingWithInteractivity(Sam3MultiplexTracking(Sam3MultiplexBase(Sam3VideoBase)))` (`M:1170`; `T:2196`, `T:78`, `B:196`) |
| Tracker | `Sam3MultiplexPredictorWrapper(Sam3MultiplexTrackerPredictor)` around `Sam3VideoTrackingMultiplexDemo` (`M:987,1135-1141`; `B:2887`). Its own backbone is deleted (`M:1131-1133`). |
| Detector | `Sam3MultiplexDetector`, backbone `SAM3VLBackboneTri(scalp=0, visual=Sam3TriViTDetNeck, text=...)` (`M:1144-1167`) |
| Not used | `Sam3MultiplexTrackingProd` (`T:1823`), `Sam3MultiplexTrackerPredictor.__init__` (Hydra) |

Request flow [V]: `start_session` -> `Sam3MultiplexTracking.init_state` (`T:208-250`; frames resized to 1008, mean/std 0.5; see README
for the fp16 loader) -> `add_prompt(text, frame_index=f)` (`BP:170-228` -> `T:2694-2745` -> `T:1656-1738`) ->
`propagate_in_video` (`BP:302-361`, default direction "both"; forward `T:2297-2329` -> `T:302-583`). `output_prob_thresh` (0.5) is
accepted but **unused** by the multiplex model (`T:2297-2329` does not forward it; masks are `logit > 0`).

Action-history logic (`T:2515-2599`): with a text-only prompt `reset_state` clears history (`T:1696`, `T:2245-2252`), so the first
propagate is `propagation_full`; a second call (the reverse leg) is also `propagation_full` because the recorded `frame_idx` is None
(`T:2572-2584`). Partial/fetch branches only occur after point refinement or remove_object, not on the text path.

Backbone sharing [V]: the ViT trunk runs once per frame inside the detector call (`B:687-707`). `Sam3TriViTDetNeck` yields three
FPN stacks (`vl_combiner.py:244-271`):
- `convs` -> `output["backbone_fpn"]` -> **detector** encoder (uses only the last level, 72x72, `num_feature_levels=1`, `M:1156`; `I:126`) and segmentation head (all three levels).
- `interactive_convs` -> `output["interactive"]` -> `interactive_backbone_fpn_*` -> tracker **interactive** SAM decoder (mask-prompt -> object pointer), levels 0/1 through `tracker.interactive_sam_mask_decoder.conv_s0/conv_s1` (`B:762-784`).
- `propagation_convs` -> `output["sam2_backbone_out"]` -> `sam2_backbone_fpn_*` -> tracker memory attention / multiplex decoder / memory encoder, levels 0/1 through `tracker.sam_mask_decoder.conv_s0/conv_s1` (`B:785-799`).
Result is stored once as `feature_cache[frame_idx] = (image, {"interactive":..., "sam2_backbone_out":...})` (`B:802-806`) and the
tracker reads it (`VD:2630-2651`; the cache dict is passed as `cached_features`, `B:2711-2716`). `feature_cache.pop(frame-1)` (`B:808`).
Text features are computed once per prompt (`T:1750`, reused `T:604-612`).

Detector cadence [V]: every frame, but batched 16 at a time: `forward_video_grounding_batched_multigpu` processes chunk
`[16*floor(f/16), +16)` in one `forward_grounding` call (`D:741-762,784-826`). Chunking only changes batch composition (numerics),
not semantics; a per-frame port is equivalent [I]. Exception: during `add_prompt` batching is switched off
(`T:2729-2745`), i.e. frame f is detected twice (batch 1 then in a chunk) [V].

## 2. Per-frame algorithm (forward direction, `_det_track_one_frame_impl`, `B:477-642`)

Inputs: image t, tracker states `S` (list), metadata `M`. Order is exactly:

1. **Detect** (`B:509`, `B:646-809`): text-conditioned image detector on frame t with the empty geometric prompt
   (`T:599-603`, `T:163-170`) -> per query `pred_logits`, `pred_boxes_xyxy` (normalized), `pred_masks` (288x288 logits [I: finest of
   the 4x/2x/1x FPN levels, `maskformer_segmentation.py:216-225`; association code interpolates only if sizes differ, `B:1999-2015`]).
   Joint score: `logit = clamp(inverse_sigmoid(sigmoid(cls) * sigmoid(presence)), -10, 10)` (`I:345-353`, eps 1e-3 `model_misc.py:30-38`).
2. **Detection NMS** inside the detector (`D:828-848`, `DU:90-132,185-236`): valid = `prob > 0.4`; pairwise **mask IoM** (`inter/(min_area+1e-8)`, binarized `mask > 0`)
   greater than **0.1** suppresses the lower-scored one (greedy in descending-score order; only kept ones suppress). Suppressed
   queries get `pred_logits -= 1e4` (prob ~ 0).
3. **Keep mask** (`B:739-755`): `det_keep = prob > 0.4` and not "close to boundary" (box center within 0.025 of any image edge, `V:526-543`). `det_out = {bbox, mask, scores=prob}`.
4. **Propagate every existing state one frame** without memory encoding (`B:529`, `B:811-863`, `B:1731-1889`): per state
   `tracker.propagate_in_video(start=t, max_frame_num_to_track=0, run_mem_encoder=False)` (`B:1755-1764`) =
   `Sam3VideoTrackingMultiplexDemo.propagate_in_video` (`VD:3394-3481`). Frames already in `consolidated_frame_inds["cond"/"non_cond"]`
   reuse their stored output (`VD:3434-3446`); otherwise `track_step` in propagation-only mode with memory attention. Results from all states are concatenated and sorted by object id (`B:1854-1889`); `fill_holes_in_mask_scores` is a no-op (`fill_hole_area=0`, `sam3_tracker_utils.py:395`). Output per track: 288x288 mask logits (object score <= 0 -> mask = -1024, `VM:839-848`) and the object-score logit.
5. **Reorder detections** so kept ones come first: `argsort(bool, descending)` (`B:545-552`). Order among kept (and among dropped) is
   not specified by torch [U]; a stable sort (original query order) is the natural reading. This order decides new object ids and slots.
6. **Associate** (`B:1043`, `B:1893-2071`, `V:163-262`): details in section 4.
7. **Hot-start bookkeeping and removal flags** (`B:1062`, `B:2094-2366`): section 4.
8. **Reconditioning** (only if `frame_idx % 16 == 0`, `B:1132-1176`): section 6. Edits `tracker_low_res_masks_global` in place and calls
   `tracker.add_new_masks(..., reconditioning=True)` then `propagate_in_video_preflight(run_mem_encoder=True)` on affected states.
9. **Occlusion-based suppression** (threshold 0.7 > 0, `B:1183-1197`, `B:1424-1514`, `B:1635-1729`): binarize track masks (`> 0`), pairwise
   **mask IoU** (`perflib/masks_ops.py:50-72`, `union.clamp(min=1)`), pairs (i<j) with IoU >= 0.7. Let `last_occ[i]` be the last frame
   where track i had an empty mask or was suppressed (init -1; hot-start-removed this frame = 100000). With `allow_unoccluded_to_suppress=False`:
   i is suppressed if `last_occ[i] > last_occ[j]` and `last_occ[j] > -1` (and symmetrically for j) (`cmp` is `lt` in reverse). Suppressed
   masks are overwritten **in place** with -10 in `tracker_low_res_masks_global` (`B:1512`), so the suppression also changes the
   emitted output (same tensor reaches `build_outputs`, `B:603-617`). `last_occ` is updated with
   `empty | suppressed -> frame_idx` (`B:1494-1502`). Note this uses the *current frame's* masks after reconditioning.
10. **Memory encode for all existing tracks** (`B:1198-1204` -> `B:2496-2666`): `low_res` masks -> bilinear to 1152 -> `_suppress_object_pw_area_shrinkage`
    (`VD:3325-3368`; if batch > 1: pixel-wise argmax non-overlap computed only to measure area shrinkage; objects whose area after/before < 0.3 get `clamp(max=-10)`, others are **left overlapping**) ->
    `object_score_logits = +10 if mask has any positive pixel else -10` (area proxy, not the tracker's score, `B:2531-2533`) ->
    per state `tracker._run_memory_encoder(..., is_mask_from_pts=False, conditioning_objects=<looked up from the stored output>)` (`VD:2869-2931`),
    stored bf16 into `output_dict[...][t]` (`B:2603-2624`). `conditioning_objects` for a propagated frame is the empty set, so the condition channel is 0.0 for all (`VM:1705-1727`, `VM:2089` only for mask-as-output frames).
11. **Realize association** and pick births (`B:1207-1214`, `V:54-94`): `new_det_fa_inds` = detections flagged `is_new_det`; cap
    (section 3); ids `max_obj_id+1, +2, ...` in kept-detection order (ids start at 0, `V:1932`); all on rank 0.
12. **Metadata update** (`B:1224-1282,1287-1407`): scores (`obj_id_to_score[new] = det score`, removed -> -1e4), confirmation status
    (`V:1976-2027` / `B:2824-2884`), compaction of per-position tensors, new objects appended with `first_frame=t`, `keep_alive=0`, `last_occ=-1`.
13. **Execute** (`V:1176-1223`): (a) add new objects (section 5), (b) remove hot-start-removed ids from tracker states (`B:2800-2822`; a state with no objects left is dropped).
14. **Build outputs** (`B:1544-1631`): existing tracks = `tracker_low_res_masks_global` (post recondition/suppression), bilinear (align_corners False) to video HxW, `> 0`; new tracks = detector mask (288 logits) bilinear to video size, `> 0`.
    Order of `obj_id_to_mask`: previous ids first, new ids after.
15. **Yield with delay** (`T:337-481`): section 7.

Per-frame stats: `tracker_obj_scores_global.sigmoid()` is stored as `obj_id_to_sam2_score_frame_wise[t]` for previous ids
(`B:626-633`); new ids get their detection score (`B:1255-1257`).

## 3. Detection post-processing and score combination (Q2)

| Item | Value / rule | Evidence |
|---|---|---|
| Presence | multiplied into the class probability before thresholding: `p = sigmoid(cls) * sigmoid(presence_logit_dec)`, re-expressed as a clamped logit (+-10, eps 1e-3). No separate presence gate. | [V] `I:345-353` |
| Detection threshold | `prob > 0.4` (strict) for NMS validity and for `det_keep` | [V] `M:1173`, `B:744`, `DU:112` |
| NMS | mask **IoM** (`nms_use_iom=True`), `> 0.1` suppresses (strict), greedy by descending score, per frame, on `mask > 0` at 288x288 | [V] `M:1174-1175`, `B:698-702`, `DU:160-182,206-236` |
| Boundary filter | box center `x_c, y_c` must be in `(0.025, 0.975)` else not kept | [V] `V:526-543`, `B:746-749` |
| Birth threshold | `score >= 0.65`, in addition to `det_keep` | [V] `V:210-213`, `M:1177` |
| Max objects | 16 total. If `prev_count + new > 16` (count includes tracks removed this frame; they free capacity next frame): keep the top-scoring new detections (`np.argsort(score)[::-1]`), drop the rest; dropped ones are simply ignored and can be born later | [V] `V:63-76,2086-2100`, `M:1075,1191` |
| Score stored per object | the birth detection score; never updated afterwards (removed -> -1e4). This is `out_probs` | [V] `B:1249-1251,1264-1265`; `T:686-688` |
| Image-only threshold | `image_only_det_thresh=0.5` only for single-image input, not used for video | [V] `B:1049-1052` |

Order matters: NMS (validity `prob > 0.4` only, `DU:112`) runs **before** the boundary filter (`B:746-749`), so a boundary
detection can still suppress a neighbour inside NMS and is dropped only afterwards.

`max_num_kboxes=0`, `use_iom_recondition=True` etc. are builder arguments (`M:1170-1202`).

## 4. Association, hot-start, confirmation (Q3)

`_associate_det_trk_compilable` (`V:163-262`), inputs: det masks (`>0`, rows of non-kept detections zeroed, `V:177-178`), track masks (`>0`),
`IoM[d,t] = inter / (min(area_d, area_t) + 1e-8)` (`train/masks_ops.py:76-87`; empty masks give 0). Let `A` = IoM.

- `trk_is_nonempty[t] = any(track mask > 0)`.
- `trk_is_matched[t] = any_d (A[d,t] >= 0.5)`; `trk_is_unmatched[t] = nonempty & ~matched` (empty tracks are never "unmatched"). [V `V:203-206`]
- `is_new_det[d] = det_score >= 0.65 & det_keep & ~any_t (A[d,t] >= 0.1)` using the **original** `A`. [V `V:210-213`]
- Ambiguity zeroing (`V:215-244`): `det_many = (#t with A>=0.5) > 1` and `trk_many = (#d with A>=0.5) > 1`; columns of `trk_many` tracks and rows of `det_many` detections of `A` are set to 0 -> `A'`.
- `im_mask = A' >= 0.1` (det-track matches used for keep-alive, confirmation and overlap counting). `det_is_high_conf = score >= 0.8 & keep & ~is_new_det`; `det_is_high_iou = max_t A' >= 0.5`; `det_to_max_iou_trk_idx = argmax_t A'` (first on ties [U]). [V `V:246-251`]
- `o2o_matching_masklets_enable=False` and the code asserts it cannot be on (`V:186-188`): no Hungarian.
- Track masks are padded with zero rows up to `max_num_objects` for a fixed compile shape and trimmed after (`B:2018-2060`); compile-only, skip.

Hot-start state per track position (`_process_hotstart_gpu`, `B:2094-2366`), all counters are per position, new tracks start at 0:
- `trk_keep_alive`: +1 if `im_mask` column any, else -1, clamped to `[-4, 8]`, init 0 (`B:2211-2219`). **Only feeds `to_suppress`, which is unused** (finding 3).
- `consecutive_unmatch_count` (really a cumulative count, never reset): `+1` when `trk_is_unmatched` (`B:2235-2239`).
- `overlap_pair_counts[i,j]` (i<j by position): `+= number of detections whose im_mask row has >= 2 tracks including both i and j` (`B:2246-2268`). Because of the ambiguity zeroing, only weak overlaps (IoM in [0.1,0.5) for at least one of the tracks) accumulate in practice [I].
- `is_within_hotstart = obj_first_frame > t - 15` (forward) (`B:2275-2286`).
- **Remove** a track if within hot-start and (`unmatch_count >= 8` or `max overlap count with an earlier-born track >= 8`), where "earlier" is strictly smaller `first_frame` (`B:2289-2345`). Same-frame births never remove each other.
- Removal effects: tracker removal in execution phase (`V:1218-1220`), `obj_id_to_score = -1e4`, dropped from `obj_ids_*` for the next frame (`B:1234-1269`); the removed track **is still emitted on the frame it is removed** (outputs are built from the previous id list, `B:1570-1608`).
- Reverse direction mirrors with `t + 15`, `first_frame < ...`, `>` comparisons.

Masklet confirmation (`masklet_confirmation_enable=True`, `M:1188`; `B:2824-2884` overrides `V:1976`):
- Per track `consecutive_det_num`: `+1` if the track is a new det id **or** appears in any `det_to_matched_trk_obj_ids` (the `im_mask` rows of kept dets), else reset to 0; `status=CONFIRMED` when `>= 3` (default `masklet_confirmation_consecutive_det_thresh=3`, builder does not set it). CONFIRMED is sticky. [V `B:2862-2879`]
- `out["unconfirmed_obj_ids"]` for frame t = ids with status UNCONFIRMED after frame t's update, computed from the *new* id list (`T:657-661`).
- A track born on frame b matched on b, b+1, b+2 is confirmed on b+2, so frames b.. are shown (hide test looks two frames ahead).

Unmatched tracks: no explicit "occlusion state". Empty mask (tracker score <= 0) or suppression (step 9) just removes it from the output (`has_area` filter, `T:704`/`T:928`); the track keeps being propagated.
Duplicate tracks of the same object are handled only by step 9 and by hot-start duplicate removal.

## 5. Entry of a newborn into the multiplex tracker (Q4)

`_tracker_add_new_objects` (`B:2668-2798`), called with `new_obj_ids`, detector `mask[new_inds]` (288 logits):

1. **State choice** (`B:2685-2723`): among `tracker_states_local` pick the state with the **smallest** `available_slots` that is `>= n_new` (strict `<` best so far, so first one wins ties); `available_slots = num_buckets*16 - total_non_padding_entries` (`MU:118-124`; removed slots count as occupied and are never reused). If none fits, `tracker.init_state(cached_features=feature_cache, ...)` creates a new empty state and appends it.
2. **Mask prep** (`B:2761-2769`): detector 288 logits -> bilinear (align_corners False, no antialias) to `input_mask_size = 1152` -> `> 0` binary.
3. `tracker.add_new_masks(state, frame_idx, obj_ids, masks, add_mask_to_memory=True)` (`B:2776-2782`; `VD:1665-1904`):
   - object idx = insertion order in the state (`VD:154-166`); mask at video resolution = bilinear **antialias** to (H,W), `> 0.5`, later stored as `+-1024` (`NO_OBJ_SCORE=-1024`, `VM:49`, `VD:1796-1802`).
   - **New state** (`multiplex_state is None`): `MultiplexController.get_state(n, random=False)` (`VD:1727-1736`): deterministic, `ceil(n/16)` buckets, object k -> bucket k//16, slot k%16, padding at the end (`MU:521-546`). The frame is an **initial conditioning frame** (`frame not in frames_already_tracked`) -> `cond_frame_outputs` (`VD:1749-1764`), mode "mask-as-output" (`VM:2068-2089`): `_use_mask_as_output` = logits `mask*20-10` at 1152, low-res = antialiased bilinear to 288, IoU=1, object-score logit `+-10` from "mask non-empty", object pointer from the interactive decoder fed `interactive_mask_downsample(mask)` (`VM:935-1023`); all objects are `conditioning_objects` (same as the already-ported VOS frame-0 path).
   - **Existing state** (frame t already propagated in it, so `t in frames_already_tracked`): `is_cond = False` (`add_all_frames_to_correct_as_cond=False`, `VM:149`; builder does not set it) -> **non-conditioning** frame; the stored output of frame t is modified in place by `add_new_masks_to_existing_state` (`VM:3142-3299`, `VD:2682-2790`): `multiplex_state.add_objects` fills PADDING slots in bucket order (`MU:198-209`), `new indices = total_valid_entries..+n` (`MU:142-147`), new masks/pointers/score logits are *appended*, `conditioning_objects` gains the new indices. `allow_new_buckets` is False unless the state lacks room, which the best-fit rule prevents (`VD:1766-1770`).
   - Same-frame newborns suppress each other's video-res masks (union of the *other* new masks is set to -1024 for each, `VD:1840-1886`); existing (non-temp) objects are untouched.
4. `propagate_in_video_preflight(state, run_mem_encoder=True)` (`B:2796`, `VD:2277-2361`) -> `_consolidate_temp_output_across_obj` (`VD:1933-2274`): rebuilds the frame's masks at 288 (the state's own masks; for an existing state these come from `pred_masks_video_res`, i.e. video-res and pixel-wise non-overlapped, then antialias-downsampled back to 288, `VD:2069-2086`), overwrites new objects' masks with the +-1024 video-res masks resized to 288 with antialias (`VD:2184-2204`), upsamples to **1008** (`VD:2249-2254`), applies hard pixel-wise non-overlap across *that state's* objects (`VM:2668-2686`, ties -> lowest index), runs the memory encoder with `is_mask_from_pts=True` (no effect, `binarize_mask_from_pts_for_mem_enc` is forced False, `VM:357-366,1672-1675`) and `conditioning_objects` = {newborn idxs (+ any other conditioning on t)} -> condition channel 1.0 for those objects, 0.0 for the rest (`VM:1705-1727`). Mask for memory is `sigmoid(logit)*2-1` (`M:1017-1019`). This **overwrites** the frame-t memory that step 10 wrote for that state.

   **Two different resample chains reach the memory encoder** (the downsampler resizes its 32-channel input to `interpol_size=[1152,1152]`
   with bilinear `antialias=True` only when the size differs, `memory.py:80-88`):
   - step 10 (propagated frames, `B:2515-2533`): 288 logits -> bilinear 1152 (logits) -> area-shrink suppression -> `sigmoid*2-1` at 1152 -> no further resample.
   - preflight (births, recondition preflight, `VD:2249-2254`): 288 logits -> bilinear **1008** (logits) -> hard non-overlap -> `sigmoid*2-1` at 1008 -> the downsampler upsamples the *probability-space* mask (and the condition-flag channels) to 1152.
   Same object on the same frame therefore gets two slightly different 1152 inputs; replicate whichever applies to the final writer of that frame's memory.

Slot/bucket rules summary: deterministic; objects get state-local indices in id order; slot = first free PADDING slot scanning buckets then slots ascending. Removing objects marks slots `-1116` (not reusable), deletes fully empty buckets and densifies indices (`MU:242-344`). With the 16-object cap, every state has exactly one bucket; a second state appears when a birth does not fit any state's free slots.

## 6. Re-conditioning (Q5)

Condition (`B:1132-1147`): `recondition_every_nth_frame=16` and `frame_idx % 16 == 0` (absolute index, also in reverse) and at least one candidate. The bbox-IoU trigger is off (`reconstruction_bbox_iou_thresh=-1`, `B:1091`). Candidates are `trk_id_to_max_iou_high_conf_det` (`V:138-160`): detection kept, score >= 0.8, not a new det, unique match `A' >= 0.5` (section 4). `_recondition_masklets` (`B:867-1008`) then further requires the tracker's own `sigmoid(object score logit) > 0.8` (`B:905-906`).

For each surviving (track, det):
1. `tracker_low_res_masks_global[track] = where((det>0)==(trk>0), trk, det)` (pixelwise: keep tracker logit where both agree on the binary mask, otherwise take the detector logit) (`B:943-958`); hole fill no-op. This mask is what step 10 encodes into memory and what step 14 emits.
2. `tracker.add_new_masks(state, t, obj_ids, det_mask_bilinear_1152 > 0, reconditioning=True)` per state (`B:961-990`; `VD:1665-1904`): the stored frame-t output of those objects gets `pred_masks`, `object_score_logits` (+-10), **object pointer** replaced via `recondition_masks_in_existing_state` (`VM:3301-3428`: interactive decoder on the *detector* mask), and `conditioning_objects` updated. Stored in non-cond storage unless t is the state's cond frame (`VD:1763`, `VD:2695-2700`).
3. `propagate_in_video_preflight(state, run_mem_encoder=True)` (`B:1168-1176`), then step 10 re-encodes the memory from the hybrid masks (overwriting the preflight encode, `B:2603-2624`) with condition channel 1.0 for reconditioned objects.

So detector masks refresh tracker memory/pointers for matched tracks every 16th frame only; in between they never influence a matched track. `clear_detector_added_cond_frame_in_sam2` is only reached from point refinement (`T:3003`) and is **not** on the text path [V by `rg`].

## 7. Output stage (Q6)

Delay and hiding (`T:337-481`): `hotstart_delay=15` buffers frames; frame y is yielded when 15 frames are buffered (i.e. after frame y+14 was processed) or at the end of propagation. Its hide list is `suppressed_obj_ids` (always empty here) U `hotstart_removed_obj_ids` (always empty here, finding 3) U `unconfirmed_obj_ids_per_frame[clamp(y+2, 0, N-1)]` (`+2 = masklet_confirmation_consecutive_det_thresh - 1`, reverse uses `y-2`). Equivalent behavior needs only a 2-frame look-ahead [I]. Quirk [I]: a track removed by hot-start on frame r is still present in outputs for frames r-2..r if it was not UNCONFIRMED at the moment checked.

Per yielded frame (`T:799-1210` batched; `T:668-797` single, same logic), `postprocess_batch_size=16`:
1. ids = sorted keys of `obj_id_to_mask`; probs = `obj_id_to_score` (birth score); tracker scores = `obj_id_to_sam2_score[id]` or 0.
2. keep = mask has any pixel & id not hidden.
3. If more than one kept: **object-wise winner-take-all** (`VD:3370-3391`): pixels covered by several masks go to the object with the highest tracker score (ties -> lowest id index); losers lose the pixel. [V; numeric treatment of the bool tensors in `torch.clamp` is [U]]
4. Boxes from final masks: inclusive min/max pixel indices (`perflib/masks_ops.py:10-48`), xywh with `w = X - x` (no +1), divided by W,H.
5. Return `out_obj_ids`, `out_probs`, `out_boxes_xywh` (normalized), `out_binary_masks` (video resolution, bool).

No hole filling, no sprinkle removal (`fill_hole_area=0`, `sprinkle_removal_area=0`; the function returns early for `max_area <= 0`, `sam3_tracker_utils.py:395`; `sprinkle_removal_area` is not even read), no min-area filter, no per-frame score filter. The `add_prompt` call returns frame f's output **without** the hide list (`T:1738`).

## 8. Hyperparameters as the builder sets them (Q7)

Source column: **M** = `build_sam3_multiplex_video_predictor` (`M:1072-1202`); **D** = class default not overridden by the builder; **T** = tracker builder `build_sam3_multiplex_video_model` (`M:939-1049`).

| Name | Value | Src | Effect |
|---|---|---|---|
| `score_threshold_detection` | 0.4 | M `M:1173` | det keep, NMS validity (strict `>`) |
| `det_nms_thresh` / `det_nms_use_iom` | 0.1 / True | M `M:1174-1175` | mask IoM NMS, strict `>` |
| `assoc_iou_thresh` | 0.1 | M `M:1176` | det-track match (IoM `>=`), birth veto, `im_mask` |
| `trk_assoc_iou_thresh` | 0.5 | D `B:217` | track "matched" test for unmatched counting |
| `new_det_thresh` | 0.65 | M `M:1177` | birth score (`>=`) |
| `use_iom_recondition` / `iom_thresh_recondition` | True / 0.5 | M `M:1186-1187` | metric = IoM; ambiguity and recondition cutoff |
| `iou_thresh_recondition` | 0.8 | D `B:245` | unused (IoM branch) |
| `HIGH_CONF_THRESH` | 0.8 | const `B:878,1919` | det score and tracker `sigmoid(obj score)` for reconditioning |
| `recondition_every_nth_frame` | 16 | M `M:1185` | `frame % 16 == 0` |
| `reconstruction_bbox_iou_thresh` / `_det_score` | -1 / 0.8 | M `M:1189-1190` | bbox trigger disabled |
| `hotstart_delay` | 15 | M `M:1178` | yield buffer; hot-start window |
| `hotstart_unmatch_thresh` / `hotstart_dup_thresh` | 8 / 8 | M `M:1179-1180` | removal counts within window |
| `suppress_unmatched_only_within_hotstart` | False | M `M:1181` | only affects the unused `to_suppress` |
| `init/max/min_trk_keep_alive` | 0 / 8 / -4 | D `B:228-230` | feeds only the unused suppress flag |
| `decrease_trk_keep_alive_for_empty_masklets` | False | D `B:234` | off |
| `suppress_overlapping_based_on_recent_occlusion_threshold` | 0.7 | M `M:1182` | mask IoU `>=` for occlusion suppression |
| `allow_unoccluded_to_suppress` | False | D `B:233` | other track must have been occluded (`last_occ > -1`) |
| `suppress_det_close_to_boundary` | True (margin 0.025) | M `M:1183`, `V:526` | center-in-edge filter |
| `masklet_confirmation_enable` / `_consecutive_det_thresh` | True / 3 | M `M:1188` / D `B:251` | output hiding, look-ahead 2 |
| `max_num_objects` | 16 | M `M:1075,1191` | global cap; one bucket per state |
| `multiplex_count` / `eval_multiplex_count` | 16 / 16 | M `M:1076`, `M:979-982` | bucket capacity |
| `fill_hole_area` / `sprinkle_removal_area` | 0 / 0 | M `M:1184,1196` | no-ops |
| `postprocess_batch_size` | 16 | M `M:1192` | batching only |
| `use_batched_grounding` / `batched_grounding_batch_size` | True / 16 | M `M:1193-1194` | detector chunking (off in `add_prompt`) |
| `max_num_kboxes` | 0 | M `M:1195` | unused on text path |
| `image_size`, mean/std | 1008, 0.5/0.5 | M `M:1198-1200` | |
| `default_output_prob_thresh` | 0.5 | M `M:1082` | unused by the multiplex model |
| `session_expiration_sec`, `async_loading_frames` | 1200, True | M `M:1081,1083` | session plumbing only |
| `use_prev_mem_frame`, `use_stateless_refinement`, `refinement_detector_cond_frame_removal_window` | False, False, 120 | D `T:2199-2201` | interaction only |
| `use_fa3`, `use_rope_real` | True, True (predictor); tracker build gets the same | M `M:1077-1078,1125-1126` | kernel choice (README: FA3 off / complex RoPE reproduces fp32 CPU) |
| `num_maskmem` / `max_cond_frames_in_attn` / `max_obj_ptrs_in_encoder` | 7 / 4 / 16 | T `M:994,998,1038` | memory window |
| `apply_sigmoid_to_mask_logits_for_mem_enc`, scale, bias | True, 2.0, -1.0 | T `M:1017-1019` | memory mask `sigmoid*2-1` |
| `non_overlap_masks_for_mem_enc` | False | T `M:1020` | memory masks are not made non-overlapping on the propagation path (only the area-shrink suppression, step 10) |
| `non_overlap_masks_for_output` | True (tracker-internal video-res only; not the final output) | D `VD:46,1924` | final output uses the object-wise rule (section 7) |
| `condition_as_mask_input` fg/bg | True, 1.0 / 0.0 | T `M:1024-1026` | condition channel |
| `object_score_logit_threshold` | 0.0 | D `VM:268` | mask gating |
| `add_all_frames_to_correct_as_cond` | False | D `VM:149` | births/recond in existing states are non-cond |
| `clear_non_cond_mem_around_input` | False | D `VD:33` | no memory clearing |
| `use_memory_selection`, `offload_output_to_cpu_for_eval`, `trim_past_non_cond_mem_for_eval` | False, False, False | T `M:1035,1037,1048` | no trimming |
| `always_start_from_first_ann_frame`, `keep_first_cond_frame` | False, False | D `VD:40`, `VM:146` | |
| `low_res_mask_size` / `input_mask_size` | 288 / 1152 | derived `VM:396-400` | `1008/14*4`, `x4` |
| shrinkage threshold | 0.3 | const `VD:3357` | `_suppress_object_pw_area_shrinkage` |
| `NO_OBJ_SCORE`, `NO_OBJ_LOGIT` (suppression) | -1024, -10 | const `VM:49`, `B:1511` | |

Quantities not tabulated (decoder query count, presence head internals) are in the already ported detector and are unchanged on the video path [I].

## 9. CUDA-only, multi-GPU, compile-only, nondeterminism (Q8)

Skippable for a single-device port [V]:
- All `world_size > 1` paths (all-gather, `broadcast_python_obj_cpu`, `_assign_new_det_to_gpus`, `forward_video_grounding_multigpu` async gather); `rank0_metadata` is just local state.
- `torch.compile` (`compile=False`), `_compile_model`, `warm_up_compilation`, the padding of track masks to `max_num_objects` (`B:2018-2033`), `perflib` fast paths (`USE_PERFLIB=1` default, `perflib/__init__.py`): the NMS/IoU/box functions have equal semantics in both branches (`DU:115-132`).
- bf16 autocast contexts (`B:171,2938`, `BP:226`) and TF32 flags; `pin_memory()/non_blocking` transfers; async frame loading; profiler hooks; `cached_frame_outputs` bookkeeping; `postprocess_batch_size` batching; `action_history` (text path is always "full").
- Heads-up: the CPU patch left `.pin_memory()` call sites on the builder path (`B:2299,2570`, `T:719,1031,1096-1140`). Whether the full predictor runs on a torch build without an accelerator was not tested here [U].

Simplification a port can rely on: every tracker state has exactly one conditioning frame (its creation frame). Births into existing
states and reconditioning land in non-cond storage (`add_all_frames_to_correct_as_cond=False`), and a recondition that hits the cond
frame stays in the cond dict (`VD:2326-2333`). So `max_cond_frames_in_attn=4` / `select_closest_cond_frames` never bind here [I].
The non-batched `add_prompt` detector path (`forward_video_grounding_multigpu`) also prefetches frame f+1 into a cache that is never
read afterwards (`D:468-494`): skippable compute, no semantics.

Must be reproduced:
- Memory kept in **bf16** (`VD:2820,2920`); the per-state memory dict (`cond` / `non_cond` outputs per frame, per object pointers) and the `conditioning_objects` sets.
- Several tracker states in one session, each with its own multiplex state, dict of frame outputs and object-id order.

Nondeterminism / unspecified order [U]:
- `argsort` on bool (kept-first reordering, `B:545`) and score argsorts (`DU:206`, `V:2098`) have unspecified tie order; use stable sorts. This determines new-object id and slot order within one frame.
- `argmax` tie-breaking (`V:246`, non-overlap, `VM:2668`) is first index on CPU in practice, not guaranteed.
- `torch.randperm` is **not** used (`random=False`, `VD:1731`). GPU batched matmul IoU (`bmm` / `mm`) is exact on 0/1 float within 2^24 pixels [I].
- Batch size changes detector numerics slightly (frame f is detected at batch 1 in `add_prompt`, in a 16-chunk afterwards).

## 10. Open items / not determinable from code

- Reverse leg when the prompt is on f > 0: reuses the forward end-state (no reset in `Sam3BasePredictor.propagate_in_video`, `BP:342-361`), processes `f-1 .. 0` (`T:291-294`), hot-start constants flip sign. Behavior of old tracks propagating backwards beyond their births was not traced [U]. Prompt on frame 0 makes the reverse leg empty [V].
- Exact torch semantics of `torch.clamp(bool_tensor, max=0)` in the output non-overlap (`VD:3386-3389`) [U]; the intended result is winner-take-all.
- Detector mask resolution 288x288 is inferred from the FPN structure, not from a run [I].
- The empty-geometric-prompt path (`T:163-170`) is assumed identical to the image-path text prompt already ported [I]; I did not diff the geometry-encoder inputs.
- No numbers were produced; thresholds above are read, not measured. The hot-start / suppression / recondition branches have no reference fixtures yet.

## 11. Suggested port checkpoints (parity traps ordered by risk)

1. Joint score = `sigmoid(clamp(inv_sigmoid(sigmoid(cls)*sigmoid(pres)), +-10))` before every 0.4/0.65/0.8 threshold.
2. IoM everywhere (never IoU) at 288x288 on `>0` masks; non-kept detections have empty masks.
3. Ambiguity zeroing before `im_mask`, but **not** before `is_new_det` / `trk_is_unmatched`.
4. Frame f is visited twice (add_prompt + propagate): the second visit reads the stored cond output (`+-10` low-res mask, `+-10` score), runs step 10 (re-encoding f's memory, which becomes the final one) and, for f % 16 == 0, reconditions every newborn with det score >= 0.8.
5. Order inside a frame: propagate (no mem) -> associate/hot-start -> recondition (every 16th) -> occlusion suppression (in place) -> memory encode all states (288->1152 chain) -> births (their own preflight memory encode, 288->1008 chain, overwrites that state's frame-t memory) -> removals -> outputs.
6. Births into an existing state are non-cond frames; into a new state they are cond frames. Condition channel 1.0 only for objects conditioned on that frame.
7. Output hide rule = unconfirmed at `y+2`; ignore the dead removed/suppressed sets; final per-pixel winner by tracker score.
