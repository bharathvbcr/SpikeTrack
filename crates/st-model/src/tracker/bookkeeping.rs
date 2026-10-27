//! Per-frame track bookkeeping of the text-prompted video path: births, hot-start
//! removal, reconditioning picks, occlusion-based suppression and masklet
//! confirmation, as upstream's builder path runs them (tools/reference/DETECT_TRACK_SPEC.md;
//! upstream = `Sam3MultiplexTrackingWithInteractivity`, i.e. `sam3_multiplex_tracking.py`
//! over `sam3_multiplex_base.py` over `sam3_video_base.py`). Forward direction, one
//! device (world size 1, rank 0). No network is run here: the caller supplies this
//! frame's [`Association`] (see `associate.rs`) and gets back a [`FramePlan`].
//!
//! Upstream function ported (file:line in the CPU-patched checkout):
//! - `realize_adt_result` / `LazyAssociateDetTrkResult._create_cpu_metadata`
//!   (`sam3_video_base.py:97-160`) -> [`realize_association`];
//! - `RealizedAssociateDetTrkresult.get_new_det_gpu_ids` + `_drop_new_det_with_obj_limit`
//!   (`sam3_video_base.py:54-94`, `2086-2100`) -> [`select_births`];
//! - `Sam3MultiplexBase._process_hotstart_gpu` (`sam3_multiplex_base.py:2094-2366`) and the
//!   compaction / extension of its metadata (`:1287-1405`) -> [`TrackBook::plan_frame`];
//! - the periodic reconditioning decision (`:1132-1176`, `_recondition_masklets` `:867-1008`);
//! - `_suppress_overlapping_based_on_recent_occlusion` and its helper
//!   (`:1424-1514`, `:1635-1729`) -> [`suppress_by_recent_occlusion`];
//! - `Sam3MultiplexBase.update_masklet_confirmation_status` (`:2824-2884`, the override
//!   that runs on this path, not `sam3_video_base.py:1976`);
//! - the id / score / max-id update (`:1215-1282`);
//! - the hot-start yield buffer and the look-ahead hiding of unconfirmed ids
//!   (`sam3_multiplex_tracking.py:337-481`, `657-661`) -> [`YieldBuffer`], [`HideLog`].
//!
//! Upstream quirk kept on purpose (it lives in `associate.rs`, which this builds on): when no
//! track exists, `_associate_det_trk` marks a detection new from its score alone, so a
//! boundary-suppressed detection (`det_keep` false) with a high score is still born then.
//! Not ported: the reverse direction (`reverse=True` flips the hot-start comparisons and the
//! occlusion comparison), multi-device load balancing, the model calls and the mask surgery
//! of reconditioning (its result on the binary mask is the detection's mask, which
//! [`TrackBook::plan_frame`] applies for the suppression test).
//!
//! Dead on this path, therefore not ported (verified with `rg -uu` over the checkout):
//! `to_suppress_mask` of `_process_hotstart_gpu` and the `trk_keep_alive` counters
//! behind it (assigned at `:1062`, never read), `rank0_metadata["removed_obj_ids"]` and
//! `["suppressed_obj_ids"]` (written only by `_process_hotstart`, which only the
//! overridden `Sam3VideoBase.run_tracker_update_planning_phase` calls, so the yield-time
//! hide sets are always empty and only the unconfirmed set hides), and the legacy
//! `obj_id_to_last_occluded` dict (reset to `{}` at `:1507`).
//!
//! Tie rules (each marked `TIE` where it is applied): the first index wins an `argmax` tie
//! (in `associate.rs`); `_drop_new_det_with_obj_limit` is `np.argsort(score)[::-1]`, which this
//! port reads as "stable ascending, reversed", so the later detection wins among equal scores
//! (see [`select_births`] for where that stops being upstream's behaviour).

use std::collections::{BTreeMap, VecDeque};

use crate::tracker::associate::{Association, Bitmask};

/// Upstream's `NEVER_OCCLUDED` (`sam3_multiplex_base.py:1347`, `:1454`).
pub const NEVER_OCCLUDED: i64 = -1;
/// Upstream's `ALWAYS_OCCLUDED` (`:1455`): a track the hot-start removes this frame.
pub const ALWAYS_OCCLUDED: i64 = 100_000;
/// The score a removed object keeps in `obj_id_to_score` (`:1264`).
pub const REMOVED_SCORE: f32 = -1e4;

/// The builder's values (`model_builder.py:1170-1190`, class defaults for the rest).
#[derive(Clone, Copy, Debug)]
pub struct BookParams {
    /// `hotstart_delay`: yield buffer length and the hot-start window.
    pub hotstart_delay: i64,
    /// `hotstart_unmatch_thresh`.
    pub hotstart_unmatch_thresh: u32,
    /// `hotstart_dup_thresh`.
    pub hotstart_dup_thresh: u32,
    /// `suppress_overlapping_based_on_recent_occlusion_threshold`; <= 0 turns suppression off.
    pub occlusion_iou_thresh: f32,
    /// `max_num_objects`.
    pub max_num_objects: usize,
    /// `recondition_every_nth_frame`; <= 0 turns reconditioning off.
    pub recondition_every_nth_frame: i64,
    /// `HIGH_CONF_THRESH` applied to `sigmoid(tracker object score logit)` (strict `>`).
    pub recondition_trk_score: f32,
    /// `masklet_confirmation_consecutive_det_thresh`.
    pub confirm_consecutive: u32,
}

impl Default for BookParams {
    fn default() -> Self {
        Self {
            hotstart_delay: 15,
            hotstart_unmatch_thresh: 8,
            hotstart_dup_thresh: 8,
            occlusion_iou_thresh: 0.7,
            max_num_objects: 16,
            recondition_every_nth_frame: 16,
            recondition_trk_score: 0.8,
            confirm_consecutive: 3,
        }
    }
}

/// `RealizedAssociateDetTrkresult` (without the birth fields): the association with
/// track positions turned into object ids.
#[derive(Clone, Debug, PartialEq)]
pub struct Realized {
    /// Detections with `is_new_det` set, in detection order.
    pub new_det_fa_inds: Vec<usize>,
    /// Ids of non-empty tracks no detection matched, in track order.
    pub unmatched_trk_ids: Vec<i64>,
    /// Ids of tracks with an empty mask, in track order.
    pub empty_trk_ids: Vec<i64>,
    /// For every kept detection (ascending), the ids of the tracks its `im_mask` row selects.
    pub det_to_matched_trk_ids: Vec<(usize, Vec<i64>)>,
    /// Track id -> detection for high-confidence, uniquely high-overlap kept detections.
    /// Dict semantics: keys in first-insertion order, a later detection overwrites the value.
    pub trk_id_to_max_iou_high_conf_det: Vec<(i64, usize)>,
}

/// `_create_cpu_metadata`: `trk_ids` are the previous frame's ids, one per track row of `a`.
pub fn realize_association(a: &Association, det_keep: &[bool], trk_ids: &[i64]) -> Realized {
    let (nd, nt) = (det_keep.len(), trk_ids.len());
    assert_eq!(a.trk_is_unmatched.len(), nt, "one unmatched flag per track");
    assert_eq!(a.trk_is_nonempty.len(), nt, "one non-empty flag per track");
    assert_eq!(a.is_new_det.len(), nd, "one new-det flag per detection");
    assert_eq!(a.im_mask.len(), nd * nt, "im_mask is [num_det, num_trk]");
    let mut det_to_matched_trk_ids = Vec::new();
    let mut trk_id_to_max_iou_high_conf_det: Vec<(i64, usize)> = Vec::new();
    for d in 0..nd {
        if !det_keep[d] {
            continue;
        }
        let matched = (0..nt).filter(|&t| a.im_mask[d * nt + t]).map(|t| trk_ids[t]).collect();
        det_to_matched_trk_ids.push((d, matched));
        if a.det_is_high_conf[d] && a.det_is_high_overlap[d] {
            let id = trk_ids[a.det_to_max_trk[d]];
            match trk_id_to_max_iou_high_conf_det.iter_mut().find(|(k, _)| *k == id) {
                Some(entry) => entry.1 = d,
                None => trk_id_to_max_iou_high_conf_det.push((id, d)),
            }
        }
    }
    Realized {
        new_det_fa_inds: (0..nd).filter(|&d| a.is_new_det[d]).collect(),
        unmatched_trk_ids: (0..nt).filter(|&t| a.trk_is_unmatched[t]).map(|t| trk_ids[t]).collect(),
        empty_trk_ids: (0..nt).filter(|&t| !a.trk_is_nonempty[t]).map(|t| trk_ids[t]).collect(),
        det_to_matched_trk_ids,
        trk_id_to_max_iou_high_conf_det,
    }
}

/// A newborn: which detection it comes from and the object id it gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Birth {
    pub det: usize,
    pub id: i64,
}

/// `get_new_det_gpu_ids` with one rank: choose which new detections become objects.
///
/// `prev_count` is the number of ids before this frame's removals (removed tracks still
/// count, they free capacity next frame). When `prev_count + new > max_objects`, only
/// `max_objects - prev_count` survive, taken from `_drop_new_det_with_obj_limit`:
/// `np.argsort(score[new])[::-1][:keep]`, so the survivors come in descending score and
/// that order, not detection order, assigns the ids. Otherwise all candidates keep detection
/// order. Ids are `max_obj_id + 1, + 2, ...` in the order chosen. Returns the births and the
/// number dropped.
///
/// TIE: equal scores among capped candidates. Upstream leaves the order to numpy's `argsort`
/// (quicksort kind, not stable). On the reference machine (numpy 2.5.4, arm64) it returns the
/// stable order for up to 16 elements, so "stable ascending, then reversed" (the later
/// detection wins a tie) is upstream's behaviour there, checked by the parity test. From 17
/// elements on numpy scrambles ties and this port does not follow it; the oracle never
/// generates a tied cap that large. With more than 16 new candidates in one frame and a tie at
/// the cut, expect the surviving set to differ from upstream's.
pub fn select_births(new_det_fa_inds: &[usize], det_scores: &[f32], prev_count: usize, max_obj_id: i64, max_objects: usize) -> (Vec<Birth>, usize) {
    let new_count = new_det_fa_inds.len();
    let mut kept: Vec<usize> = new_det_fa_inds.to_vec();
    let mut dropped = 0;
    if prev_count + new_count > max_objects {
        assert!(prev_count <= max_objects, "more tracked objects than the cap");
        let keep = max_objects - prev_count;
        dropped = new_count - keep;
        if keep < new_count {
            let mut order: Vec<usize> = (0..new_count).collect();
            order.sort_by(|&a, &b| det_scores[new_det_fa_inds[a]].partial_cmp(&det_scores[new_det_fa_inds[b]]).unwrap_or(std::cmp::Ordering::Equal));
            order.reverse();
            kept = order[..keep].iter().map(|&i| new_det_fa_inds[i]).collect();
        }
    }
    let births = kept.iter().enumerate().map(|(k, &det)| Birth { det, id: max_obj_id + 1 + k as i64 }).collect();
    (births, dropped)
}

/// `mask_iou`: `intersection / max(1, area_a + area_b - intersection)` in f32, row-major `[n, n]`.
fn mask_iou_self(masks: &[Bitmask]) -> Vec<f32> {
    let n = masks.len();
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let inter = masks[i].intersection(&masks[j]);
            let union = (masks[i].area() + masks[j].area() - inter).max(1);
            out[i * n + j] = inter as f32 / union as f32;
        }
    }
    out
}

/// `_get_objects_to_suppress_based_on_most_recently_occluded` (forward, and the builder's
/// `allow_unoccluded_to_suppress=False`): among pairs `i < j` whose mask IoU is `>= thresh`,
/// the track that was occluded more recently is suppressed, but only when the other one
/// was occluded at some point (`last_occluded > -1`). Equal values suppress nobody.
pub fn suppress_by_recent_occlusion(masks: &[Bitmask], last_occluded: &[i64], thresh: f32) -> Vec<bool> {
    let n = masks.len();
    assert_eq!(last_occluded.len(), n, "one last-occluded frame per track");
    let mut suppress = vec![false; n];
    if n <= 1 {
        return suppress;
    }
    let iou = mask_iou_self(masks);
    for i in 0..n {
        for j in i + 1..n {
            if iou[i * n + j] >= thresh {
                if last_occluded[i] > last_occluded[j] && last_occluded[j] > NEVER_OCCLUDED {
                    suppress[i] = true;
                }
                if last_occluded[j] > last_occluded[i] && last_occluded[i] > NEVER_OCCLUDED {
                    suppress[j] = true;
                }
            }
        }
    }
    suppress
}

/// `torch.sigmoid` on f32. Upstream compares `sigmoid(score) > 0.8`; a logit within an ulp
/// or two of `ln 4` could round differently from torch's vectorized exp (the parity oracle
/// probes the neighbourhood, see tests/bookkeeping_parity.rs).
pub fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// This frame's inputs. Track rows follow [`TrackBook::obj_ids`] (the previous frame's list).
pub struct FrameInput<'a> {
    pub frame_idx: i64,
    /// `associate()` over this frame's kept detections and the tracks' masks.
    pub association: &'a Association,
    /// Kept flags per detection (`det_keep`; `Association` does not carry them).
    pub det_keep: &'a [bool],
    pub det_scores: &'a [f32],
    /// The detector's binarized masks (needed: a reconditioned track takes the detection's mask).
    pub det_masks: &'a [Bitmask],
    /// The propagated tracker masks, before reconditioning and suppression.
    pub trk_masks: &'a [Bitmask],
    /// The tracker's object-score logits per track (`tracker_obj_scores_global`).
    pub trk_score_logits: &'a [f32],
}

/// What the frame decides.
#[derive(Clone, Debug, PartialEq)]
pub struct FramePlan {
    pub frame_idx: i64,
    pub realized: Realized,
    /// Newborns in id order, after the cap.
    pub births: Vec<Birth>,
    pub num_dropped_due_to_limit: usize,
    /// Ids removed by the hot-start (ascending, as in the previous list). They are still
    /// emitted for this frame; the id list returned below no longer has them.
    pub removed_ids: Vec<i64>,
    /// Track rows (positions in the previous list) whose mask is overwritten with -10.
    pub suppressed: Vec<usize>,
    /// Reconditioned tracks with the detection whose mask they take, in dict order.
    pub recondition: Vec<(i64, usize)>,
    /// The id list after this frame: previous ids then newborns, minus removals.
    pub obj_ids: Vec<i64>,
    /// Ids of `obj_ids` still unconfirmed after this frame's update.
    pub unconfirmed_ids: Vec<i64>,
}

/// Everything carried from frame to frame.
#[derive(Clone, Debug)]
pub struct TrackBook {
    params: BookParams,
    obj_ids: Vec<i64>,
    max_obj_id: i64,
    /// Birth frame per track row (`obj_first_frame`).
    first_frame: Vec<i64>,
    /// Cumulative unmatched frames per row (`consecutive_unmatch_count`, never reset).
    unmatch_count: Vec<u32>,
    /// `[n, n]` row-major; entry `[i][j]`, `i < j`, counts frames a detection covered both.
    overlap: Vec<u32>,
    /// Last frame the row was empty or suppressed (`last_occluded_tensor`).
    last_occluded: Vec<i64>,
    /// Frames in a row each row was matched, capped by being confirmed (`consecutive_det_num`).
    consecutive_det: Vec<u32>,
    confirmed: Vec<bool>,
    /// `obj_id_to_score`: the birth score; removed ids keep [`REMOVED_SCORE`].
    scores: BTreeMap<i64, f32>,
}

impl TrackBook {
    pub fn new(params: BookParams) -> Self {
        Self {
            params,
            obj_ids: vec![],
            max_obj_id: -1,
            first_frame: vec![],
            unmatch_count: vec![],
            overlap: vec![],
            last_occluded: vec![],
            consecutive_det: vec![],
            confirmed: vec![],
            scores: BTreeMap::new(),
        }
    }

    pub fn params(&self) -> &BookParams {
        &self.params
    }

    /// The live ids in track-row order (`obj_ids_all_gpu`); ascending.
    pub fn obj_ids(&self) -> &[i64] {
        &self.obj_ids
    }

    pub fn max_obj_id(&self) -> i64 {
        self.max_obj_id
    }

    /// `obj_id_to_score`.
    pub fn scores(&self) -> &BTreeMap<i64, f32> {
        &self.scores
    }

    pub fn first_frame(&self) -> &[i64] {
        &self.first_frame
    }

    pub fn unmatch_count(&self) -> &[u32] {
        &self.unmatch_count
    }

    /// `[n, n]` row-major pair counts.
    pub fn overlap_pair_counts(&self) -> &[u32] {
        &self.overlap
    }

    pub fn last_occluded(&self) -> &[i64] {
        &self.last_occluded
    }

    pub fn consecutive_det(&self) -> &[u32] {
        &self.consecutive_det
    }

    pub fn confirmed(&self) -> &[bool] {
        &self.confirmed
    }

    /// Run one frame of `run_tracker_update_planning_phase` and carry the state to the next.
    pub fn plan_frame(&mut self, input: &FrameInput) -> FramePlan {
        let n = self.obj_ids.len();
        let nd = input.det_keep.len();
        let p = self.params;
        let a = input.association;
        assert_eq!(input.det_scores.len(), nd, "one score per detection");
        assert_eq!(input.det_masks.len(), nd, "one mask per detection");
        assert_eq!(input.trk_masks.len(), n, "one mask per tracked object");
        assert_eq!(input.trk_score_logits.len(), n, "one score logit per tracked object");
        let realized = realize_association(a, input.det_keep, &self.obj_ids);

        // `_process_hotstart_gpu` (:2211-2345). Counters first, then the two removal tests.
        for t in 0..n {
            if a.trk_is_unmatched[t] {
                self.unmatch_count[t] += 1;
            }
        }
        for d in 0..nd {
            let row: Vec<usize> = (0..n).filter(|&t| a.im_mask[d * n + t]).collect();
            if row.len() > 1 {
                for (k, &i) in row.iter().enumerate() {
                    for &j in &row[k + 1..] {
                        self.overlap[i * n + j] += 1;
                    }
                }
            }
        }
        // `is_within_hotstart = obj_first_frame > frame_idx - hotstart_delay`.
        let hotstart_start = input.frame_idx - p.hotstart_delay;
        let to_remove: Vec<bool> = (0..n)
            .map(|j| {
                let within = self.first_frame[j] > hotstart_start;
                let by_unmatch = within && self.unmatch_count[j] >= p.hotstart_unmatch_thresh;
                // Max overlap count with any strictly earlier-born row (equal first frames never remove).
                let max_overlap = (0..n).filter(|&i| self.first_frame[i] < self.first_frame[j]).map(|i| self.overlap[i * n + j]).max().unwrap_or(0);
                let by_overlap = within && max_overlap >= p.hotstart_dup_thresh;
                by_unmatch || by_overlap
            })
            .collect();

        // Periodic reconditioning (:1132-1176, :867-1008): the pair is picked from the realized
        // association, then gated on the tracker's own score. The reconditioned track's binary
        // mask becomes the detection's: upstream writes `where(det>0 == trk>0, trk, det)`.
        let mut post_masks: Vec<Bitmask> = input.trk_masks.to_vec();
        let mut recondition = Vec::new();
        if p.recondition_every_nth_frame > 0 && input.frame_idx % p.recondition_every_nth_frame == 0 {
            for &(id, det) in &realized.trk_id_to_max_iou_high_conf_det {
                let row = self.obj_ids.iter().position(|&x| x == id).expect("a matched track id is in the id list");
                if sigmoid_f32(input.trk_score_logits[row]) > p.recondition_trk_score {
                    post_masks[row] = input.det_masks[det].clone();
                    recondition.push((id, det));
                }
            }
        }

        // `_suppress_overlapping_based_on_recent_occlusion` (:1424-1514), on the post-recondition masks.
        let mut suppressed_rows = vec![false; n];
        if p.occlusion_iou_thresh > 0.0 && n > 0 {
            let mut last_occ: Vec<i64> = (0..n).map(|t| if to_remove[t] { ALWAYS_OCCLUDED } else { self.last_occluded[t] }).collect();
            suppressed_rows = suppress_by_recent_occlusion(&post_masks, &last_occ, p.occlusion_iou_thresh);
            for t in 0..n {
                if post_masks[t].area() == 0 || suppressed_rows[t] {
                    last_occ[t] = input.frame_idx;
                }
            }
            self.last_occluded = last_occ;
        }

        // Births (get_new_det_gpu_ids).
        let (births, num_dropped_due_to_limit) = select_births(&realized.new_det_fa_inds, input.det_scores, n, self.max_obj_id, p.max_num_objects);

        // Id list, scores, max id (:1224-1282). Removed ids come only from the previous list.
        let removed_ids: Vec<i64> = (0..n).filter(|&t| to_remove[t]).map(|t| self.obj_ids[t]).collect();
        let keep_rows: Vec<usize> = (0..n).filter(|&t| !to_remove[t]).collect();
        let mut updated_ids: Vec<i64> = keep_rows.iter().map(|&t| self.obj_ids[t]).collect();
        updated_ids.extend(births.iter().map(|b| b.id));
        for b in &births {
            self.scores.insert(b.id, input.det_scores[b.det]);
        }
        if let Some(last) = births.iter().map(|b| b.id).max() {
            self.max_obj_id = self.max_obj_id.max(last);
        }
        for &id in &removed_ids {
            self.scores.insert(id, REMOVED_SCORE);
        }

        // `update_masklet_confirmation_status` (:2824-2884): carry by id, then count matches.
        let mut matched: Vec<i64> = births.iter().map(|b| b.id).collect();
        for (_, ids) in &realized.det_to_matched_trk_ids {
            matched.extend(ids);
        }
        let mut consecutive_det: Vec<u32> = Vec::with_capacity(updated_ids.len());
        let mut confirmed: Vec<bool> = Vec::with_capacity(updated_ids.len());
        for (k, &id) in updated_ids.iter().enumerate() {
            let (prev_count, prev_confirmed) = if k < keep_rows.len() { (self.consecutive_det[keep_rows[k]], self.confirmed[keep_rows[k]]) } else { (0, false) };
            let count = if matched.contains(&id) { prev_count + 1 } else { 0 };
            consecutive_det.push(count);
            confirmed.push(prev_confirmed || count >= p.confirm_consecutive);
        }
        self.consecutive_det = consecutive_det;
        self.confirmed = confirmed;
        let unconfirmed_ids: Vec<i64> = updated_ids.iter().zip(&self.confirmed).filter(|(_, &c)| !c).map(|(&id, _)| id).collect();

        // Compaction (:1287-1331) then extension for the newborns (:1334-1405).
        let m = keep_rows.len();
        let k = births.len();
        let mut overlap = vec![0u32; (m + k) * (m + k)];
        for (ni, &i) in keep_rows.iter().enumerate() {
            for (nj, &j) in keep_rows.iter().enumerate() {
                overlap[ni * (m + k) + nj] = self.overlap[i * n + j];
            }
        }
        self.overlap = overlap;
        self.first_frame = keep_rows.iter().map(|&t| self.first_frame[t]).chain(std::iter::repeat_n(input.frame_idx, k)).collect();
        self.unmatch_count = keep_rows.iter().map(|&t| self.unmatch_count[t]).chain(std::iter::repeat_n(0, k)).collect();
        self.last_occluded = keep_rows.iter().map(|&t| self.last_occluded[t]).chain(std::iter::repeat_n(NEVER_OCCLUDED, k)).collect();
        self.obj_ids = updated_ids;

        FramePlan {
            frame_idx: input.frame_idx,
            realized,
            births,
            num_dropped_due_to_limit,
            removed_ids,
            suppressed: (0..n).filter(|&t| suppressed_rows[t]).collect(),
            recondition,
            obj_ids: self.obj_ids.clone(),
            unconfirmed_ids,
        }
    }
}

/// The hot-start yield buffer of `propagate_in_video` (`sam3_multiplex_tracking.py:337-364`):
/// frame `y` is released when the buffer holds `delay` frames (so after `y + delay - 1` was
/// processed), and everything left is released with the last frame. `delay == 0` releases
/// each frame at once.
#[derive(Clone, Debug)]
pub struct YieldBuffer {
    delay: usize,
    pending: VecDeque<i64>,
}

impl YieldBuffer {
    pub fn new(delay: usize) -> Self {
        Self { delay, pending: VecDeque::new() }
    }

    /// Frame `frame` has been processed; `is_last` marks the end of propagation. Returns the
    /// frames released now, oldest first.
    pub fn push(&mut self, frame: i64, is_last: bool) -> Vec<i64> {
        if self.delay == 0 {
            return vec![frame];
        }
        self.pending.push_back(frame);
        if is_last {
            self.pending.drain(..).collect()
        } else if self.pending.len() >= self.delay {
            self.pending.pop_front().into_iter().collect()
        } else {
            vec![]
        }
    }
}

/// `unconfirmed_obj_ids_per_frame` and its look-ahead lookup (`:331-336`, `:395-405`).
/// A frame is shown without the ids that are still unconfirmed `confirm_consecutive - 1`
/// frames later (clamped into the video), because a track only counts once it was matched
/// in `confirm_consecutive` frames in a row.
#[derive(Clone, Debug)]
pub struct HideLog {
    look_ahead: i64,
    num_frames: i64,
    per_frame: BTreeMap<i64, Vec<i64>>,
}

impl HideLog {
    pub fn new(confirm_consecutive: u32, num_frames: i64) -> Self {
        Self { look_ahead: confirm_consecutive as i64 - 1, num_frames, per_frame: BTreeMap::new() }
    }

    /// Record the unconfirmed ids after processing `frame` (`FramePlan::unconfirmed_ids`).
    pub fn record(&mut self, frame: i64, unconfirmed_ids: &[i64]) {
        self.per_frame.insert(frame, unconfirmed_ids.to_vec());
    }

    /// The ids to hide when frame `yield_frame` is released; `None` if the look-ahead frame was
    /// never processed (then nothing is hidden).
    pub fn hidden_for(&self, yield_frame: i64) -> Option<&[i64]> {
        let at = (yield_frame + self.look_ahead).clamp(0, self.num_frames - 1);
        self.per_frame.get(&at).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::associate::{associate, AssocParams};

    const W: usize = 20;

    fn rect(x0: usize, y0: usize, x1: usize, y1: usize) -> Bitmask {
        Bitmask::from_fn(W * W, |i| {
            let (x, y) = (i % W, i / W);
            x >= x0 && x < x1 && y >= y0 && y < y1
        })
    }

    /// Run one frame the way the caller would: associate, then plan.
    fn step(book: &mut TrackBook, frame: i64, dets: &[(Bitmask, f32)], trks: &[Bitmask], logits: &[f32]) -> FramePlan {
        let det_masks: Vec<Bitmask> = dets.iter().map(|d| d.0.clone()).collect();
        let scores: Vec<f32> = dets.iter().map(|d| d.1).collect();
        let keep = vec![true; dets.len()];
        let assoc = associate(&det_masks, &scores, &keep, trks, &AssocParams::default());
        book.plan_frame(&FrameInput { frame_idx: frame, association: &assoc, det_keep: &keep, det_scores: &scores, det_masks: &det_masks, trk_masks: trks, trk_score_logits: logits })
    }

    #[test]
    fn births_take_consecutive_ids_and_are_hidden_until_confirmed() {
        let mut book = TrackBook::new(BookParams::default());
        let a = rect(0, 0, 6, 6);
        let b = rect(10, 10, 16, 16);
        let plan = step(&mut book, 3, &[(a.clone(), 0.9), (b.clone(), 0.7), (rect(0, 12, 3, 15), 0.6)], &[], &[]);
        assert_eq!(plan.births, vec![Birth { det: 0, id: 0 }, Birth { det: 1, id: 1 }]);
        assert_eq!(book.obj_ids(), &[0, 1]);
        assert_eq!(plan.unconfirmed_ids, vec![0, 1]);
        assert_eq!(book.first_frame(), &[3, 3]);
        assert_eq!(book.scores().get(&1), Some(&0.7));
        // Matched on the next two frames: confirmed on the third matched frame.
        let p4 = step(&mut book, 4, &[(a.clone(), 0.9), (b.clone(), 0.9)], &[a.clone(), b.clone()], &[5.0, 5.0]);
        assert_eq!(p4.unconfirmed_ids, vec![0, 1]);
        let p5 = step(&mut book, 5, &[(a.clone(), 0.9)], &[a.clone(), b.clone()], &[5.0, 5.0]);
        assert_eq!(p5.unconfirmed_ids, vec![1]);
        assert_eq!(book.consecutive_det(), &[3, 0]);
        // Confirmation is sticky once reached.
        let p6 = step(&mut book, 6, &[], &[a.clone(), b.clone()], &[5.0, 5.0]);
        assert_eq!(p6.unconfirmed_ids, vec![1]);
        assert_eq!(book.confirmed(), &[true, false]);
    }

    #[test]
    fn cap_keeps_the_top_scores_and_ids_follow_score_order() {
        // Three candidates, room for two: the two best survive, best first.
        let (births, dropped) = select_births(&[0, 2, 5], &[0.7, 0.0, 0.9, 0.0, 0.0, 0.8], 14, 20, 16);
        assert_eq!(births, vec![Birth { det: 2, id: 21 }, Birth { det: 5, id: 22 }]);
        assert_eq!(dropped, 1);
        // Everything fits: detection order, nothing dropped.
        let (births, dropped) = select_births(&[0, 2, 5], &[0.7, 0.0, 0.9, 0.0, 0.0, 0.8], 13, -1, 16);
        assert_eq!(births.iter().map(|b| (b.det, b.id)).collect::<Vec<_>>(), vec![(0, 0), (2, 1), (5, 2)]);
        assert_eq!(dropped, 0);
        // Equal scores at the cut: the later detection wins (stable ascending order, reversed).
        let (births, _) = select_births(&[0, 1, 2, 3], &[0.7, 0.7, 0.7, 0.9], 14, -1, 16);
        assert_eq!(births.iter().map(|b| (b.det, b.id)).collect::<Vec<_>>(), vec![(3, 0), (2, 1)]);
        // Full already: nobody is born.
        let (births, dropped) = select_births(&[1, 2], &[0.0, 0.9, 0.8], 16, 40, 16);
        assert!(births.is_empty());
        assert_eq!(dropped, 2);
    }

    #[test]
    fn unmatched_track_in_hotstart_is_removed_on_the_eighth_unmatched_frame() {
        let mut book = TrackBook::new(BookParams::default());
        let a = rect(0, 0, 6, 6);
        step(&mut book, 0, &[(a.clone(), 0.9)], &[], &[]);
        for f in 1..=7 {
            let plan = step(&mut book, f, &[], &[a.clone()], &[5.0]);
            assert!(plan.removed_ids.is_empty(), "frame {f}");
        }
        let plan = step(&mut book, 8, &[], &[a.clone()], &[5.0]);
        assert_eq!(plan.removed_ids, vec![0]);
        assert!(plan.obj_ids.is_empty() && book.obj_ids().is_empty());
        assert_eq!(book.scores().get(&0), Some(&REMOVED_SCORE));
        assert_eq!(book.max_obj_id(), 0);
        // A re-detection is born under a new id.
        let plan = step(&mut book, 9, &[(a, 0.9)], &[], &[]);
        assert_eq!(plan.births, vec![Birth { det: 0, id: 1 }]);
    }

    #[test]
    fn unmatched_track_after_the_hotstart_window_is_never_removed() {
        let mut book = TrackBook::new(BookParams::default());
        let a = rect(0, 0, 6, 6);
        step(&mut book, 0, &[(a.clone(), 0.9)], &[], &[]);
        // Matched through the window, then unmatched forever.
        for f in 1..=15 {
            step(&mut book, f, &[(a.clone(), 0.9)], &[a.clone()], &[5.0]);
        }
        for f in 16..=60 {
            let plan = step(&mut book, f, &[], &[a.clone()], &[5.0]);
            assert!(plan.removed_ids.is_empty(), "frame {f}");
        }
        assert_eq!(book.unmatch_count(), &[45]);
    }

    #[test]
    fn an_ambiguous_detection_does_not_count_as_a_duplicate_pair() {
        let mut book = TrackBook::new(BookParams::default());
        let old = rect(0, 0, 10, 10);
        let young = rect(10, 0, 20, 10);
        step(&mut book, 0, &[(old.clone(), 0.9)], &[], &[]);
        step(&mut book, 1, &[(old.clone(), 0.9), (young.clone(), 0.9)], &[old.clone()], &[5.0]);
        // One detection covering both tracks fully: IoM 1.0 with each, so its row is zeroed.
        let both = rect(0, 0, 20, 10);
        let trks = vec![old.clone(), young.clone()];
        for f in 2..=14 {
            let plan = step(&mut book, f, &[(both.clone(), 0.9)], &trks, &[5.0, 5.0]);
            assert!(plan.removed_ids.is_empty(), "frame {f}");
        }
        assert_eq!(book.overlap_pair_counts(), &[0, 0, 0, 0]);
    }

    #[test]
    fn weak_double_overlap_counts_and_removes_the_younger_track_on_the_eighth_frame() {
        let mut book = TrackBook::new(BookParams::default());
        let old = rect(0, 0, 10, 10);
        let young = rect(10, 0, 20, 10);
        step(&mut book, 0, &[(old.clone(), 0.9)], &[], &[]);
        step(&mut book, 1, &[(old.clone(), 0.9), (young.clone(), 0.9)], &[old.clone()], &[5.0]);
        // 20 of old's 100 pixels and 20 of young's: IoM 0.2 with each, below the 0.5 ambiguity cut.
        let straddle = Bitmask::from_fn(W * W, |i| {
            let (x, y) = (i % W, i / W);
            (x >= 8 && x < 12 && y < 10) || y >= 12
        });
        let trks = vec![old.clone(), young.clone()];
        let mut gone = None;
        for f in 2..=14 {
            let plan = step(&mut book, f, &[(straddle.clone(), 0.5), (old.clone(), 0.9), (young.clone(), 0.9)], &trks, &[5.0, 5.0]);
            if !plan.removed_ids.is_empty() {
                gone = Some((f, plan.removed_ids));
                break;
            }
        }
        // The pair counter reaches 8 on frame 9 (frames 2..=9); only the later-born track goes.
        assert_eq!(gone, Some((9, vec![1])));
        assert_eq!(book.obj_ids(), &[0]);
        assert_eq!(book.first_frame(), &[0]);
    }

    #[test]
    fn same_frame_twins_never_remove_each_other() {
        let mut book = TrackBook::new(BookParams::default());
        let t0 = rect(0, 0, 10, 10);
        let t1 = rect(10, 0, 20, 10);
        step(&mut book, 0, &[(t0.clone(), 0.9), (t1.clone(), 0.9)], &[], &[]);
        let straddle = Bitmask::from_fn(W * W, |i| {
            let (x, y) = (i % W, i / W);
            (x >= 8 && x < 12 && y < 10) || y >= 12
        });
        let trks = vec![t0.clone(), t1.clone()];
        for f in 1..=14 {
            let plan = step(&mut book, f, &[(straddle.clone(), 0.5), (t0.clone(), 0.9), (t1.clone(), 0.9)], &trks, &[5.0, 5.0]);
            assert!(plan.removed_ids.is_empty(), "frame {f}");
        }
        assert_eq!(book.overlap_pair_counts()[1], 14);
    }

    #[test]
    fn suppression_goes_to_the_more_recently_occluded_of_two_overlapping_tracks() {
        let m = rect(0, 0, 10, 10);
        // Identical masks: IoU 1. Row 1 was occluded at frame 7, row 0 never: nobody is suppressed
        // (the other track must have been occluded before) ...
        assert_eq!(suppress_by_recent_occlusion(&[m.clone(), m.clone()], &[NEVER_OCCLUDED, 7], 0.7), vec![false, false]);
        // ... both occluded: the later one is suppressed.
        assert_eq!(suppress_by_recent_occlusion(&[m.clone(), m.clone()], &[9, 3], 0.7), vec![true, false]);
        assert_eq!(suppress_by_recent_occlusion(&[m.clone(), m.clone()], &[3, 9], 0.7), vec![false, true]);
        // Equal frames suppress nobody.
        assert_eq!(suppress_by_recent_occlusion(&[m.clone(), m.clone()], &[4, 4], 0.7), vec![false, false]);
        // IoU exactly 0.7 suppresses (>=): 70 of 100 pixels in common, union 100.
        let sub = rect(0, 0, 10, 7);
        assert_eq!(suppress_by_recent_occlusion(&[m.clone(), sub.clone()], &[2, 1], 0.7), vec![true, false]);
        // Just below: 69 of 100.
        let less = Bitmask::from_fn(W * W, |i| rect(0, 0, 10, 7).get(i) && i != 0);
        assert_eq!(suppress_by_recent_occlusion(&[m.clone(), less], &[2, 1], 0.7), vec![false, false]);
        // Two empty masks have IoU 0.
        assert_eq!(suppress_by_recent_occlusion(&[Bitmask::empty(100), Bitmask::empty(100)], &[1, 5], 0.7), vec![false, false]);
        // A lone track is never suppressed.
        assert_eq!(suppress_by_recent_occlusion(&[m], &[5], 0.7), vec![false]);
    }

    #[test]
    fn occlusion_updates_last_occluded_and_suppresses_in_the_plan() {
        let mut book = TrackBook::new(BookParams::default());
        let a = rect(0, 0, 10, 10);
        let b = rect(12, 12, 18, 18);
        let empty = Bitmask::empty(W * W);
        step(&mut book, 0, &[(a.clone(), 0.9), (b.clone(), 0.9)], &[], &[]);
        step(&mut book, 1, &[(b.clone(), 0.9)], &[empty.clone(), b.clone()], &[5.0, 5.0]);
        assert_eq!(book.last_occluded(), &[1, NEVER_OCCLUDED]);
        step(&mut book, 2, &[(a.clone(), 0.9)], &[a.clone(), empty.clone()], &[5.0, 5.0]);
        assert_eq!(book.last_occluded(), &[1, 2]);
        // Both come back on the same spot: track 1 was occluded more recently and track 0 earlier,
        // so track 1 is suppressed and counts as occluded again.
        let plan = step(&mut book, 3, &[(a.clone(), 0.9)], &[a.clone(), a.clone()], &[5.0, 5.0]);
        assert_eq!(plan.suppressed, vec![1]);
        assert_eq!(book.last_occluded(), &[1, 3]);
    }

    #[test]
    fn periodic_recondition_pairs_unique_high_confidence_matches_gated_by_the_tracker_score() {
        let mut book = TrackBook::new(BookParams::default());
        let a = rect(0, 0, 10, 10);
        let b = rect(10, 10, 20, 20);
        step(&mut book, 14, &[(a.clone(), 0.9), (b.clone(), 0.9)], &[], &[]);
        // Frame 15 is not a multiple of 16: the pairs exist but nothing is reconditioned.
        let det = rect(1, 1, 9, 9);
        let p15 = step(&mut book, 15, &[(det.clone(), 0.85), (b.clone(), 0.9)], &[a.clone(), b.clone()], &[5.0, 5.0]);
        assert!(p15.recondition.is_empty());
        assert_eq!(p15.realized.trk_id_to_max_iou_high_conf_det, vec![(0, 0), (1, 1)]);
        // Frame 16: both pairs qualify, but track 1's own score is low (sigmoid(1.0) < 0.8).
        let p16 = step(&mut book, 16, &[(det.clone(), 0.85), (b.clone(), 0.9)], &[a.clone(), b.clone()], &[2.0, 1.0]);
        assert_eq!(p16.recondition, vec![(0, 0)]);
        // A detection below 0.8 is not high confidence.
        let p32 = step(&mut book, 32, &[(det, 0.79)], &[a, b], &[5.0, 5.0]);
        assert!(p32.recondition.is_empty());
    }

    #[test]
    fn suppression_sees_the_reconditioned_mask() {
        let mut book = TrackBook::new(BookParams::default());
        let big = rect(0, 0, 10, 10);
        let small = rect(0, 0, 10, 7); // IoU 0.7 with `big`
        let empty = Bitmask::empty(W * W);
        step(&mut book, 11, &[(big.clone(), 0.9), (small.clone(), 0.9)], &[], &[]);
        // Small, then big, are occluded: last_occluded = [13, 12].
        step(&mut book, 12, &[(big.clone(), 0.9)], &[big.clone(), empty.clone()], &[5.0, 5.0]);
        step(&mut book, 13, &[(small.clone(), 0.9)], &[empty, small.clone()], &[5.0, 5.0]);
        assert_eq!(book.last_occluded(), &[13, 12]);
        // Frame 16: a detection sits on the part of `big` outside `small` and matches only `big`.
        let det = rect(0, 8, 6, 10);
        let dets = [(det.clone(), 0.9)];
        let mut without = book.clone();
        let plain = step(&mut without, 16, &dets, &[big.clone(), small.clone()], &[0.0, 5.0]);
        assert!(plain.recondition.is_empty());
        assert_eq!(plain.suppressed, vec![0]); // IoU(big, small) = 0.7, big was occluded last
        let recond = step(&mut book, 16, &dets, &[big, small], &[5.0, 5.0]);
        assert_eq!(recond.recondition, vec![(0, 0)]);
        assert!(recond.suppressed.is_empty()); // big now carries the detection's mask: IoU 0 with small
        assert_eq!(book.last_occluded(), &[13, 12]);
    }

    #[test]
    fn yield_buffer_releases_after_delay_frames_and_flushes_at_the_end() {
        let mut buf = YieldBuffer::new(3);
        assert_eq!(buf.push(0, false), Vec::<i64>::new());
        assert_eq!(buf.push(1, false), Vec::<i64>::new());
        assert_eq!(buf.push(2, false), vec![0]);
        assert_eq!(buf.push(3, false), vec![1]);
        assert_eq!(buf.push(4, true), vec![2, 3, 4]);
        let mut now = YieldBuffer::new(0);
        assert_eq!(now.push(7, false), vec![7]);
    }

    #[test]
    fn hide_log_looks_two_frames_ahead_and_clamps() {
        let mut log = HideLog::new(3, 5);
        log.record(0, &[0]);
        log.record(1, &[0, 1]);
        log.record(2, &[1]);
        log.record(3, &[]);
        log.record(4, &[2]);
        assert_eq!(log.hidden_for(0), Some(&[1][..]));
        assert_eq!(log.hidden_for(2), Some(&[2][..]));
        assert_eq!(log.hidden_for(4), Some(&[2][..])); // 6 clamps to the last frame
        let mut short = HideLog::new(3, 100);
        short.record(0, &[5]);
        assert_eq!(short.hidden_for(0), None); // frame 2 never processed: nothing hidden
    }
}
