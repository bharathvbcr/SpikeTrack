//! Detection post-processing and detection-to-track association for the
//! text-prompted video path, as upstream's builder path runs them
//! (tools/reference/DETECT_TRACK_SPEC.md):
//! - `sam3_multiplex_detector_utils._nms_masks_core_batched` (NMS by mask IoM);
//! - `sam3_video_base._suppress_detections_close_to_boundary`;
//! - `sam3_video_base._associate_det_trk_compilable` with `use_iom_recondition`
//!   and no one-to-one matching (upstream asserts it off).
//!
//! Masks are binarized at logit > 0 and bit-packed; overlaps are popcounts.
//! Ratios are computed as upstream does: f32 `intersection / (min_area + 1e-8)`,
//! where the epsilon vanishes for any non-empty mask and turns 0/0 into 0.

/// A binary mask, row-major, one bit per pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmask {
    words: Vec<u64>,
    pixels: usize,
    area: u32,
}

impl Bitmask {
    /// `logits > 0`.
    pub fn from_logits(logits: &[f32]) -> Self {
        Self::from_fn(logits.len(), |i| logits[i] > 0.0)
    }

    pub fn from_fn(pixels: usize, mut on: impl FnMut(usize) -> bool) -> Self {
        let mut words = vec![0u64; pixels.div_ceil(64)];
        let mut area = 0u32;
        for i in 0..pixels {
            if on(i) {
                words[i / 64] |= 1 << (i % 64);
                area += 1;
            }
        }
        Self { words, pixels, area }
    }

    pub fn empty(pixels: usize) -> Self {
        Self { words: vec![0; pixels.div_ceil(64)], pixels, area: 0 }
    }

    pub fn area(&self) -> u32 {
        self.area
    }

    pub fn pixels(&self) -> usize {
        self.pixels
    }

    pub fn get(&self, i: usize) -> bool {
        self.words[i / 64] >> (i % 64) & 1 == 1
    }

    pub fn intersection(&self, other: &Self) -> u32 {
        assert_eq!(self.pixels, other.pixels, "masks of different sizes");
        self.words.iter().zip(&other.words).map(|(a, b)| (a & b).count_ones()).sum()
    }
}

/// `intersection / (min(area_a, area_b) + 1e-8)` in f32, row-major `[a.len(), b.len()]`.
pub fn mask_iom(a: &[Bitmask], b: &[Bitmask]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for x in a {
        for y in b {
            let min_area = x.area.min(y.area) as f32;
            out.push(x.intersection(y) as f32 / (min_area + 1e-8));
        }
    }
    out
}

/// Greedy NMS in descending-score order over detections with `prob > prob_threshold`:
/// a kept detection suppresses every later one whose IoM with it exceeds `iom_threshold`.
/// Returns the keep flags in the original order. Ties in score keep the original order.
pub fn nms_iom(probs: &[f32], masks: &[Bitmask], prob_threshold: f32, iom_threshold: f32) -> Vec<bool> {
    assert_eq!(probs.len(), masks.len(), "one mask per score");
    let n = probs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut keep_sorted: Vec<bool> = order.iter().map(|&i| probs[i] > prob_threshold).collect();
    for i in 0..n {
        if !keep_sorted[i] {
            continue;
        }
        let mi = &masks[order[i]];
        for j in i + 1..n {
            if keep_sorted[j] {
                let mj = &masks[order[j]];
                let min_area = mi.area.min(mj.area) as f32;
                if mi.intersection(mj) as f32 / (min_area + 1e-8) > iom_threshold {
                    keep_sorted[j] = false;
                }
            }
        }
    }
    let mut keep = vec![false; n];
    for (k, &i) in order.iter().enumerate() {
        keep[i] = keep_sorted[k];
    }
    keep
}

/// Box centers (normalized xyxy) strictly inside `[margin, 1 - margin]` on both axes.
pub fn not_close_to_boundary(boxes_xyxy: &[[f32; 4]], margin: f32) -> Vec<bool> {
    boxes_xyxy
        .iter()
        .map(|&[x0, y0, x1, y1]| {
            let (xc, yc) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
            xc > margin && xc < 1.0 - margin && yc > margin && yc < 1.0 - margin
        })
        .collect()
}

/// The builder path's association thresholds.
#[derive(Clone, Copy, Debug)]
pub struct AssocParams {
    /// `new_det_thresh`: a detection can start a track at or above this score.
    pub new_det_thresh: f32,
    /// `assoc_iou_thresh` (IoM here): a detection overlapping a track this much is not new.
    pub assoc_thresh: f32,
    /// `trk_assoc_iou_thresh`: a track overlapping a detection this much is matched.
    pub trk_assoc_thresh: f32,
    /// `iom_thresh_recondition`: unique high-overlap matches for reconditioning.
    pub recondition_thresh: f32,
    /// `HIGH_CONF_THRESH`.
    pub high_conf: f32,
}

impl Default for AssocParams {
    fn default() -> Self {
        Self { new_det_thresh: 0.65, assoc_thresh: 0.1, trk_assoc_thresh: 0.5, recondition_thresh: 0.5, high_conf: 0.8 }
    }
}

/// `_associate_det_trk`'s result, field for field.
#[derive(Clone, Debug, PartialEq)]
pub struct Association {
    /// Per track: non-empty and matched by no detection at `trk_assoc_thresh`.
    pub trk_is_unmatched: Vec<bool>,
    pub trk_is_nonempty: Vec<bool>,
    /// Per detection: kept, score >= `new_det_thresh`, and overlapping no track at `assoc_thresh`.
    pub is_new_det: Vec<bool>,
    /// Per detection: the track with the largest overlap after ambiguous rows and
    /// columns are zeroed (first index on ties; 0 when the row is all zero).
    pub det_to_max_trk: Vec<usize>,
    pub det_is_high_conf: Vec<bool>,
    pub det_is_high_overlap: Vec<bool>,
    /// `[num_det, num_trk]`: overlap >= `assoc_thresh` after the ambiguity zeroing.
    pub im_mask: Vec<bool>,
}

/// Match this frame's detections (`det_keep` marks those past the score and
/// boundary filters) with the tracks' masks. Detections that are not kept take
/// part as empty masks, as upstream zeroes them.
pub fn associate(det_masks: &[Bitmask], det_scores: &[f32], det_keep: &[bool], trk_masks: &[Bitmask], p: &AssocParams) -> Association {
    let (nd, nt) = (det_masks.len(), trk_masks.len());
    assert!(det_scores.len() == nd && det_keep.len() == nd, "one score and keep flag per detection");
    let trk_is_nonempty: Vec<bool> = trk_masks.iter().map(|m| m.area() > 0).collect();
    if nt == 0 {
        return Association {
            trk_is_unmatched: vec![],
            trk_is_nonempty,
            is_new_det: det_scores.iter().map(|&s| s >= p.new_det_thresh).collect(),
            det_to_max_trk: vec![usize::MAX; nd],
            det_is_high_conf: det_scores.iter().map(|&s| s >= p.high_conf).collect(),
            det_is_high_overlap: vec![false; nd],
            im_mask: vec![],
        };
    }
    if nd == 0 {
        return Association {
            trk_is_unmatched: vec![true; nt],
            trk_is_nonempty,
            is_new_det: vec![],
            det_to_max_trk: vec![],
            det_is_high_conf: vec![],
            det_is_high_overlap: vec![],
            im_mask: vec![],
        };
    }
    let empty = Bitmask::empty(det_masks[0].pixels());
    let dets: Vec<Bitmask> = det_masks.iter().zip(det_keep).map(|(m, &k)| if k { m.clone() } else { empty.clone() }).collect();
    let mut metric = mask_iom(&dets, trk_masks);
    let at = |m: &[f32], d: usize, t: usize| m[d * nt + t];

    let trk_is_matched: Vec<bool> = (0..nt).map(|t| (0..nd).any(|d| at(&metric, d, t) >= p.trk_assoc_thresh)).collect();
    let trk_is_unmatched: Vec<bool> = (0..nt).map(|t| trk_is_nonempty[t] && !trk_is_matched[t]).collect();
    let is_new_det: Vec<bool> = (0..nd).map(|d| det_scores[d] >= p.new_det_thresh && det_keep[d] && !(0..nt).any(|t| at(&metric, d, t) >= p.assoc_thresh)).collect();

    let det_many: Vec<bool> = (0..nd).map(|d| (0..nt).filter(|&t| at(&metric, d, t) >= p.recondition_thresh).count() > 1).collect();
    let trk_many: Vec<bool> = (0..nt).map(|t| (0..nd).filter(|&d| at(&metric, d, t) >= p.recondition_thresh).count() > 1).collect();
    for d in 0..nd {
        for t in 0..nt {
            if trk_many[t] || det_many[d] {
                metric[d * nt + t] = 0.0;
            }
        }
    }
    let det_to_max_trk: Vec<usize> = (0..nd)
        .map(|d| {
            let row = &metric[d * nt..(d + 1) * nt];
            // torch.argmax: the first index of the maximum.
            let mut best = 0;
            for (t, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = t;
                }
            }
            best
        })
        .collect();
    let det_is_high_conf: Vec<bool> = (0..nd).map(|d| det_scores[d] >= p.high_conf && det_keep[d] && !is_new_det[d]).collect();
    let det_is_high_overlap: Vec<bool> = (0..nd).map(|d| metric[d * nt..(d + 1) * nt].iter().copied().fold(f32::NEG_INFINITY, f32::max) >= p.recondition_thresh).collect();
    let im_mask: Vec<bool> = metric.iter().map(|&v| v >= p.assoc_thresh).collect();
    Association { trk_is_unmatched, trk_is_nonempty, is_new_det, det_to_max_trk, det_is_high_conf, det_is_high_overlap, im_mask }
}

/// Not upstream: the detection whose box a track reports under `--boxes detector`. Per track,
/// the highest-scoring kept detection (first on ties) that `a` assigns to it: not a birth, its
/// `det_to_max_trk` is the track, and that pair passes `im_mask` (overlap >= `assoc_thresh`
/// after the ambiguity zeroing). `None` for a track no detection is assigned to. `nt` tracks.
pub fn track_detections(a: &Association, det_scores: &[f32], det_keep: &[bool], nt: usize) -> Vec<Option<usize>> {
    let mut best: Vec<Option<usize>> = vec![None; nt];
    if nt == 0 {
        return best;
    }
    for d in 0..det_scores.len() {
        let t = a.det_to_max_trk[d];
        if !det_keep[d] || a.is_new_det[d] || t >= nt || !a.im_mask[d * nt + t] {
            continue;
        }
        if best[t].is_none_or(|b| det_scores[d] > det_scores[b]) {
            best[t] = Some(d);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Bitmask {
        Bitmask::from_fn(w * h, |i| {
            let (x, y) = (i % w, i / w);
            x >= x0 && x < x1 && y >= y0 && y < y1
        })
    }

    #[test]
    fn iom_is_intersection_over_the_smaller_area() {
        let a = rect(20, 20, 0, 0, 10, 10); // 100
        let b = rect(20, 20, 5, 0, 7, 10); // 20, inside a
        let c = rect(20, 20, 15, 15, 20, 20); // disjoint
        let e = Bitmask::empty(400);
        let m = mask_iom(&[a.clone()], &[b, c, e]);
        assert_eq!(m, vec![1.0, 0.0, 0.0]);
        assert_eq!(a.area(), 100);
    }

    #[test]
    fn nms_suppresses_contained_lower_scores_and_drops_low_probs() {
        let big = rect(20, 20, 0, 0, 10, 10);
        let inside = rect(20, 20, 2, 2, 4, 4);
        let apart = rect(20, 20, 12, 12, 16, 16);
        let keep = nms_iom(&[0.6, 0.9, 0.5, 0.3], &[big.clone(), inside, apart.clone(), apart], 0.4, 0.1);
        // 0.9 (inside) kept; big contains it (IoM 1) and is suppressed; apart kept; 0.3 below threshold.
        assert_eq!(keep, vec![false, true, true, false]);
    }

    #[test]
    fn boundary_margin_is_strict() {
        let k = not_close_to_boundary(&[[0.0, 0.0, 0.05, 0.05], [0.4, 0.4, 0.6, 0.6], [0.0, 0.4, 0.06, 0.6]], 0.025);
        assert_eq!(k, vec![false, true, true]);
    }

    #[test]
    fn association_new_matched_and_ambiguous() {
        let p = AssocParams::default();
        let t0 = rect(20, 20, 0, 0, 10, 10);
        let t1 = rect(20, 20, 10, 10, 20, 20);
        let d_on_t0 = rect(20, 20, 1, 1, 9, 9); // IoM 1 with t0
        let d_new = rect(20, 20, 0, 12, 5, 18); // disjoint from both
        let d_low = rect(20, 20, 12, 0, 18, 5); // disjoint but low score
        let a = associate(&[d_on_t0, d_new, d_low], &[0.9, 0.7, 0.5], &[true, true, true], &[t0, t1], &p);
        assert_eq!(a.is_new_det, vec![false, true, false]);
        assert_eq!(a.trk_is_unmatched, vec![false, true]);
        assert_eq!(a.det_to_max_trk[0], 0);
        assert_eq!(a.det_is_high_conf, vec![true, false, false]);
        assert_eq!(a.det_is_high_overlap, vec![true, false, false]);
        assert_eq!(a.im_mask, vec![true, false, false, false, false, false]);

        // One detection covering two tracks: ambiguous, its row is zeroed.
        let both = rect(20, 20, 0, 0, 20, 20);
        let a = associate(&[both], &[0.9], &[true], &[rect(20, 20, 0, 0, 10, 10), rect(20, 20, 10, 10, 20, 20)], &p);
        assert_eq!(a.trk_is_unmatched, vec![false, false]);
        assert_eq!(a.is_new_det, vec![false]);
        assert_eq!(a.det_is_high_overlap, vec![false]);
        assert_eq!(a.im_mask, vec![false, false]);
    }

    #[test]
    fn each_track_takes_its_best_scoring_assigned_detection() {
        let p = AssocParams::default();
        let t0 = rect(20, 20, 0, 0, 10, 10);
        let t1 = rect(20, 20, 10, 10, 20, 20);
        let t2 = rect(20, 20, 0, 12, 6, 18);
        // IoM 1/3 with t0: assigned to it, and not a second >= 0.5 overlap (which would make t0 ambiguous).
        let low_on_t0 = rect(20, 20, 8, 0, 14, 2);
        let high_on_t0 = rect(20, 20, 5, 5, 9, 9);
        let unkept_on_t1 = rect(20, 20, 12, 12, 18, 18);
        let newborn = rect(20, 20, 12, 0, 18, 5);
        let dets = [low_on_t0, high_on_t0, unkept_on_t1, newborn];
        let (scores, keep) = ([0.5, 0.7, 0.95, 0.9], [true, true, false, true]);
        let a = associate(&dets, &scores, &keep, &[t0, t1, t2], &p);
        assert_eq!(a.is_new_det, vec![false, false, false, true]);
        // t0: the 0.7 detection beats the 0.5; t1: its only detection is not kept; t2: nothing.
        assert_eq!(track_detections(&a, &scores, &keep, 3), vec![Some(1), None, None]);
        // No tracks: nothing to assign.
        let a = associate(&dets, &scores, &keep, &[], &p);
        assert_eq!(track_detections(&a, &scores, &keep, 0), Vec::<Option<usize>>::new());
    }

    #[test]
    fn an_ambiguous_detection_is_assigned_to_no_track() {
        let p = AssocParams::default();
        let both = rect(20, 20, 0, 0, 20, 20);
        let a = associate(&[both], &[0.9], &[true], &[rect(20, 20, 0, 0, 10, 10), rect(20, 20, 10, 10, 20, 20)], &p);
        assert_eq!(track_detections(&a, &[0.9], &[true], 2), vec![None, None]);
    }

    #[test]
    fn unkept_detections_match_nothing() {
        let t0 = rect(20, 20, 0, 0, 10, 10);
        let a = associate(&[t0.clone()], &[0.95], &[false], &[t0], &AssocParams::default());
        assert_eq!(a.trk_is_unmatched, vec![true]);
        assert_eq!(a.is_new_det, vec![false]);
        assert_eq!(a.det_is_high_conf, vec![false]);
    }
}
