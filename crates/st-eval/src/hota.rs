//! HOTA (Luiten et al., IJCV 2021) as TrackEval computes it
//! (`trackeval/metrics/hota.py`), for box tracks with IoU similarity.
//!
//! Per sequence: a global alignment score per (gt id, tracker id) pair, a
//! per-frame Hungarian match on `alignment * IoU`, then detection and
//! association accuracy at the 19 thresholds alpha = 0.05, 0.10, ..., 0.95;
//! reported numbers are the means over alpha. [`combine`] merges sequences as
//! TrackEval's `combine_sequences` does (sums of counts, TP-weighted AssA/LocA).

use std::collections::HashMap;
use std::hash::Hash;

use crate::assign::linear_sum_assignment;

pub const ALPHAS: usize = 19;
const EPS: f64 = f64::EPSILON;

pub fn alpha(a: usize) -> f64 {
    0.05 * (a as f64 + 1.0)
}

/// Axis-aligned box, top-left corner plus size (TrackEval's `xywh`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xywh {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

pub fn iou(a: &Xywh, b: &Xywh) -> f64 {
    let ix = ((a.x + a.w).min(b.x + b.w) - a.x.max(b.x)).max(0.0);
    let iy = ((a.y + a.h).min(b.y + b.h) - a.y.max(b.y)).max(0.0);
    let inter = ix * iy;
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// One frame: ground-truth and tracker detections with their track ids.
#[derive(Clone, Debug, Default)]
pub struct Frame<Id> {
    pub gt: Vec<(Id, Xywh)>,
    pub pred: Vec<(Id, Xywh)>,
}

/// Per-alpha counts and accuracies for one sequence (or a combination).
#[derive(Clone, Debug, PartialEq)]
pub struct Hota {
    pub tp: [f64; ALPHAS],
    pub fn_: [f64; ALPHAS],
    pub fp: [f64; ALPHAS],
    pub ass_a: [f64; ALPHAS],
    pub ass_re: [f64; ALPHAS],
    pub ass_pr: [f64; ALPHAS],
    pub loc_a: [f64; ALPHAS],
}

/// The numbers TrackEval prints: means over the 19 alphas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HotaSummary {
    pub hota: f64,
    pub det_a: f64,
    pub ass_a: f64,
    pub det_re: f64,
    pub det_pr: f64,
    pub ass_re: f64,
    pub ass_pr: f64,
    pub loc_a: f64,
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = xs.collect();
    v.iter().sum::<f64>() / v.len() as f64
}

impl Hota {
    fn zero() -> Self {
        Self {
            tp: [0.0; ALPHAS],
            fn_: [0.0; ALPHAS],
            fp: [0.0; ALPHAS],
            ass_a: [0.0; ALPHAS],
            ass_re: [0.0; ALPHAS],
            ass_pr: [0.0; ALPHAS],
            loc_a: [0.0; ALPHAS],
        }
    }

    pub fn det_re(&self, a: usize) -> f64 {
        self.tp[a] / (self.tp[a] + self.fn_[a]).max(1.0)
    }
    pub fn det_pr(&self, a: usize) -> f64 {
        self.tp[a] / (self.tp[a] + self.fp[a]).max(1.0)
    }
    pub fn det_a(&self, a: usize) -> f64 {
        self.tp[a] / (self.tp[a] + self.fn_[a] + self.fp[a]).max(1.0)
    }
    pub fn hota(&self, a: usize) -> f64 {
        (self.det_a(a) * self.ass_a[a]).sqrt()
    }

    pub fn summary(&self) -> HotaSummary {
        let r = 0..ALPHAS;
        HotaSummary {
            hota: mean(r.clone().map(|a| self.hota(a))),
            det_a: mean(r.clone().map(|a| self.det_a(a))),
            ass_a: mean(r.clone().map(|a| self.ass_a[a])),
            det_re: mean(r.clone().map(|a| self.det_re(a))),
            det_pr: mean(r.clone().map(|a| self.det_pr(a))),
            ass_re: mean(r.clone().map(|a| self.ass_re[a])),
            ass_pr: mean(r.clone().map(|a| self.ass_pr[a])),
            loc_a: mean(r.map(|a| self.loc_a[a])),
        }
    }
}

fn dense<Id: Eq + Hash + Clone>(map: &mut HashMap<Id, usize>, id: &Id) -> usize {
    let n = map.len();
    *map.entry(id.clone()).or_insert(n)
}

/// HOTA of one sequence. Ids are per sequence; a frame may not repeat an id.
pub fn hota_sequence<Id: Eq + Hash + Clone + std::fmt::Debug>(frames: &[Frame<Id>]) -> Hota {
    let (mut gmap, mut tmap) = (HashMap::new(), HashMap::new());
    // Dense ids and IoU matrices per frame.
    let mut per_frame = Vec::with_capacity(frames.len());
    for (t, f) in frames.iter().enumerate() {
        let g: Vec<usize> = f.gt.iter().map(|(id, _)| dense(&mut gmap, id)).collect();
        let p: Vec<usize> = f.pred.iter().map(|(id, _)| dense(&mut tmap, id)).collect();
        for (ids, what) in [(&g, "gt"), (&p, "tracker")] {
            let mut s = ids.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), ids.len(), "frame {t}: a {what} id appears twice");
        }
        let sim: Vec<f64> = f.gt.iter().flat_map(|(_, gb)| f.pred.iter().map(move |(_, pb)| iou(gb, pb))).collect();
        per_frame.push((g, p, sim));
    }
    let (ng, nt) = (gmap.len(), tmap.len());
    let mut res = Hota::zero();

    // Pass 1: global alignment between every gt and tracker id.
    let mut potential = vec![0.0f64; ng * nt];
    let mut gt_count = vec![0.0f64; ng];
    let mut tr_count = vec![0.0f64; nt];
    for (g, p, sim) in &per_frame {
        let (rows, cols) = (g.len(), p.len());
        let row_sum: Vec<f64> = (0..rows).map(|i| (0..cols).map(|j| sim[i * cols + j]).sum()).collect();
        let col_sum: Vec<f64> = (0..cols).map(|j| (0..rows).map(|i| sim[i * cols + j]).sum()).collect();
        for i in 0..rows {
            for j in 0..cols {
                let s = sim[i * cols + j];
                let denom = row_sum[i] + col_sum[j] - s;
                if denom > EPS {
                    potential[g[i] * nt + p[j]] += s / denom;
                }
            }
        }
        g.iter().for_each(|&i| gt_count[i] += 1.0);
        p.iter().for_each(|&j| tr_count[j] += 1.0);
    }
    let global: Vec<f64> = (0..ng * nt)
        .map(|k| {
            let (i, j) = (k / nt, k % nt);
            potential[k] / (gt_count[i] + tr_count[j] - potential[k])
        })
        .collect();

    // Pass 2: per-frame matching, counted per alpha.
    let mut matches = vec![vec![0.0f64; ng * nt]; ALPHAS];
    for (g, p, sim) in &per_frame {
        let (rows, cols) = (g.len(), p.len());
        if rows == 0 || cols == 0 {
            for a in 0..ALPHAS {
                res.fp[a] += cols as f64;
                res.fn_[a] += rows as f64;
            }
            continue;
        }
        let neg_score: Vec<f64> =
            (0..rows * cols).map(|k| -(global[g[k / cols] * nt + p[k % cols]] * sim[k])).collect();
        let pairs = linear_sum_assignment(&neg_score, rows, cols);
        for a in 0..ALPHAS {
            let ok: Vec<&(usize, usize)> = pairs.iter().filter(|&&(i, j)| sim[i * cols + j] >= alpha(a) - EPS).collect();
            let n = ok.len() as f64;
            res.tp[a] += n;
            res.fn_[a] += rows as f64 - n;
            res.fp[a] += cols as f64 - n;
            for &&(i, j) in &ok {
                res.loc_a[a] += sim[i * cols + j];
                matches[a][g[i] * nt + p[j]] += 1.0;
            }
        }
    }

    for a in 0..ALPHAS {
        let (mut ass_a, mut ass_re, mut ass_pr) = (0.0, 0.0, 0.0);
        for k in 0..ng * nt {
            let m = matches[a][k];
            if m == 0.0 {
                continue;
            }
            let (i, j) = (k / nt, k % nt);
            ass_a += m * (m / (gt_count[i] + tr_count[j] - m).max(1.0));
            ass_re += m * (m / gt_count[i].max(1.0));
            ass_pr += m * (m / tr_count[j].max(1.0));
        }
        let tp = res.tp[a].max(1.0);
        res.ass_a[a] = ass_a / tp;
        res.ass_re[a] = ass_re / tp;
        res.ass_pr[a] = ass_pr / tp;
        res.loc_a[a] = res.loc_a[a].max(1e-10) / res.tp[a].max(1e-10);
    }
    res
}

/// TrackEval's `combine_sequences` for HOTA: counts add; AssA, AssRe, AssPr and
/// LocA are TP-weighted means.
pub fn combine(seqs: &[Hota]) -> Hota {
    let mut out = Hota::zero();
    for a in 0..ALPHAS {
        let tp: f64 = seqs.iter().map(|s| s.tp[a]).sum();
        out.tp[a] = tp;
        out.fn_[a] = seqs.iter().map(|s| s.fn_[a]).sum();
        out.fp[a] = seqs.iter().map(|s| s.fp[a]).sum();
        let w = |f: fn(&Hota, usize) -> f64| seqs.iter().map(|s| f(s, a) * s.tp[a]).sum::<f64>() / tp.max(1.0);
        out.ass_a[a] = w(|s, a| s.ass_a[a]);
        out.ass_re[a] = w(|s, a| s.ass_re[a]);
        out.ass_pr[a] = w(|s, a| s.ass_pr[a]);
        out.loc_a[a] = seqs.iter().map(|s| s.loc_a[a] * s.tp[a]).sum::<f64>().max(1e-10) / tp.max(1e-10);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x: f64, y: f64) -> Xywh {
        Xywh { x, y, w: 10.0, h: 10.0 }
    }

    #[test]
    fn perfect_tracking_scores_one() {
        let frames: Vec<Frame<u32>> = (0..5)
            .map(|t| Frame {
                gt: vec![(1, b(t as f64, 0.0)), (2, b(50.0, t as f64))],
                pred: vec![(7, b(t as f64, 0.0)), (9, b(50.0, t as f64))],
            })
            .collect();
        let s = hota_sequence(&frames).summary();
        for v in [s.hota, s.det_a, s.ass_a, s.loc_a, s.det_re, s.det_pr] {
            assert!((v - 1.0).abs() < 1e-12, "{s:?}");
        }
    }

    #[test]
    fn id_switch_halves_association() {
        // One object, tracker switches id halfway: detection perfect, AssA = 0.5.
        let frames: Vec<Frame<u32>> = (0..10)
            .map(|t| Frame { gt: vec![(1, b(0.0, 0.0))], pred: vec![(if t < 5 { 1 } else { 2 }, b(0.0, 0.0))] })
            .collect();
        let s = hota_sequence(&frames).summary();
        assert!((s.det_a - 1.0).abs() < 1e-12);
        assert!((s.ass_a - 0.5).abs() < 1e-12, "{s:?}");
        assert!((s.hota - 0.5f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn empty_tracker_scores_zero() {
        let frames: Vec<Frame<u32>> = vec![Frame { gt: vec![(1, b(0.0, 0.0))], pred: vec![] }];
        let h = hota_sequence(&frames);
        assert_eq!(h.fn_[0], 1.0);
        assert_eq!(h.summary().hota, 0.0);
    }
}
