//! DAVIS region similarity J and boundary F-measure, as
//! davis2017-evaluation computes them (`davis2017/metrics.py`, ac7c43f):
//! `db_eval_iou` and `f_measure` with `_seg2bmap` boundaries and a
//! `skimage.morphology.disk` dilation of radius `ceil(0.008 * diagonal)`.

/// A binary mask, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn new(width: usize, height: usize, data: Vec<bool>) -> Self {
        assert_eq!(data.len(), width * height, "mask data is not {width} x {height}");
        Self { width, height, data }
    }

    pub fn empty(width: usize, height: usize) -> Self {
        Self::new(width, height, vec![false; width * height])
    }

    fn at(&self, y: usize, x: usize) -> bool {
        self.data[y * self.width + x]
    }

    fn same_size(&self, other: &Mask, what: &str) {
        assert!(
            self.width == other.width && self.height == other.height,
            "{what}: {}x{} vs {}x{}",
            self.width,
            self.height,
            other.width,
            other.height
        );
    }
}

pub const DAVIS_BOUND_TH: f64 = 0.008;

/// Jaccard index of `segmentation` against `annotation`, ignoring `void`
/// pixels; 1 when the union is empty.
pub fn region_j(annotation: &Mask, segmentation: &Mask, void: Option<&Mask>) -> f64 {
    annotation.same_size(segmentation, "region_j");
    if let Some(v) = void {
        annotation.same_size(v, "region_j void");
    }
    let (mut inter, mut union) = (0u64, 0u64);
    for i in 0..annotation.data.len() {
        if void.is_some_and(|v| v.data[i]) {
            continue;
        }
        let (a, s) = (annotation.data[i], segmentation.data[i]);
        inter += u64::from(a && s);
        union += u64::from(a || s);
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// `_seg2bmap`: 1-pixel boundaries offset half a pixel towards the origin.
pub fn seg2bmap(seg: &Mask) -> Mask {
    let (w, h) = (seg.width, seg.height);
    let mut b = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let s = seg.at(y, x);
            let e = x + 1 < w && seg.at(y, x + 1);
            let so = y + 1 < h && seg.at(y + 1, x);
            let se = x + 1 < w && y + 1 < h && seg.at(y + 1, x + 1);
            b[y * w + x] = (s ^ e) | (s ^ so) | (s ^ se);
        }
    }
    // Upstream overwrites the last row and column, then clears the corner.
    if h > 0 {
        for x in 0..w {
            let e = x + 1 < w && seg.at(h - 1, x + 1);
            b[(h - 1) * w + x] = seg.at(h - 1, x) ^ e;
        }
    }
    if w > 0 {
        for y in 0..h {
            let so = y + 1 < h && seg.at(y + 1, w - 1);
            b[y * w + w - 1] = seg.at(y, w - 1) ^ so;
        }
    }
    if w > 0 && h > 0 {
        b[(h - 1) * w + w - 1] = false;
    }
    Mask::new(w, h, b)
}

/// Binary dilation by `skimage.morphology.disk(radius)` (offsets with
/// `dx^2 + dy^2 <= radius^2`); pixels outside the image count as unset.
fn dilate_disk(m: &Mask, radius: usize) -> Mask {
    let r = radius as isize;
    let offsets: Vec<(isize, isize)> =
        (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dy, dx))).filter(|(dy, dx)| dy * dy + dx * dx <= r * r).collect();
    let (w, h) = (m.width as isize, m.height as isize);
    let mut out = vec![false; m.data.len()];
    for y in 0..h {
        for x in 0..w {
            if !m.data[(y * w + x) as usize] {
                continue;
            }
            for &(dy, dx) in &offsets {
                let (yy, xx) = (y + dy, x + dx);
                if (0..h).contains(&yy) && (0..w).contains(&xx) {
                    out[(yy * w + xx) as usize] = true;
                }
            }
        }
    }
    Mask::new(m.width, m.height, out)
}

/// Boundary F-measure of `foreground` (the segmentation) against `gt`.
pub fn boundary_f(foreground: &Mask, gt: &Mask, void: Option<&Mask>, bound_th: f64) -> f64 {
    foreground.same_size(gt, "boundary_f");
    let keep = |m: &Mask| -> Mask {
        match void {
            Some(v) => {
                m.same_size(v, "boundary_f void");
                Mask::new(m.width, m.height, m.data.iter().zip(&v.data).map(|(&a, &vd)| a && !vd).collect())
            }
            None => m.clone(),
        }
    };
    let bound_pix = if bound_th >= 1.0 {
        bound_th
    } else {
        (bound_th * ((foreground.height as f64).powi(2) + (foreground.width as f64).powi(2)).sqrt()).ceil()
    };
    let fg_b = seg2bmap(&keep(foreground));
    let gt_b = seg2bmap(&keep(gt));
    let fg_dil = dilate_disk(&fg_b, bound_pix as usize);
    let gt_dil = dilate_disk(&gt_b, bound_pix as usize);
    let count = |m: &Mask| m.data.iter().filter(|&&b| b).count() as f64;
    let both = |a: &Mask, b: &Mask| a.data.iter().zip(&b.data).filter(|(&x, &y)| x && y).count() as f64;
    let (n_fg, n_gt) = (count(&fg_b), count(&gt_b));
    let (precision, recall) = match (n_fg == 0.0, n_gt == 0.0) {
        (true, false) => (1.0, 0.0),
        (false, true) => (0.0, 1.0),
        (true, true) => (1.0, 1.0),
        (false, false) => (both(&fg_b, &gt_dil) / n_fg, both(&gt_b, &fg_dil) / n_gt),
    };
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(w: usize, h: usize, x0: usize, y0: usize, s: usize) -> Mask {
        Mask::new(w, h, (0..w * h).map(|i| (x0..x0 + s).contains(&(i % w)) && (y0..y0 + s).contains(&(i / w))).collect())
    }

    #[test]
    fn identical_masks_score_one() {
        let m = square(64, 48, 10, 10, 20);
        assert_eq!(region_j(&m, &m, None), 1.0);
        assert_eq!(boundary_f(&m, &m, None, DAVIS_BOUND_TH), 1.0);
    }

    #[test]
    fn empty_cases_follow_upstream() {
        let e = Mask::empty(32, 32);
        let m = square(32, 32, 4, 4, 8);
        assert_eq!(region_j(&e, &e, None), 1.0);
        assert_eq!(boundary_f(&e, &e, None, DAVIS_BOUND_TH), 1.0);
        assert_eq!(boundary_f(&e, &m, None, DAVIS_BOUND_TH), 0.0); // precision 1, recall 0
        assert_eq!(boundary_f(&m, &e, None, DAVIS_BOUND_TH), 0.0); // precision 0, recall 1
    }
}
