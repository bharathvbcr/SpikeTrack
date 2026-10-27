//! Mask post-processing for `spiketrack image`: the detector's `288 x 288` mask
//! logits to the image's own resolution, and run-length encoding.
//!
//! Upstream (`Sam3Processor._forward_grounding`, sam3_image_processor.py) does
//! `torch.nn.functional.interpolate(logits, (img_h, img_w), mode="bilinear",
//! align_corners=False)` and then `.sigmoid() > 0.5`. Plain bilinear on the
//! logits, **not** antialiased (a larger-than-288 image is upsampled, a smaller
//! one is not area-filtered). [`ResizedMask`] reproduces the interpolation's
//! index and weight rule (`area_pixel_compute_source_index` for the linear
//! modes: `src = (dst + 0.5) * in / out - 0.5`, clamped below at 0; the upper
//! neighbour clamped to `in - 1`) in f32, and thresholds the interpolated
//! logit at 0, which is `sigmoid(x) > 0.5` except where the sigmoid rounds to
//! exactly 0.5 in f32 (|x| under about 6e-8).

/// One axis of the interpolation: for each output index, the two source
/// indices and the weight of the second.
struct Axis {
    i0: Vec<usize>,
    i1: Vec<usize>,
    frac: Vec<f32>,
}

impl Axis {
    fn new(input: usize, output: usize) -> Axis {
        let scale = input as f32 / output as f32;
        let mut axis = Axis { i0: Vec::with_capacity(output), i1: Vec::with_capacity(output), frac: Vec::with_capacity(output) };
        for d in 0..output {
            let src = ((d as f32 + 0.5) * scale - 0.5).max(0.0);
            let i0 = (src as usize).min(input - 1);
            axis.i0.push(i0);
            axis.i1.push((i0 + 1).min(input - 1));
            axis.frac.push(src - i0 as f32);
        }
        axis
    }
}

/// The bilinear resize of one square-or-not logit plane to `height x width`,
/// evaluated lazily: [`ResizedMask::is_set`] computes a pixel from its four
/// source logits, so no full-resolution float plane is ever allocated.
pub struct ResizedMask<'a> {
    logits: &'a [f32],
    in_w: usize,
    rows: Axis,
    cols: Axis,
    pub height: usize,
    pub width: usize,
}

impl<'a> ResizedMask<'a> {
    /// `logits` is `in_h x in_w` row-major. All sizes must be non-zero.
    pub fn new(logits: &'a [f32], in_h: usize, in_w: usize, height: usize, width: usize) -> ResizedMask<'a> {
        assert!(in_h > 0 && in_w > 0 && height > 0 && width > 0, "empty mask size");
        assert_eq!(logits.len(), in_h * in_w, "logits are not {in_h} x {in_w}");
        ResizedMask { logits, in_w, rows: Axis::new(in_h, height), cols: Axis::new(in_w, width), height, width }
    }

    /// The interpolated logit at row `y`, column `x`. Rows are interpolated
    /// first, as ATen's `upsample_bilinear2d` does (`h0lambda * (w0 * v00 + w1 * v01) + h1lambda * (...)`).
    pub fn logit(&self, y: usize, x: usize) -> f32 {
        let (r0, r1, ly) = (self.rows.i0[y], self.rows.i1[y], self.rows.frac[y]);
        let (c0, c1, lx) = (self.cols.i0[x], self.cols.i1[x], self.cols.frac[x]);
        let l = |r: usize, c: usize| self.logits[r * self.in_w + c];
        let top = (1.0 - lx) * l(r0, c0) + lx * l(r0, c1);
        let bottom = (1.0 - lx) * l(r1, c0) + lx * l(r1, c1);
        (1.0 - ly) * top + ly * bottom
    }

    pub fn is_set(&self, y: usize, x: usize) -> bool {
        self.logit(y, x) > 0.0
    }

    /// COCO uncompressed RLE of the thresholded mask: see [`Rle`].
    pub fn rle(&self) -> Rle {
        Rle::from_fn(self.height, self.width, |y, x| self.is_set(y, x))
    }
}

/// COCO uncompressed run-length encoding (`pycocotools` `{"size": [h, w],
/// "counts": [...]}` with a list of integers): the mask is read in
/// **column-major** order (down each column, columns left to right), `counts`
/// alternates run lengths starting with a run of zeros (so it starts with 0 when
/// the first pixel is set), and sums to `height * width`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rle {
    pub height: usize,
    pub width: usize,
    pub counts: Vec<u64>,
    /// Number of set pixels.
    pub area: u64,
}

impl Rle {
    /// Encode a `height x width` mask given pixel by pixel as `set(row, column)`.
    pub fn from_fn(height: usize, width: usize, mut set: impl FnMut(usize, usize) -> bool) -> Rle {
        let mut counts = Vec::new();
        let (mut current, mut run, mut area) = (false, 0u64, 0u64);
        for x in 0..width {
            for y in 0..height {
                let v = set(y, x);
                if v != current {
                    counts.push(run);
                    current = v;
                    run = 0;
                }
                run += 1;
                area += u64::from(v);
            }
        }
        counts.push(run);
        Rle { height, width, counts, area }
    }

    /// `counts` as the text of a JSON array.
    pub fn counts_json(&self) -> String {
        self.counts.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
    }

    /// Expand to row-major booleans (for tests and overlays).
    pub fn decode(&self) -> Vec<bool> {
        let mut out = vec![false; self.height * self.width];
        let (mut pos, mut value) = (0usize, false);
        for &c in &self.counts {
            for _ in 0..c {
                let (x, y) = (pos / self.height, pos % self.height);
                out[y * self.width + x] = value;
                pos += 1;
            }
            value = !value;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_size_returns_the_logits() {
        let logits: Vec<f32> = (0..12).map(|i| i as f32 - 5.5).collect();
        let m = ResizedMask::new(&logits, 3, 4, 3, 4);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(m.logit(y, x), logits[y * 4 + x]);
            }
        }
    }

    #[test]
    fn upsampling_follows_the_align_corners_false_rule() {
        // 1 x 2 logits [0, 4] to 1 x 4: sources (d + 0.5) / 2 - 0.5 = -0.25 (clamped to 0), 0.25, 0.75, 1.25 (neighbour clamped).
        let m = ResizedMask::new(&[0.0, 4.0], 1, 2, 1, 4);
        let got: Vec<f32> = (0..4).map(|x| m.logit(0, x)).collect();
        assert_eq!(got, vec![0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn downsampling_samples_without_antialiasing() {
        // 1 x 4 to 1 x 2: sources (d + 0.5) * 2 - 0.5 = 0.5 and 2.5, so the means of neighbours, not of all four.
        let m = ResizedMask::new(&[0.0, 2.0, 4.0, 10.0], 1, 4, 1, 2);
        assert_eq!((m.logit(0, 0), m.logit(0, 1)), (1.0, 7.0));
    }

    #[test]
    fn rle_is_column_major_and_starts_with_zeros() {
        // 2 rows x 3 columns, row-major:  1 0 0 / 1 1 0  -> column-major 1 1 | 0 1 | 0 0.
        let logits = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
        let rle = ResizedMask::new(&logits, 2, 3, 2, 3).rle();
        assert_eq!(rle.counts, vec![0, 2, 1, 1, 2]);
        assert_eq!(rle.counts.iter().sum::<u64>(), 6);
        assert_eq!(rle.area, 3);
        assert_eq!(rle.decode(), vec![true, false, false, true, true, false]);
    }

    #[test]
    fn rle_of_an_empty_and_a_full_mask() {
        let empty = ResizedMask::new(&[-1.0; 4], 2, 2, 2, 2).rle();
        assert_eq!((empty.counts.clone(), empty.area), (vec![4], 0));
        let full = ResizedMask::new(&[1.0; 4], 2, 2, 2, 2).rle();
        assert_eq!((full.counts.clone(), full.area), (vec![0, 4], 4));
    }

    #[test]
    fn rle_round_trips_a_resized_mask() {
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let m = ResizedMask::new(&logits, 8, 8, 13, 21);
        let rle = m.rle();
        let flat = rle.decode();
        for y in 0..13 {
            for x in 0..21 {
                assert_eq!(flat[y * 21 + x], m.is_set(y, x));
            }
        }
        assert_eq!(rle.area as usize, flat.iter().filter(|&&b| b).count());
    }
}
