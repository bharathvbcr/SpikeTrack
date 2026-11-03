//! Overlapping 2x2 tile geometry, interior-seam truncation filtering, and box/mask projection
//! back into full-frame coordinates for dense small-object detection.

use crate::host;

/// Default fractional side length of each crop in a 2x2 overlapping grid (`0.60` -> `20%` overlap band).
pub const DEFAULT_TILE_SPAN: f32 = 0.60;
/// Default tile-local margin from an interior cut edge within which partial boxes are discarded.
pub const DEFAULT_SEAM_MARGIN: f32 = 0.015;
/// Background mask logit placed outside a tile's footprint when pasting into the full-frame mask grid.
pub const OUTSIDE_MASK_LOGIT: f32 = -10.0;

/// One crop tile inside a `(full_w, full_h)` image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSpec {
    pub full_w: usize,
    pub full_h: usize,
    pub px0: usize,
    pub py0: usize,
    pub pw: usize,
    pub ph: usize,
    /// Normalized `[x0, y0, x1, y1]` in `[0, 1]` of the full image.
    pub norm_xyxy: [f32; 4],
}

impl TileSpec {
    /// Construct a `2x2` overlapping tile grid covering `(full_w, full_h)`.
    ///
    /// `span_frac` must lie in `(0.5, 1.0)` so the four tiles overlap across the central seam.
    pub fn grid_2x2(full_w: usize, full_h: usize, span_frac: f32) -> Result<[Self; 4], String> {
        if full_w < 2 || full_h < 2 {
            return Err(format!(
                "2x2 tiling needs an image at least 2x2, got {full_w}x{full_h}"
            ));
        }
        if !span_frac.is_finite() || span_frac <= 0.5 || span_frac >= 1.0 {
            return Err(format!(
                "tile span_frac must be finite in (0.5, 1.0), got {span_frac}"
            ));
        }
        let tw = ((full_w as f64 * f64::from(span_frac)).round() as usize).clamp(1, full_w);
        let th = ((full_h as f64 * f64::from(span_frac)).round() as usize).clamp(1, full_h);
        let xs = [0usize, full_w - tw];
        let ys = [0usize, full_h - th];
        let mut out = [Self {
            full_w,
            full_h,
            px0: 0,
            py0: 0,
            pw: tw,
            ph: th,
            norm_xyxy: [0.0; 4],
        }; 4];
        let mut idx = 0;
        for &py0 in &ys {
            for &px0 in &xs {
                out[idx] = Self {
                    full_w,
                    full_h,
                    px0,
                    py0,
                    pw: tw,
                    ph: th,
                    norm_xyxy: [
                        px0 as f32 / full_w as f32,
                        py0 as f32 / full_h as f32,
                        (px0 + tw) as f32 / full_w as f32,
                        (py0 + th) as f32 / full_h as f32,
                    ],
                };
                idx += 1;
            }
        }
        Ok(out)
    }

    /// True when a tile-local `[cx, cy, w, h]` box comes within `seam_margin` of an **interior**
    /// cut edge of this tile (outer image borders are not cut seams and are never suppressed here).
    pub fn touches_interior_seam(&self, box_cxcywh: [f32; 4], seam_margin: f32) -> bool {
        let [cx, cy, bw, bh] = box_cxcywh;
        let (bx0, by0) = (cx - 0.5 * bw, cy - 0.5 * bh);
        let (bx1, by1) = (cx + 0.5 * bw, cy + 0.5 * bh);
        (self.px0 > 0 && bx0 < seam_margin)
            || (self.px0 + self.pw < self.full_w && bx1 > 1.0 - seam_margin)
            || (self.py0 > 0 && by0 < seam_margin)
            || (self.py0 + self.ph < self.full_h && by1 > 1.0 - seam_margin)
    }

    /// Map a tile-local normalized `[cx, cy, w, h]` box into full-image normalized `[cx, cy, w, h]`.
    pub fn project_box_cxcywh(&self, box_cxcywh: [f32; 4]) -> [f32; 4] {
        let [x0, y0, x1, y1] = self.norm_xyxy;
        let (tw, th) = (x1 - x0, y1 - y0);
        [
            x0 + box_cxcywh[0] * tw,
            y0 + box_cxcywh[1] * th,
            box_cxcywh[2] * tw,
            box_cxcywh[3] * th,
        ]
    }

    /// Map a tile-local `[mask_side, mask_side]` mask logit plane into full-image `[mask_side, mask_side]`
    /// coordinates by pixel-center bilinear interpolation inside `self.norm_xyxy`, filling outside pixels
    /// with `outside_logit`.
    pub fn paste_mask_bilinear(
        &self,
        tile_mask: &[f32],
        mask_side: usize,
        outside_logit: f32,
    ) -> Vec<f32> {
        assert_eq!(
            tile_mask.len(),
            mask_side * mask_side,
            "tile_mask length must equal mask_side^2"
        );
        if mask_side == 0 {
            return Vec::new();
        }
        let [x0, y0, x1, y1] = self.norm_xyxy;
        let (tw, th) = (x1 - x0, y1 - y0);
        let mut full = vec![outside_logit; mask_side * mask_side];
        if !(tw > 0.0 && th > 0.0) {
            return full;
        }
        let s_f = mask_side as f32;
        let max_c = (mask_side - 1) as f32;
        for fy in 0..mask_side {
            let vf = (fy as f32 + 0.5) / s_f;
            if vf < y0 || vf > y1 {
                continue;
            }
            let vt = (vf - y0) / th;
            let sy = (vt * s_f - 0.5).clamp(0.0, max_c);
            let iy0 = sy.floor() as usize;
            let iy1 = (iy0 + 1).min(mask_side - 1);
            let wy = sy - iy0 as f32;
            let row0 = iy0 * mask_side;
            let row1 = iy1 * mask_side;
            let dst_row = fy * mask_side;
            for fx in 0..mask_side {
                let uf = (fx as f32 + 0.5) / s_f;
                if uf < x0 || uf > x1 {
                    continue;
                }
                let ut = (uf - x0) / tw;
                let sx = (ut * s_f - 0.5).clamp(0.0, max_c);
                let ix0 = sx.floor() as usize;
                let ix1 = (ix0 + 1).min(mask_side - 1);
                let wx = sx - ix0 as f32;
                let v00 = tile_mask[row0 + ix0];
                let v01 = tile_mask[row0 + ix1];
                let v10 = tile_mask[row1 + ix0];
                let v11 = tile_mask[row1 + ix1];
                let top = v00 + wx * (v01 - v00);
                let bot = v10 + wx * (v11 - v10);
                full[dst_row + fx] = top + wy * (bot - top);
            }
        }
        full
    }
}

/// Filter and project candidate queries (`sigmoid(logit) > min_prob`, not truncated on an interior
/// tile seam) from one tile into full-frame `(logits, boxes_cxcywh, masks)`.
pub fn collect_tile_candidates(
    spec: &TileSpec,
    tile_logits: &[f32],
    tile_boxes_cxcywh: &[f32],
    tile_masks: &[f32],
    mask_side: usize,
    min_prob: f32,
    seam_margin: f32,
    out_logits: &mut Vec<f32>,
    out_boxes_cxcywh: &mut Vec<f32>,
    out_masks: &mut Vec<f32>,
) {
    let q_count = tile_logits.len();
    let px = mask_side * mask_side;
    assert_eq!(tile_boxes_cxcywh.len(), q_count * 4);
    assert_eq!(tile_masks.len(), q_count * px);
    for q in 0..q_count {
        let logit = tile_logits[q];
        if !(host::sigmoid(logit) > min_prob) {
            continue;
        }
        let b = [
            tile_boxes_cxcywh[q * 4],
            tile_boxes_cxcywh[q * 4 + 1],
            tile_boxes_cxcywh[q * 4 + 2],
            tile_boxes_cxcywh[q * 4 + 3],
        ];
        if spec.touches_interior_seam(b, seam_margin) {
            continue;
        }
        let fb = spec.project_box_cxcywh(b);
        let fm = spec.paste_mask_bilinear(
            &tile_masks[q * px..(q + 1) * px],
            mask_side,
            OUTSIDE_MASK_LOGIT,
        );
        out_logits.push(logit);
        out_boxes_cxcywh.extend_from_slice(&fb);
        out_masks.extend_from_slice(&fm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_2x2_covers_corners_and_refuses_bad_inputs() {
        assert!(TileSpec::grid_2x2(1, 480, 0.6).is_err());
        assert!(TileSpec::grid_2x2(640, 1, 0.6).is_err());
        assert!(TileSpec::grid_2x2(640, 480, 0.5).is_err());
        assert!(TileSpec::grid_2x2(640, 480, 1.0).is_err());
        assert!(TileSpec::grid_2x2(640, 480, f32::NAN).is_err());

        let g = TileSpec::grid_2x2(640, 480, 0.60).unwrap();
        assert_eq!((g[0].px0, g[0].py0, g[0].pw, g[0].ph), (0, 0, 384, 288));
        assert_eq!((g[1].px0, g[1].py0, g[1].pw, g[1].ph), (256, 0, 384, 288));
        assert_eq!((g[2].px0, g[2].py0, g[2].pw, g[2].ph), (0, 192, 384, 288));
        assert_eq!((g[3].px0, g[3].py0, g[3].pw, g[3].ph), (256, 192, 384, 288));
        assert_eq!(g[0].norm_xyxy, [0.0, 0.0, 0.6, 0.6]);
        assert_eq!(g[3].norm_xyxy, [0.4, 0.4, 1.0, 1.0]);
    }

    #[test]
    fn interior_seam_filter_preserves_outer_image_borders_and_drops_cut_edges() {
        let g = TileSpec::grid_2x2(640, 480, 0.60).unwrap();
        let tl = g[0]; // [0..0.6, 0..0.6]: outer left & top, interior right & bottom
                       // Near outer left border (x0 = 0.002): kept!
        assert!(!tl.touches_interior_seam([0.02, 0.5, 0.036, 0.05], 0.015));
        // Near interior right seam (x1 = 0.995 > 0.985): dropped!
        assert!(tl.touches_interior_seam([0.97, 0.5, 0.05, 0.05], 0.015));

        let br = g[3]; // [0.4..1.0, 0.4..1.0]: interior left & top, outer right & bottom
                       // Near outer right/bottom border: kept!
        assert!(!br.touches_interior_seam([0.98, 0.98, 0.036, 0.036], 0.015));
        // Near interior left seam (x0 = 0.005 < 0.015): dropped!
        assert!(br.touches_interior_seam([0.02, 0.5, 0.03, 0.05], 0.015));
    }

    #[test]
    fn box_projection_and_bilinear_mask_paste_align_in_full_frame() {
        let g = TileSpec::grid_2x2(100, 100, 0.60).unwrap();
        let br = g[3]; // [0.4, 0.4, 1.0, 1.0]
        let fb = br.project_box_cxcywh([0.5, 0.5, 0.2, 0.1]);
        assert!((fb[0] - 0.7).abs() < 1e-6 && (fb[1] - 0.7).abs() < 1e-6);
        assert!((fb[2] - 0.12).abs() < 1e-6 && (fb[3] - 0.06).abs() < 1e-6);

        // 10x10 mask where tile is positive (+5.0) everywhere: full-frame pixels [4..10, 4..10]
        // have centers 0.45..0.95 inside [0.4, 1.0] -> +5.0; pixels 0..4 have centers 0.05..0.35 -> -10.0.
        let m = vec![5.0f32; 100];
        let pasted = br.paste_mask_bilinear(&m, 10, OUTSIDE_MASK_LOGIT);
        for y in 0..10 {
            for x in 0..10 {
                let want = if x >= 4 && y >= 4 {
                    5.0
                } else {
                    OUTSIDE_MASK_LOGIT
                };
                assert_eq!(pasted[y * 10 + x], want, "at ({x}, {y})");
            }
        }
    }
}
