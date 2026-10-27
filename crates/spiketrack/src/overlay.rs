//! `overlay.png`: the image with every kept mask tinted and every box outlined,
//! one colour per detection. No text is drawn (there is no font); scores and
//! query indices are in `detections.json`, and the colour order is the order of
//! the `detections` array.

use st_video::Rgb8;

const PALETTE: [[u8; 3]; 8] = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]];
/// Weight of the mask colour over the pixel underneath, in 1/256ths.
const MASK_ALPHA: u32 = 100;
/// Box outline thickness in pixels.
const OUTLINE: usize = 2;

/// What is drawn for one detection.
pub struct Drawn<'a> {
    /// Row-major `height * width` mask at the image's resolution.
    pub mask: &'a [bool],
    /// `[x0, y0, x1, y1]` in pixels; may extend past the image (it is clipped).
    pub box_px: [f64; 4],
}

pub fn colour(index: usize) -> [u8; 3] {
    PALETTE[index % PALETTE.len()]
}

/// A copy of `image` with `detections` drawn over it, in order (later ones on top).
pub fn render(image: &Rgb8, detections: &[Drawn<'_>]) -> Rgb8 {
    let (w, h) = (image.width, image.height);
    let mut out = image.data.clone();
    for (i, d) in detections.iter().enumerate() {
        assert_eq!(d.mask.len(), w * h, "mask is not {w} x {h}");
        let c = colour(i);
        for (px, &set) in out.chunks_exact_mut(3).zip(d.mask) {
            if set {
                for k in 0..3 {
                    px[k] = ((u32::from(px[k]) * (256 - MASK_ALPHA) + u32::from(c[k]) * MASK_ALPHA) / 256) as u8;
                }
            }
        }
    }
    for (i, d) in detections.iter().enumerate() {
        outline(&mut out, w, h, d.box_px, colour(i));
    }
    Rgb8::new(w, h, out)
}

fn outline(out: &mut [u8], w: usize, h: usize, b: [f64; 4], c: [u8; 3]) {
    // Pixel indices of the box, clipped to the image; an entirely outside box draws nothing.
    let clip = |v: f64, max: usize| v.round().clamp(0.0, max as f64 - 1.0) as usize;
    if b[2] < 0.0 || b[3] < 0.0 || b[0] > w as f64 - 1.0 || b[1] > h as f64 - 1.0 {
        return;
    }
    let (x0, y0, x1, y1) = (clip(b[0], w), clip(b[1], h), clip(b[2], w), clip(b[3], h));
    let mut put = |x: usize, y: usize| out[(y * w + x) * 3..][..3].copy_from_slice(&c);
    for t in 0..OUTLINE {
        for x in x0..=x1 {
            put(x, (y0 + t).min(h - 1));
            put(x, y1.saturating_sub(t));
        }
        for y in y0..=y1 {
            put((x0 + t).min(w - 1), y);
            put(x1.saturating_sub(t), y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(w: usize, h: usize) -> Rgb8 {
        Rgb8::new(w, h, vec![100; w * h * 3])
    }

    #[test]
    fn a_mask_is_tinted_toward_its_colour_and_the_rest_is_untouched() {
        let img = grey(4, 2);
        let mut mask = vec![false; 8];
        mask[1] = true;
        let out = render(&img, &[Drawn { mask: &mask, box_px: [-10.0, -10.0, -5.0, -5.0] }]);
        let c = colour(0);
        for k in 0..3 {
            assert_eq!(u32::from(out.data[3 + k]), (100 * 156 + u32::from(c[k]) * 100) / 256);
        }
        assert_eq!(&out.data[..3], &[100, 100, 100]);
        assert_eq!(&out.data[6..], &img.data[6..], "a box entirely outside the image draws nothing");
    }

    #[test]
    fn a_box_is_outlined_inside_the_image_and_clipped() {
        let img = grey(10, 10);
        let mask = vec![false; 100];
        let out = render(&img, &[Drawn { mask: &mask, box_px: [2.0, 3.0, 20.0, 7.0] }]);
        let c = colour(0);
        let at = |x: usize, y: usize| out.data[(y * 10 + x) * 3..][..3].to_vec();
        assert_eq!(at(2, 3), c);
        assert_eq!(at(9, 3), c);
        assert_eq!(at(5, 7), c);
        assert_eq!(at(5, 4), c, "outline is 2 pixels thick");
        assert_eq!(at(5, 5), vec![100; 3], "the inside is not filled");
        assert_eq!(at(1, 3), vec![100; 3]);
    }
}
