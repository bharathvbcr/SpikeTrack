//! `overlay.png`: the image with every kept mask tinted and every box outlined,
//! one colour per detection. The only text is an optional number per box (a track
//! id), drawn with a built-in 3x5 digit font; scores and query indices are in
//! `detections.json`, and the colour order is the order of the `detections` array.

use st_video::Rgb8;

const PALETTE: [[u8; 3]; 8] = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]];
/// Weight of the mask colour over the pixel underneath, in 1/256ths.
const MASK_ALPHA: u32 = 100;
/// Box outline thickness in pixels.
const OUTLINE: usize = 2;
/// Digits 0-9 in a 3 x 5 bitmap, row-major from the top, the most significant of 15 bits first.
const DIGITS: [u16; 10] = [
    0b111_101_101_101_111, // 0
    0b010_110_010_010_111, // 1
    0b111_001_111_100_111, // 2
    0b111_001_111_001_111, // 3
    0b101_101_111_001_001, // 4
    0b111_100_111_001_111, // 5
    0b111_100_111_101_111, // 6
    0b111_001_010_010_010, // 7
    0b111_101_111_101_111, // 8
    0b111_101_111_001_111, // 9
];
/// Each font pixel is drawn as a SCALE x SCALE block; digits are one block apart, on a dark
/// backing with a one-block margin so the label reads over bright frames.
const SCALE: usize = 2;

/// What is drawn for one detection.
pub struct Drawn<'a> {
    /// Row-major `height * width` mask at the image's resolution.
    pub mask: &'a [bool],
    /// `[x0, y0, x1, y1]` in pixels; may extend past the image (it is clipped).
    pub box_px: [f64; 4],
    /// Selects the colour (modulo the palette size): the detection's position for stills,
    /// the track id for video, so a track keeps its colour from frame to frame.
    pub colour: usize,
    /// A number drawn in the box colour at the box's top-left corner (above the box, or just
    /// inside it when there is no room above): the track id for video, none for stills.
    pub label: Option<u64>,
}

pub fn colour(index: usize) -> [u8; 3] {
    PALETTE[index % PALETTE.len()]
}

/// A copy of `image` with `detections` drawn over it, in order (later ones on top).
pub fn render(image: &Rgb8, detections: &[Drawn<'_>]) -> Rgb8 {
    let (w, h) = (image.width, image.height);
    let mut out = image.data.clone();
    for d in detections {
        assert_eq!(d.mask.len(), w * h, "mask is not {w} x {h}");
        let c = colour(d.colour);
        for (px, &set) in out.chunks_exact_mut(3).zip(d.mask) {
            if set {
                for k in 0..3 {
                    px[k] = ((u32::from(px[k]) * (256 - MASK_ALPHA) + u32::from(c[k]) * MASK_ALPHA) / 256) as u8;
                }
            }
        }
    }
    for d in detections {
        outline(&mut out, w, h, d.box_px, colour(d.colour));
    }
    for d in detections {
        if let Some(n) = d.label {
            label(&mut out, w, h, d.box_px, n, colour(d.colour));
        }
    }
    Rgb8::new(w, h, out)
}

/// Draws `n` with its top-left at the box's top-left corner shifted up by the label height when
/// it fits above the image top, else at the corner (inside the box); clipped at the image edges.
/// A box entirely outside the image gets no label.
fn label(out: &mut [u8], w: usize, h: usize, b: [f64; 4], n: u64, c: [u8; 3]) {
    if b[2] < 0.0 || b[3] < 0.0 || b[0] > w as f64 - 1.0 || b[1] > h as f64 - 1.0 {
        return;
    }
    let digits: Vec<usize> = n.to_string().bytes().map(|d| usize::from(d - b'0')).collect();
    let (cell_w, cell_h) = (3 * SCALE, 5 * SCALE);
    let (label_w, label_h) = (digits.len() * (cell_w + SCALE) + SCALE, cell_h + 2 * SCALE);
    let x0 = b[0].round().max(0.0) as usize;
    let top = b[1].round().max(0.0) as usize;
    let y0 = if top >= label_h { top - label_h } else { top };
    let mut put = |x: usize, y: usize, colour: [u8; 3]| {
        if x < w && y < h {
            out[(y * w + x) * 3..][..3].copy_from_slice(&colour);
        }
    };
    for y in y0..y0 + label_h {
        for x in x0..x0 + label_w {
            put(x, y, [0, 0, 0]);
        }
    }
    for (i, &d) in digits.iter().enumerate() {
        let gx = x0 + SCALE + i * (cell_w + SCALE);
        for row in 0..5 {
            for col in 0..3 {
                if DIGITS[d] >> (14 - (row * 3 + col)) & 1 == 1 {
                    for dy in 0..SCALE {
                        for dx in 0..SCALE {
                            put(gx + col * SCALE + dx, y0 + SCALE + row * SCALE + dy, c);
                        }
                    }
                }
            }
        }
    }
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
        let out = render(&img, &[Drawn { mask: &mask, box_px: [-10.0, -10.0, -5.0, -5.0], colour: 0, label: Some(7) }]);
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
        let out = render(&img, &[Drawn { mask: &mask, box_px: [2.0, 3.0, 20.0, 7.0], colour: 0, label: None }]);
        let c = colour(0);
        let at = |x: usize, y: usize| out.data[(y * 10 + x) * 3..][..3].to_vec();
        assert_eq!(at(2, 3), c);
        assert_eq!(at(9, 3), c);
        assert_eq!(at(5, 7), c);
        assert_eq!(at(5, 4), c, "outline is 2 pixels thick");
        assert_eq!(at(5, 5), vec![100; 3], "the inside is not filled");
        assert_eq!(at(1, 3), vec![100; 3]);
    }

    #[test]
    fn a_label_is_drawn_above_the_box_in_its_colour_on_a_dark_backing() {
        let img = grey(40, 40);
        let mask = vec![false; 1600];
        // Box top at y = 20: the 14-pixel-high label sits at y 6..20, x 5..; "10" is 2 digits.
        let out = render(&img, &[Drawn { mask: &mask, box_px: [5.0, 20.0, 30.0, 35.0], colour: 1, label: Some(10) }]);
        let c = colour(1).to_vec();
        let at = |x: usize, y: usize| out.data[(y * 40 + x) * 3..][..3].to_vec();
        // Backing: the margin corner is black; past the label width (2 * 8 + 2 = 18) it is untouched.
        assert_eq!(at(5, 6), vec![0, 0, 0]);
        assert_eq!(at(5 + 18, 6), vec![100; 3]);
        // "1": its top row is 010, so its first lit block is the middle column, at x 7 + 2..
        assert_eq!(at(9, 8), c, "the top of the 1");
        assert_eq!(at(7, 8), vec![0, 0, 0], "left of the 1's top is backing");
        // "0" starts at x 7 + 8 = 15; its top row 111 is lit across.
        assert_eq!(at(15, 8), c);
        assert_eq!(at(19, 8), c);
        // The 0's hollow middle (row 1, col 1) is backing.
        assert_eq!(at(17, 10), vec![0, 0, 0]);
        // The box outline below is still drawn.
        assert_eq!(at(10, 20), c);
    }

    #[test]
    fn a_label_with_no_room_above_goes_inside_the_box_and_is_clipped_at_the_edge() {
        let img = grey(20, 20);
        let mask = vec![false; 400];
        let out = render(&img, &[Drawn { mask: &mask, box_px: [15.0, 2.0, 19.0, 19.0], colour: 0, label: Some(123) }]);
        // Top at y = 2 < 14: drawn from y = 2; x from 15, three digits run past x = 19 and clip.
        assert_eq!(&out.data[(2 * 20 + 15) * 3..][..3], &[0, 0, 0]);
        assert_eq!(out.data.len(), 20 * 20 * 3);
    }

    #[test]
    fn every_digit_glyph_is_distinct_and_nonempty() {
        for (i, a) in DIGITS.iter().enumerate() {
            assert!(*a != 0 && *a < 1 << 15, "digit {i}");
            for b in &DIGITS[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
