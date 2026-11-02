//! Pillow's bilinear resampling for 8-bit RGB (`libImaging/Resample.c`):
//! `Image.resize(size, Image.BILINEAR)` with no reducing gap, which is what
//! upstream SAM 3's video loader does (`torchvision.transforms.functional.resize`
//! on a PIL image gives identical output; checked in tools/reference/README.md).
//!
//! The algorithm: per output pixel, triangle-filter weights over the source
//! window (support widened by the downscale factor), normalized in f64 and
//! converted to 22-bit fixed point; a horizontal pass into a uint8
//! intermediate, then a vertical pass, each rounding and clamping to [0, 255].

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Interleaved 8-bit RGB, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgb8 {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height * 3,
            "RGB data is not {width} x {height} x 3"
        );
        Self {
            width,
            height,
            data,
        }
    }
}

fn bilinear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc`: per output index, the first
/// source index, the tap count, and `ksize` fixed-point weights.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    k: Vec<i32>,
}

fn coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 1.0 * filterscale; // bilinear support is 1
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut pre = vec![0f64; out_size * ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C casts to int truncate toward zero; Rust `as` does the same.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let row = &mut pre[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0;
        for x in 0..xmax {
            let w = bilinear((x as f64 + xmin as f64 - center + 0.5) * ss);
            row[x] = w;
            ww += w;
        }
        if ww != 0.0 {
            row[..xmax].iter_mut().for_each(|w| *w /= ww);
        }
        bounds.push((xmin, xmax));
    }
    let one = f64::from(1u32 << PRECISION_BITS);
    let k = pre
        .iter()
        .map(|&w| {
            if w < 0.0 {
                (-0.5 + w * one) as i32
            } else {
                (0.5 + w * one) as i32
            }
        })
        .collect();
    Coeffs { ksize, bounds, k }
}

fn clip8(v: i64) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((out_w, out_h), Image.BILINEAR)` for an RGB image.
pub fn resize_bilinear_pil(src: &Rgb8, out_w: usize, out_h: usize) -> Rgb8 {
    assert!(out_w > 0 && out_h > 0, "empty output size");
    let half = 1i64 << (PRECISION_BITS - 1);
    let need_h = out_w != src.width;
    let need_v = out_h != src.height;
    let cv = coeffs(src.height, out_h);
    // Only the source rows the vertical pass reads go through the horizontal pass.
    let (ybox_first, ybox_last) = if need_v {
        let first = cv.bounds[0].0;
        let (lb, lc) = cv.bounds[out_h - 1];
        (first, lb + lc)
    } else {
        (0, src.height)
    };
    let tmp = if need_h {
        let ch = coeffs(src.width, out_w);
        let rows = ybox_last - ybox_first;
        let mut out = vec![0u8; out_w * rows * 3];
        for y in 0..rows {
            let srow =
                &src.data[(y + ybox_first) * src.width * 3..(y + ybox_first + 1) * src.width * 3];
            for xx in 0..out_w {
                let (xmin, xmax) = ch.bounds[xx];
                let k = &ch.k[xx * ch.ksize..];
                for c in 0..3 {
                    let mut ss = half;
                    for x in 0..xmax {
                        ss += i64::from(srow[(x + xmin) * 3 + c]) * i64::from(k[x]);
                    }
                    out[(y * out_w + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        Rgb8::new(out_w, rows, out)
    } else {
        Rgb8::new(
            src.width,
            ybox_last - ybox_first,
            src.data[ybox_first * src.width * 3..ybox_last * src.width * 3].to_vec(),
        )
    };
    if !need_v {
        return tmp;
    }
    let w = tmp.width;
    let mut out = vec![0u8; w * out_h * 3];
    for yy in 0..out_h {
        let (ymin, ymax) = cv.bounds[yy];
        let ymin = ymin - ybox_first;
        let k = &cv.k[yy * cv.ksize..];
        for xx in 0..w {
            for c in 0..3 {
                let mut ss = half;
                for y in 0..ymax {
                    ss += i64::from(tmp.data[((y + ymin) * w + xx) * 3 + c]) * i64::from(k[y]);
                }
                out[(yy * w + xx) * 3 + c] = clip8(ss);
            }
        }
    }
    Rgb8::new(w, out_h, out)
}

/// Per output index: first source index and normalized f64 weights (torch's
/// `_compute_indices_min_size_weights_aa` for bilinear: Pillow's triangle filter).
fn aa_weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..out_size)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|j| bilinear((j as f64 - center + 0.5) * inv))
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (xmin, w)
        })
        .collect()
}

/// `F.interpolate(x, size, mode="bilinear", align_corners=False, antialias=True)`
/// for `planes` channel planes of `h x w` f32 (row-major, plane after plane).
/// Width first, then height, accumulating in f64 and rounding to f32 once per pass.
pub fn resize_bilinear_aa_f32(
    src: &[f32],
    planes: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    assert_eq!(
        src.len(),
        planes * h * w,
        "source is not {planes} x {h} x {w}"
    );
    let wx = aa_weights(w, ow);
    let wy = aa_weights(h, oh);
    let mut out = vec![0f32; planes * oh * ow];
    let mut tmp = vec![0f32; h * ow];
    for p in 0..planes {
        let s = &src[p * h * w..(p + 1) * h * w];
        for y in 0..h {
            for (x, (x0, ws)) in wx.iter().enumerate() {
                let row = &s[y * w + x0..y * w + x0 + ws.len()];
                tmp[y * ow + x] = row
                    .iter()
                    .zip(ws)
                    .map(|(&v, &k)| f64::from(v) * k)
                    .sum::<f64>() as f32;
            }
        }
        let o = &mut out[p * oh * ow..(p + 1) * oh * ow];
        for (y, (y0, ws)) in wy.iter().enumerate() {
            for x in 0..ow {
                o[y * ow + x] = ws
                    .iter()
                    .enumerate()
                    .map(|(k, &kw)| f64::from(tmp[(y0 + k) * ow + x]) * kw)
                    .sum::<f64>() as f32;
            }
        }
    }
    out
}

/// SAM 3's model input from 8-bit RGB: resize to `size x size`, scale to
/// [0, 1], normalize with mean 0.5 and std 0.5: NCHW `[1, 3, size, size]`.
pub fn sam_input(src: &Rgb8, size: usize) -> Vec<f32> {
    let r = resize_bilinear_pil(src, size, size);
    let n = size * size;
    let mut out = vec![0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = (f32::from(r.data[i * 3 + c]) / 255.0 - 0.5) / 0.5;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_size_is_a_copy() {
        let src = Rgb8::new(3, 2, (0..18).map(|v| v as u8 * 10).collect());
        assert_eq!(resize_bilinear_pil(&src, 3, 2), src);
    }

    #[test]
    fn constant_image_stays_constant() {
        let src = Rgb8::new(17, 9, vec![77; 17 * 9 * 3]);
        for (w, h) in [(40, 30), (5, 3), (17, 30), (100, 1)] {
            assert!(
                resize_bilinear_pil(&src, w, h)
                    .data
                    .iter()
                    .all(|&v| v == 77),
                "{w}x{h}"
            );
        }
    }
}
