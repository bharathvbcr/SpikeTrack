//! Small host-side (f32) math for the detector's per-query work: 200 boxes,
//! a 512-wide sine embedding, the box-RPB tables. Everything large runs on the
//! Ojas backend.

/// `y = x @ w^T + b` for `x` `[n, inp]` and `w` `[out, inp]`.
pub fn linear(x: &[f32], n: usize, w: &[f32], b: &[f32], out: usize) -> Vec<f32> {
    let inp = w.len() / out;
    assert_eq!(x.len(), n * inp, "linear input");
    assert_eq!(b.len(), out, "linear bias");
    let mut y = vec![0f32; n * out];
    for r in 0..n {
        let xr = &x[r * inp..(r + 1) * inp];
        for o in 0..out {
            let wr = &w[o * inp..(o + 1) * inp];
            let mut acc = 0f32;
            for i in 0..inp {
                acc += xr[i] * wr[i];
            }
            y[r * out + o] = acc + b[o];
        }
    }
    y
}

pub fn relu_in_place(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
}

/// LayerNorm over rows of width `dim` (biased variance), torch's formula.
pub fn layer_norm(x: &[f32], dim: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for (row, out) in x.chunks_exact(dim).zip(y.chunks_exact_mut(dim)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv = 1.0 / (var + f64::from(eps)).sqrt();
        for i in 0..dim {
            out[i] = ((f64::from(row[i]) - mean) * inv) as f32 * w[i] + b[i];
        }
    }
    y
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `model_misc.inverse_sigmoid` with its `eps = 1e-3` clamps.
pub fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    (x.max(1e-3) / (1.0 - x).max(1e-3)).ln()
}

/// `model_misc.gen_sineembed_for_position` for `[n, 4]` boxes (cx, cy, w, h)
/// and `num_feats = 256`: `[n, 512]` laid out (y, x, w, h), each 128 wide as
/// interleaved sin/cos of `v * 2pi / 10000^(2 floor(i / 2) / 128)`.
pub fn sine_embed_boxes(boxes: &[f32]) -> Vec<f32> {
    const HALF: usize = 128;
    let dim_t: Vec<f32> = (0..HALF).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / HALF as f32)).collect();
    let scale = 2.0 * std::f32::consts::PI;
    let n = boxes.len() / 4;
    let mut out = Vec::with_capacity(n * 4 * HALF);
    for b in boxes.chunks_exact(4) {
        for &coord in &[b[1], b[0], b[2], b[3]] {
            let e = coord * scale;
            for (i, &d) in dim_t.iter().enumerate() {
                let p = e / d;
                out.push(if i % 2 == 0 { p.sin() } else { p.cos() });
            }
        }
    }
    out
}

/// `PositionEmbeddingSine(num_pos_feats=256, normalize=True)` for an `h x w`
/// map: `[256, h, w]`, channels (y then x), each 128 interleaved sin/cos.
pub fn sine_position_map(h: usize, w: usize) -> Vec<f32> {
    const HALF: usize = 128;
    let dim_t: Vec<f32> = (0..HALF).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / HALF as f32)).collect();
    let scale = 2.0 * std::f32::consts::PI;
    let eps = 1e-6f32;
    let mut out = vec![0f32; 2 * HALF * h * w];
    for y in 0..h {
        let ye = (y + 1) as f32 / (h as f32 + eps) * scale;
        for x in 0..w {
            let xe = (x + 1) as f32 / (w as f32 + eps) * scale;
            for (i, &d) in dim_t.iter().enumerate() {
                let (py, px) = (ye / d, xe / d);
                let (vy, vx) = if i % 2 == 0 { (py.sin(), px.sin()) } else { (py.cos(), px.cos()) };
                out[(i * h + y) * w + x] = vy;
                out[((HALF + i) * h + y) * w + x] = vx;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_sigmoid_round_trips_inside_the_clamp() {
        for &p in &[0.01f32, 0.3, 0.5, 0.9, 0.99] {
            assert!((sigmoid(inverse_sigmoid(p)) - p).abs() < 1e-6);
        }
        assert_eq!(inverse_sigmoid(0.0), (1e-3f32 / 1.0).ln());
    }

    #[test]
    fn sine_map_first_pixel() {
        let m = sine_position_map(2, 3);
        // channel 0: sin of y-embed at y=0: (1 / (2 + 1e-6)) * 2pi / 1
        let want = (1.0f32 / (2.0 + 1e-6) * 2.0 * std::f32::consts::PI).sin();
        assert!((m[0] - want).abs() < 1e-7);
    }
}
