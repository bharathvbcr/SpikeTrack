//! Small host-side (f32) math for the detector's per-query work: 200 boxes,
//! a 512-wide sine embedding, the box-RPB tables. Everything large runs on the
//! Ojas backend.

/// `y = x @ w^T + b` for `x` `[n, inp]` and `w` `[out, inp]`.
///
/// Every output is `(((0 + x0 w0) + x1 w1) + ...) + b`: one mul and one add per
/// term, in input order, with no fused multiply-add. The result is the same bit
/// for bit however the work is split. Outputs are computed in blocks of 8 or 4
/// that share one pass over the input row, which gives independent add chains
/// without reordering any one of them. Rows go to up to [`MAX_THREADS`] scoped
/// threads once a call holds [`THREAD_MIN_MACS`] multiply-adds.
pub fn linear(x: &[f32], n: usize, w: &[f32], b: &[f32], out: usize) -> Vec<f32> {
    assert!(out > 0, "linear with no outputs");
    let inp = w.len() / out;
    assert_eq!(w.len(), out * inp, "linear weight");
    assert_eq!(x.len(), n * inp, "linear input");
    assert_eq!(b.len(), out, "linear bias");
    let mut y = vec![0f32; n * out];
    if n == 0 {
        return y;
    }
    if inp == 0 {
        y.chunks_exact_mut(out).for_each(|yr| yr.iter_mut().zip(b).for_each(|(v, &bo)| *v = 0f32 + bo));
        return y;
    }
    let threads = if n * inp * out >= THREAD_MIN_MACS {
        std::thread::available_parallelism().map_or(1, |p| p.get()).min(MAX_THREADS).min(n)
    } else {
        1
    };
    if threads <= 1 {
        linear_rows(x, w, b, inp, out, &mut y);
        return y;
    }
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (xc, yc) in x.chunks(per * inp).zip(y.chunks_mut(per * out)) {
            s.spawn(move || linear_rows(xc, w, b, inp, out, yc));
        }
    });
    y
}

/// The most threads one [`linear`] or [`for_each_chunk_mut`] call uses.
pub const MAX_THREADS: usize = 8;

/// `f(start, piece)` over contiguous pieces of `dst`, `start` being the piece's first index, on up
/// to [`MAX_THREADS`] scoped threads with at least `min_per_thread` elements each (one piece, on
/// the calling thread, below that). Every element lies in exactly one piece, so an `f` that
/// computes each element from its index alone gives the same result however `dst` is split.
pub fn for_each_chunk_mut<T: Send>(dst: &mut [T], min_per_thread: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let len = dst.len();
    let threads = std::thread::available_parallelism()
        .map_or(1, |p| p.get())
        .min(MAX_THREADS)
        .min(len / min_per_thread.max(1));
    if threads <= 1 {
        f(0, dst);
        return;
    }
    let per = len.div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        for (k, piece) in dst.chunks_mut(per).enumerate() {
            s.spawn(move || f(k * per, piece));
        }
    });
}
/// The multiply-add count from which [`linear`] splits rows across threads.
pub const THREAD_MIN_MACS: usize = 1 << 20;

fn linear_rows(x: &[f32], w: &[f32], b: &[f32], inp: usize, out: usize, y: &mut [f32]) {
    for (xr, yr) in x.chunks_exact(inp).zip(y.chunks_exact_mut(out)) {
        let mut o = 0;
        while o + 8 <= out {
            linear_block::<8>(xr, w, b, inp, o, yr);
            o += 8;
        }
        if o + 4 <= out {
            linear_block::<4>(xr, w, b, inp, o, yr);
            o += 4;
        }
        while o < out {
            linear_block::<1>(xr, w, b, inp, o, yr);
            o += 1;
        }
    }
}

/// Outputs `o0..o0 + N` of one row.
#[inline(always)]
fn linear_block<const N: usize>(xr: &[f32], w: &[f32], b: &[f32], inp: usize, o0: usize, yr: &mut [f32]) {
    let wr: [&[f32]; N] = std::array::from_fn(|k| &w[(o0 + k) * inp..(o0 + k + 1) * inp]);
    let mut acc = [0f32; N];
    for i in 0..inp {
        let v = xr[i];
        for k in 0..N {
            acc[k] += v * wr[k][i];
        }
    }
    for k in 0..N {
        yr[o0 + k] = acc[k] + b[o0 + k];
    }
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

    /// The pre-blocking `linear`, one output at a time: the oracle the blocked,
    /// threaded one must match bit for bit.
    fn linear_scalar(x: &[f32], n: usize, w: &[f32], b: &[f32], out: usize) -> Vec<f32> {
        let inp = w.len() / out;
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

    /// Deterministic values over several binades, signs mixed, so that a
    /// reordered sum would round differently.
    fn values(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let m = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                m * [1e-3f32, 1.0, 37.0, 1e3][(s & 3) as usize]
            })
            .collect()
    }

    #[test]
    fn blocked_threaded_linear_is_bit_identical_to_the_scalar_one() {
        // (rows, in, out): the detector's shapes (box RPB 2 -> 256 -> 8 over 200 x 72 rows,
        // ref_point_head 512 -> 256, bbox_embed 256 -> 4), single rows, and out not a multiple
        // of 4 or 8. The 14400-row cases cross THREAD_MIN_MACS, with rows not divisible by the
        // thread count.
        for (case, &(n, inp, out)) in [(14_400, 2, 256), (14_400, 256, 8), (200, 512, 256), (200, 256, 4), (1, 256, 256), (3, 7, 13), (14_401, 5, 15), (2, 1, 1)].iter().enumerate() {
            let x = values(n * inp, case as u32);
            let w = values(out * inp, 100 + case as u32);
            let b = values(out, 200 + case as u32);
            let (got, want) = (linear(&x, n, &w, &b, out), linear_scalar(&x, n, &w, &b, out));
            assert_eq!(got.len(), want.len());
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "case {case} ({n}, {inp}, {out}) element {i}: {g} vs {e}");
            }
        }
        assert!(14_400 * 256 * 8 >= THREAD_MIN_MACS, "the threaded path is exercised");
    }

    #[test]
    fn linear_edge_shapes() {
        assert!(linear(&[], 0, &[1.0, 2.0], &[0.5, 0.25], 2).is_empty());
        assert_eq!(linear(&[], 3, &[], &[0.5, -0.0], 2), vec![0.5, 0.0, 0.5, 0.0, 0.5, 0.0]);
    }

    #[test]
    fn chunked_pieces_cover_every_index_once_at_their_start() {
        // Lengths below, at and above the threading threshold, and not divisible by the thread count.
        for &(len, min) in &[(0usize, 1usize), (1, 1), (7, 1), (1000, 64), (1001, 64), (100_003, 1 << 12), (5, 100)] {
            let mut v = vec![usize::MAX; len];
            for_each_chunk_mut(&mut v, min, |start, piece| {
                for (j, x) in piece.iter_mut().enumerate() {
                    assert_eq!(*x, usize::MAX, "index {} written twice", start + j);
                    *x = start + j;
                }
            });
            assert!(v.iter().enumerate().all(|(i, &x)| x == i), "len {len} min {min}");
        }
    }

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
