//! Deterministic weight-space inspection primitives: scalar statistics, row/column norms,
//! cosine-similarity search across embedding rows, matrix-free `f64` top singular directions,
//! and base-vs-edited delta analysis.
//!
//! All reductions run in `f64` in fixed ascending index order so inspection outputs are
//! reproducible across machines. Non-finite elements are refused rather than silently propagated.

use crate::Weight;

/// Default power-iteration steps per singular triplet.
pub const DEFAULT_SVD_ITERS: usize = 32;

/// Summary statistics of one tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub l2_norm: f64,
    pub rms: f64,
    pub mean: f64,
    pub std: f64,
    pub min: f32,
    pub max: f32,
    pub max_abs: f32,
    pub zero_count: usize,
}

impl TensorStats {
    pub fn compute(w: &Weight) -> Result<Self, String> {
        let numel = w.numel();
        if w.data.len() != numel {
            return Err(format!(
                "{}: data length {} disagrees with shape {:?} ({numel})",
                w.name,
                w.data.len(),
                w.shape
            ));
        }
        if numel == 0 {
            return Err(format!("{}: empty tensor {:?}", w.name, w.shape));
        }
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut max_abs = 0.0f32;
        let mut zero_count = 0usize;
        for (i, &v) in w.data.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!(
                    "{}: non-finite value {v} at flat index {i}",
                    w.name
                ));
            }
            let x = f64::from(v);
            sum += x;
            sum_sq += x * x;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
            if v == 0.0 {
                zero_count += 1;
            }
        }
        let n = numel as f64;
        let mean = sum / n;
        let mut var_acc = 0.0f64;
        for &v in &w.data {
            let d = f64::from(v) - mean;
            var_acc += d * d;
        }
        let std = (var_acc / n).sqrt();
        Ok(Self {
            name: w.name.clone(),
            shape: w.shape.clone(),
            numel,
            l2_norm: sum_sq.sqrt(),
            rms: (sum_sq / n).sqrt(),
            mean,
            std,
            min,
            max,
            max_abs,
            zero_count,
        })
    }
}

/// Summary of one row or column of a 2-D weight matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceStats {
    pub index: usize,
    pub l2_norm: f64,
    pub mean: f64,
    pub min: f32,
    pub max: f32,
    pub max_abs: f32,
}

fn require_2d(w: &Weight) -> Result<(usize, usize), String> {
    let &[rows, cols] = w.shape.as_slice() else {
        return Err(format!("{}: shape {:?} is not 2-D", w.name, w.shape));
    };
    if rows == 0 || cols == 0 {
        return Err(format!("{}: empty 2-D shape {:?}", w.name, w.shape));
    }
    if w.data.len() != rows * cols {
        return Err(format!(
            "{}: data length {} disagrees with [{rows}, {cols}]",
            w.name,
            w.data.len()
        ));
    }
    if let Some((i, v)) = w.data.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "{}: non-finite value {v} at flat index {i}",
            w.name
        ));
    }
    Ok((rows, cols))
}

/// Per-row norms and extrema of a 2-D `[rows, cols]` weight.
pub fn row_stats(w: &Weight) -> Result<Vec<SliceStats>, String> {
    let (rows, cols) = require_2d(w)?;
    let inv = 1.0 / cols as f64;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let slice = &w.data[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut max_abs = 0.0f32;
        for &v in slice {
            let x = f64::from(v);
            sum += x;
            sum_sq += x * x;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        out.push(SliceStats {
            index: r,
            l2_norm: sum_sq.sqrt(),
            mean: sum * inv,
            min,
            max,
            max_abs,
        });
    }
    Ok(out)
}

/// Per-column norms and extrema of a 2-D `[rows, cols]` weight.
pub fn col_stats(w: &Weight) -> Result<Vec<SliceStats>, String> {
    let (rows, cols) = require_2d(w)?;
    let inv = 1.0 / rows as f64;
    let mut sums = vec![0.0f64; cols];
    let mut sum_sqs = vec![0.0f64; cols];
    let mut mins = vec![f32::INFINITY; cols];
    let mut maxs = vec![f32::NEG_INFINITY; cols];
    let mut max_abss = vec![0.0f32; cols];
    for r in 0..rows {
        let row = &w.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let v = row[c];
            let x = f64::from(v);
            sums[c] += x;
            sum_sqs[c] += x * x;
            if v < mins[c] {
                mins[c] = v;
            }
            if v > maxs[c] {
                maxs[c] = v;
            }
            let a = v.abs();
            if a > max_abss[c] {
                max_abss[c] = a;
            }
        }
    }
    Ok((0..cols)
        .map(|c| SliceStats {
            index: c,
            l2_norm: sum_sqs[c].sqrt(),
            mean: sums[c] * inv,
            min: mins[c],
            max: maxs[c],
            max_abs: max_abss[c],
        })
        .collect())
}

/// Cosine similarity `a · b / (||a||_2 ||b||_2)` in `f64`; returns `0.0` when either vector is identically zero.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64, String> {
    if a.len() != b.len() || a.is_empty() {
        return Err(format!(
            "cosine_similarity: lengths {} and {} must match and be non-empty",
            a.len(),
            b.len()
        ));
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return Err("cosine_similarity: non-finite element".to_string());
        }
        let (xf, yf) = (f64::from(x), f64::from(y));
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        Ok(0.0)
    } else {
        Ok((dot / denom).clamp(-1.0, 1.0))
    }
}

/// One row matched by cosine similarity.
#[derive(Clone, Debug, PartialEq)]
pub struct NearestRow {
    pub row: usize,
    pub cosine: f64,
    pub l2_norm: f64,
}

/// Top `top_k` rows of 2-D `w` ordered by descending cosine similarity to `query`.
pub fn nearest_rows(w: &Weight, query: &[f32], top_k: usize) -> Result<Vec<NearestRow>, String> {
    let (rows, cols) = require_2d(w)?;
    if query.len() != cols {
        return Err(format!(
            "{}: query length {} disagrees with width {cols}",
            w.name,
            query.len()
        ));
    }
    if query.iter().any(|v| !v.is_finite()) {
        return Err(format!("{}: query holds a non-finite value", w.name));
    }
    if top_k == 0 {
        return Ok(Vec::new());
    }
    let qn: f64 = query
        .iter()
        .map(|&v| {
            let x = f64::from(v);
            x * x
        })
        .sum::<f64>()
        .sqrt();
    if qn == 0.0 {
        return Err(format!("{}: zero-norm query vector", w.name));
    }
    let mut all: Vec<NearestRow> = Vec::with_capacity(rows);
    for r in 0..rows {
        let slice = &w.data[r * cols..(r + 1) * cols];
        let mut dot = 0.0f64;
        let mut rn_sq = 0.0f64;
        for (&x, &q) in slice.iter().zip(query) {
            let (xf, qf) = (f64::from(x), f64::from(q));
            dot += xf * qf;
            rn_sq += xf * xf;
        }
        let rn = rn_sq.sqrt();
        let cos = if rn == 0.0 {
            0.0
        } else {
            (dot / (rn * qn)).clamp(-1.0, 1.0)
        };
        all.push(NearestRow {
            row: r,
            cosine: cos,
            l2_norm: rn,
        });
    }
    all.sort_by(|a, b| b.cosine.total_cmp(&a.cosine).then(a.row.cmp(&b.row)));
    all.truncate(top_k);
    Ok(all)
}

/// One singular value and its unit left (`u`, length `rows`) and right (`v`, length `cols`) vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct SingularTriplet {
    pub sigma: f64,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

fn vec_norm(x: &[f64]) -> f64 {
    let max_abs = x.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return max_abs;
    }
    let inv = 1.0 / max_abs;
    let sum_sq: f64 = x
        .iter()
        .map(|&v| {
            let s = v * inv;
            s * s
        })
        .sum();
    max_abs * sum_sq.sqrt()
}

fn normalize_in_place(x: &mut [f64]) -> f64 {
    let n = vec_norm(x);
    if n > 0.0 {
        let inv = 1.0 / n;
        for v in x {
            *v *= inv;
        }
    }
    n
}

fn orthogonalize(vec: &mut [f64], basis: &[Vec<f64>]) {
    // Two passes of classical Gram-Schmidt keep orthogonality to machine precision.
    for _ in 0..2 {
        for b in basis {
            let dot: f64 = vec.iter().zip(b).map(|(&x, &y)| x * y).sum();
            for (x, &y) in vec.iter_mut().zip(b) {
                *x -= dot * y;
            }
        }
    }
}

fn seed_unit_vector(len: usize, mode: usize, basis: &[Vec<f64>]) -> Option<Vec<f64>> {
    // Try deterministic trigonometric seeds first, then standard basis vectors if a tiny subspace
    // happened to align with the trigonometric probe.
    for attempt in 0..4 {
        let mut v: Vec<f64> = (0..len)
            .map(|j| {
                let a = (j + 1) as f64 * (mode + attempt + 1) as f64 * 0.618_033_988_749_895;
                let b = (j + 2) as f64 * (mode + 2 * attempt + 3) as f64 * 0.414_213_562_373_095;
                a.sin() + 0.3125 * b.cos()
            })
            .collect();
        orthogonalize(&mut v, basis);
        if normalize_in_place(&mut v) > 1e-12 {
            return Some(v);
        }
    }
    for e in 0..len {
        let mut v = vec![0.0f64; len];
        v[e] = 1.0;
        orthogonalize(&mut v, basis);
        if normalize_in_place(&mut v) > 1e-12 {
            return Some(v);
        }
    }
    None
}

/// Deterministic column-pivoted two-pass Gram-Schmidt orthonormal basis for `cols_in`.
/// Drops numerically dependent directions below `1e-14 * max_col_norm`.
fn orthonormal_span(cols_in: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = cols_in.len();
    if m == 0 {
        return Vec::new();
    }
    let mut rem: Vec<Vec<f64>> = cols_in.to_vec();
    let max_norm = rem.iter().fold(0.0f64, |acc, v| acc.max(vec_norm(v)));
    if max_norm == 0.0 {
        return Vec::new();
    }
    let tol = 1e-14 * max_norm;
    let mut used = vec![false; m];
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m);
    for _ in 0..m {
        let mut best: Option<(usize, f64)> = None;
        for (j, r) in rem.iter().enumerate() {
            if used[j] {
                continue;
            }
            let n = vec_norm(r);
            if best.is_none_or(|(_, bn)| n > bn) {
                best = Some((j, n));
            }
        }
        let Some((piv, piv_norm)) = best else {
            break;
        };
        if piv_norm <= tol {
            break;
        }
        used[piv] = true;
        let mut q = std::mem::take(&mut rem[piv]);
        orthogonalize(&mut q, &basis);
        if normalize_in_place(&mut q) <= tol {
            continue;
        }
        for (j, r) in rem.iter_mut().enumerate() {
            if !used[j] {
                let dot: f64 = r.iter().zip(&q).map(|(&x, &y)| x * y).sum();
                for (x, &y) in r.iter_mut().zip(&q) {
                    *x -= dot * y;
                }
            }
        }
        basis.push(q);
    }
    basis
}

/// Matrix-free top-`k` singular triplets of an operator given by `matvec(v, u)` (`u = A v`) and
/// `rmatvec(u, v)` (`v = A^T u`).
///
/// Uses simultaneous subspace iteration on a block of width `p = min(min(rows, cols), max(k + 6, 2k))`
/// followed by column-pivoted CGS2 QR and exact one-sided cyclic Jacobi SVD on the projected `q x p`
/// factor, avoiding singular-value squaring and converging cleanly on close or repeated singular values.
/// Canonical sign rule: the largest-magnitude entry of `u` (first index on ties) is made non-negative.
fn svd_operator(
    rows: usize,
    cols: usize,
    k: usize,
    iters: usize,
    matvec: impl Fn(&[f64], &mut [f64]),
    rmatvec: impl Fn(&[f64], &mut [f64]),
) -> Vec<SingularTriplet> {
    let max_rank = rows.min(cols);
    let limit = k.min(max_rank);
    if limit == 0 {
        return Vec::new();
    }
    let p = (limit + 6).max(limit * 2).min(max_rank);
    let steps = iters.max(8);

    let mut v_basis: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut seed_cursor = 0usize;
    while v_basis.len() < p {
        let Some(v) = seed_unit_vector(cols, seed_cursor, &v_basis) else {
            break;
        };
        seed_cursor += 1;
        v_basis.push(v);
    }
    if v_basis.is_empty() {
        return Vec::new();
    }

    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(p);
    let mut w_cols: Vec<Vec<f64>> = Vec::with_capacity(p);
    for _ in 0..steps {
        u_cols.resize_with(v_basis.len(), || vec![0.0; rows]);
        u_cols.truncate(v_basis.len());
        for (v, u) in v_basis.iter().zip(&mut u_cols) {
            matvec(v, u);
        }
        let u_basis = orthonormal_span(&u_cols);
        if u_basis.is_empty() {
            if seed_cursor >= cols + p {
                break;
            }
            v_basis.clear();
            while v_basis.len() < p {
                let Some(v) = seed_unit_vector(cols, seed_cursor, &v_basis) else {
                    break;
                };
                seed_cursor += 1;
                v_basis.push(v);
            }
            continue;
        }
        w_cols.resize_with(u_basis.len(), || vec![0.0; cols]);
        w_cols.truncate(u_basis.len());
        for (u, w) in u_basis.iter().zip(&mut w_cols) {
            rmatvec(u, w);
        }
        let mut next_v = orthonormal_span(&w_cols);
        while next_v.len() < p {
            let Some(v) = seed_unit_vector(cols, seed_cursor, &next_v) else {
                break;
            };
            seed_cursor += 1;
            next_v.push(v);
        }
        v_basis = next_v;
    }

    let m = v_basis.len();
    let mut y_cols = vec![vec![0.0f64; rows]; m];
    for (v, y) in v_basis.iter().zip(&mut y_cols) {
        matvec(v, y);
    }
    let q_basis = orthonormal_span(&y_cols);
    let q = q_basis.len();
    if q == 0 {
        return (0..limit)
            .map(|_| SingularTriplet {
                sigma: 0.0,
                u: vec![0.0; rows],
                v: vec![0.0; cols],
            })
            .collect();
    }

    // Projected q x m matrix B = Q_U^T Y stored by column, plus orthogonal Z = I_m.
    let mut w_small = vec![vec![0.0f64; q]; m];
    for j in 0..m {
        for i in 0..q {
            w_small[j][i] = q_basis[i]
                .iter()
                .zip(&y_cols[j])
                .map(|(&qi, &yi)| qi * yi)
                .sum();
        }
    }
    let mut z_cols = vec![vec![0.0f64; m]; m];
    for (j, z) in z_cols.iter_mut().enumerate() {
        z[j] = 1.0;
    }

    // One-sided cyclic Jacobi orthogonalizes the columns of W = B Z without squaring singular values.
    for _sweep in 0..32 {
        let mut rotated = false;
        for a in 0..m {
            for b in (a + 1)..m {
                let scale = w_small[a]
                    .iter()
                    .chain(&w_small[b])
                    .fold(0.0f64, |acc, &x| acc.max(x.abs()));
                if scale == 0.0 {
                    continue;
                }
                let inv = 1.0 / scale;
                let mut aa = 0.0f64;
                let mut bb = 0.0f64;
                let mut ab = 0.0f64;
                for i in 0..q {
                    let wa = w_small[a][i] * inv;
                    let wb = w_small[b][i] * inv;
                    aa += wa * wa;
                    bb += wb * wb;
                    ab += wa * wb;
                }
                if ab.abs() <= 1e-15 * (aa * bb).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (bb - aa) / (2.0 * ab);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..q {
                    let wa = w_small[a][i];
                    let wb = w_small[b][i];
                    w_small[a][i] = c * wa - s * wb;
                    w_small[b][i] = s * wa + c * wb;
                }
                for l in 0..m {
                    let za = z_cols[a][l];
                    let zb = z_cols[b][l];
                    z_cols[a][l] = c * za - s * zb;
                    z_cols[b][l] = s * za + c * zb;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    let sigmas: Vec<f64> = w_small.iter().map(|col| vec_norm(col)).collect();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| sigmas[b].total_cmp(&sigmas[a]).then(a.cmp(&b)));

    let mut out = Vec::with_capacity(limit);
    for (rank, &idx) in order.iter().take(limit).enumerate() {
        let sigma = sigmas[idx];
        if rank >= q || sigma == 0.0 {
            out.push(SingularTriplet {
                sigma: 0.0,
                u: vec![0.0; rows],
                v: vec![0.0; cols],
            });
            continue;
        }
        let mut v = vec![0.0f64; cols];
        for (l, &zl) in z_cols[idx].iter().enumerate() {
            if zl == 0.0 {
                continue;
            }
            for (vc, &vb) in v.iter_mut().zip(&v_basis[l]) {
                *vc += zl * vb;
            }
        }
        normalize_in_place(&mut v);

        let inv_sigma = 1.0 / sigma;
        let mut u = vec![0.0f64; rows];
        for (i, &wi) in w_small[idx].iter().enumerate() {
            let coeff = wi * inv_sigma;
            if coeff == 0.0 {
                continue;
            }
            for (ur, &qr) in u.iter_mut().zip(&q_basis[i]) {
                *ur += coeff * qr;
            }
        }
        normalize_in_place(&mut u);

        if let Some((_, &val)) = u
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.abs().total_cmp(&b.abs()).then(ib.cmp(ia)))
        {
            if val < 0.0 {
                for x in &mut u {
                    *x = -*x;
                }
                for x in &mut v {
                    *x = -*x;
                }
            }
        }
        out.push(SingularTriplet { sigma, u, v });
    }
    out
}

/// Deterministic top-`k` singular triplets of a 2-D `[rows, cols]` weight matrix.
pub fn top_singular_triplets(
    w: &Weight,
    k: usize,
    iters: usize,
) -> Result<Vec<SingularTriplet>, String> {
    let (rows, cols) = require_2d(w)?;
    if k == 0 {
        return Ok(Vec::new());
    }
    let data = &w.data;
    Ok(svd_operator(
        rows,
        cols,
        k,
        iters,
        |v, u| {
            for r in 0..rows {
                let row = &data[r * cols..(r + 1) * cols];
                let mut acc = 0.0f64;
                for c in 0..cols {
                    acc += f64::from(row[c]) * v[c];
                }
                u[r] = acc;
            }
        },
        |u, v| {
            v.fill(0.0);
            for r in 0..rows {
                let ur = u[r];
                if ur == 0.0 {
                    continue;
                }
                let row = &data[r * cols..(r + 1) * cols];
                for c in 0..cols {
                    v[c] += ur * f64::from(row[c]);
                }
            }
        },
    ))
}

/// Summary of `current - base` for one modified tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorDiff {
    pub name: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub values_changed: usize,
    pub base_l2: f64,
    pub delta_l2: f64,
    pub rel_l2: f64,
    pub max_abs_delta: f64,
    pub top_singular_values: Vec<f64>,
    /// `(row_index, row_delta_l2)` sorted by descending `row_delta_l2` (2-D weights only).
    pub top_changed_rows: Vec<(usize, f64)>,
}

/// Compare `base` and `current`; returns `Ok(None)` when every element is bit-identical (`to_bits`).
pub fn diff_weights(
    base: &Weight,
    current: &Weight,
    svd_k: usize,
    top_rows: usize,
) -> Result<Option<TensorDiff>, String> {
    if base.shape != current.shape {
        return Err(format!(
            "{}: base shape {:?} != current shape {:?}",
            base.name, base.shape, current.shape
        ));
    }
    let numel = base.numel();
    if base.data.len() != numel || current.data.len() != numel {
        return Err(format!("{}: data length disagrees with shape", base.name));
    }
    let mut values_changed = 0usize;
    let mut base_sq = 0.0f64;
    let mut delta_sq = 0.0f64;
    let mut max_abs_delta = 0.0f64;
    for (&b, &c) in base.data.iter().zip(&current.data) {
        if !b.is_finite() || !c.is_finite() {
            return Err(format!("{}: non-finite value in diff", base.name));
        }
        if b.to_bits() != c.to_bits() {
            values_changed += 1;
        }
        let bf = f64::from(b);
        let df = f64::from(c) - bf;
        base_sq += bf * bf;
        delta_sq += df * df;
        let a = df.abs();
        if a > max_abs_delta {
            max_abs_delta = a;
        }
    }
    if values_changed == 0 {
        return Ok(None);
    }
    let base_l2 = base_sq.sqrt();
    let delta_l2 = delta_sq.sqrt();
    let rel_l2 = if base_l2 == 0.0 {
        f64::INFINITY
    } else {
        delta_l2 / base_l2
    };
    let mut top_singular_values = Vec::new();
    let mut top_changed_rows = Vec::new();
    if let &[rows, cols] = base.shape.as_slice() {
        let need_row_scan = top_rows > 0 || (svd_k > 0 && rows > 0 && cols > 0);
        let mut nonzero_rows = Vec::new();
        if need_row_scan {
            let mut r_norms: Vec<(usize, f64)> = (0..rows)
                .filter_map(|r| {
                    let s0 = &base.data[r * cols..(r + 1) * cols];
                    let s1 = &current.data[r * cols..(r + 1) * cols];
                    let sq: f64 = s0
                        .iter()
                        .zip(s1)
                        .map(|(&b, &c)| {
                            let d = f64::from(c) - f64::from(b);
                            d * d
                        })
                        .sum();
                    (sq > 0.0).then_some((r, sq.sqrt()))
                })
                .collect();
            nonzero_rows = r_norms.iter().map(|&(r, _)| r).collect();
            if top_rows > 0 {
                r_norms.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                r_norms.truncate(top_rows);
                top_changed_rows = r_norms;
            }
        }
        if svd_k > 0 && rows > 0 && cols > 0 {
            let b_data = &base.data;
            let c_data = &current.data;
            let triplets = svd_operator(
                rows,
                cols,
                svd_k,
                DEFAULT_SVD_ITERS,
                |v, u| {
                    u.fill(0.0);
                    for &r in &nonzero_rows {
                        let br = &b_data[r * cols..(r + 1) * cols];
                        let cr = &c_data[r * cols..(r + 1) * cols];
                        let mut acc = 0.0f64;
                        for col in 0..cols {
                            acc += (f64::from(cr[col]) - f64::from(br[col])) * v[col];
                        }
                        u[r] = acc;
                    }
                },
                |u, v| {
                    v.fill(0.0);
                    for &r in &nonzero_rows {
                        let ur = u[r];
                        if ur == 0.0 {
                            continue;
                        }
                        let br = &b_data[r * cols..(r + 1) * cols];
                        let cr = &c_data[r * cols..(r + 1) * cols];
                        for col in 0..cols {
                            v[col] += ur * (f64::from(cr[col]) - f64::from(br[col]));
                        }
                    }
                },
            );
            top_singular_values = triplets.into_iter().map(|t| t.sigma).collect();
        }
    }
    Ok(Some(TensorDiff {
        name: base.name.clone(),
        shape: base.shape.clone(),
        numel,
        values_changed,
        base_l2,
        delta_l2,
        rel_l2,
        max_abs_delta,
        top_singular_values,
        top_changed_rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_and_row_col_norms_match_hand_values() {
        let w = Weight {
            name: "m".into(),
            shape: vec![2, 3],
            data: vec![3.0, 0.0, 4.0, -6.0, 8.0, 0.0],
        };
        let s = TensorStats::compute(&w).unwrap();
        assert_eq!(s.numel, 6);
        assert_eq!(s.zero_count, 2);
        assert_eq!(s.min, -6.0);
        assert_eq!(s.max, 8.0);
        assert_eq!(s.max_abs, 8.0);
        // 9 + 16 + 36 + 64 = 125
        assert!((s.l2_norm - 125.0f64.sqrt()).abs() <= 1e-12);
        let rs = row_stats(&w).unwrap();
        assert!((rs[0].l2_norm - 5.0).abs() <= 1e-12);
        assert!((rs[1].l2_norm - 10.0).abs() <= 1e-12);
        let cs = col_stats(&w).unwrap();
        assert!((cs[0].l2_norm - 45.0f64.sqrt()).abs() <= 1e-12);
        assert!((cs[1].l2_norm - 8.0).abs() <= 1e-12);
        assert!((cs[2].l2_norm - 4.0).abs() <= 1e-12);
    }

    #[test]
    fn top_singular_triplets_recover_known_orthogonal_spectrum() {
        // A = 7 * e0 * e1^T + 3 * e2 * e0^T on 3x4
        let mut data = vec![0.0f32; 12];
        data[1] = 7.0; // (0, 1)
        data[2 * 4] = 3.0; // (2, 0)
        let w = Weight {
            name: "a".into(),
            shape: vec![3, 4],
            data,
        };
        let t = top_singular_triplets(&w, 3, 32).unwrap();
        assert_eq!(t.len(), 3);
        assert!((t[0].sigma - 7.0).abs() <= 1e-10, "{:?}", t[0]);
        assert!((t[1].sigma - 3.0).abs() <= 1e-10, "{:?}", t[1]);
        assert!(t[2].sigma <= 1e-10, "{:?}", t[2]);
        assert!((t[0].u[0] - 1.0).abs() <= 1e-10);
        assert!((t[0].v[1] - 1.0).abs() <= 1e-10);
        assert!((t[1].u[2] - 1.0).abs() <= 1e-10);
        assert!((t[1].v[0] - 1.0).abs() <= 1e-10);
    }

    #[test]
    fn cosine_and_nearest_rows_order_and_refuse_bad_inputs() {
        let w = Weight {
            name: "emb".into(),
            shape: vec![4, 2],
            data: vec![1.0, 0.0, 0.0, 2.0, -1.0, 0.0, 3.0, 0.0],
        };
        let nr = nearest_rows(&w, &[2.0, 0.0], 4).unwrap();
        assert_eq!(nr[0].row, 0);
        assert!((nr[0].cosine - 1.0).abs() <= 1e-12);
        // Row 3 ties row 0 at cosine 1.0; ascending row index breaks ties.
        assert_eq!(nr[1].row, 3);
        assert!((nr[2].cosine - 0.0).abs() <= 1e-12);
        assert!((nr[3].cosine - (-1.0)).abs() <= 1e-12);
        assert!(nearest_rows(&w, &[0.0, 0.0], 2).is_err());
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).unwrap(), 0.0);
    }

    #[test]
    fn diff_weights_reports_none_for_identical_and_rank_spectrum_for_edits() {
        let base = Weight {
            name: "w".into(),
            shape: vec![3, 3],
            data: vec![1.0; 9],
        };
        assert!(diff_weights(&base, &base, 2, 2).unwrap().is_none());
        let mut cur = base.clone();
        // Add rank-1 outer product [2, 0, 0]^T * [1, 1, 1] -> sigma_0 = 2 * sqrt(3).
        cur.data[0] += 2.0;
        cur.data[1] += 2.0;
        cur.data[2] += 2.0;
        let d = diff_weights(&base, &cur, 2, 2).unwrap().unwrap();
        assert_eq!(d.values_changed, 3);
        assert!((d.delta_l2 - (12.0f64).sqrt()).abs() <= 1e-10);
        assert!((d.top_singular_values[0] - (12.0f64).sqrt()).abs() <= 1e-10);
        assert!(d.top_singular_values[1] <= 1e-10);
        assert_eq!(d.top_changed_rows.len(), 1);
        assert_eq!(d.top_changed_rows[0].0, 0);
    }

    fn householder_apply_mat(n: usize, v: &[f64]) -> Vec<Vec<f64>> {
        let nrm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let u: Vec<f64> = v.iter().map(|&x| x / nrm).collect();
        (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| (if r == c { 1.0 } else { 0.0 }) - 2.0 * u[r] * u[c])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn svd_converges_monotonically_on_close_and_degenerate_spectra() {
        // 1. Close singular values matching the failure mode seen on SAM 3.1 LoRA deltas:
        //    sigma = [0.6839, 0.3786, 0.3690, 0.3595, 0.25, 0.15, 0.05, 0.01].
        let sigmas = [0.6839, 0.3786, 0.3690, 0.3595, 0.25, 0.15, 0.05, 0.01];
        let u_mat = householder_apply_mat(8, &[1.0, -2.0, 0.5, 3.0, -1.5, 0.7, 2.2, -0.9]);
        let v_mat = householder_apply_mat(8, &[-0.8, 1.7, 2.3, -1.1, 0.4, -2.5, 1.3, 0.6]);
        let mut data = vec![0.0f32; 64];
        for r in 0..8 {
            for c in 0..8 {
                let mut acc = 0.0f64;
                for i in 0..8 {
                    acc += u_mat[r][i] * sigmas[i] * v_mat[c][i];
                }
                data[r * 8 + c] = acc as f32;
            }
        }
        let w = Weight {
            name: "close_sv".into(),
            shape: vec![8, 8],
            data,
        };
        let t = top_singular_triplets(&w, 4, DEFAULT_SVD_ITERS).unwrap();
        assert_eq!(t.len(), 4);
        for i in 0..3 {
            assert!(
                t[i].sigma >= t[i + 1].sigma,
                "non-monotonic singular values: {:?}",
                t.iter().map(|x| x.sigma).collect::<Vec<_>>()
            );
        }
        for i in 0..4 {
            assert!(
                (t[i].sigma - sigmas[i]).abs() <= 1e-6,
                "mode {i}: got {} vs expected {}",
                t[i].sigma,
                sigmas[i]
            );
        }

        // 2. Exact multiplicity-2 degenerate singular values on rectangular 6x7:
        //    sigma = [5.0, 5.0, 2.0, 2.0, 0.5, 0.0].
        let deg = [5.0, 5.0, 2.0, 2.0, 0.5, 0.0];
        let u6 = householder_apply_mat(6, &[1.0, 2.0, -1.0, 0.5, -1.5, 0.8]);
        let v7 = householder_apply_mat(7, &[-1.2, 0.7, 1.9, -0.4, 1.1, -2.0, 0.3]);
        let mut deg_data = vec![0.0f32; 42];
        for r in 0..6 {
            for c in 0..7 {
                let mut acc = 0.0f64;
                for i in 0..6 {
                    acc += u6[r][i] * deg[i] * v7[c][i];
                }
                deg_data[r * 7 + c] = acc as f32;
            }
        }
        let w_deg = Weight {
            name: "deg_sv".into(),
            shape: vec![6, 7],
            data: deg_data,
        };
        let td = top_singular_triplets(&w_deg, 5, DEFAULT_SVD_ITERS).unwrap();
        assert_eq!(td.len(), 5);
        for i in 0..5 {
            assert!(
                (td[i].sigma - deg[i]).abs() <= 1e-5,
                "deg mode {i}: got {} vs expected {}",
                td[i].sigma,
                deg[i]
            );
            // Verify A v_i == sigma_i u_i and unit norms.
            for r in 0..6 {
                let av_r: f64 = (0..7)
                    .map(|c| f64::from(w_deg.data[r * 7 + c]) * td[i].v[c])
                    .sum();
                assert!(
                    (av_r - td[i].sigma * td[i].u[r]).abs() <= 1e-5,
                    "residual mode {i} row {r}: {av_r} vs {}",
                    td[i].sigma * td[i].u[r]
                );
            }
            for j in (i + 1)..5 {
                let u_dot: f64 = td[i].u.iter().zip(&td[j].u).map(|(a, b)| a * b).sum();
                let v_dot: f64 = td[i].v.iter().zip(&td[j].v).map(|(a, b)| a * b).sum();
                assert!(u_dot.abs() <= 1e-10, "u_{i} . u_{j} = {u_dot}");
                assert!(v_dot.abs() <= 1e-10, "v_{i} . v_{j} = {v_dot}");
            }
        }

        // 3. Scaled Euclidean norm avoids overflow/underflow at f64 extremes.
        assert!((vec_norm(&[3e200, 4e200]) - 5e200).abs() <= 1e186);
        assert!((vec_norm(&[3e-200, 4e-200]) - 5e-200).abs() <= 1e-214);
    }
}
