//! Rectangular linear assignment (minimum cost), the Hungarian method with
//! potentials, O(n^2 m). Same problem as `scipy.optimize.linear_sum_assignment`,
//! which TrackEval's HOTA uses; ties may be broken differently.

/// Minimum-cost assignment for a `rows x cols` cost matrix (row-major).
/// Returns `(row, col)` pairs, `min(rows, cols)` of them, sorted by row.
/// Costs must be finite.
pub fn linear_sum_assignment(cost: &[f64], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    assert_eq!(
        cost.len(),
        rows * cols,
        "cost matrix is {} long, not {rows} x {cols}",
        cost.len()
    );
    assert!(cost.iter().all(|c| c.is_finite()), "non-finite cost");
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    // The solver below needs n <= m; transpose otherwise.
    let transposed = rows > cols;
    let (n, m) = if transposed {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let a = |i: usize, j: usize| {
        if transposed {
            cost[j * cols + i]
        } else {
            cost[i * cols + j]
        }
    };

    // 1-based arrays as in the classic formulation; index 0 is the virtual column.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1]; // p[j] = row matched to column j
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let cur = a(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out: Vec<(usize, usize)> = (1..=m)
        .filter(|&j| p[j] != 0)
        .map(|j| {
            if transposed {
                (j - 1, p[j] - 1)
            } else {
                (p[j] - 1, j - 1)
            }
        })
        .collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(cost: &[f64], cols: usize, pairs: &[(usize, usize)]) -> f64 {
        pairs.iter().map(|&(r, c)| cost[r * cols + c]).sum()
    }

    /// Exhaustive minimum over all injections of the smaller side, for small sizes.
    fn brute(cost: &[f64], rows: usize, cols: usize) -> f64 {
        fn go(
            cost: &[f64],
            rows: usize,
            cols: usize,
            r: usize,
            used: &mut Vec<bool>,
            acc: f64,
            best: &mut f64,
        ) {
            if r == rows {
                *best = best.min(acc);
                return;
            }
            for c in 0..cols {
                if !used[c] {
                    used[c] = true;
                    go(
                        cost,
                        rows,
                        cols,
                        r + 1,
                        used,
                        acc + cost[r * cols + c],
                        best,
                    );
                    used[c] = false;
                }
            }
        }
        if rows <= cols {
            let mut best = f64::INFINITY;
            go(cost, rows, cols, 0, &mut vec![false; cols], 0.0, &mut best);
            best
        } else {
            let t: Vec<f64> = (0..cols)
                .flat_map(|c| (0..rows).map(move |r| (r, c)))
                .map(|(r, c)| cost[r * cols + c])
                .collect();
            brute(&t, cols, rows)
        }
    }

    #[test]
    fn matches_brute_force_on_random_rectangles() {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for rows in 0..6 {
            for cols in 0..6 {
                for _ in 0..20 {
                    // Coarse values make ties common.
                    let cost: Vec<f64> = (0..rows * cols)
                        .map(|_| (rnd() * 4.0).floor() - 2.0)
                        .collect();
                    let pairs = linear_sum_assignment(&cost, rows, cols);
                    assert_eq!(pairs.len(), rows.min(cols));
                    let mut rs: Vec<_> = pairs.iter().map(|p| p.0).collect();
                    let mut cs: Vec<_> = pairs.iter().map(|p| p.1).collect();
                    rs.dedup();
                    cs.sort_unstable();
                    cs.dedup();
                    assert_eq!(rs.len(), pairs.len(), "row used twice");
                    assert_eq!(cs.len(), pairs.len(), "column used twice");
                    if rows.min(cols) > 0 {
                        let got = total(&cost, cols, &pairs);
                        let want = brute(&cost, rows, cols);
                        assert!(
                            (got - want).abs() < 1e-9,
                            "{rows}x{cols}: {got} != {want} for {cost:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn refuses_nan() {
        linear_sum_assignment(&[0.0, f64::NAN], 1, 2);
    }
}
