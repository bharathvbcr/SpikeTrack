//! Declarative, gradient-free weight-space editing (`spiketrack-weight-edit-v1`).
//!
//! Reads a pure-JSON manifest from disk (`--weight-edit FILE`, no Python) and applies surgical
//! parameter transformations across any tensor in [`Sam31Weights`] — including typed
//! [`crate::VisionBackbone`] and [`crate::TextEncoder`] weights (`token_embedding.weight`,
//! `resizer`, attention projections) as well as untyped detector and tracker groups.
//!
//! # Operations
//!
//! * `scale_rows`: scale rows `[start, end)` of a 2-D `[rows, cols]` weight (`scale = 0.0` ablates).
//! * `scale_cols`: scale columns `[start, end)` of a 2-D `[rows, cols]` weight (`scale = 0.0` zeroes
//!   writer columns, e.g. MLP `down_proj` / `c_proj` or `out_proj` columns).
//! * `scale_1d`: scale `[start, end)` of a 1-D `[len]` parameter (biases, LayerNorm scales).
//! * `add_1d`: add an explicit slice `values` to `[start, end)` of a 1-D `[len]` parameter.
//! * `graft_row`: set row `dst` of a 2-D `[rows, cols]` weight to
//!   `dst_scale * W[dst] + sum_k weight_k * W[row_k]`, optionally rescaling to preserve the
//!   pre-edit L2 norm of `W[dst]` (`"preserve_norm": true`).
//! * `project_out_direction`: remove strength `alpha` along unit direction `u = d / ||d||_2`:
//!   for `"axis": "cols"` (`d.len() == rows`), `W[:, j] -= alpha * (u^T W[:, j]) * u` on `j` in
//!   `range`; for `"axis": "rows"` (`d.len() == cols`), `W[i, :] -= alpha * (W[i, :] . u) * u` on
//!   `i` in `range`.
//! * `rank1_update`: add `scale * u * v^T` onto `W[r0..r1, c0..c1]` of a 2-D `[rows, cols]` weight.
//!
//! # Numerics and Atomicity
//!
//! Every touched tensor is promoted into an `f64` working buffer, all edits targeting that tensor
//! execute in `f64` in manifest order, and the result is rounded once to `f32` and checked for
//! finiteness before any tensor in [`Sam31Weights`] is mutated. Untouched elements keep their
//! exact `f32` bit pattern (`f32 -> f64 -> f32` is lossless in IEEE 754, and unvisited indices
//! are never written). Any validation or overflow error leaves [`Sam31Weights`] untouched.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use ojas_io::{parse_json, IoError, JsonValue};

use crate::names::Sam31Weights;

/// Manifest format identifier required under `"format"`.
pub const WEIGHT_EDIT_FORMAT: &str = "spiketrack-weight-edit-v1";

#[derive(Debug)]
pub enum EditError {
    Io(String),
    Format(String),
    UnknownTarget { param: String },
    Shape { param: String, detail: String },
    NonFinite { param: String, detail: String },
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "weight edit I/O: {s}"),
            Self::Format(s) => write!(f, "weight edit manifest: {s}"),
            Self::UnknownTarget { param } => {
                write!(f, "weight edit target {param:?} is not a checkpoint tensor")
            }
            Self::Shape { param, detail } => write!(f, "weight edit on {param}: {detail}"),
            Self::NonFinite { param, detail } => {
                write!(
                    f,
                    "weight edit on {param} produced non-finite output: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for EditError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Rows,
    Cols,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RowSource {
    pub row: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EditOp {
    ScaleRows {
        param: String,
        row_start: usize,
        row_end: usize,
        scale: f64,
    },
    ScaleCols {
        param: String,
        col_start: usize,
        col_end: usize,
        scale: f64,
    },
    Scale1d {
        param: String,
        start: usize,
        end: usize,
        scale: f64,
    },
    Add1d {
        param: String,
        start: usize,
        end: usize,
        values: Vec<f64>,
    },
    GraftRow {
        param: String,
        dst: usize,
        dst_scale: f64,
        sources: Vec<RowSource>,
        preserve_norm: bool,
    },
    ProjectOutDirection {
        param: String,
        axis: Axis,
        range: Option<(usize, usize)>,
        /// Unit-norm `f64` direction `d / ||d||_2`, normalized at parse time.
        unit_direction: Vec<f64>,
        strength: f64,
    },
    Rank1Update {
        param: String,
        rows: Option<(usize, usize)>,
        cols: Option<(usize, usize)>,
        scale: f64,
        u: Vec<f64>,
        v: Vec<f64>,
    },
}

impl EditOp {
    pub fn param(&self) -> &str {
        match self {
            Self::ScaleRows { param, .. }
            | Self::ScaleCols { param, .. }
            | Self::Scale1d { param, .. }
            | Self::Add1d { param, .. }
            | Self::GraftRow { param, .. }
            | Self::ProjectOutDirection { param, .. }
            | Self::Rank1Update { param, .. } => param,
        }
    }
}

/// Validated sequence of surgical weight-space edits.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightEditPlan {
    pub description: Option<String>,
    pub edits: Vec<EditOp>,
}

/// Summary of an applied [`WeightEditPlan`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditReport {
    pub description: Option<String>,
    pub edits: usize,
    pub tensors: BTreeSet<String>,
    pub values_changed: usize,
    pub total_values: usize,
}

impl EditReport {
    pub fn fraction_changed(&self) -> f64 {
        if self.total_values == 0 {
            0.0
        } else {
            self.values_changed as f64 / self.total_values as f64
        }
    }
}

fn fmt_err<T>(detail: impl Into<String>) -> Result<T, EditError> {
    Err(EditError::Format(detail.into()))
}

fn as_fmt(e: IoError) -> EditError {
    EditError::Format(e.detail().to_string())
}

fn check_allowed_keys(
    obj: &[(String, JsonValue)],
    allowed: &[&str],
    where_desc: &str,
) -> Result<(), EditError> {
    let mut seen = BTreeSet::new();
    for (k, _) in obj {
        if !seen.insert(k.as_str()) {
            return fmt_err(format!("{where_desc}: duplicate key {k:?}"));
        }
        if !allowed.contains(&k.as_str()) {
            return fmt_err(format!(
                "{where_desc}: unknown key {k:?} (allowed: {})",
                allowed.join(", ")
            ));
        }
    }
    Ok(())
}

fn json_usize(v: &JsonValue, what: &str) -> Result<usize, EditError> {
    let n = v
        .as_u64()
        .map_err(|e| EditError::Format(format!("{what}: {}", e.detail())))?;
    usize::try_from(n).map_err(|_| EditError::Format(format!("{what}: {n} overflows usize")))
}

fn json_finite_f64(v: &JsonValue, what: &str) -> Result<f64, EditError> {
    let x = v
        .as_f64()
        .map_err(|e| EditError::Format(format!("{what}: {}", e.detail())))?;
    if !x.is_finite() {
        return fmt_err(format!("{what}: {x} is not finite"));
    }
    Ok(x)
}

fn parse_range(v: &JsonValue, what: &str) -> Result<(usize, usize), EditError> {
    let arr = v
        .as_array()
        .map_err(|e| EditError::Format(format!("{what}: {}", e.detail())))?;
    let [a, b] = arr else {
        return fmt_err(format!("{what}: expected [start, end]"));
    };
    let start = json_usize(a, &format!("{what}[0]"))?;
    let end = json_usize(b, &format!("{what}[1]"))?;
    if start >= end {
        return fmt_err(format!("{what}: empty or reversed range [{start}, {end}]"));
    }
    Ok((start, end))
}

fn parse_f64_vec(v: &JsonValue, what: &str) -> Result<Vec<f64>, EditError> {
    let arr = v
        .as_array()
        .map_err(|e| EditError::Format(format!("{what}: {}", e.detail())))?;
    if arr.is_empty() {
        return fmt_err(format!("{what}: vector must not be empty"));
    }
    arr.iter()
        .enumerate()
        .map(|(i, item)| json_finite_f64(item, &format!("{what}[{i}]")))
        .collect()
}

fn parse_op(idx: usize, v: &JsonValue) -> Result<EditOp, EditError> {
    let ctx = format!("edits[{idx}]");
    let obj = v
        .as_object()
        .map_err(|e| EditError::Format(format!("{ctx}: {}", e.detail())))?;
    let op = v
        .get_str("op")
        .map_err(|e| EditError::Format(format!("{ctx}.op: {}", e.detail())))?;
    let param = v
        .get_str("param")
        .map_err(|e| EditError::Format(format!("{ctx}.param: {}", e.detail())))?
        .to_string();
    if param.is_empty() {
        return fmt_err(format!("{ctx}.param: empty tensor name"));
    }
    match op {
        "scale_rows" => {
            check_allowed_keys(obj, &["op", "param", "rows", "scale"], &ctx)?;
            let (row_start, row_end) =
                parse_range(v.field("rows").map_err(as_fmt)?, &format!("{ctx}.rows"))?;
            let scale = json_finite_f64(v.field("scale").map_err(as_fmt)?, &format!("{ctx}.scale"))?;
            Ok(EditOp::ScaleRows {
                param,
                row_start,
                row_end,
                scale,
            })
        }
        "scale_cols" => {
            check_allowed_keys(obj, &["op", "param", "cols", "scale"], &ctx)?;
            let (col_start, col_end) =
                parse_range(v.field("cols").map_err(as_fmt)?, &format!("{ctx}.cols"))?;
            let scale = json_finite_f64(v.field("scale").map_err(as_fmt)?, &format!("{ctx}.scale"))?;
            Ok(EditOp::ScaleCols {
                param,
                col_start,
                col_end,
                scale,
            })
        }
        "scale_1d" => {
            check_allowed_keys(obj, &["op", "param", "range", "scale"], &ctx)?;
            let (start, end) =
                parse_range(v.field("range").map_err(as_fmt)?, &format!("{ctx}.range"))?;
            let scale = json_finite_f64(v.field("scale").map_err(as_fmt)?, &format!("{ctx}.scale"))?;
            Ok(EditOp::Scale1d {
                param,
                start,
                end,
                scale,
            })
        }
        "add_1d" => {
            check_allowed_keys(obj, &["op", "param", "range", "values"], &ctx)?;
            let (start, end) =
                parse_range(v.field("range").map_err(as_fmt)?, &format!("{ctx}.range"))?;
            let values =
                parse_f64_vec(v.field("values").map_err(as_fmt)?, &format!("{ctx}.values"))?;
            if values.len() != end - start {
                return fmt_err(format!(
                    "{ctx}.values: length {} disagrees with range [{start}, {end}) ({})",
                    values.len(),
                    end - start
                ));
            }
            Ok(EditOp::Add1d {
                param,
                start,
                end,
                values,
            })
        }
        "graft_row" => {
            check_allowed_keys(
                obj,
                &[
                    "op",
                    "param",
                    "dst",
                    "dst_scale",
                    "sources",
                    "preserve_norm",
                ],
                &ctx,
            )?;
            let dst = json_usize(v.field("dst").map_err(as_fmt)?, &format!("{ctx}.dst"))?;
            let dst_scale = match v.field_opt("dst_scale").map_err(as_fmt)? {
                Some(s) => json_finite_f64(s, &format!("{ctx}.dst_scale"))?,
                None => 0.0,
            };
            let preserve_norm = match v.field_opt("preserve_norm").map_err(as_fmt)? {
                Some(b) => b
                    .as_bool()
                    .map_err(|e| EditError::Format(format!("{ctx}.preserve_norm: {}", e.detail())))?,
                None => false,
            };
            let src_arr = v
                .field("sources")
                .map_err(as_fmt)?
                .as_array()
                .map_err(|e| EditError::Format(format!("{ctx}.sources: {}", e.detail())))?;
            if src_arr.is_empty() {
                return fmt_err(format!("{ctx}.sources: must list at least one source row"));
            }
            let mut sources = Vec::with_capacity(src_arr.len());
            for (si, item) in src_arr.iter().enumerate() {
                let sctx = format!("{ctx}.sources[{si}]");
                let sobj = item
                    .as_object()
                    .map_err(|e| EditError::Format(format!("{sctx}: {}", e.detail())))?;
                check_allowed_keys(sobj, &["row", "weight"], &sctx)?;
                let row = json_usize(item.field("row").map_err(as_fmt)?, &format!("{sctx}.row"))?;
                let weight = json_finite_f64(
                    item.field("weight").map_err(as_fmt)?,
                    &format!("{sctx}.weight"),
                )?;
                sources.push(RowSource { row, weight });
            }
            Ok(EditOp::GraftRow {
                param,
                dst,
                dst_scale,
                sources,
                preserve_norm,
            })
        }
        "project_out_direction" => {
            check_allowed_keys(
                obj,
                &["op", "param", "axis", "range", "direction", "strength"],
                &ctx,
            )?;
            let axis = match v.get_str("axis").map_err(as_fmt)? {
                "rows" => Axis::Rows,
                "cols" => Axis::Cols,
                other => {
                    return fmt_err(format!(
                        "{ctx}.axis: {other:?} is invalid (expected \"rows\" or \"cols\")"
                    ))
                }
            };
            let range = v
                .field_opt("range")
                .map_err(as_fmt)?
                .map(|r| parse_range(r, &format!("{ctx}.range")))
                .transpose()?;
            let mut unit_direction = parse_f64_vec(
                v.field("direction").map_err(as_fmt)?,
                &format!("{ctx}.direction"),
            )?;
            let dir_max = unit_direction.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
            if !dir_max.is_finite() || dir_max == 0.0 {
                return fmt_err(format!(
                    "{ctx}.direction: norm {dir_max} must be positive and finite"
                ));
            }
            for x in &mut unit_direction {
                *x /= dir_max;
            }
            let unit_norm: f64 = unit_direction.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if !unit_norm.is_finite() || unit_norm == 0.0 {
                return fmt_err(format!(
                    "{ctx}.direction: norm {unit_norm} must be positive and finite"
                ));
            }
            for x in &mut unit_direction {
                *x /= unit_norm;
            }
            let strength = json_finite_f64(
                v.field("strength").map_err(as_fmt)?,
                &format!("{ctx}.strength"),
            )?;
            Ok(EditOp::ProjectOutDirection {
                param,
                axis,
                range,
                unit_direction,
                strength,
            })
        }
        "rank1_update" => {
            check_allowed_keys(
                obj,
                &["op", "param", "rows", "cols", "scale", "u", "v"],
                &ctx,
            )?;
            let rows = v
                .field_opt("rows")
                .map_err(as_fmt)?
                .map(|r| parse_range(r, &format!("{ctx}.rows")))
                .transpose()?;
            let cols = v
                .field_opt("cols")
                .map_err(as_fmt)?
                .map(|c| parse_range(c, &format!("{ctx}.cols")))
                .transpose()?;
            let scale =
                json_finite_f64(v.field("scale").map_err(as_fmt)?, &format!("{ctx}.scale"))?;
            let u = parse_f64_vec(v.field("u").map_err(as_fmt)?, &format!("{ctx}.u"))?;
            let v_vec = parse_f64_vec(v.field("v").map_err(as_fmt)?, &format!("{ctx}.v"))?;
            if let Some((r0, r1)) = rows {
                if u.len() != r1 - r0 {
                    return fmt_err(format!(
                        "{ctx}.u: length {} disagrees with rows [{r0}, {r1})",
                        u.len()
                    ));
                }
            }
            if let Some((c0, c1)) = cols {
                if v_vec.len() != c1 - c0 {
                    return fmt_err(format!(
                        "{ctx}.v: length {} disagrees with cols [{c0}, {c1})",
                        v_vec.len()
                    ));
                }
            }
            Ok(EditOp::Rank1Update {
                param,
                rows,
                cols,
                scale,
                u,
                v: v_vec,
            })
        }
        other => fmt_err(format!(
            "{ctx}.op: unknown operation {other:?} (supported: scale_rows, scale_cols, scale_1d, add_1d, graft_row, project_out_direction, rank1_update)"
        )),
    }
}

struct StagedTensor {
    shape: Vec<usize>,
    orig: Vec<f32>,
    work: Vec<f64>,
    touched: Vec<bool>,
}

impl StagedTensor {
    fn from_slice(shape: Vec<usize>, orig: Vec<f32>) -> Self {
        let work = orig.iter().map(|&x| f64::from(x)).collect();
        let touched = vec![false; orig.len()];
        Self {
            shape,
            orig,
            work,
            touched,
        }
    }

    fn require_1d(&self, param: &str) -> Result<usize, EditError> {
        let &[len] = self.shape.as_slice() else {
            return Err(EditError::Shape {
                param: param.to_string(),
                detail: format!("shape {:?} is not 1-D", self.shape),
            });
        };
        Ok(len)
    }

    fn require_2d(&self, param: &str) -> Result<(usize, usize), EditError> {
        let &[rows, cols] = self.shape.as_slice() else {
            return Err(EditError::Shape {
                param: param.to_string(),
                detail: format!("shape {:?} is not 2-D", self.shape),
            });
        };
        Ok((rows, cols))
    }

    fn apply_op(&mut self, op: &EditOp) -> Result<(), EditError> {
        let param = op.param();
        let shape_err = |detail: String| EditError::Shape {
            param: param.to_string(),
            detail,
        };
        match op {
            EditOp::ScaleRows {
                row_start,
                row_end,
                scale,
                ..
            } => {
                let (rows, cols) = self.require_2d(param)?;
                if *row_end > rows {
                    return Err(shape_err(format!(
                        "rows [{row_start}, {row_end}) exceeds {rows} rows"
                    )));
                }
                for r in *row_start..*row_end {
                    let base = r * cols;
                    for c in 0..cols {
                        self.work[base + c] *= *scale;
                        self.touched[base + c] = true;
                    }
                }
            }
            EditOp::ScaleCols {
                col_start,
                col_end,
                scale,
                ..
            } => {
                let (rows, cols) = self.require_2d(param)?;
                if *col_end > cols {
                    return Err(shape_err(format!(
                        "cols [{col_start}, {col_end}) exceeds {cols} cols"
                    )));
                }
                for r in 0..rows {
                    let base = r * cols;
                    for c in *col_start..*col_end {
                        self.work[base + c] *= *scale;
                        self.touched[base + c] = true;
                    }
                }
            }
            EditOp::Scale1d {
                start, end, scale, ..
            } => {
                let len = self.require_1d(param)?;
                if *end > len {
                    return Err(shape_err(format!(
                        "range [{start}, {end}) exceeds length {len}"
                    )));
                }
                for i in *start..*end {
                    self.work[i] *= *scale;
                    self.touched[i] = true;
                }
            }
            EditOp::Add1d {
                start, end, values, ..
            } => {
                let len = self.require_1d(param)?;
                if *end > len {
                    return Err(shape_err(format!(
                        "range [{start}, {end}) exceeds length {len}"
                    )));
                }
                for (offset, &dv) in values.iter().enumerate() {
                    let idx = *start + offset;
                    self.work[idx] += dv;
                    self.touched[idx] = true;
                }
            }
            EditOp::GraftRow {
                dst,
                dst_scale,
                sources,
                preserve_norm,
                ..
            } => {
                let (rows, cols) = self.require_2d(param)?;
                if *dst >= rows {
                    return Err(shape_err(format!("dst row {dst} >= {rows} rows")));
                }
                for s in sources {
                    if s.row >= rows {
                        return Err(shape_err(format!("source row {} >= {rows} rows", s.row)));
                    }
                }
                let dst_slice = &self.work[*dst * cols..(*dst + 1) * cols];
                if dst_slice.iter().any(|x| !x.is_finite()) {
                    return Err(EditError::NonFinite {
                        param: param.to_string(),
                        detail: format!("graft_row on row {dst} encountered non-finite dst value"),
                    });
                }
                let dst_max = dst_slice.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
                let orig_norm = if dst_max == 0.0 {
                    0.0
                } else {
                    let inv = 1.0 / dst_max;
                    dst_max
                        * dst_slice
                            .iter()
                            .map(|&x| {
                                let u = x * inv;
                                u * u
                            })
                            .sum::<f64>()
                            .sqrt()
                };
                if !orig_norm.is_finite() {
                    return Err(EditError::NonFinite {
                        param: param.to_string(),
                        detail: format!("graft_row on row {dst} overflowed pre-edit norm"),
                    });
                }
                let mut next: Vec<f64> = dst_slice.iter().map(|&x| *dst_scale * x).collect();
                for s in sources {
                    let src_slice = &self.work[s.row * cols..(s.row + 1) * cols];
                    for (acc, &x) in next.iter_mut().zip(src_slice) {
                        *acc += s.weight * x;
                    }
                }
                if next.iter().any(|x| !x.is_finite()) {
                    return Err(EditError::NonFinite {
                        param: param.to_string(),
                        detail: format!(
                            "graft_row on row {dst} produced a non-finite intermediate value"
                        ),
                    });
                }
                if *preserve_norm {
                    if orig_norm == 0.0 {
                        return Err(EditError::NonFinite {
                            param: param.to_string(),
                            detail: format!(
                                "graft_row on row {dst} has zero pre-edit norm with preserve_norm=true"
                            ),
                        });
                    }
                    let next_max = next.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
                    if next_max == 0.0 {
                        return Err(EditError::NonFinite {
                            param: param.to_string(),
                            detail: format!(
                                "graft_row on row {dst} cancelled to zero norm with preserve_norm=true"
                            ),
                        });
                    }
                    let inv_max = 1.0 / next_max;
                    for x in &mut next {
                        *x *= inv_max;
                    }
                    let unit_norm: f64 = next.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    let scale = orig_norm / unit_norm;
                    for x in &mut next {
                        *x *= scale;
                        if !x.is_finite() {
                            return Err(EditError::NonFinite {
                                param: param.to_string(),
                                detail: format!("graft_row on row {dst} overflowed norm rescaling"),
                            });
                        }
                    }
                }
                let base = *dst * cols;
                self.work[base..base + cols].copy_from_slice(&next);
                self.touched[base..base + cols].fill(true);
            }
            EditOp::ProjectOutDirection {
                axis,
                range,
                unit_direction,
                strength,
                ..
            } => {
                let (rows, cols) = self.require_2d(param)?;
                match axis {
                    Axis::Cols => {
                        if unit_direction.len() != rows {
                            return Err(shape_err(format!(
                                "direction length {} != {rows} rows for axis=\"cols\"",
                                unit_direction.len()
                            )));
                        }
                        let (c0, c1) = range.unwrap_or((0, cols));
                        if c1 > cols {
                            return Err(shape_err(format!(
                                "range [{c0}, {c1}) exceeds {cols} cols"
                            )));
                        }
                        for c in c0..c1 {
                            let mut dot = 0.0f64;
                            for r in 0..rows {
                                dot += unit_direction[r] * self.work[r * cols + c];
                            }
                            let coeff = *strength * dot;
                            for r in 0..rows {
                                let idx = r * cols + c;
                                self.work[idx] -= coeff * unit_direction[r];
                                self.touched[idx] = true;
                            }
                        }
                    }
                    Axis::Rows => {
                        if unit_direction.len() != cols {
                            return Err(shape_err(format!(
                                "direction length {} != {cols} cols for axis=\"rows\"",
                                unit_direction.len()
                            )));
                        }
                        let (r0, r1) = range.unwrap_or((0, rows));
                        if r1 > rows {
                            return Err(shape_err(format!(
                                "range [{r0}, {r1}) exceeds {rows} rows"
                            )));
                        }
                        for r in r0..r1 {
                            let base = r * cols;
                            let mut dot = 0.0f64;
                            for c in 0..cols {
                                dot += self.work[base + c] * unit_direction[c];
                            }
                            let coeff = *strength * dot;
                            for c in 0..cols {
                                self.work[base + c] -= coeff * unit_direction[c];
                                self.touched[base + c] = true;
                            }
                        }
                    }
                }
            }
            EditOp::Rank1Update {
                rows: r_opt,
                cols: c_opt,
                scale,
                u,
                v,
                ..
            } => {
                let (rows, cols) = self.require_2d(param)?;
                let (r0, r1) = r_opt.unwrap_or((0, rows));
                let (c0, c1) = c_opt.unwrap_or((0, cols));
                if r1 > rows || c1 > cols {
                    return Err(shape_err(format!(
                        "slice [{r0}..{r1}, {c0}..{c1}] exceeds [{rows}, {cols}]"
                    )));
                }
                if u.len() != r1 - r0 || v.len() != c1 - c0 {
                    return Err(shape_err(format!(
                        "u ({}) or v ({}) length disagrees with [{r0}..{r1}, {c0}..{c1}]",
                        u.len(),
                        v.len()
                    )));
                }
                for (i, &ui) in u.iter().enumerate() {
                    let base = (r0 + i) * cols;
                    let s_ui = *scale * ui;
                    for (j, &vj) in v.iter().enumerate() {
                        let idx = base + c0 + j;
                        self.work[idx] += s_ui * vj;
                        self.touched[idx] = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn finalize(&self, param: &str) -> Result<(Vec<f32>, usize), EditError> {
        let mut out = self.orig.clone();
        let mut changed = 0usize;
        for (i, &was_touched) in self.touched.iter().enumerate() {
            if !was_touched {
                continue;
            }
            let w64 = self.work[i];
            if !w64.is_finite() {
                return Err(EditError::NonFinite {
                    param: param.to_string(),
                    detail: format!("{w64} at flat index {i}"),
                });
            }
            let f = w64 as f32;
            if !f.is_finite() {
                return Err(EditError::NonFinite {
                    param: param.to_string(),
                    detail: format!("{w64} overflows f32 at flat index {i}"),
                });
            }
            if f.to_bits() != out[i].to_bits() {
                changed += 1;
            }
            out[i] = f;
        }
        Ok((out, changed))
    }
}

impl WeightEditPlan {
    pub fn open(path: &Path) -> Result<Self, EditError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| EditError::Io(format!("{}: {e}", path.display())))?;
        Self::parse(&text)
    }

    pub fn parse(json_text: &str) -> Result<Self, EditError> {
        let root = parse_json(json_text)
            .map_err(|e| EditError::Format(format!("not valid JSON: {}", e.detail())))?;
        let obj = root
            .as_object()
            .map_err(|e| EditError::Format(format!("root: {}", e.detail())))?;
        check_allowed_keys(obj, &["format", "description", "edits"], "root")?;
        let format = root.get_str("format").map_err(as_fmt)?;
        if format != WEIGHT_EDIT_FORMAT {
            return fmt_err(format!(
                "format {format:?}, expected {WEIGHT_EDIT_FORMAT:?}"
            ));
        }
        let description = root
            .field_opt("description")
            .map_err(as_fmt)?
            .map(|v| v.as_str().map(str::to_string))
            .transpose()
            .map_err(as_fmt)?;
        let raw_edits = root.get_array("edits").map_err(as_fmt)?;
        if raw_edits.is_empty() {
            return fmt_err("edits list is empty");
        }
        let mut edits = Vec::with_capacity(raw_edits.len());
        for (i, item) in raw_edits.iter().enumerate() {
            edits.push(parse_op(i, item)?);
        }
        Ok(Self { description, edits })
    }

    /// Apply all edits atomically to `w`. On any error, `w` is unchanged bit for bit.
    pub fn apply(&self, w: &mut Sam31Weights) -> Result<EditReport, EditError> {
        let total_values = w.total_numel();
        let mut staged: BTreeMap<String, StagedTensor> = BTreeMap::new();
        for op in &self.edits {
            let param = op.param();
            if !staged.contains_key(param) {
                let Some(existing) = w.tensor(param) else {
                    return Err(EditError::UnknownTarget {
                        param: param.to_string(),
                    });
                };
                if existing.data.len() != existing.numel() {
                    return Err(EditError::Shape {
                        param: param.to_string(),
                        detail: format!(
                            "weight data length {} disagrees with shape {:?}",
                            existing.data.len(),
                            existing.shape
                        ),
                    });
                }
                if existing.data.iter().any(|v| !v.is_finite()) {
                    return Err(EditError::NonFinite {
                        param: param.to_string(),
                        detail: "base tensor already holds a non-finite value".to_string(),
                    });
                }
                staged.insert(
                    param.to_string(),
                    StagedTensor::from_slice(existing.shape.clone(), existing.data.clone()),
                );
            }
            staged.get_mut(param).expect("staged above").apply_op(op)?;
        }

        let mut finalized: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let mut values_changed = 0usize;
        for (param, buf) in &staged {
            let (data, changed) = buf.finalize(param)?;
            values_changed += changed;
            finalized.insert(param.clone(), data);
        }

        let mut tensors = BTreeSet::new();
        for (param, data) in finalized {
            let dst = w
                .tensor_mut(&param)
                .expect("verified present via w.tensor() above");
            dst.data = data;
            tensors.insert(param);
        }
        Ok(EditReport {
            description: self.description.clone(),
            edits: self.edits.len(),
            tensors,
            values_changed,
            total_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SchemaRecorder, Weight};

    fn synthetic_weights() -> Sam31Weights {
        let rec = SchemaRecorder::default();
        let mut w = Sam31Weights::load(&rec).unwrap();
        // Fill every typed tensor with tiny deterministic finite data matching its schema shape,
        // except compress large tensors in test fixture? No: keep token_embedding tiny for fast unit tests.
        w.for_each_tensor_mut(|t| {
            if t.name.ends_with("token_embedding.weight") {
                t.shape = vec![8, 4];
            } else {
                t.shape = vec![2, 2];
            }
            let n = t.numel();
            t.data = (0..n).map(|i| (i as f32 + 1.0) * 0.25).collect();
        });
        w.detector_transformer.tensors.insert(
            "detector.transformer.decoder.bbox_embed.layers.2.weight".into(),
            Weight {
                name: "detector.transformer.decoder.bbox_embed.layers.2.weight".into(),
                shape: vec![4, 3],
                data: vec![
                    1.0, 2.0, 3.0, // cx
                    4.0, 5.0, 6.0, // cy
                    7.0, 8.0, 9.0, // w
                    -0.0, 1.5, -2.5, // h (includes -0.0 bit pattern)
                ],
            },
        );
        w.detector_transformer.tensors.insert(
            "detector.transformer.decoder.bbox_embed.layers.2.bias".into(),
            Weight {
                name: "detector.transformer.decoder.bbox_embed.layers.2.bias".into(),
                shape: vec![4],
                data: vec![0.1, 0.2, 0.5, 0.6],
            },
        );
        w
    }

    #[test]
    fn graft_row_and_1d_bias_edit_reach_typed_and_untyped_tensors_preserving_untouched_bits() {
        let mut w = synthetic_weights();
        let orig_bbox_w = w
            .tensor("detector.transformer.decoder.bbox_embed.layers.2.weight")
            .unwrap()
            .data
            .clone();
        let orig_row2_norm: f64 = w
            .tensor("detector.backbone.language_backbone.encoder.token_embedding.weight")
            .unwrap()
            .data[2 * 4..3 * 4]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();

        let manifest = r#"{
          "format": "spiketrack-weight-edit-v1",
          "description": "graft token 5 into token 2 and shrink w/h bias",
          "edits": [
            {
              "op": "graft_row",
              "param": "detector.backbone.language_backbone.encoder.token_embedding.weight",
              "dst": 2,
              "dst_scale": 0.0,
              "sources": [{"row": 5, "weight": 1.0}],
              "preserve_norm": true
            },
            {
              "op": "add_1d",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.bias",
              "range": [2, 4],
              "values": [-0.25, -0.35]
            },
            {
              "op": "scale_rows",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "rows": [2, 3],
              "scale": 0.5
            }
          ]
        }"#;

        let plan = WeightEditPlan::parse(manifest).unwrap();
        let rep = plan.apply(&mut w).unwrap();
        assert_eq!(rep.edits, 3);
        assert_eq!(rep.tensors.len(), 3);
        assert_eq!(rep.values_changed, 4 + 2 + 3);

        let emb = &w
            .tensor("detector.backbone.language_backbone.encoder.token_embedding.weight")
            .unwrap()
            .data;
        let row2 = &emb[2 * 4..3 * 4];
        let row5 = &emb[5 * 4..6 * 4];
        let new_norm: f64 = row2
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        assert!((new_norm - orig_row2_norm).abs() <= 1e-6);
        let cos = crate::inspect::cosine_similarity(row2, row5).unwrap();
        assert!((cos - 1.0).abs() <= 1e-6, "cos={cos}");

        let bias = &w
            .tensor("detector.transformer.decoder.bbox_embed.layers.2.bias")
            .unwrap()
            .data;
        assert_eq!(bias[0..2], [0.1, 0.2]);
        assert!((bias[2] - 0.25).abs() <= 1e-6);
        assert!((bias[3] - 0.25).abs() <= 1e-6);

        let bw = &w
            .tensor("detector.transformer.decoder.bbox_embed.layers.2.weight")
            .unwrap()
            .data;
        // Row 3 index 9 was -0.0f32 and untouched: must preserve its exact sign bit!
        assert_eq!(bw[9].to_bits(), orig_bbox_w[9].to_bits());
        assert_eq!(bw[6..9], [3.5, 4.0, 4.5]);
    }

    #[test]
    fn project_out_direction_removes_subspace_on_cols_and_rows() {
        let mut w = synthetic_weights();
        // bbox_embed.layers.2.weight is [4, 3]. Project out [1, 1, 0, 0] across cols (strength 1.0),
        // then project out [0, 0, 1] across rows [2, 4) (strength 1.0).
        let manifest = r#"{
          "format": "spiketrack-weight-edit-v1",
          "edits": [
            {
              "op": "project_out_direction",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "axis": "cols",
              "direction": [2.0, 2.0, 0.0, 0.0],
              "strength": 1.0
            },
            {
              "op": "project_out_direction",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "axis": "rows",
              "range": [2, 4],
              "direction": [0.0, 0.0, 5.0],
              "strength": 1.0
            }
          ]
        }"#;
        WeightEditPlan::parse(manifest)
            .unwrap()
            .apply(&mut w)
            .unwrap();
        let d = &w
            .tensor("detector.transformer.decoder.bbox_embed.layers.2.weight")
            .unwrap()
            .data;
        // For every column c, d[0, c] + d[1, c] == 0.
        for c in 0..3 {
            assert!((d[c] + d[3 + c]).abs() <= 1e-6, "col {c}: {d:?}");
        }
        // For rows 2 and 3, column 2 was projected out -> 0.
        assert!(d[2 * 3 + 2].abs() <= 1e-6);
        assert!(d[3 * 3 + 2].abs() <= 1e-6);
    }

    #[test]
    fn scale_cols_and_rank1_update_match_exact_arithmetic() {
        let mut w = synthetic_weights();
        let manifest = r#"{
          "format": "spiketrack-weight-edit-v1",
          "edits": [
            {
              "op": "scale_cols",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "cols": [1, 2],
              "scale": 0.0
            },
            {
              "op": "rank1_update",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "rows": [0, 2],
              "cols": [0, 2],
              "scale": 2.0,
              "u": [1.0, -1.0],
              "v": [0.5, 3.0]
            }
          ]
        }"#;
        WeightEditPlan::parse(manifest)
            .unwrap()
            .apply(&mut w)
            .unwrap();
        let d = &w
            .tensor("detector.transformer.decoder.bbox_embed.layers.2.weight")
            .unwrap()
            .data;
        // Row 0 started [1, 2, 3] -> col 1 zeroed [1, 0, 3] -> + 2*1*[0.5, 3.0] = [2, 6, 3]
        assert_eq!(&d[0..3], &[2.0, 6.0, 3.0]);
        // Row 1 started [4, 5, 6] -> col 1 zeroed [4, 0, 6] -> + 2*(-1)*[0.5, 3.0] = [3, -6, 6]
        assert_eq!(&d[3..6], &[3.0, -6.0, 6.0]);
        // Rows 2, 3 had col 1 zeroed and were outside rank1_update rows.
        assert_eq!(d[2 * 3 + 1], 0.0);
        assert_eq!(d[3 * 3 + 1], 0.0);
    }

    #[test]
    fn refusals_reject_typos_bounds_and_overflow_leaving_weights_untouched() {
        let base = synthetic_weights();
        let bad_manifests = [
            r#"{"format": "wrong-v1", "edits": [{"op": "scale_1d", "param": "p", "range": [0, 1], "scale": 1.0}]}"#,
            r#"{"format": "spiketrack-weight-edit-v1", "edits": []}"#,
            r#"{"format": "spiketrack-weight-edit-v1", "extra": 1, "edits": [{"op": "scale_1d", "param": "p", "range": [0, 1], "scale": 1.0}]}"#,
            r#"{"format": "spiketrack-weight-edit-v1", "edits": [{"op": "scale_1d", "param": "p", "range": [2, 2], "scale": 1.0}]}"#,
            r#"{"format": "spiketrack-weight-edit-v1", "edits": [{"op": "scale_rows", "param": "p", "rows": [0, 1], "scale": 1.0, "typo": 2}]}"#,
            r#"{"format": "spiketrack-weight-edit-v1", "edits": [{"op": "project_out_direction", "param": "p", "axis": "cols", "direction": [0.0, 0.0], "strength": 1.0}]}"#,
            r#"{"format": "spiketrack-weight-edit-v1", "edits": [{"op": "add_1d", "param": "p", "range": [0, 2], "values": [1.0]}]}"#,
        ];
        for m in bad_manifests {
            assert!(
                WeightEditPlan::parse(m).is_err(),
                "accepted bad manifest: {m}"
            );
        }

        // Apply-time failures after a valid first op must roll back everything!
        let rollback_cases = [
            // Unknown second tensor
            r#"{
              "format": "spiketrack-weight-edit-v1",
              "edits": [
                {"op": "scale_1d", "param": "detector.transformer.decoder.bbox_embed.layers.2.bias", "range": [0, 4], "scale": 0.0},
                {"op": "scale_1d", "param": "detector.no.such.bias", "range": [0, 1], "scale": 0.0}
              ]
            }"#,
            // Out-of-bounds second op
            r#"{
              "format": "spiketrack-weight-edit-v1",
              "edits": [
                {"op": "scale_1d", "param": "detector.transformer.decoder.bbox_embed.layers.2.bias", "range": [0, 4], "scale": 0.0},
                {"op": "scale_rows", "param": "detector.transformer.decoder.bbox_embed.layers.2.weight", "rows": [0, 5], "scale": 0.0}
              ]
            }"#,
            // f32 overflow on second op
            r#"{
              "format": "spiketrack-weight-edit-v1",
              "edits": [
                {"op": "scale_1d", "param": "detector.transformer.decoder.bbox_embed.layers.2.bias", "range": [0, 4], "scale": 0.0},
                {"op": "scale_rows", "param": "detector.transformer.decoder.bbox_embed.layers.2.weight", "rows": [0, 1], "scale": 1e300}
              ]
            }"#,
            // Zero-norm cancellation under preserve_norm=true
            r#"{
              "format": "spiketrack-weight-edit-v1",
              "edits": [
                {"op": "graft_row", "param": "detector.transformer.decoder.bbox_embed.layers.2.weight", "dst": 0, "dst_scale": 1.0, "sources": [{"row": 0, "weight": -1.0}], "preserve_norm": true}
              ]
            }"#,
        ];
        for m in rollback_cases {
            let mut w = base.clone();
            let plan = WeightEditPlan::parse(m).unwrap();
            assert!(plan.apply(&mut w).is_err(), "should fail: {m}");
            for (a, b) in w.tensors().into_iter().zip(base.tensors()) {
                assert_eq!(a.shape, b.shape);
                let a_bits: Vec<u32> = a.data.iter().map(|v| v.to_bits()).collect();
                let b_bits: Vec<u32> = b.data.iter().map(|v| v.to_bits()).collect();
                assert_eq!(a_bits, b_bits, "mutated {}", a.name);
            }
        }
    }

    #[test]
    fn graft_row_preserve_norm_refuses_zero_norm_destination_and_rolls_back() {
        let mut base = synthetic_weights();
        // Zero row 0 in the base tensor.
        base.tensor_mut("detector.backbone.language_backbone.encoder.token_embedding.weight")
            .unwrap()
            .data[0..4]
            .fill(0.0);

        let cases = [
            // Case 1: dst row is already zero in the checkpoint.
            r#"{
              "format": "spiketrack-weight-edit-v1",
              "edits": [
                {"op": "scale_1d", "param": "detector.transformer.decoder.bbox_embed.layers.2.bias", "range": [0, 4], "scale": 0.0},
                {"op": "graft_row", "param": "detector.backbone.language_backbone.encoder.token_embedding.weight", "dst": 0, "dst_scale": 0.0, "sources": [{"row": 5, "weight": 1.0}], "preserve_norm": true}
              ]
            }"#,
            // Case 2: dst row 2 is zeroed by an earlier op in the same manifest.
            r#"{
              "format": "spiketrack-weight-edit-v1",
              "edits": [
                {"op": "scale_rows", "param": "detector.backbone.language_backbone.encoder.token_embedding.weight", "rows": [2, 3], "scale": 0.0},
                {"op": "graft_row", "param": "detector.backbone.language_backbone.encoder.token_embedding.weight", "dst": 2, "dst_scale": 0.0, "sources": [{"row": 5, "weight": 1.0}], "preserve_norm": true}
              ]
            }"#,
        ];

        for m in cases {
            let mut w = base.clone();
            let plan = WeightEditPlan::parse(m).unwrap();
            let err = plan
                .apply(&mut w)
                .expect_err("zero-norm dst with preserve_norm=true must fail closed");
            assert!(matches!(err, EditError::NonFinite { .. }), "{err:?}");
            for (a, b) in w.tensors().into_iter().zip(base.tensors()) {
                let a_bits: Vec<u32> = a.data.iter().map(|v| v.to_bits()).collect();
                let b_bits: Vec<u32> = b.data.iter().map(|v| v.to_bits()).collect();
                assert_eq!(a_bits, b_bits, "mutated {}", a.name);
            }
        }
    }

    #[test]
    fn extreme_finite_f64_scales_and_directions_do_not_overflow_or_underflow_norms() {
        // 1. project_out_direction with 1e-200 and 1e200 finite JSON vectors must normalize cleanly.
        let mut w_dir = synthetic_weights();
        let dir_manifest = r#"{
          "format": "spiketrack-weight-edit-v1",
          "edits": [
            {
              "op": "project_out_direction",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "axis": "cols",
              "direction": [1e200, 1e200, 0.0, 0.0],
              "strength": 1.0
            },
            {
              "op": "project_out_direction",
              "param": "detector.transformer.decoder.bbox_embed.layers.2.weight",
              "axis": "rows",
              "range": [2, 4],
              "direction": [0.0, 0.0, 1e-200],
              "strength": 1.0
            }
          ]
        }"#;
        WeightEditPlan::parse(dir_manifest)
            .expect("extreme finite directions must parse")
            .apply(&mut w_dir)
            .unwrap();
        let d = &w_dir
            .tensor("detector.transformer.decoder.bbox_embed.layers.2.weight")
            .unwrap()
            .data;
        for c in 0..3 {
            assert!((d[c] + d[3 + c]).abs() <= 1e-6);
        }
        assert!(d[2 * 3 + 2].abs() <= 1e-6);

        // 2. Intermediate 1e200 source row grafted with preserve_norm=true must normalize to
        //    orig_row2_norm instead of overflowing sum(x*x) to +inf and dividing to [0, 0, 0, 0].
        let mut w_graft = synthetic_weights();
        let orig_row2_norm: f64 = w_graft
            .tensor("detector.backbone.language_backbone.encoder.token_embedding.weight")
            .unwrap()
            .data[2 * 4..3 * 4]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        let graft_manifest = r#"{
          "format": "spiketrack-weight-edit-v1",
          "edits": [
            {"op": "scale_rows", "param": "detector.backbone.language_backbone.encoder.token_embedding.weight", "rows": [5, 6], "scale": 1e200},
            {"op": "graft_row", "param": "detector.backbone.language_backbone.encoder.token_embedding.weight", "dst": 2, "dst_scale": 0.0, "sources": [{"row": 5, "weight": 1.0}], "preserve_norm": true},
            {"op": "scale_rows", "param": "detector.backbone.language_backbone.encoder.token_embedding.weight", "rows": [5, 6], "scale": 1e-200}
          ]
        }"#;
        WeightEditPlan::parse(graft_manifest)
            .unwrap()
            .apply(&mut w_graft)
            .unwrap();
        let emb = &w_graft
            .tensor("detector.backbone.language_backbone.encoder.token_embedding.weight")
            .unwrap()
            .data;
        let row2 = &emb[2 * 4..3 * 4];
        let row5 = &emb[5 * 4..6 * 4];
        let new_norm: f64 = row2
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        assert!(
            (new_norm - orig_row2_norm).abs() <= 1e-6,
            "got {new_norm} vs {orig_row2_norm}"
        );
        let cos = crate::inspect::cosine_similarity(row2, row5).unwrap();
        assert!((cos - 1.0).abs() <= 1e-6, "cos={cos}");
    }
}
