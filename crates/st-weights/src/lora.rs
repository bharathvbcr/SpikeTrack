//! LoRA adapter for the SAM 3.1 detector: load a safetensors adapter and fold it into the
//! checkpoint tensors it names, `W' = W + (alpha / r) * B A`, in f32, at load time.
//!
//! The adapter is written offline by `tools/reference/train_lora.py` (PyTorch, MPS). Nothing
//! here runs Python.
//!
//! # Layout (the PEFT layout, as `ojas-model/src/lora.rs` and its adapter checkpoint use it)
//!
//! * `<module>.lora_A.weight`: `A`, f32 `[r, in]`.
//! * `<module>.lora_B.weight`: `B`, f32 `[rows, r]`.
//! * scale `alpha / r`, with `alpha` and `r` per module.
//! * metadata key [`LORA_METADATA_KEY`]: JSON
//!   `{"format": "spiketrack-lora-v1", "base_sha256": "<hex>", "dropout": 0.1,
//!     "modules": [{"module": "...", "param": "<exact checkpoint tensor>", "rows": [start, end],
//!                  "rank": 16, "alpha": 32.0}, ...]}`.
//!
//! SAM's attention keeps q, k and v in one `in_proj_weight [768, 256]`, so each of q, k, v is its
//! own module over a row slice (`rows`) of that tensor and has its own `A` and `B`; `out_proj`
//! targets the whole `[256, 256]` tensor (`rows` is `[0, 256]`). `param` is the exact name in
//! the checkpoint (`detector.transformer.encoder.layers.0.self_attn.in_proj_weight`). The
//! `module` string only keys the two tensors.
//!
//! # Refusals
//!
//! Opening refuses: a missing or malformed metadata key, another format, a module without both
//! tensors, a tensor no module claims, a non-f32 tensor, a non-finite value, `A` or `B` whose
//! shape disagrees with `rank` or `rows`, rank 0, a rank above `min(in, rows)`, a negative or
//! non-finite alpha, an empty or reversed row range, and two modules whose row ranges overlap on
//! one parameter. Applying refuses a target that names no tensor in the checkpoint
//! ([`LoraError::UnknownTarget`]), a weight that is not 2-D, a row range past the weight, an `A`
//! whose width is not the weight's, and a non-finite merged value. Applying checks every target
//! before it changes any tensor, so a refusal leaves the weights untouched.
//!
//! Any 2-D tensor in [`Sam31Weights`] can be targeted (the detector transformer, geometry encoder,
//! segmentation head, scoring head, tracker, text encoder/resizer, and vision trunk). Adapters that
//! touch `detector.backbone.vision_backbone.trunk.*` are flagged by
//! [`LoraAdapter::touches_vision_trunk`] so CLI commands using an exported CoreML trunk can refuse
//! rather than silently bypassing those edits.
//!
//! # Numerics
//!
//! `delta[i][j] = sum_k B[i][k] * A[k][j]`, summed in f32 in ascending `k`; then
//! `W'[i][j] = W[i][j] + s * delta[i][j]` with `s = alpha / r` computed in f64 and rounded once
//! to f32. PyTorch's `B @ A` (BLAS) sums in a different order, so the two agree to f32 rounding,
//! not bit for bit: `tests/lora_merge.rs` states and checks the tolerance.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use ojas_io::{parse_json, IoError, JsonValue, SafeTensors, StDtype};

use crate::names::Sam31Weights;
use crate::{Weight, WeightsError};

/// Safetensors metadata key holding the adapter's JSON description.
pub const LORA_METADATA_KEY: &str = "spiketrack_lora";
/// The only adapter format this module reads.
pub const LORA_FORMAT: &str = "spiketrack-lora-v1";

#[derive(Debug)]
pub enum LoraError {
    Io(IoError),
    Weights(WeightsError),
    /// The adapter file is malformed or inconsistent with itself.
    Format(String),
    /// A target names a tensor the checkpoint (its adaptable groups) does not have.
    UnknownTarget {
        param: String,
    },
    /// A target does not fit the tensor it names.
    Shape {
        param: String,
        detail: String,
    },
    /// The merge produced a NaN or infinity.
    NonFinite {
        param: String,
    },
}

impl fmt::Display for LoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "lora adapter I/O: {}", e.detail()),
            Self::Weights(e) => write!(f, "lora: {e}"),
            Self::Format(detail) => write!(f, "lora adapter: {detail}"),
            Self::UnknownTarget { param } => {
                write!(f, "lora target {param:?} is not a tensor of the checkpoint")
            }
            Self::Shape { param, detail } => write!(f, "lora target {param}: {detail}"),
            Self::NonFinite { param } => write!(f, "lora merge into {param} is not finite"),
        }
    }
}

impl std::error::Error for LoraError {}

impl From<IoError> for LoraError {
    fn from(e: IoError) -> Self {
        Self::Io(e)
    }
}

impl From<WeightsError> for LoraError {
    fn from(e: WeightsError) -> Self {
        Self::Weights(e)
    }
}

/// One adapted slice of one checkpoint tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraTarget {
    pub module: String,
    /// Exact checkpoint tensor name.
    pub param: String,
    pub row_start: usize,
    pub row_end: usize,
    pub rank: usize,
    pub alpha: f32,
    /// `[rank, in]`, row-major.
    pub a: Weight,
    /// `[row_end - row_start, rank]`, row-major.
    pub b: Weight,
}

impl LoraTarget {
    /// `alpha / rank`, divided in f64 and rounded once.
    pub fn scale(&self) -> f32 {
        (f64::from(self.alpha) / self.rank as f64) as f32
    }

    pub fn in_features(&self) -> usize {
        self.a.shape[1]
    }

    /// `W[row_start..row_end] += scale * (B A)` on a row-major `[_, in]` slice of the weight.
    fn add_into(&self, w: &mut [f32], cols: usize) {
        let scale = self.scale();
        let rows = self.row_end - self.row_start;
        let mut delta = vec![0f32; cols];
        for i in 0..rows {
            delta.iter_mut().for_each(|d| *d = 0.0);
            for k in 0..self.rank {
                let b = self.b.data[i * self.rank + k];
                let a_row = &self.a.data[k * cols..(k + 1) * cols];
                for (d, &a) in delta.iter_mut().zip(a_row) {
                    *d += b * a;
                }
            }
            let start = (self.row_start + i) * cols;
            for (wv, &d) in w[start..start + cols].iter_mut().zip(&delta) {
                *wv += scale * d;
            }
        }
    }
}

/// What a merge touched.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub targets: usize,
    /// Distinct checkpoint tensors changed.
    pub tensors: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub struct LoraAdapter {
    pub base_sha256: Option<String>,
    pub dropout: Option<f64>,
    pub targets: Vec<LoraTarget>,
}

fn fmt_err<T>(detail: impl Into<String>) -> Result<T, LoraError> {
    Err(LoraError::Format(detail.into()))
}

fn json_usize(v: &JsonValue, what: &str) -> Result<usize, LoraError> {
    let n = v
        .as_u64()
        .map_err(|e| LoraError::Format(format!("{what}: {}", e.detail())))?;
    usize::try_from(n).map_err(|_| LoraError::Format(format!("{what}: {n} does not fit usize")))
}

fn read_f32_2d(st: &SafeTensors<'_>, name: &str) -> Result<Weight, LoraError> {
    let info = st
        .info(name)
        .map_err(|_| LoraError::Format(format!("tensor {name:?} is missing")))?;
    if info.dtype != StDtype::F32 {
        return fmt_err(format!("{name}: dtype {:?}, expected F32", info.dtype));
    }
    let (shape, data) = st.read_f32(name)?;
    if shape.len() != 2 || shape.contains(&0) {
        return fmt_err(format!("{name}: shape {shape:?}, expected non-empty 2-D"));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return fmt_err(format!("{name}: holds a non-finite value"));
    }
    let shape = shape.into_iter().map(|d| d as usize).collect();
    Ok(Weight {
        name: name.to_string(),
        shape,
        data,
    })
}

impl LoraAdapter {
    pub fn open(path: &Path) -> Result<Self, LoraError> {
        let st = SafeTensors::open(path)?;
        Self::from_safetensors(&st)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, LoraError> {
        let st = SafeTensors::parse(bytes)?;
        Self::from_safetensors(&st)
    }

    fn from_safetensors(st: &SafeTensors<'_>) -> Result<Self, LoraError> {
        let text = st
            .metadata()
            .get(LORA_METADATA_KEY)
            .ok_or_else(|| LoraError::Format(format!("no {LORA_METADATA_KEY:?} metadata")))?;
        let meta = parse_json(text)
            .map_err(|e| LoraError::Format(format!("metadata is not JSON: {}", e.detail())))?;
        let as_fmt = |e: IoError| LoraError::Format(e.detail().to_string());
        let format = meta.get_str("format").map_err(as_fmt)?;
        if format != LORA_FORMAT {
            return fmt_err(format!("format {format:?}, expected {LORA_FORMAT:?}"));
        }
        let base_sha256 = meta
            .field_opt("base_sha256")
            .map_err(as_fmt)?
            .map(|v| v.as_str().map(str::to_string))
            .transpose()
            .map_err(as_fmt)?;
        let dropout = meta
            .field_opt("dropout")
            .map_err(as_fmt)?
            .map(|v| v.as_f64())
            .transpose()
            .map_err(as_fmt)?;
        let modules = meta.get_array("modules").map_err(as_fmt)?;
        if modules.is_empty() {
            return fmt_err("no modules");
        }

        let mut targets = Vec::with_capacity(modules.len());
        let mut claimed: BTreeSet<String> = BTreeSet::new();
        let mut seen_modules: BTreeSet<String> = BTreeSet::new();
        for m in modules {
            let module = m.get_str("module").map_err(as_fmt)?.to_string();
            let param = m.get_str("param").map_err(as_fmt)?.to_string();
            if module.is_empty() || param.is_empty() {
                return fmt_err("empty module or param name");
            }
            if !seen_modules.insert(module.clone()) {
                return fmt_err(format!("module {module:?} listed twice"));
            }
            let rows = m.get_array("rows").map_err(as_fmt)?;
            let [rs, re] = rows else {
                return fmt_err(format!("{module}: rows must be [start, end]"));
            };
            let (row_start, row_end) = (json_usize(rs, "rows[0]")?, json_usize(re, "rows[1]")?);
            if row_start >= row_end {
                return fmt_err(format!(
                    "{module}: empty or reversed rows [{row_start}, {row_end}]"
                ));
            }
            let rank = json_usize(m.field("rank").map_err(as_fmt)?, "rank")?;
            if rank == 0 {
                return fmt_err(format!("{module}: rank 0"));
            }
            let alpha = m.get_f64("alpha").map_err(as_fmt)?;
            if !alpha.is_finite() || alpha < 0.0 || alpha > f64::from(f32::MAX) {
                return fmt_err(format!(
                    "{module}: alpha {alpha} is not a finite non-negative f32"
                ));
            }
            let a_name = format!("{module}.lora_A.weight");
            let b_name = format!("{module}.lora_B.weight");
            let a = read_f32_2d(st, &a_name)?;
            let b = read_f32_2d(st, &b_name)?;
            claimed.insert(a_name);
            claimed.insert(b_name);
            let rows_n = row_end - row_start;
            if a.shape[0] != rank {
                return fmt_err(format!("{module}: A {:?} but rank {rank}", a.shape));
            }
            if b.shape != [rows_n, rank] {
                return fmt_err(format!(
                    "{module}: B {:?}, expected [{rows_n}, {rank}]",
                    b.shape
                ));
            }
            if rank > a.shape[1].min(rows_n) {
                return fmt_err(format!(
                    "{module}: rank {rank} exceeds min(in {}, rows {rows_n})",
                    a.shape[1]
                ));
            }
            targets.push(LoraTarget {
                module,
                param,
                row_start,
                row_end,
                rank,
                alpha: alpha as f32,
                a,
                b,
            });
        }
        if let Some(stray) = st.names().find(|n| !claimed.contains(*n)) {
            return fmt_err(format!("tensor {stray:?} belongs to no module"));
        }
        let mut by_param: BTreeMap<&str, Vec<(usize, usize, &str)>> = BTreeMap::new();
        for t in &targets {
            by_param
                .entry(&t.param)
                .or_default()
                .push((t.row_start, t.row_end, &t.module));
        }
        for (param, ranges) in &mut by_param {
            ranges.sort();
            for pair in ranges.windows(2) {
                if pair[0].1 > pair[1].0 {
                    return fmt_err(format!(
                        "modules {:?} and {:?} overlap on {param}: rows [{}, {}) and [{}, {})",
                        pair[0].2, pair[1].2, pair[0].0, pair[0].1, pair[1].0, pair[1].1
                    ));
                }
            }
        }
        Ok(Self {
            base_sha256,
            dropout,
            targets,
        })
    }

    /// Fold the adapter into `maps` (tensors by checkpoint name). All targets are checked first;
    /// on any refusal no tensor is changed.
    pub fn merge_into(
        &self,
        maps: &mut [&mut BTreeMap<String, Weight>],
    ) -> Result<MergeReport, LoraError> {
        let mut plan: Vec<(usize, &LoraTarget)> = Vec::with_capacity(self.targets.len());
        for t in &self.targets {
            let Some(mi) = maps.iter().position(|m| m.contains_key(&t.param)) else {
                return Err(LoraError::UnknownTarget {
                    param: t.param.clone(),
                });
            };
            let w = &maps[mi][&t.param];
            let shape_err = |detail: String| LoraError::Shape {
                param: t.param.clone(),
                detail,
            };
            let &[rows, cols] = w.shape.as_slice() else {
                return Err(shape_err(format!("weight shape {:?} is not 2-D", w.shape)));
            };
            if t.row_end > rows {
                return Err(shape_err(format!(
                    "rows [{}, {}) past the weight's {rows} rows",
                    t.row_start, t.row_end
                )));
            }
            if t.in_features() != cols {
                return Err(shape_err(format!(
                    "A is {} wide, the weight {cols}",
                    t.in_features()
                )));
            }
            if w.data.len() != rows * cols {
                return Err(shape_err(
                    "weight data length disagrees with its shape".to_string(),
                ));
            }
            plan.push((mi, t));
        }
        // Merge into copies of the touched tensors first, so a non-finite result changes nothing.
        let mut merged: BTreeMap<(usize, String), Vec<f32>> = BTreeMap::new();
        for &(mi, t) in &plan {
            let cols = t.in_features();
            let buf = merged
                .entry((mi, t.param.clone()))
                .or_insert_with(|| maps[mi][&t.param].data.clone());
            t.add_into(buf, cols);
        }
        for ((_, param), data) in &merged {
            if data.iter().any(|v| !v.is_finite()) {
                return Err(LoraError::NonFinite {
                    param: param.clone(),
                });
            }
        }
        let mut report = MergeReport {
            targets: plan.len(),
            tensors: BTreeSet::new(),
        };
        for ((mi, param), data) in merged {
            if let Some(w) = maps[mi].get_mut(&param) {
                w.data = data;
            }
            report.tensors.insert(param);
        }
        Ok(report)
    }

    /// First target parameter inside `detector.backbone.vision_backbone.trunk.`, if any.
    ///
    /// Callers running an external pre-compiled ViT trunk (e.g. CoreML) use this to fail closed
    /// rather than silently bypassing trunk LoRA modules.
    pub fn touches_vision_trunk(&self) -> Option<&str> {
        self.targets
            .iter()
            .map(|t| t.param.as_str())
            .find(|p| p.starts_with("detector.backbone.vision_backbone.trunk."))
    }

    /// Fold the adapter into a loaded checkpoint (both the untyped detector/tracker groups and
    /// typed 2-D weights such as `detector.backbone.language_backbone.resizer.weight`).
    /// All targets are checked and staged first; on any refusal no tensor is changed.
    pub fn apply(&self, w: &mut Sam31Weights) -> Result<MergeReport, LoraError> {
        for t in &self.targets {
            let Some(existing) = w.tensor(&t.param) else {
                return Err(LoraError::UnknownTarget {
                    param: t.param.clone(),
                });
            };
            let shape_err = |detail: String| LoraError::Shape {
                param: t.param.clone(),
                detail,
            };
            let &[rows, cols] = existing.shape.as_slice() else {
                return Err(shape_err(format!(
                    "weight shape {:?} is not 2-D",
                    existing.shape
                )));
            };
            if t.row_end > rows {
                return Err(shape_err(format!(
                    "rows [{}, {}) past the weight's {rows} rows",
                    t.row_start, t.row_end
                )));
            }
            if t.in_features() != cols {
                return Err(shape_err(format!(
                    "A is {} wide, the weight {cols}",
                    t.in_features()
                )));
            }
            if existing.data.len() != rows * cols {
                return Err(shape_err(
                    "weight data length disagrees with its shape".to_string(),
                ));
            }
        }
        let mut merged: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for t in &self.targets {
            let cols = t.in_features();
            let buf = merged.entry(t.param.clone()).or_insert_with(|| {
                w.tensor(&t.param)
                    .expect("verified present above")
                    .data
                    .clone()
            });
            t.add_into(buf, cols);
        }
        for (param, data) in &merged {
            if data.iter().any(|v| !v.is_finite()) {
                return Err(LoraError::NonFinite {
                    param: param.clone(),
                });
            }
        }
        let mut report = MergeReport {
            targets: self.targets.len(),
            tensors: BTreeSet::new(),
        };
        for (param, data) in merged {
            let dst = w.tensor_mut(&param).expect("verified present above");
            dst.data = data;
            report.tensors.insert(param);
        }
        Ok(report)
    }
}

/// Load the whole checkpoint (every tensor claimed exactly once) and fold `adapter` into it.
pub fn load_merged(checkpoint: &Path, adapter: &LoraAdapter) -> Result<Sam31Weights, LoraError> {
    let mut w = crate::Checkpoint::load_all(checkpoint)?;
    adapter.apply(&mut w)?;
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_io::{encode_safetensors, TensorOut};

    struct Spec {
        module: &'static str,
        param: &'static str,
        rows: [usize; 2],
        rank: usize,
        alpha: f64,
        a: Vec<f32>,
        b: Vec<f32>,
        cols: usize,
    }

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn spec(
        module: &'static str,
        param: &'static str,
        rows: [usize; 2],
        rank: usize,
        alpha: f64,
        cols: usize,
        seed: u32,
    ) -> Spec {
        Spec {
            module,
            param,
            rows,
            rank,
            alpha,
            a: pseudo(rank * cols, seed),
            b: pseudo((rows[1] - rows[0]) * rank, seed + 1),
            cols,
        }
    }

    fn bytes(
        specs: &[Spec],
        mutate_meta: impl Fn(String) -> String,
        extra: &[(&str, Vec<u64>, Vec<f32>)],
    ) -> Vec<u8> {
        let mut names = Vec::new();
        let mut shapes = Vec::new();
        let mut datas = Vec::new();
        for s in specs {
            names.push(format!("{}.lora_A.weight", s.module));
            shapes.push(vec![s.rank as u64, s.cols as u64]);
            datas.push(
                s.a.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            names.push(format!("{}.lora_B.weight", s.module));
            shapes.push(vec![(s.rows[1] - s.rows[0]) as u64, s.rank as u64]);
            datas.push(
                s.b.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        }
        for (n, sh, d) in extra {
            names.push((*n).to_string());
            shapes.push(sh.clone());
            datas.push(d.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        }
        let items: Vec<TensorOut<'_>> = (0..names.len())
            .map(|i| TensorOut {
                name: &names[i],
                dtype: StDtype::F32,
                shape: &shapes[i],
                data: &datas[i],
            })
            .collect();
        let modules: Vec<String> = specs
            .iter()
            .map(|s| {
                format!(
                    "{{\"module\":\"{}\",\"param\":\"{}\",\"rows\":[{},{}],\"rank\":{},\"alpha\":{:?}}}",
                    s.module, s.param, s.rows[0], s.rows[1], s.rank, s.alpha
                )
            })
            .collect();
        let meta = format!("{{\"format\":\"{LORA_FORMAT}\",\"base_sha256\":\"abc\",\"dropout\":0.1,\"modules\":[{}]}}", modules.join(","));
        let meta = mutate_meta(meta);
        encode_safetensors(&items, &[(LORA_METADATA_KEY, meta.as_str())]).unwrap()
    }

    fn weight(name: &str, rows: usize, cols: usize, seed: u32) -> Weight {
        Weight {
            name: name.to_string(),
            shape: vec![rows, cols],
            data: pseudo(rows * cols, seed),
        }
    }

    fn map(ws: Vec<Weight>) -> BTreeMap<String, Weight> {
        ws.into_iter().map(|w| (w.name.clone(), w)).collect()
    }

    const P: &str = "detector.transformer.encoder.layers.0.self_attn.in_proj_weight";
    const O: &str = "detector.transformer.encoder.layers.0.self_attn.out_proj.weight";

    fn reference(w: &Weight, t: &Spec) -> Vec<f64> {
        let s = t.alpha / t.rank as f64;
        let mut out: Vec<f64> = w.data.iter().map(|&v| f64::from(v)).collect();
        for i in 0..(t.rows[1] - t.rows[0]) {
            for j in 0..t.cols {
                let d: f64 = (0..t.rank)
                    .map(|k| f64::from(t.b[i * t.rank + k]) * f64::from(t.a[k * t.cols + j]))
                    .sum();
                out[(t.rows[0] + i) * t.cols + j] += s * d;
            }
        }
        out
    }

    #[test]
    fn merge_matches_an_f64_reference_and_leaves_other_rows_alone() {
        let specs = [
            spec("q", P, [0, 8], 4, 8.0, 16, 1),
            spec("v", P, [16, 24], 4, 2.0, 16, 7),
        ];
        let ad = LoraAdapter::parse(&bytes(&specs, |m| m, &[])).unwrap();
        assert_eq!(ad.base_sha256.as_deref(), Some("abc"));
        assert_eq!(ad.dropout, Some(0.1));
        let base = weight(P, 24, 16, 99);
        let mut m = map(vec![base.clone()]);
        let rep = ad.merge_into(&mut [&mut m]).unwrap();
        assert_eq!(rep.targets, 2);
        assert_eq!(rep.tensors.iter().collect::<Vec<_>>(), vec![&P.to_string()]);
        let mut want: Vec<f64> = base.data.iter().map(|&v| f64::from(v)).collect();
        for s in &specs {
            let r = reference(&base, s);
            for (w, (r, b)) in want.iter_mut().zip(r.iter().zip(&base.data)) {
                *w += r - f64::from(*b);
            }
        }
        let got = &m[P].data;
        for (i, (&g, &e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(g) - e).abs() <= 1e-6 + 1e-6 * e.abs(),
                "element {i}: {g} vs {e}"
            );
        }
        // Rows 8..16 (the "k" slice) carry no module: bit for bit unchanged.
        assert_eq!(got[8 * 16..16 * 16], base.data[8 * 16..16 * 16]);
        assert_ne!(got[..8 * 16], base.data[..8 * 16]);
    }

    #[test]
    fn scale_is_alpha_over_rank() {
        let s1 = spec("q", P, [0, 4], 4, 4.0, 8, 3);
        let s2 = spec("q", P, [0, 4], 4, 8.0, 8, 3);
        let base = weight(P, 4, 8, 5);
        let (mut m1, mut m2) = (map(vec![base.clone()]), map(vec![base.clone()]));
        LoraAdapter::parse(&bytes(&[s1], |m| m, &[]))
            .unwrap()
            .merge_into(&mut [&mut m1])
            .unwrap();
        LoraAdapter::parse(&bytes(&[s2], |m| m, &[]))
            .unwrap()
            .merge_into(&mut [&mut m2])
            .unwrap();
        for ((a, b), w) in m1[P].data.iter().zip(&m2[P].data).zip(&base.data) {
            // alpha 8 / r 4 = 2, twice the alpha 4 / r 4 = 1 delta.
            assert!(((b - w) - 2.0 * (a - w)).abs() <= 1e-6, "{a} {b} {w}");
        }
    }

    #[test]
    fn zero_b_is_the_identity_bit_for_bit() {
        let mut s = spec("o", O, [0, 8], 2, 32.0, 8, 4);
        s.b = vec![0.0; s.b.len()];
        let base = weight(O, 8, 8, 6);
        let mut m = map(vec![base.clone()]);
        LoraAdapter::parse(&bytes(&[s], |m| m, &[]))
            .unwrap()
            .merge_into(&mut [&mut m])
            .unwrap();
        assert_eq!(m[O].data, base.data);
    }

    #[test]
    fn unknown_target_is_refused_and_nothing_changes() {
        let specs = [
            spec("q", P, [0, 8], 4, 8.0, 16, 1),
            spec(
                "x",
                "detector.transformer.nope.in_proj_weight",
                [0, 8],
                4,
                8.0,
                16,
                2,
            ),
        ];
        let ad = LoraAdapter::parse(&bytes(&specs, |m| m, &[])).unwrap();
        let base = weight(P, 24, 16, 99);
        let mut m = map(vec![base.clone()]);
        let err = ad.merge_into(&mut [&mut m]).unwrap_err();
        assert!(
            matches!(err, LoraError::UnknownTarget { ref param } if param.ends_with("nope.in_proj_weight")),
            "{err}"
        );
        assert_eq!(m[P].data, base.data);
    }

    #[test]
    fn targets_found_across_several_maps() {
        let specs = [
            spec("q", P, [0, 8], 4, 8.0, 16, 1),
            spec("o", O, [0, 8], 2, 2.0, 8, 2),
        ];
        let ad = LoraAdapter::parse(&bytes(&specs, |m| m, &[])).unwrap();
        let (mut m1, mut m2) = (
            map(vec![weight(P, 24, 16, 1)]),
            map(vec![weight(O, 8, 8, 2)]),
        );
        let rep = ad.merge_into(&mut [&mut m1, &mut m2]).unwrap();
        assert_eq!(rep.tensors.len(), 2);
    }

    #[test]
    fn shape_mismatches_are_refused_without_changes() {
        let ad =
            LoraAdapter::parse(&bytes(&[spec("q", P, [0, 8], 4, 8.0, 16, 1)], |m| m, &[])).unwrap();
        for (rows, cols) in [(4usize, 16usize), (24, 12), (24, 17)] {
            let base = weight(P, rows, cols, 9);
            let mut m = map(vec![base.clone()]);
            assert!(
                matches!(ad.merge_into(&mut [&mut m]), Err(LoraError::Shape { .. })),
                "{rows}x{cols}"
            );
            assert_eq!(m[P].data, base.data);
        }
        let mut one_d = map(vec![Weight {
            name: P.to_string(),
            shape: vec![24],
            data: vec![0.0; 24],
        }]);
        assert!(matches!(
            ad.merge_into(&mut [&mut one_d]),
            Err(LoraError::Shape { .. })
        ));
    }

    #[test]
    fn overflow_to_infinity_is_refused() {
        let mut s = spec("q", P, [0, 4], 2, 2.0, 4, 1);
        s.a = vec![1.0e30; s.a.len()];
        s.b = vec![1.0e30; s.b.len()];
        let ad = LoraAdapter::parse(&bytes(&[s], |m| m, &[])).unwrap();
        let base = weight(P, 4, 4, 1);
        let mut m = map(vec![base.clone()]);
        assert!(matches!(
            ad.merge_into(&mut [&mut m]),
            Err(LoraError::NonFinite { .. })
        ));
        assert_eq!(m[P].data, base.data);
    }

    fn refuses(
        specs: &[Spec],
        meta: impl Fn(String) -> String,
        extra: &[(&str, Vec<u64>, Vec<f32>)],
        needle: &str,
    ) {
        let err = LoraAdapter::parse(&bytes(specs, meta, extra)).unwrap_err();
        let text = err.to_string();
        assert!(text.contains(needle), "{text:?} lacks {needle:?}");
    }

    #[test]
    fn malformed_adapters_are_refused() {
        let ok = || spec("q", P, [0, 8], 4, 8.0, 16, 1);
        refuses(
            &[ok()],
            |m| m.replace(LORA_FORMAT, "other-v9"),
            &[],
            "format",
        );
        refuses(
            &[ok()],
            |m| m.replace("\"rank\":4", "\"rank\":0"),
            &[],
            "rank 0",
        );
        refuses(
            &[ok()],
            |m| m.replace("\"rank\":4", "\"rank\":3"),
            &[],
            "rank 3",
        );
        refuses(
            &[ok()],
            |m| m.replace("\"rows\":[0,8]", "\"rows\":[0,6]"),
            &[],
            "expected [6, 4]",
        );
        refuses(
            &[ok()],
            |m| m.replace("\"rows\":[0,8]", "\"rows\":[8,8]"),
            &[],
            "empty or reversed",
        );
        refuses(
            &[ok()],
            |m| m.replace("\"alpha\":8.0", "\"alpha\":-1.0"),
            &[],
            "alpha",
        );
        refuses(
            &[ok()],
            |m| m.replace("\"module\":\"q\"", "\"module\":\"z\""),
            &[],
            "missing",
        );
        refuses(
            &[ok()],
            |m| m,
            &[("stray", vec![1, 1], vec![0.0])],
            "belongs to no module",
        );
        refuses(
            &[ok(), spec("k", P, [4, 12], 4, 8.0, 16, 3)],
            |m| m,
            &[],
            "overlap",
        );
        refuses(
            &[ok(), spec("k", P, [8, 16], 4, 8.0, 16, 3)],
            |m| m.replace("\"module\":\"k\"", "\"module\":\"q\""),
            &[],
            "listed twice",
        );
        let mut nan = ok();
        nan.a[3] = f32::NAN;
        refuses(&[nan], |m| m, &[], "non-finite");
        refuses(
            &[ok()],
            |m| m.replace("\"modules\":[", "\"modules\":[],\"unused\":["),
            &[],
            "no modules",
        );
    }

    #[test]
    fn missing_metadata_and_non_f32_are_refused() {
        let items = [TensorOut {
            name: "x",
            dtype: StDtype::F32,
            shape: &[1],
            data: &0f32.to_le_bytes(),
        }];
        let no_meta = encode_safetensors(&items, &[]).unwrap();
        assert!(LoraAdapter::parse(&no_meta)
            .unwrap_err()
            .to_string()
            .contains(LORA_METADATA_KEY));
        let bad_json = encode_safetensors(&items, &[(LORA_METADATA_KEY, "{")]).unwrap();
        assert!(LoraAdapter::parse(&bad_json)
            .unwrap_err()
            .to_string()
            .contains("JSON"));
        // A bf16 A tensor under a valid module description.
        let meta = format!(
            "{{\"format\":\"{LORA_FORMAT}\",\"modules\":[{{\"module\":\"q\",\"param\":\"{P}\",\"rows\":[0,2],\"rank\":1,\"alpha\":1.0}}]}}"
        );
        let a = [0u8; 4];
        let b = [0u8; 8];
        let items = [
            TensorOut {
                name: "q.lora_A.weight",
                dtype: StDtype::BF16,
                shape: &[1, 2],
                data: &a,
            },
            TensorOut {
                name: "q.lora_B.weight",
                dtype: StDtype::F32,
                shape: &[2, 1],
                data: &b,
            },
        ];
        let bytes = encode_safetensors(&items, &[(LORA_METADATA_KEY, meta.as_str())]).unwrap();
        assert!(LoraAdapter::parse(&bytes)
            .unwrap_err()
            .to_string()
            .contains("dtype"));
    }

    #[test]
    fn open_reads_a_file() {
        let specs = [spec("q", P, [0, 8], 4, 8.0, 16, 1)];
        let path = std::env::temp_dir().join(format!(
            "st-weights-lora-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes(&specs, |m| m, &[])).unwrap();
        let ad = LoraAdapter::open(&path);
        std::fs::remove_file(&path).unwrap();
        let ad = ad.unwrap();
        assert_eq!(ad.targets.len(), 1);
        assert_eq!(ad.targets[0].scale(), 2.0);
        assert_eq!(ad.targets[0].a.shape, vec![4, 16]);
    }

    fn tiny_checkpoint() -> Sam31Weights {
        let rec = crate::SchemaRecorder::default();
        let mut w = Sam31Weights::load(&rec).unwrap();
        w.for_each_tensor_mut(|t| {
            t.shape = if t.name.ends_with(".bias") {
                vec![4]
            } else {
                vec![4, 6]
            };
            let n = t.numel();
            t.data = pseudo(n, 42);
        });
        w.detector_transformer
            .tensors
            .insert(P.to_string(), weight(P, 8, 6, 77));
        w
    }

    #[test]
    fn apply_reaches_typed_language_resizer_and_rolls_back_atomically() {
        const RESIZER: &str = "detector.backbone.language_backbone.resizer.weight";
        const RESIZER_BIAS: &str = "detector.backbone.language_backbone.resizer.bias";
        const TRUNK_QKV: &str = "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight";

        let mut w = tiny_checkpoint();
        let orig_resizer = w.text.resizer.weight.clone();
        let orig_det = w.detector_transformer.tensors[P].clone();

        // 1. Valid adapter touching both typed resizer and untyped detector attention.
        let s_res = spec("resizer", RESIZER, [0, 4], 2, 4.0, 6, 11);
        let s_det = spec("det_q", P, [0, 4], 2, 4.0, 6, 19);
        let ad = LoraAdapter::parse(&bytes(&[s_res, s_det], |m| m, &[])).unwrap();
        assert_eq!(ad.touches_vision_trunk(), None);

        let rep = ad.apply(&mut w).unwrap();
        assert_eq!(rep.targets, 2);
        assert!(rep.tensors.contains(RESIZER) && rep.tensors.contains(P));
        assert_ne!(w.text.resizer.weight.data, orig_resizer.data);
        assert_ne!(w.detector_transformer.tensors[P].data, orig_det.data);
        // Untouched rows 4..8 of P remain bit-identical.
        assert_eq!(
            w.detector_transformer.tensors[P].data[4 * 6..],
            orig_det.data[4 * 6..]
        );

        // 2. 1-D typed bias is refused without mutating prior state.
        let snap_resizer = w.text.resizer.weight.data.clone();
        let bad_1d = LoraAdapter::parse(&bytes(
            &[
                spec("resizer", RESIZER, [0, 4], 2, 4.0, 6, 3),
                spec("bias", RESIZER_BIAS, [0, 4], 1, 1.0, 1, 5),
            ],
            |m| m,
            &[],
        ))
        .unwrap();
        assert!(matches!(bad_1d.apply(&mut w), Err(LoraError::Shape { .. })));
        assert_eq!(w.text.resizer.weight.data, snap_resizer);

        // 3. Overflow to Inf on second target leaves first target untouched.
        let mut s_inf = spec("det_q", P, [0, 4], 1, 1e38, 6, 9);
        s_inf.a = vec![1e20; 6];
        s_inf.b = vec![1e20; 4];
        let bad_inf = LoraAdapter::parse(&bytes(
            &[spec("resizer", RESIZER, [0, 4], 2, 4.0, 6, 3), s_inf],
            |m| m,
            &[],
        ))
        .unwrap();
        assert!(matches!(
            bad_inf.apply(&mut w),
            Err(LoraError::NonFinite { .. })
        ));
        assert_eq!(w.text.resizer.weight.data, snap_resizer);

        // 4. Trunk target detection for external CoreML guard.
        let trunk_ad = LoraAdapter::parse(&bytes(
            &[spec("t", TRUNK_QKV, [0, 4], 2, 2.0, 6, 1)],
            |m| m,
            &[],
        ))
        .unwrap();
        assert_eq!(trunk_ad.touches_vision_trunk(), Some(TRUNK_QKV));
    }
}
