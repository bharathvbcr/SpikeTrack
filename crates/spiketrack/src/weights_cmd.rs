//! `spiketrack weights`: native weight-space inspection, vocabulary geometry, SVD spectra,
//! and base-vs-modified delta analysis for SAM 3.1 checkpoints.
//!
//! Supports inspecting the raw checkpoint, folding `--lora FILE`, applying `--weight-edit FILE`,
//! comparing modified tensors against the untouched base (`--diff base`), inspecting individual
//! tensors (`--tensor NAME`), filtering tensor tables (`--filter SUBSTR`), and probing BPE token
//! embeddings plus post-24-layer resized detector prompt representations (`--tokens A,B,C`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use ojas_core::Budget;
use ojas_cpu::CpuBackend;
use st_model::detector::D;
use st_model::{Text, Tokenizer};
use st_weights::inspect::{
    col_stats, cosine_similarity, diff_weights, nearest_rows, row_stats, top_singular_triplets,
    NearestRow, SingularTriplet, SliceStats, TensorDiff, TensorStats, DEFAULT_SVD_ITERS,
};
use st_weights::{Sam31Weights, Weight};

use crate::cli::{existing_path, Args, CliError};
use crate::json;
use crate::weights::{self, edit_json, lora_json, Applied, AppliedEdit};

const TEXT_BUDGET_BYTES: u64 = 2 << 30;
const TOKEN_EMB_NAME: &str = "detector.backbone.language_backbone.encoder.token_embedding.weight";

pub const USAGE: &str = "\
spiketrack weights [--weights FILE] [--lora FILE] [--weight-edit FILE] [--model-dir DIR]
                   [--filter SUBSTR] [--tensor NAME] [--tokens W1,W2,...]
                   [--diff none|base] [--svd-k K] [--top-k K] [--format text|json]

  Read and inspect SAM 3.1 weights natively in Rust (norms, row/col slices, top singular
  directions, BPE token-embedding neighborhoods, 24-layer text-resizer cosines, and LoRA or
  surgical weight-edit deltas).

  --weights FILE      the converted checkpoint (default: $SPIKETRACK_WEIGHTS)
  --lora FILE         fold this detector LoRA adapter before inspecting
  --weight-edit FILE  apply this surgical weight-edit manifest (spiketrack-weight-edit-v1)
  --model-dir DIR     folder with vocab.json and merges.txt (default: $SPIKETRACK_MODEL_DIR; used by --tokens)
  --filter SUBSTR     list every tensor whose name contains SUBSTR with f64 summary stats
  --tensor NAME       deeply inspect one tensor (values if <=32 elems, row/col norms, top SVD)
  --tokens WORDS      comma-separated prompts/words: inspect BPE ids, token_embedding rows,
                      nearest vocabulary neighbors, and post-transformer 256-D prompt cosines
  --diff MODE         base (default when --lora or --weight-edit is given) or none
  --svd-k K           singular triplets to extract for --tensor and --diff, 0 to 16 (default 4)
  --top-k K           top rows/cols/neighbors to show, 1 to 100 (default 10)
  --format FMT        text (default) or json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(CliError::usage(format!(
                "--format: {other:?} is invalid (choices: text, json)"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffMode {
    None,
    Base,
}

impl DiffMode {
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s {
            "none" => Ok(Self::None),
            "base" => Ok(Self::Base),
            other => Err(CliError::usage(format!(
                "--diff: {other:?} is invalid (choices: none, base)"
            ))),
        }
    }
}

pub struct FamilySummary {
    pub family: &'static str,
    pub tensors: usize,
    pub numel: usize,
}

pub fn summarize_families(w: &Sam31Weights) -> Vec<FamilySummary> {
    let prefixes: [(&str, fn(&str) -> bool); 8] = [
        ("vision.trunk", |n| {
            n.starts_with("detector.backbone.vision_backbone.trunk.")
        }),
        ("vision.necks", |n| {
            n.starts_with("detector.backbone.vision_backbone.")
                && !n.starts_with("detector.backbone.vision_backbone.trunk.")
        }),
        ("language_backbone", |n| {
            n.starts_with("detector.backbone.language_backbone.")
        }),
        ("detector.transformer", |n| {
            n.starts_with("detector.transformer.")
        }),
        ("detector.geometry_encoder", |n| {
            n.starts_with("detector.geometry_encoder.")
        }),
        ("detector.segmentation_head", |n| {
            n.starts_with("detector.segmentation_head.")
        }),
        ("detector.dot_prod_scoring", |n| {
            n.starts_with("detector.dot_prod_scoring.")
        }),
        ("tracker.model", |n| n.starts_with("tracker.model.")),
    ];
    let mut counts = vec![(0usize, 0usize); prefixes.len()];
    w.for_each_tensor(|t| {
        for (i, (_, pred)) in prefixes.iter().enumerate() {
            if pred(&t.name) {
                counts[i].0 += 1;
                counts[i].1 += t.numel();
                break;
            }
        }
    });
    prefixes
        .iter()
        .zip(counts)
        .map(|((family, _), (tensors, numel))| FamilySummary {
            family,
            tensors,
            numel,
        })
        .collect()
}

pub struct TensorDetail {
    pub stats: TensorStats,
    pub small_values: Option<Vec<f32>>,
    pub rows: Option<Vec<SliceStats>>,
    pub cols: Option<Vec<SliceStats>>,
    pub svd: Vec<SingularTriplet>,
}

pub fn inspect_tensor_detail(w: &Weight, svd_k: usize) -> Result<TensorDetail, String> {
    let stats = TensorStats::compute(w)?;
    let small_values = (stats.numel <= 32).then(|| w.data.clone());
    let (rows, cols, svd) = if w.shape.len() == 2 {
        (
            Some(row_stats(w)?),
            Some(col_stats(w)?),
            top_singular_triplets(w, svd_k, DEFAULT_SVD_ITERS)?,
        )
    } else {
        (None, None, Vec::new())
    };
    Ok(TensorDetail {
        stats,
        small_values,
        rows,
        cols,
        svd,
    })
}

pub struct TokenProbeEntry {
    pub prompt: String,
    pub bpe_ids: Vec<u32>,
    pub bpe_pieces: Vec<String>,
    pub token_norms: Vec<f64>,
    pub mean_emb: Vec<f32>,
    pub mean_emb_norm: f64,
    pub neighbors: Vec<(u32, String, f64, f64)>,
    pub resized_mean: Vec<f32>,
    pub resized_mean_norm: f64,
    pub resized_eot: Vec<f32>,
    pub resized_eot_norm: f64,
}

pub struct TokenProbeReport {
    pub entries: Vec<TokenProbeEntry>,
    pub emb_cosines: Vec<Vec<f64>>,
    pub resized_mean_cosines: Vec<Vec<f64>>,
    pub resized_eot_cosines: Vec<Vec<f64>>,
}

fn l2_norm_f32(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

pub fn probe_tokens(
    w: &Sam31Weights,
    tok: &Tokenizer,
    prompts: &[String],
    top_k: usize,
) -> Result<TokenProbeReport, String> {
    let emb = w
        .tensor(TOKEN_EMB_NAME)
        .ok_or_else(|| format!("missing {TOKEN_EMB_NAME}"))?;
    let &[vocab, dim] = emb.shape.as_slice() else {
        return Err(format!(
            "{TOKEN_EMB_NAME}: unexpected shape {:?}",
            emb.shape
        ));
    };
    let pieces = tok.vocab_pieces();
    let piece_of =
        |id: u32| -> String { pieces.get(&id).copied().unwrap_or("<unknown>").to_string() };

    let be =
        CpuBackend::with_threads(Budget::new(TEXT_BUDGET_BYTES), 4).map_err(|e| e.to_string())?;
    let text_enc = Text::new(&be, &w.text).map_err(|e| e.to_string())?;

    let mut tokenized_batch = Vec::with_capacity(prompts.len());
    let mut bpe_per_prompt = Vec::with_capacity(prompts.len());
    for p in prompts {
        let bpe = tok.encode(p).map_err(|e| format!("{p:?}: {e}"))?;
        if bpe.is_empty() {
            return Err(format!("prompt {p:?} encoded to zero BPE tokens"));
        }
        let ctx = tok.tokenize(p).map_err(|e| format!("{p:?}: {e}"))?;
        tokenized_batch.push(ctx);
        bpe_per_prompt.push(bpe);
    }

    let tf = text_enc
        .encode(&tokenized_batch)
        .map_err(|e| e.to_string())?;
    let resized_host = tf.resized.to_host(be.budget()).map_err(|e| e.to_string())?;
    let resized_slice = resized_host.f32_slice().map_err(|e| e.to_string())?;
    let ctx_len = st_model::tokenizer::CONTEXT;

    let mut entries = Vec::with_capacity(prompts.len());
    for (bi, p) in prompts.iter().enumerate() {
        let bpe_ids = &bpe_per_prompt[bi];
        let mut bpe_pieces = Vec::with_capacity(bpe_ids.len());
        let mut token_norms = Vec::with_capacity(bpe_ids.len());
        let mut mean_emb = vec![0.0f64; dim];
        for &id in bpe_ids {
            let r = id as usize;
            if r >= vocab {
                return Err(format!("token id {id} >= vocab {vocab}"));
            }
            let row = &emb.data[r * dim..(r + 1) * dim];
            bpe_pieces.push(piece_of(id));
            token_norms.push(l2_norm_f32(row));
            for (acc, &x) in mean_emb.iter_mut().zip(row) {
                *acc += f64::from(x);
            }
        }
        let inv_n = 1.0 / bpe_ids.len() as f64;
        let mean_emb_f32: Vec<f32> = mean_emb.into_iter().map(|v| (v * inv_n) as f32).collect();
        let mean_emb_norm = l2_norm_f32(&mean_emb_f32);
        let nr: Vec<NearestRow> = nearest_rows(emb, &mean_emb_f32, top_k)?;
        let neighbors = nr
            .into_iter()
            .map(|n| (n.row as u32, piece_of(n.row as u32), n.cosine, n.l2_norm))
            .collect();

        // Post-24-layer + resizer (256-D): valid tokens are non-padded positions in [0..ctx_len).
        let pad_row = &tf.padding[bi * ctx_len..(bi + 1) * ctx_len];
        let mut valid_positions: Vec<usize> = (0..ctx_len).filter(|&t| !pad_row[t]).collect();
        if valid_positions.is_empty() {
            valid_positions.push(0);
        }
        let eot_pos = *valid_positions.last().expect("non-empty");
        let eot_off = (bi * ctx_len + eot_pos) * D;
        let resized_eot = resized_slice[eot_off..eot_off + D].to_vec();
        let resized_eot_norm = l2_norm_f32(&resized_eot);

        let mut r_mean = vec![0.0f64; D];
        for &t in &valid_positions {
            let off = (bi * ctx_len + t) * D;
            for (acc, &x) in r_mean.iter_mut().zip(&resized_slice[off..off + D]) {
                *acc += f64::from(x);
            }
        }
        let inv_v = 1.0 / valid_positions.len() as f64;
        let resized_mean: Vec<f32> = r_mean.into_iter().map(|v| (v * inv_v) as f32).collect();
        let resized_mean_norm = l2_norm_f32(&resized_mean);

        entries.push(TokenProbeEntry {
            prompt: p.clone(),
            bpe_ids: bpe_ids.clone(),
            bpe_pieces,
            token_norms,
            mean_emb: mean_emb_f32,
            mean_emb_norm,
            neighbors,
            resized_mean,
            resized_mean_norm,
            resized_eot,
            resized_eot_norm,
        });
    }

    let n = entries.len();
    let mut emb_cosines = vec![vec![0.0; n]; n];
    let mut resized_mean_cosines = vec![vec![0.0; n]; n];
    let mut resized_eot_cosines = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            emb_cosines[i][j] = cosine_similarity(&entries[i].mean_emb, &entries[j].mean_emb)?;
            resized_mean_cosines[i][j] =
                cosine_similarity(&entries[i].resized_mean, &entries[j].resized_mean)?;
            resized_eot_cosines[i][j] =
                cosine_similarity(&entries[i].resized_eot, &entries[j].resized_eot)?;
        }
    }

    Ok(TokenProbeReport {
        entries,
        emb_cosines,
        resized_mean_cosines,
        resized_eot_cosines,
    })
}

fn top_abs_coords(v: &[f64], top_k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0)));
    indexed.truncate(top_k);
    indexed
}

fn render_text(
    weights_path: &Path,
    total_tensors: usize,
    total_values: usize,
    applied_lora: Option<&Applied>,
    applied_edit: Option<&AppliedEdit>,
    families: &[FamilySummary],
    filtered: Option<&[TensorStats]>,
    detail: Option<&TensorDetail>,
    diffs: Option<&[TensorDiff]>,
    tokens: Option<&TokenProbeReport>,
    top_k: usize,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "checkpoint: {} ({total_tensors} tensors, {total_values} parameters)",
        weights_path.display()
    );
    if let Some(l) = applied_lora {
        let _ = writeln!(
            out,
            "lora: {} ({} targets across {} tensors)",
            l.path.display(),
            l.targets,
            l.tensors
        );
    }
    if let Some(e) = applied_edit {
        let _ = writeln!(
            out,
            "weight_edit: {} ({} op(s), {} tensor(s), {} / {} values = {:.6}%)",
            e.path.display(),
            e.edits,
            e.tensors,
            e.values_changed,
            e.total_values,
            100.0 * e.fraction_changed()
        );
        if let Some(desc) = &e.description {
            let _ = writeln!(out, "  description: {desc}");
        }
    }

    if filtered.is_none() && detail.is_none() && diffs.is_none() && tokens.is_none() {
        let _ = writeln!(out, "\nmodule families:");
        for f in families {
            let _ = writeln!(
                out,
                "  {:<28} {:>5} tensors  {:>12} params",
                f.family, f.tensors, f.numel
            );
        }
    }

    if let Some(list) = filtered {
        let _ = writeln!(out, "\nfiltered tensors ({}):", list.len());
        for s in list {
            let _ = writeln!(
                out,
                "  {:<68} {:<16} l2={:>9.4} rms={:>8.5} [{:>8.4}, {:>8.4}]",
                s.name,
                format!("{:?}", s.shape),
                s.l2_norm,
                s.rms,
                s.min,
                s.max
            );
        }
    }

    if let Some(d) = detail {
        let s = &d.stats;
        let _ = writeln!(
            out,
            "\ntensor: {} shape={:?} numel={}",
            s.name, s.shape, s.numel
        );
        let _ = writeln!(
            out,
            "  l2={:.6} rms={:.6} mean={:.6} std={:.6} min={:.6} max={:.6} zeros={}",
            s.l2_norm, s.rms, s.mean, s.std, s.min, s.max, s.zero_count
        );
        if let Some(vals) = &d.small_values {
            let formatted: Vec<String> = vals.iter().map(|v| format!("{v:.6}")).collect();
            let _ = writeln!(out, "  values: [{}]", formatted.join(", "));
        }
        if !d.svd.is_empty() {
            let _ = writeln!(out, "  top singular triplets:");
            for (i, t) in d.svd.iter().enumerate() {
                let u_top: Vec<String> = top_abs_coords(&t.u, 5)
                    .into_iter()
                    .map(|(idx, v)| format!("{idx}:{v:+.3}"))
                    .collect();
                let v_top: Vec<String> = top_abs_coords(&t.v, 5)
                    .into_iter()
                    .map(|(idx, v)| format!("{idx}:{v:+.3}"))
                    .collect();
                let _ = writeln!(
                    out,
                    "    sigma[{i}] = {:.6}  u_top=[{}]  v_top=[{}]",
                    t.sigma,
                    u_top.join(", "),
                    v_top.join(", ")
                );
            }
        }
        if let Some(rs) = &d.rows {
            if rs.len() <= 16 {
                let _ = writeln!(out, "  all {} row(s):", rs.len());
                for r in rs {
                    let _ = writeln!(
                        out,
                        "    row {:>3}: l2={:.6} mean={:+.6} [{:+.6}, {:+.6}]",
                        r.index, r.l2_norm, r.mean, r.min, r.max
                    );
                }
            } else {
                let mut sorted = rs.clone();
                sorted.sort_by(|a, b| b.l2_norm.total_cmp(&a.l2_norm).then(a.index.cmp(&b.index)));
                let _ = writeln!(
                    out,
                    "  top {} row(s) by L2 norm (of {}):",
                    top_k.min(sorted.len()),
                    rs.len()
                );
                for r in sorted.iter().take(top_k) {
                    let _ = writeln!(
                        out,
                        "    row {:>5}: l2={:.6} mean={:+.6} [{:+.6}, {:+.6}]",
                        r.index, r.l2_norm, r.mean, r.min, r.max
                    );
                }
            }
        }
        if let Some(cs) = &d.cols {
            let mut sorted = cs.clone();
            sorted.sort_by(|a, b| b.l2_norm.total_cmp(&a.l2_norm).then(a.index.cmp(&b.index)));
            let _ = writeln!(
                out,
                "  top {} col(s) by L2 norm (of {}):",
                top_k.min(sorted.len()),
                cs.len()
            );
            for c in sorted.iter().take(top_k) {
                let _ = writeln!(
                    out,
                    "    col {:>5}: l2={:.6} mean={:+.6} [{:+.6}, {:+.6}]",
                    c.index, c.l2_norm, c.mean, c.min, c.max
                );
            }
        }
    }

    if let Some(ds) = diffs {
        let total_changed: usize = ds.iter().map(|d| d.values_changed).sum();
        let _ = writeln!(
            out,
            "\ndiff vs base: {} modified tensor(s), {total_changed} scalar(s) changed ({:.6}%)",
            ds.len(),
            if total_values == 0 {
                0.0
            } else {
                100.0 * total_changed as f64 / total_values as f64
            }
        );
        for d in ds {
            let sigs: Vec<String> = d
                .top_singular_values
                .iter()
                .map(|s| format!("{s:.4}"))
                .collect();
            let top_rows_suffix = if d.top_changed_rows.is_empty() {
                String::new()
            } else {
                let items: Vec<String> = d
                    .top_changed_rows
                    .iter()
                    .take(4)
                    .map(|(r, n)| format!("{r}:{n:.4}"))
                    .collect();
                format!(" top_rows=[{}]", items.join(", "))
            };
            let _ = writeln!(
                out,
                "  {:<68} changed={:>7}/{} delta_l2={:>8.4} rel_l2={:>7.4} max_abs={:>7.4} sv=[{}]{top_rows_suffix}",
                d.name,
                d.values_changed,
                d.numel,
                d.delta_l2,
                d.rel_l2,
                d.max_abs_delta,
                sigs.join(", ")
            );
        }
    }

    if let Some(tp) = tokens {
        let _ = writeln!(
            out,
            "\ntoken & prompt weight-space probe ({} item(s)):",
            tp.entries.len()
        );
        for e in &tp.entries {
            let toks: Vec<String> = e
                .bpe_ids
                .iter()
                .zip(&e.bpe_pieces)
                .zip(&e.token_norms)
                .map(|((id, p), n)| format!("{id}:{p:?}(l2={n:.4})"))
                .collect();
            let _ = writeln!(
                out,
                "  prompt {:?}: bpe=[{}] emb_l2={:.4} resized_mean_l2={:.4} resized_eot_l2={:.4}",
                e.prompt,
                toks.join(", "),
                e.mean_emb_norm,
                e.resized_mean_norm,
                e.resized_eot_norm
            );
            let nbrs: Vec<String> = e
                .neighbors
                .iter()
                .map(|(id, p, cos, n)| format!("{id}:{p:?}({cos:.3},l2={n:.3})"))
                .collect();
            let _ = writeln!(out, "    nearest token_embedding rows: {}", nbrs.join(", "));
        }
        if tp.entries.len() > 1 {
            let names: Vec<String> = tp
                .entries
                .iter()
                .map(|e| format!("{:>14}", e.prompt))
                .collect();
            let _ = writeln!(out, "\n  pairwise cosine — raw token_embedding (1024-D):");
            let _ = writeln!(out, "  {:>14} {}", "", names.join(" "));
            for (i, e) in tp.entries.iter().enumerate() {
                let row: Vec<String> = tp.emb_cosines[i]
                    .iter()
                    .map(|c| format!("{c:>14.4}"))
                    .collect();
                let _ = writeln!(out, "  {:>14} {}", e.prompt, row.join(" "));
            }
            let _ = writeln!(
                out,
                "\n  pairwise cosine — post-24-layer Text+resizer mean valid tokens (256-D):"
            );
            let _ = writeln!(out, "  {:>14} {}", "", names.join(" "));
            for (i, e) in tp.entries.iter().enumerate() {
                let row: Vec<String> = tp.resized_mean_cosines[i]
                    .iter()
                    .map(|c| format!("{c:>14.4}"))
                    .collect();
                let _ = writeln!(out, "  {:>14} {}", e.prompt, row.join(" "));
            }
            let _ = writeln!(
                out,
                "\n  pairwise cosine — post-24-layer Text+resizer EOT token (256-D):"
            );
            let _ = writeln!(out, "  {:>14} {}", "", names.join(" "));
            for (i, e) in tp.entries.iter().enumerate() {
                let row: Vec<String> = tp.resized_eot_cosines[i]
                    .iter()
                    .map(|c| format!("{c:>14.4}"))
                    .collect();
                let _ = writeln!(out, "  {:>14} {}", e.prompt, row.join(" "));
            }
        }
    }

    out
}

fn shape_json(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", parts.join(", "))
}

fn slice_stats_json(slices: &[SliceStats], limit: Option<usize>) -> Result<String, String> {
    let chosen: Vec<&SliceStats> = match limit {
        None => slices.iter().collect(),
        Some(k) => {
            let mut sorted: Vec<&SliceStats> = slices.iter().collect();
            sorted.sort_by(|a, b| b.l2_norm.total_cmp(&a.l2_norm).then(a.index.cmp(&b.index)));
            sorted.truncate(k);
            sorted
        }
    };
    let mut items = Vec::with_capacity(chosen.len());
    for s in chosen {
        items.push(format!(
            "{{\"index\": {}, \"l2_norm\": {}, \"mean\": {}, \"min\": {}, \"max\": {}, \"max_abs\": {}}}",
            s.index,
            json::number(s.l2_norm, "l2_norm")?,
            json::number(s.mean, "mean")?,
            json::number32(s.min, "min")?,
            json::number32(s.max, "max")?,
            json::number32(s.max_abs, "max_abs")?,
        ));
    }
    Ok(format!("[{}]", items.join(", ")))
}

fn coords_json(vec: &[f64], top_n: usize) -> Result<String, String> {
    let top = top_abs_coords(vec, top_n);
    let mut items = Vec::with_capacity(top.len());
    for (idx, val) in top {
        items.push(format!(
            "{{\"index\": {idx}, \"value\": {}}}",
            json::number(val, "coord")?
        ));
    }
    Ok(format!("[{}]", items.join(", ")))
}

#[allow(clippy::too_many_arguments)]
fn render_json(
    weights_path: &Path,
    total_tensors: usize,
    total_values: usize,
    applied_lora: Option<&Applied>,
    applied_edit: Option<&AppliedEdit>,
    families: &[FamilySummary],
    filtered: Option<&[TensorStats]>,
    detail: Option<&TensorDetail>,
    diffs: Option<&[TensorDiff]>,
    tokens: Option<&TokenProbeReport>,
    top_k: usize,
) -> Result<String, String> {
    let fam_items: Vec<String> = families
        .iter()
        .map(|f| {
            format!(
                "{{\"family\": {}, \"tensors\": {}, \"numel\": {}}}",
                json::string(f.family),
                f.tensors,
                f.numel
            )
        })
        .collect();

    let filtered_json = match filtered {
        None => "null".to_string(),
        Some(list) => {
            let mut items = Vec::with_capacity(list.len());
            for s in list {
                items.push(format!(
                    "{{\"name\": {}, \"shape\": {}, \"numel\": {}, \"l2_norm\": {}, \"rms\": {}, \"mean\": {}, \"std\": {}, \"min\": {}, \"max\": {}, \"max_abs\": {}, \"zero_count\": {}}}",
                    json::string(&s.name),
                    shape_json(&s.shape),
                    s.numel,
                    json::number(s.l2_norm, "l2_norm")?,
                    json::number(s.rms, "rms")?,
                    json::number(s.mean, "mean")?,
                    json::number(s.std, "std")?,
                    json::number32(s.min, "min")?,
                    json::number32(s.max, "max")?,
                    json::number32(s.max_abs, "max_abs")?,
                    s.zero_count,
                ));
            }
            format!("[{}]", items.join(", "))
        }
    };

    let detail_json = match detail {
        None => "null".to_string(),
        Some(d) => {
            let s = &d.stats;
            let vals_json = match &d.small_values {
                None => "null".to_string(),
                Some(vs) => {
                    let parts: Result<Vec<String>, String> =
                        vs.iter().map(|&v| json::number32(v, "value")).collect();
                    format!("[{}]", parts?.join(", "))
                }
            };
            let mut svd_items = Vec::with_capacity(d.svd.len());
            for t in &d.svd {
                svd_items.push(format!(
                    "{{\"sigma\": {}, \"u_top\": {}, \"v_top\": {}}}",
                    json::number(t.sigma, "sigma")?,
                    coords_json(&t.u, 5)?,
                    coords_json(&t.v, 5)?,
                ));
            }
            let rows_json = match &d.rows {
                None => "null".to_string(),
                Some(rs) => {
                    let lim = (rs.len() > 16).then_some(top_k);
                    slice_stats_json(rs, lim)?
                }
            };
            let cols_json = match &d.cols {
                None => "null".to_string(),
                Some(cs) => slice_stats_json(cs, Some(top_k))?,
            };
            format!(
                "{{\"name\": {}, \"shape\": {}, \"numel\": {}, \"l2_norm\": {}, \"rms\": {}, \"mean\": {}, \"std\": {}, \"min\": {}, \"max\": {}, \"max_abs\": {}, \"zero_count\": {}, \"values\": {}, \"svd\": [{}], \"rows\": {}, \"cols\": {}}}",
                json::string(&s.name),
                shape_json(&s.shape),
                s.numel,
                json::number(s.l2_norm, "l2_norm")?,
                json::number(s.rms, "rms")?,
                json::number(s.mean, "mean")?,
                json::number(s.std, "std")?,
                json::number32(s.min, "min")?,
                json::number32(s.max, "max")?,
                json::number32(s.max_abs, "max_abs")?,
                s.zero_count,
                vals_json,
                svd_items.join(", "),
                rows_json,
                cols_json,
            )
        }
    };

    let diffs_json = match diffs {
        None => "null".to_string(),
        Some(ds) => {
            let mut items = Vec::with_capacity(ds.len());
            for d in ds {
                let rel = if d.rel_l2.is_finite() {
                    json::number(d.rel_l2, "rel_l2")?
                } else {
                    "null".to_string()
                };
                let mut row_items = Vec::with_capacity(d.top_changed_rows.len());
                for &(r, dl2) in &d.top_changed_rows {
                    row_items.push(format!(
                        "{{\"row\": {r}, \"delta_l2\": {}}}",
                        json::number(dl2, "row_delta_l2")?
                    ));
                }
                items.push(format!(
                    "{{\"name\": {}, \"shape\": {}, \"numel\": {}, \"values_changed\": {}, \"base_l2\": {}, \"delta_l2\": {}, \"rel_l2\": {}, \"max_abs_delta\": {}, \"top_singular_values\": {}, \"top_changed_rows\": [{}]}}",
                    json::string(&d.name),
                    shape_json(&d.shape),
                    d.numel,
                    d.values_changed,
                    json::number(d.base_l2, "base_l2")?,
                    json::number(d.delta_l2, "delta_l2")?,
                    rel,
                    json::number(d.max_abs_delta, "max_abs_delta")?,
                    json::number_array(&d.top_singular_values, "top_singular_values")?,
                    row_items.join(", "),
                ));
            }
            format!("[{}]", items.join(", "))
        }
    };

    let tokens_json = match tokens {
        None => "null".to_string(),
        Some(tp) => {
            let mut ent = Vec::with_capacity(tp.entries.len());
            for e in &tp.entries {
                let ids: Vec<String> = e.bpe_ids.iter().map(u32::to_string).collect();
                let pcs: Vec<String> = e.bpe_pieces.iter().map(|s| json::string(s)).collect();
                let mut nb = Vec::with_capacity(e.neighbors.len());
                for (id, p, cos, n) in &e.neighbors {
                    nb.push(format!(
                        "{{\"id\": {id}, \"piece\": {}, \"cosine\": {}, \"l2_norm\": {}}}",
                        json::string(p),
                        json::number(*cos, "cosine")?,
                        json::number(*n, "l2_norm")?,
                    ));
                }
                ent.push(format!(
                    "{{\"prompt\": {}, \"bpe_ids\": [{}], \"bpe_pieces\": [{}], \"emb_l2\": {}, \"resized_mean_l2\": {}, \"resized_eot_l2\": {}, \"neighbors\": [{}]}}",
                    json::string(&e.prompt),
                    ids.join(", "),
                    pcs.join(", "),
                    json::number(e.mean_emb_norm, "emb_l2")?,
                    json::number(e.resized_mean_norm, "resized_mean_l2")?,
                    json::number(e.resized_eot_norm, "resized_eot_l2")?,
                    nb.join(", "),
                ));
            }
            let matrix_json = |m: &[Vec<f64>]| -> Result<String, String> {
                let rows: Result<Vec<String>, String> =
                    m.iter().map(|r| json::number_array(r, "cosine")).collect();
                Ok(format!("[{}]", rows?.join(", ")))
            };
            format!(
                "{{\"entries\": [{}], \"emb_cosines\": {}, \"resized_mean_cosines\": {}, \"resized_eot_cosines\": {}}}",
                ent.join(", "),
                matrix_json(&tp.emb_cosines)?,
                matrix_json(&tp.resized_mean_cosines)?,
                matrix_json(&tp.resized_eot_cosines)?,
            )
        }
    };

    Ok(format!(
        "{{\n  \"weights\": {},\n  \"total_tensors\": {},\n  \"total_values\": {},\n  \"lora\": {},\n  \"weight_edit\": {},\n  \"families\": [{}],\n  \"filtered\": {},\n  \"tensor\": {},\n  \"diffs\": {},\n  \"tokens\": {}\n}}\n",
        json::string(&weights_path.to_string_lossy()),
        total_tensors,
        total_values,
        lora_json(applied_lora),
        edit_json(applied_edit)?,
        fam_items.join(", "),
        filtered_json,
        detail_json,
        diffs_json,
        tokens_json,
    ))
}

pub fn run(raw: &[String]) -> Result<(), CliError> {
    let args = Args::parse(
        raw,
        &[
            "weights",
            "lora",
            "weight-edit",
            "model-dir",
            "filter",
            "tensor",
            "tokens",
            "diff",
            "svd-k",
            "top-k",
            "format",
        ],
        0,
    )?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }
    let format = OutputFormat::parse(args.get("format").unwrap_or("text"))?;
    let svd_k: usize = args.parsed("svd-k", 4)?;
    if svd_k > 16 {
        return Err(CliError::usage(format!("--svd-k {svd_k} exceeds 16")));
    }
    let top_k: usize = args.parsed("top-k", 10)?;
    if !(1..=100).contains(&top_k) {
        return Err(CliError::usage(format!(
            "--top-k {top_k} is outside 1..=100"
        )));
    }
    let lora = weights::lora_path(args.get("lora"))?;
    let weight_edit = weights::edit_path(args.get("weight-edit"))?;
    let has_modifier = lora.is_some() || weight_edit.is_some();
    let diff_mode = match args.get("diff") {
        Some(d) => DiffMode::parse(d)?,
        None => {
            if has_modifier {
                DiffMode::Base
            } else {
                DiffMode::None
            }
        }
    };
    if diff_mode == DiffMode::Base && !has_modifier {
        return Err(CliError::usage(
            "--diff base requires --lora or --weight-edit",
        ));
    }

    let token_prompts: Option<Vec<String>> = match args.get("tokens") {
        None => None,
        Some(raw_list) => {
            let list: Vec<String> = raw_list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if list.is_empty() {
                return Err(CliError::usage("--tokens has no non-empty items"));
            }
            Some(list)
        }
    };
    let tokenizer = match &token_prompts {
        None => None,
        Some(_) => {
            let model_dir = existing_path(
                args.get("model-dir"),
                "SPIKETRACK_MODEL_DIR",
                "tokenizer folder",
                "model-dir",
            )?;
            let vocab = model_dir.join("vocab.json");
            let merges = model_dir.join("merges.txt");
            Some(Tokenizer::from_hf_files(&vocab, &merges).map_err(|e| {
                CliError::failed(format!("tokenizer files in {}: {e}", model_dir.display()))
            })?)
        }
    };

    let weights_path = existing_path(
        args.get("weights"),
        "SPIKETRACK_WEIGHTS",
        "weights",
        "weights",
    )?;
    let (adapter, plan) = weights::open_modifiers(lora.as_deref(), weight_edit.as_deref())?;
    let mut w = weights::load_raw(&weights_path)?;

    // Snapshot only the tensors named by --lora or --weight-edit when --diff base is requested.
    let mut base_snapshots: BTreeMap<String, Weight> = BTreeMap::new();
    if diff_mode == DiffMode::Base {
        let mut touched_names = BTreeSet::new();
        if let Some(ad) = &adapter {
            for t in &ad.targets {
                touched_names.insert(t.param.clone());
            }
        }
        if let Some(pl) = &plan {
            for e in &pl.edits {
                touched_names.insert(e.param().to_string());
            }
        }
        for name in touched_names {
            if let Some(orig) = w.tensor(&name) {
                base_snapshots.insert(name, orig.clone());
            }
        }
    }

    let (applied_lora, applied_edit) = weights::apply_modifiers(
        &mut w,
        &weights_path,
        lora.as_deref().zip(adapter),
        weight_edit.as_deref().zip(plan),
    )?;

    let all_tensors = w.tensors();
    let total_tensors = all_tensors.len();
    let total_values = w.total_numel();
    let families = summarize_families(&w);

    let filtered = match args.get("filter") {
        None => None,
        Some(substr) => {
            let mut stats = Vec::new();
            for t in &all_tensors {
                if t.name.contains(substr) {
                    stats.push(TensorStats::compute(t).map_err(CliError::failed)?);
                }
            }
            Some(stats)
        }
    };

    let detail = match args.get("tensor") {
        None => None,
        Some(name) => {
            let t = w.tensor(name).ok_or_else(|| {
                CliError::failed(format!("checkpoint has no tensor named {name:?}"))
            })?;
            Some(inspect_tensor_detail(t, svd_k).map_err(CliError::failed)?)
        }
    };

    let diffs = if diff_mode == DiffMode::Base {
        let mut list = Vec::new();
        for (name, base_t) in &base_snapshots {
            let cur_t = w
                .tensor(name)
                .ok_or_else(|| CliError::failed(format!("missing modified tensor {name:?}")))?;
            if let Some(d) = diff_weights(base_t, cur_t, svd_k, top_k).map_err(CliError::failed)? {
                list.push(d);
            }
        }
        list.sort_by(|a, b| b.delta_l2.total_cmp(&a.delta_l2).then(a.name.cmp(&b.name)));
        Some(list)
    } else {
        None
    };

    let token_report = match (&token_prompts, &tokenizer) {
        (Some(prompts), Some(tok)) => {
            Some(probe_tokens(&w, tok, prompts, top_k).map_err(CliError::failed)?)
        }
        _ => None,
    };

    match format {
        OutputFormat::Text => {
            let text = render_text(
                &weights_path,
                total_tensors,
                total_values,
                applied_lora.as_ref(),
                applied_edit.as_ref(),
                &families,
                filtered.as_deref(),
                detail.as_ref(),
                diffs.as_deref(),
                token_report.as_ref(),
                top_k,
            );
            print!("{text}");
        }
        OutputFormat::Json => {
            let j = render_json(
                &weights_path,
                total_tensors,
                total_values,
                applied_lora.as_ref(),
                applied_edit.as_ref(),
                &families,
                filtered.as_deref(),
                detail.as_ref(),
                diffs.as_deref(),
                token_report.as_ref(),
                top_k,
            )
            .map_err(CliError::failed)?;
            print!("{j}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_io::parse_json;
    use st_weights::SchemaRecorder;

    fn vec_s(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn cli_rejects_invalid_options_before_model_load() {
        assert!(run(&vec_s(&["--help"])).is_ok());
        assert!(matches!(
            run(&vec_s(&["positional"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            run(&vec_s(&["--format", "yaml"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            run(&vec_s(&["--diff", "base"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            run(&vec_s(&["--svd-k", "99"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            run(&vec_s(&["--top-k", "0"])),
            Err(CliError::Usage(_))
        ));
        assert!(matches!(
            run(&vec_s(&["--tokens", " , "])),
            Err(CliError::Usage(_))
        ));
    }

    #[test]
    fn json_and_text_renderers_produce_valid_output() {
        let rec = SchemaRecorder::default();
        let mut w = Sam31Weights::load(&rec).unwrap();
        w.for_each_tensor_mut(|t| {
            t.shape = vec![2, 2];
            t.data = vec![1.0, 0.0, 0.0, 2.0];
        });
        let families = summarize_families(&w);
        assert_eq!(families.len(), 8);
        let t0 = w.tensors()[0];
        let det = inspect_tensor_detail(t0, 2).unwrap();
        let j = render_json(
            Path::new("sam.safetensors"),
            w.tensors().len(),
            w.total_numel(),
            None,
            None,
            &families,
            None,
            Some(&det),
            None,
            None,
            5,
        )
        .unwrap();
        let parsed = parse_json(&j).expect("valid JSON");
        assert_eq!(parsed.get_str("weights").unwrap(), "sam.safetensors");
        assert!(
            (parsed.field("tensor").unwrap().get_f64("l2_norm").unwrap() - 5.0f64.sqrt()).abs()
                < 1e-6
        );

        let txt = render_text(
            Path::new("sam.safetensors"),
            w.tensors().len(),
            w.total_numel(),
            None,
            None,
            &families,
            None,
            Some(&det),
            None,
            None,
            5,
        );
        assert!(txt.contains("sigma[0] = 2.000000"), "{txt}");

        let base_t = t0.clone();
        let mut cur_t = base_t.clone();
        cur_t.data[2] = 5.0; // mutate row 1 col 0
        let diff = diff_weights(&base_t, &cur_t, 2, 5).unwrap().unwrap();
        let stats = TensorStats::compute(&cur_t).unwrap();
        let j_full = render_json(
            Path::new("sam.safetensors"),
            w.tensors().len(),
            w.total_numel(),
            None,
            None,
            &families,
            Some(&[stats]),
            Some(&det),
            Some(&[diff.clone()]),
            None,
            5,
        )
        .unwrap();
        let pf = parse_json(&j_full).expect("valid full JSON");
        let tensor_obj = pf.field("tensor").unwrap();
        assert!((tensor_obj.get_f64("max_abs").unwrap() - 2.0).abs() < 1e-6);
        assert_eq!(tensor_obj.get_u64("zero_count").unwrap(), 2);
        assert_eq!(tensor_obj.get_array("rows").unwrap().len(), 2);
        assert_eq!(tensor_obj.get_array("cols").unwrap().len(), 2);
        let svd0 = &tensor_obj.get_array("svd").unwrap()[0];
        assert!(!svd0.get_array("u_top").unwrap().is_empty());
        assert!(!svd0.get_array("v_top").unwrap().is_empty());

        let filt0 = &pf.get_array("filtered").unwrap()[0];
        assert!((filt0.get_f64("max_abs").unwrap() - 5.0).abs() < 1e-6);

        let diff0 = &pf.get_array("diffs").unwrap()[0];
        let changed_rows = diff0.get_array("top_changed_rows").unwrap();
        assert_eq!(changed_rows.len(), 1);
        assert_eq!(changed_rows[0].get_u64("row").unwrap(), 1);

        let txt_diff = render_text(
            Path::new("sam.safetensors"),
            w.tensors().len(),
            w.total_numel(),
            None,
            None,
            &families,
            None,
            None,
            Some(&[diff]),
            None,
            5,
        );
        assert!(txt_diff.contains("top_rows=[1:"), "{txt_diff}");
    }
}
