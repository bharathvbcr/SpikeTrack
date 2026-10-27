//! The text-prompted video session (tracker/session.rs: detector, multiplex tracker and
//! bookkeeping) against upstream's builder-path model on the first 32 frames of VISEM video 11
//! with the prompt "cell", forward only (tools/reference/detect_track_reference.py).
//!
//! Our model runs on the oracle's own input frames (`frames`, F16 `[T, 3, 1008, 1008]`), so no
//! image decoding or resizing is compared. For every visit of upstream's
//! `_det_track_one_frame_impl` (`f{t}.v{k}`; the prompt frame has two) the table reports stage by
//! stage:
//! - detections: kept-set symmetric difference, max score and box difference, mean IoU of the
//!   kept queries' masks (upstream detects frame 0's second visit in a 16-frame batch and later
//!   frames in 16-frame chunks; ours is batch 1, so scores move in the last digits and a query
//!   near 0.4 or 0.65 can flip: that is the one place an id-set miss can come from the detector);
//! - propagation: cosine of the tracker's low-res mask logits and of the object-score logits;
//! - association and plan: exact (flags, indices, ids, births, removals, recondition picks,
//!   unconfirmed ids), from the oracle's JSON;
//! - memory update: cosine of the masks fed to it (after recondition and suppression), then per
//!   tracker state the stored memory features and object pointers (cosine; the features are
//!   bf16-rounded, so 1.0 is not expected);
//! - `build_outputs`: id sets exact, mean IoU of the per-id masks at video size.
//! Then every yielded frame: id sets exact, per-id mask IoU, max probability and box difference.
//!
//! Gates (asserted): identical id sets on every yielded frame, and the mean per-id mask IoU over
//! all yielded (frame, id) pairs >= 0.95 (the D3 and D5 image/VOS gate level). Everything else
//! is reported with an informational tolerance (detections IoU 0.95, propagation cosine 0.999,
//! memory cosine 0.99 for bf16-rounded features) and a `LOW` marker, because a deviation in
//! one stage is exactly what the table is for; it is not asserted so that one table shows
//! every stage of every visit.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... SPIKETRACK_DT_FIXTURES=fixtures/detect_track
//!  [SPIKETRACK_DT_VIDEO=11] cargo test -p st-model --release --test detect_track_parity -- --ignored --nocapture --test-threads=1`
//! (the fixtures come from `tools/locked.sh dt-ref bash tools/jobs/detect_track_ref.sh`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::{parse_json, JsonValue, SafeTensors};
use st_model::tracker::associate::Bitmask;
use st_model::tracker::session::{Session, SessionConfig, VisitTrace, YieldedFrame};
use st_model::tracker::vos::Vos;
use st_model::{Backbone, Detector, DetectorWeights, Text, Tokenizer};
use st_weights::{Checkpoint, Sam31Weights};

const IMAGE: usize = 1008;
const MASK_PX: usize = 288 * 288;
const GATE_IOU: f64 = 0.95;
const TOL_DET_IOU: f64 = 0.95;
const TOL_PROP_COS: f64 = 0.999;
const TOL_MEM_COS: f64 = 0.99;

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

struct Fixture {
    st: SafeTensors<'static>,
    names: BTreeSet<String>,
}

impl Fixture {
    fn open(path: &std::path::Path) -> Self {
        let st = SafeTensors::open(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e.detail()));
        let names = st.names().map(str::to_string).collect();
        Self { st, names }
    }

    fn has(&self, k: &str) -> bool {
        self.names.contains(k)
    }

    fn f32s(&self, k: &str) -> Option<Vec<f32>> {
        self.has(k).then(|| self.st.read_f32_widened(k).unwrap_or_else(|e| panic!("{k}: {}", e.detail())).1)
    }

    fn i64s(&self, k: &str) -> Option<Vec<i64>> {
        self.has(k).then(|| self.st.read_i64(k).unwrap_or_else(|e| panic!("{k}: {}", e.detail())).1)
    }

    fn u8s(&self, k: &str) -> Option<Vec<u8>> {
        self.has(k).then(|| self.st.read_bytes(k).unwrap_or_else(|e| panic!("{k}: {}", e.detail())))
    }

    fn with_prefix(&self, prefix: &str) -> Vec<&String> {
        self.names.range(prefix.to_string()..).take_while(|n| n.starts_with(prefix)).collect()
    }
}

/// `(cosine, max abs difference)`; two all-zero inputs have cosine 1.
fn stats(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut gg, mut ww, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        max_abs = max_abs.max((g - w).abs());
    }
    let cos = if gg == 0.0 && ww == 0.0 { 1.0 } else { dot / (gg.sqrt() * ww.sqrt()) };
    (cos, max_abs)
}

fn iou_logits(a: &[f32], b: &[f32]) -> f64 {
    let (mut i, mut u) = (0u64, 0u64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x > 0.0, y > 0.0);
        i += u64::from(x && y);
        u += u64::from(x || y);
    }
    if u == 0 {
        1.0
    } else {
        i as f64 / u as f64
    }
}

fn iou_bits(a: &Bitmask, b: &[u8]) -> f64 {
    let (mut i, mut u) = (0u64, 0u64);
    for (p, &y) in b.iter().enumerate() {
        let (x, y) = (a.get(p), y != 0);
        i += u64::from(x && y);
        u += u64::from(x || y);
    }
    if u == 0 {
        1.0
    } else {
        i as f64 / u as f64
    }
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn show(v: Option<f64>, tol: Option<f64>) -> String {
    match v {
        None => "-".to_string(),
        Some(x) => match tol {
            Some(t) if x < t => format!("{x:.5} LOW"),
            _ => format!("{x:.5}"),
        },
    }
}

fn walk<'a>(v: &'a JsonValue, path: &[&str]) -> Option<&'a JsonValue> {
    let mut cur = v;
    for key in path {
        cur = cur.as_object().ok()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)?;
    }
    Some(cur)
}

/// An integer list, from a plain array or the oracle's `{"tensor": [...]}` encoding.
fn ints(v: &JsonValue) -> Option<Vec<i64>> {
    let arr = match v.as_object() {
        Ok(o) => o.iter().find(|(k, _)| k == "tensor").map(|(_, v)| v)?.as_array().ok()?,
        Err(_) => v.as_array().ok()?,
    };
    arr.iter().map(|x| x.as_i64().ok().or_else(|| x.as_f64().ok().map(|f| f as i64))).collect()
}

/// A `{"id": ...}` object as `(id, value)` pairs.
fn keyed<'a>(v: &'a JsonValue) -> Option<Vec<(i64, &'a JsonValue)>> {
    v.as_object().ok()?.iter().map(|(k, v)| Some((k.parse::<i64>().ok()?, v))).collect()
}

/// Exact plan checks against the oracle's JSON; returns (checked, mismatched, notes).
fn check_plan(meta: Option<&JsonValue>, tr: &VisitTrace) -> (usize, usize, Vec<String>) {
    let tag = format!("f{}.v{}", tr.frame, tr.visit);
    let (mut checked, mut bad, mut notes) = (0usize, 0usize, Vec::new());
    let Some(meta) = meta else {
        return (0, 0, vec!["json unreadable".to_string()]);
    };
    let mut cmp = |name: &str, got: Option<String>, want: Option<String>| {
        if let (Some(g), Some(w)) = (got, want) {
            checked += 1;
            if g != w {
                bad += 1;
                notes.push(format!("{name}: ours {g} upstream {w}"));
            }
        }
    };
    let plan = |k: &str| walk(meta, &["plan", tag.as_str(), k]);
    let births_ids: Vec<i64> = tr.plan.births.iter().map(|b| b.id).collect();
    let births_det: Vec<i64> = tr.plan.births.iter().map(|b| b.det as i64).collect();
    cmp("new_det_obj_ids", Some(format!("{births_ids:?}")), plan("new_det_obj_ids").and_then(ints).map(|v| format!("{v:?}")));
    cmp("new_det_fa_inds", Some(format!("{births_det:?}")), plan("new_det_fa_inds").and_then(ints).map(|v| format!("{v:?}")));
    cmp("removed", Some(format!("{:?}", tr.plan.removed_ids)), plan("obj_ids_newly_removed").and_then(ints).map(|mut v| {
        v.sort_unstable();
        format!("{v:?}")
    }));
    cmp(
        "dropped",
        Some(tr.plan.num_dropped_due_to_limit.to_string()),
        plan("num_obj_dropped_due_to_limit").and_then(|v| v.as_i64().ok()).map(|v| v.to_string()),
    );
    cmp("unmatched", Some(format!("{:?}", tr.plan.realized.unmatched_trk_ids)), plan("unmatched_trk_obj_ids").and_then(ints).map(|v| format!("{v:?}")));
    let mut recond: Vec<(i64, i64)> = tr.plan.realized.trk_id_to_max_iou_high_conf_det.iter().map(|&(i, d)| (i, d as i64)).collect();
    recond.sort_unstable();
    cmp(
        "trk_id_to_max_iou_high_conf_det",
        Some(format!("{recond:?}")),
        plan("trk_id_to_max_iou_high_conf_det").and_then(keyed).and_then(|kv| {
            let mut out: Vec<(i64, i64)> = kv.iter().map(|(k, v)| Some((*k, v.as_i64().ok()?))).collect::<Option<_>>()?;
            out.sort_unstable();
            Some(format!("{out:?}"))
        }),
    );
    let matched: BTreeMap<i64, Vec<i64>> = tr.plan.realized.det_to_matched_trk_ids.iter().map(|(d, ids)| (*d as i64, ids.clone())).collect();
    cmp(
        "det_to_matched_trk_obj_ids",
        Some(format!("{matched:?}")),
        plan("det_to_matched_trk_obj_ids").and_then(keyed).and_then(|kv| {
            let out: BTreeMap<i64, Vec<i64>> = kv.iter().map(|(k, v)| Some((*k, ints(v)?))).collect::<Option<_>>()?;
            Some(format!("{out:?}"))
        }),
    );
    // Upstream lists every object of a touched state; ours are the picks. Each pick must be listed.
    let listed = plan("reconditioned_obj_ids").and_then(ints).map(|r| tr.plan.recondition.iter().all(|(id, _)| r.contains(id)).to_string());
    cmp("reconditioned picks are listed upstream", listed.as_ref().map(|_| "true".to_string()), listed);
    let md = |path: &[&str]| {
        let mut p = vec!["metadata", tag.as_str()];
        p.extend_from_slice(path);
        walk(meta, &p)
    };
    cmp("obj_ids_all_gpu", Some(format!("{:?}", tr.plan.obj_ids)), md(&["obj_ids_all_gpu"]).and_then(ints).map(|v| format!("{v:?}")));
    let status = md(&["rank0_metadata", "masklet_confirmation", "status"]).and_then(ints);
    let up_unconfirmed = status.map(|s| {
        // MaskletConfirmationStatus: UNCONFIRMED = 1, CONFIRMED = 2.
        let ids = md(&["obj_ids_all_gpu"]).and_then(ints).unwrap_or_default();
        format!("{:?}", ids.iter().zip(&s).filter(|(_, st)| **st == 1).map(|(&id, _)| id).collect::<Vec<_>>())
    });
    cmp("unconfirmed", Some(format!("{:?}", tr.plan.unconfirmed_ids)), up_unconfirmed);
    (checked, bad, notes)
}

/// Exact association checks against the recorded tensors; returns (checked, mismatched).
fn check_assoc(fx: &Fixture, tr: &VisitTrace) -> (usize, usize) {
    let tag = format!("f{}.v{}", tr.frame, tr.visit);
    let a = &tr.association;
    let (mut checked, mut bad) = (0, 0);
    let mut flags = |name: &str, got: &[bool]| {
        if let Some(want) = fx.u8s(&format!("{tag}.assoc.{name}")) {
            checked += 1;
            if want.len() != got.len() || want.iter().zip(got).any(|(&w, &g)| (w != 0) != g) {
                bad += 1;
            }
        }
    };
    flags("trk_is_unmatched", &a.trk_is_unmatched);
    flags("trk_is_nonempty", &a.trk_is_nonempty);
    flags("is_new_det", &a.is_new_det);
    flags("det_is_high_conf", &a.det_is_high_conf);
    flags("det_is_high_iou", &a.det_is_high_overlap);
    flags("im_mask", &a.im_mask);
    // `argmax` is only meaningful with tracks.
    if !a.trk_is_nonempty.is_empty() {
        if let Some(want) = fx.i64s(&format!("{tag}.assoc.det_to_max_iou_trk_idx")) {
            checked += 1;
            if want.len() != a.det_to_max_trk.len() || want.iter().zip(&a.det_to_max_trk).any(|(&w, &g)| w != g as i64) {
                bad += 1;
            }
        }
    }
    (checked, bad)
}

struct VisitRow {
    tag: String,
    cells: Vec<String>,
}

fn compare_visit(fx: &Fixture, meta: Option<&JsonValue>, tr: &VisitTrace, hw: (usize, usize)) -> VisitRow {
    let tag = format!("f{}.v{}", tr.frame, tr.visit);
    // Detections (query order, before the kept-first reorder).
    let kept_diff = fx.u8s(&format!("{tag}.det.keep")).map(|k| (0..tr.det_keep.len()).filter(|&q| (k[q] != 0) != tr.det_keep[q]).count());
    let score_diff = fx.f32s(&format!("{tag}.det.scores")).map(|s| stats(&tr.det_probs, &s).1);
    let box_diff = fx.f32s(&format!("{tag}.det.bbox")).map(|b| stats(&tr.det_boxes_xyxy, &b).1);
    let kept_q: Vec<usize> = (0..tr.det_keep.len()).filter(|&q| tr.det_keep[q]).collect();
    let mut det_ious = Vec::new();
    if let (Some(idx), Some(masks)) = (fx.i64s(&format!("{tag}.det.kept_index")), fx.f32s(&format!("{tag}.det.kept_masks"))) {
        for (j, &q) in idx.iter().enumerate() {
            if let Some(pos) = kept_q.iter().position(|&x| x as i64 == q) {
                det_ious.push(iou_logits(&tr.det_kept_masks[pos * MASK_PX..(pos + 1) * MASK_PX], &masks[j * MASK_PX..(j + 1) * MASK_PX]));
            }
        }
    }
    // Propagation.
    let prop_cos = fx.f32s(&format!("{tag}.prop.low_res_masks")).filter(|m| m.len() == tr.prop_masks.len() && !m.is_empty()).map(|m| stats(&tr.prop_masks, &m).0);
    let score_cos = fx.f32s(&format!("{tag}.prop.obj_scores")).filter(|m| m.len() == tr.prop_scores.len() && !m.is_empty()).map(|m| stats(&tr.prop_scores, &m).0);
    // Association and plan.
    let (a_checked, a_bad) = check_assoc(fx, tr);
    let (p_checked, p_bad, notes) = check_plan(meta, tr);
    // Memory update inputs and what each state stored.
    let mem_cos = fx.f32s(&format!("{tag}.mem.kw.low_res_masks")).filter(|m| m.len() == tr.mem_masks.len() && !m.is_empty()).map(|m| stats(&tr.mem_masks, &m).0);
    let (mut feat_cos, mut ptr_cos) = (Vec::new(), Vec::new());
    for (si, st) in tr.states.iter().enumerate() {
        for kind in ["cond_frame_outputs", "non_cond_frame_outputs"] {
            let base = format!("{tag}.mem.state{si}.{kind}");
            if let (Some(want), Some(got)) = (fx.f32s(&format!("{base}.maskmem_features")), st.maskmem.as_ref()) {
                if want.len() == got.len() {
                    feat_cos.push(stats(got, &want).0);
                }
            }
            if let Some(want) = fx.f32s(&format!("{base}.obj_ptr")) {
                if want.len() == st.obj_ptr.len() {
                    ptr_cos.push(stats(&st.obj_ptr, &want).0);
                }
            }
        }
    }
    // build_outputs.
    let prefix = format!("{tag}.out.");
    let ref_ids: BTreeSet<i64> = fx.with_prefix(&prefix).iter().filter_map(|n| n[prefix.len()..].parse().ok()).collect();
    let our_ids: BTreeSet<i64> = tr.outputs.iter().map(|(id, _)| *id).collect();
    let mut out_ious = Vec::new();
    for (id, mask) in &tr.outputs {
        if let Some(m) = fx.u8s(&format!("{prefix}{id}")) {
            if m.len() == hw.0 * hw.1 {
                out_ious.push(iou_bits(mask, &m));
            }
        }
    }
    let out_ids_eq = if ref_ids.is_empty() && our_ids.is_empty() { "yes".to_string() } else if ref_ids == our_ids { "yes".to_string() } else { format!("NO {our_ids:?} vs {ref_ids:?}") };
    let mut cells = vec![
        tr.states.len().to_string(),
        kept_diff.map_or("-".to_string(), |d| d.to_string()),
        score_diff.map_or("-".to_string(), |d| format!("{d:.1e}")),
        box_diff.map_or("-".to_string(), |d| format!("{d:.1e}")),
        show(mean(&det_ious), Some(TOL_DET_IOU)),
        show(prop_cos, Some(TOL_PROP_COS)),
        show(score_cos, Some(TOL_PROP_COS)),
        format!("{}/{}", a_checked - a_bad, a_checked),
        format!("{}/{}", p_checked - p_bad, p_checked),
        show(mem_cos, Some(TOL_PROP_COS)),
        show(feat_cos.iter().copied().reduce(f64::min), Some(TOL_MEM_COS)),
        show(ptr_cos.iter().copied().reduce(f64::min), Some(TOL_MEM_COS)),
        out_ids_eq,
        show(mean(&out_ious), Some(GATE_IOU)),
    ];
    if !notes.is_empty() {
        cells.push(notes.join("; "));
    }
    VisitRow { tag, cells }
}

#[test]
#[ignore = "needs the gated checkpoint, tokenizer files and the detect_track fixtures"]
fn session_matches_upstream_on_visem_11() {
    let video = std::env::var("SPIKETRACK_DT_VIDEO").unwrap_or_else(|_| "11".to_string());
    let dir = env_path("SPIKETRACK_DT_FIXTURES");
    let fx = Fixture::open(&dir.join(format!("{video}.safetensors")));
    let meta_text = std::fs::read_to_string(dir.join(format!("{video}.json"))).expect("json fixture");
    let meta = match parse_json(&meta_text) {
        Ok(m) => Some(m),
        Err(e) => {
            println!("the oracle's JSON could not be parsed ({}): plan checks are skipped", e.detail());
            None
        }
    };
    let prompt = meta.as_ref().and_then(|m| walk(m, &["prompt"])).and_then(|p| p.as_str().ok().map(str::to_string)).unwrap_or_else(|| "cell".to_string());
    let (frames_shape, frames) = fx.st.read_f32_widened("frames").unwrap_or_else(|e| panic!("frames: {}", e.detail()));
    let t_total = frames_shape[0] as usize;
    assert_eq!(&frames_shape[1..], &[3, IMAGE as u64, IMAGE as u64], "frames shape");
    let hw = match meta.as_ref().and_then(|m| walk(m, &["hw"])).and_then(ints) {
        Some(v) if v.len() == 2 => (v[0] as usize, v[1] as usize),
        _ => {
            let probe = fx.u8s("y0.masks").map(|m| m.len()).expect("no hw in the json and no y0.masks");
            panic!("video size unknown (y0.masks holds {probe} bytes)");
        }
    };

    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let be = CpuBackend::with_threads(Budget::new(28 << 30), 8).expect("cpu backend");
    let backbone = Backbone::new(&be, &w.vision).expect("backbone");
    let text = Text::new(&be, &w.text).expect("text");
    let detector = Detector::new(
        &be,
        DetectorWeights { transformer: &w.detector_transformer, geometry: &w.geometry_encoder, segmentation: &w.segmentation_head, scoring: &w.dot_prod_scoring },
    )
    .expect("detector");
    let vos = Vos::new(&be, &backbone, &w.tracker).expect("tracker");
    let mdir = env_path("SPIKETRACK_MODEL_DIR");
    let tok = Tokenizer::from_hf_files(&mdir.join("vocab.json"), &mdir.join("merges.txt")).expect("tokenizer");
    let prompt_t = Session::encode_prompt(&be, &text, &tok, &prompt).expect("prompt");

    let mut session = Session::new(&vos, &detector, prompt_t, SessionConfig::new(t_total, hw.0, hw.1));
    session.set_trace(true);

    let header = ["states", "kept diff", "score dmax", "box dmax", "det IoU", "prop cos", "score cos", "assoc ok", "plan ok", "mem-in cos", "mem cos min", "ptr cos min", "out ids", "out IoU"];
    println!("\nvideo {video}, {t_total} frames, prompt {prompt:?}, {}x{} (HxW)\n", hw.0, hw.1);
    println!("| visit | {} |\n|---|{}|", header.join(" | "), vec!["---"; header.len()].join("|"));
    let mut yields: Vec<YieldedFrame> = Vec::new();
    let mut yield_rows = Vec::new();
    let (mut pairs_iou, mut id_set_misses) = (Vec::new(), Vec::new());
    for t in 0..t_total {
        let px = 3 * IMAGE * IMAGE;
        let image = be.upload(&Tensor::from_f32(&frames[t * px..(t + 1) * px], &[1, 3, IMAGE, IMAGE], be.budget()).expect("frame")).expect("upload");
        let t0 = Instant::now();
        let released = session.push_frame(&image).unwrap_or_else(|e| panic!("frame {t}: {e}"));
        be.sync().expect("sync");
        let secs = t0.elapsed().as_secs_f64();
        for tr in session.take_traces() {
            let row = compare_visit(&fx, meta.as_ref(), &tr, hw);
            let (notes, cells) = if row.cells.len() > header.len() { (row.cells.last().cloned(), &row.cells[..header.len()]) } else { (None, &row.cells[..]) };
            println!("| {} | {} |", row.tag, cells.join(" | "));
            if let Some(n) = notes {
                println!("|   | plan: {n} |");
            }
        }
        println!("|   | frame {t}: {secs:.1} s, {} state(s), ids {:?} |", session.states().len(), session.book().obj_ids());
        for y in released {
            // Compare the released frame with upstream's yield for the same frame index.
            let key = |s: &str| format!("y{}.{s}", y.frame);
            let ref_ids = fx.i64s(&key("obj_ids")).unwrap_or_default();
            let eq = ref_ids == y.obj_ids;
            if !eq {
                id_set_misses.push((y.frame, y.obj_ids.clone(), ref_ids.clone()));
            }
            let mut ious = Vec::new();
            if let Some(masks) = fx.u8s(&key("masks")) {
                let hw_px = hw.0 * hw.1;
                for (k, id) in y.obj_ids.iter().enumerate() {
                    if let Some(r) = ref_ids.iter().position(|x| x == id) {
                        if masks.len() >= (r + 1) * hw_px {
                            let v = iou_bits(&y.masks[k], &masks[r * hw_px..(r + 1) * hw_px]);
                            ious.push(v);
                            pairs_iou.push(v);
                        }
                    }
                }
            }
            let prob_d = fx.f32s(&key("probs")).filter(|p| p.len() == y.probs.len()).map(|p| stats(&y.probs, &p).1);
            let box_d = fx.f32s(&key("boxes_xywh")).filter(|b| b.len() == y.boxes_xywh.len()).map(|b| stats(&y.boxes_xywh, &b).1);
            yield_rows.push(format!(
                "| y{} | {} | {} | {} | {} | {} |",
                y.frame,
                if eq { "yes".to_string() } else { format!("NO {:?} vs {ref_ids:?}", y.obj_ids) },
                y.obj_ids.len(),
                show(mean(&ious), Some(GATE_IOU)),
                prob_d.map_or("-".to_string(), |d| format!("{d:.1e}")),
                box_d.map_or("-".to_string(), |d| format!("{d:.1e}")),
            ));
            yields.push(y);
        }
    }
    yields.extend(session.finish());

    println!("\n| yielded frame | id set equal | objects | mean mask IoU | prob dmax | box dmax |\n|---|---|---|---|---|---|");
    yield_rows.iter().for_each(|r| println!("{r}"));
    let overall = mean(&pairs_iou);
    println!(
        "\nyielded {} frames; id-set misses {}; mean per-id mask IoU {} over {} (frame, id) pairs (gate: all id sets equal and IoU >= {GATE_IOU})",
        yields.len(),
        id_set_misses.len(),
        show(overall, None),
        pairs_iou.len()
    );
    // Upstream yields every frame once; every one of them must have been compared.
    let ref_frames = (0..t_total).filter(|t| fx.has(&format!("y{t}.obj_ids"))).count();
    assert_eq!(yields.len(), ref_frames, "released {} frames, upstream yielded {ref_frames}", yields.len());
    assert!(id_set_misses.is_empty(), "id sets differ on yielded frames: {id_set_misses:?}");
    assert!(overall.is_some_and(|m| m >= GATE_IOU), "mean per-id mask IoU {overall:?} < {GATE_IOU}");
}
