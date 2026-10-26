//! Detector parity against upstream (tools/reference/dump_detector_fixtures.py), from
//! upstream's own detector inputs (FPN levels, text features), stage by stage.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_DETECTOR_FIXTURES=fixtures/detector
//!  cargo test -p st-model --release --test detector_parity -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use st_model::detector::{Detector, DetectorTaps, DetectorWeights, D, MASK, QUERIES, TOKENS};
use st_model::host;
use st_weights::{Checkpoint, Sam31Weights};

const CASES: [&str; 3] = ["visem22_cell", "testimage_shoe", "truck_wheel"];

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

struct Cmp {
    rows: Vec<String>,
    worst: f64,
}

impl Cmp {
    fn check(&mut self, case: &str, stage: &str, got: &[f32], want: &[f32], gate: f64) {
        assert_eq!(got.len(), want.len(), "{case} {stage}: length {} vs {}", got.len(), want.len());
        let (mut dot, mut gg, mut ww, mut max_abs, mut ref_max) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&g, &w) in got.iter().zip(want) {
            let (g, w) = (f64::from(g), f64::from(w));
            assert!(g.is_finite(), "{case} {stage}: non-finite");
            dot += g * w;
            gg += g * g;
            ww += w * w;
            max_abs = max_abs.max((g - w).abs());
            ref_max = ref_max.max(w.abs());
        }
        let cos = if gg == 0.0 && ww == 0.0 { 1.0 } else { dot / (gg.sqrt() * ww.sqrt()) };
        self.worst = self.worst.min(cos);
        let ok = cos > gate;
        self.rows.push(format!("| {case} | {stage} | {cos:.7} | {max_abs:.3e} | {:.3e} | {} |", max_abs / ref_max.max(1e-30), if ok { "pass" } else { "FAIL" }));
    }
}

fn f(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    st.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail())).1
}

fn host(t: &Tensor, be: &CpuBackend) -> Vec<f32> {
    t.to_host(be.budget()).expect("host").to_f32_vec().expect("f32")
}

/// Mean IoU of binarized (logit > 0) masks over the queries upstream keeps at 0.5.
fn mask_iou(got: &[f32], want: &[f32], keep: &[usize]) -> f64 {
    let n = MASK * MASK;
    let ious: Vec<f64> = keep
        .iter()
        .map(|&q| {
            let (g, w) = (&got[q * n..(q + 1) * n], &want[q * n..(q + 1) * n]);
            let (mut i, mut u) = (0u64, 0u64);
            for (&a, &b) in g.iter().zip(w) {
                let (a, b) = (a > 0.0, b > 0.0);
                i += u64::from(a && b);
                u += u64::from(a || b);
            }
            if u == 0 { 1.0 } else { i as f64 / u as f64 }
        })
        .collect();
    ious.iter().sum::<f64>() / ious.len().max(1) as f64
}

#[test]
#[ignore = "needs the gated checkpoint and detector fixtures: set SPIKETRACK_WEIGHTS and SPIKETRACK_DETECTOR_FIXTURES"]
fn detector_matches_upstream_stage_by_stage() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let dir = env_path("SPIKETRACK_DETECTOR_FIXTURES");
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu backend");
    let det = Detector::new(
        &be,
        DetectorWeights { transformer: &w.detector_transformer, geometry: &w.geometry_encoder, segmentation: &w.segmentation_head, scoring: &w.dot_prod_scoring },
    )
    .expect("detector");
    // Every detector tensor is read by the port, apart from the ones upstream's text path never touches.
    let groups = [&w.detector_transformer, &w.geometry_encoder, &w.segmentation_head, &w.dot_prod_scoring];
    let unused: Vec<&String> = groups.iter().flat_map(|g| g.tensors.keys()).filter(|n| Detector::<CpuBackend>::unused_by_text_prompts(n)).collect();
    println!("{} detector tensors not read on the text path (point/box/mask prompt encoders, pixel_decoder stage 3)", unused.len());

    let mut cmp = Cmp { rows: Vec::new(), worst: f64::INFINITY };
    let mut iou_rows = Vec::new();
    for case in CASES {
        let st = SafeTensors::open(&dir.join(format!("{case}.safetensors"))).expect("fixture");
        let fpn: [Tensor; 3] = [0, 1, 2].map(|i| {
            let (shape, data) = st.read_f32(&format!("in.fpn{i}")).expect("fpn");
            let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
            be.upload(&Tensor::from_f32(&data, &shape, be.budget()).expect("t")).expect("up")
        });
        // Upstream's position encoding for the 72x72 level vs ours.
        let pos_chw = f(&st, "in.pos2");
        cmp.check(case, "sine pos 72x72", &host::sine_position_map(72, 72), &pos_chw, 0.99999);
        // Valid text tokens, [32, 1, 256] seq first with a [1, 32] pad mask.
        let text = f(&st, "in.text");
        let pad = st.read_bytes("in.text_pad").expect("pad");
        let valid: Vec<f32> = (0..32).filter(|&t| pad[t] == 0).flat_map(|t| text[t * D..(t + 1) * D].to_vec()).collect();
        let len = valid.len() / D;
        let text_t = Tensor::from_f32(&valid, &[1, len, D], be.budget()).expect("text");
        let mut taps = DetectorTaps::default();
        let t0 = Instant::now();
        let out = det.forward(&fpn, &text_t, &mut taps).expect("forward");
        be.sync().expect("sync");
        let secs = t0.elapsed().as_secs_f64();

        cmp.check(case, "geometry CLS", taps.geo.as_ref().expect("geo"), &f(&st, "geo.0"), 0.9999);
        for (i, t) in taps.enc_layers.iter().enumerate() {
            cmp.check(case, &format!("encoder layer {i}"), &host(t, &be), &f(&st, &format!("enc.layer{i}")), 0.999);
        }
        let hs = f(&st, "dec.0"); // [6, 200, 1, 256]
        let refs = f(&st, "dec.1"); // [6, 200, 1, 4]
        let pres = f(&st, "dec.2"); // [6, 1, 1]
        let scores = f(&st, "scoring"); // [6, 1, 200, 1]
        for l in 0..6 {
            cmp.check(case, &format!("decoder {l} queries"), &taps.hs[l], &hs[l * QUERIES * D..(l + 1) * QUERIES * D], 0.999);
            cmp.check(case, &format!("decoder {l} ref boxes in"), &taps.ref_boxes[l], &refs[l * QUERIES * 4..(l + 1) * QUERIES * 4], 0.9999);
            cmp.check(case, &format!("decoder {l} scores"), &taps.scores[l], &scores[l * QUERIES..(l + 1) * QUERIES], 0.999);
            let d = (f64::from(taps.presence_logits[l]) - f64::from(pres[l])).abs();
            assert!(d < 0.05, "{case} presence logit layer {l}: {} vs {}", taps.presence_logits[l], pres[l]);
        }
        cmp.check(case, "seg prompt cross-attn", &host(taps.seg_cross_attn.as_ref().expect("tap"), &be), &f(&st, "seg.cross_attend_prompt.0"), 0.999);
        cmp.check(case, "pixel embed", &host(taps.pixel_embed.as_ref().expect("tap"), &be), &f(&st, "seg.pixel_embed"), 0.999);
        cmp.check(case, "semantic seg", &host(&out.semantic_seg, &be), &f(&st, "seg.semantic_seg"), 0.999);
        cmp.check(case, "pred_logits", &out.pred_logits, &f(&st, "out.pred_logits"), 0.999);
        cmp.check(case, "pred_boxes", &out.pred_boxes, &f(&st, "out.pred_boxes"), 0.9999);
        let masks = host(&out.pred_masks, &be);
        let want_masks = f(&st, "out.pred_masks");
        cmp.check(case, "pred_masks", &masks, &want_masks, 0.999);
        let pres_ref = f(&st, "out.presence_logit_dec")[0];
        let want_prob: Vec<f32> = f(&st, "out.pred_logits").iter().map(|&l| host::sigmoid(l) * host::sigmoid(pres_ref)).collect();
        let keep: Vec<usize> = (0..QUERIES).filter(|&q| want_prob[q] > 0.5).collect();
        let got_prob = out.probabilities();
        let got_keep: Vec<usize> = (0..QUERIES).filter(|&q| got_prob[q] > 0.5).collect();
        let iou = mask_iou(&masks, &want_masks, &keep);
        iou_rows.push(format!("| {case} | {} | {} | {iou:.4} | {secs:.1} |", keep.len(), got_keep.len()));
        assert_eq!(keep, got_keep, "{case}: kept queries differ");
    }
    let _ = TOKENS;
    println!("\n| case | stage | cosine | max abs | max abs / max ref | gate |\n|---|---|---|---|---|---|");
    cmp.rows.iter().for_each(|r| println!("{r}"));
    println!("\n| case | upstream kept @0.5 | ours kept | mean mask IoU (kept) | detector s (CPU) |\n|---|---|---|---|---|");
    iou_rows.iter().for_each(|r| println!("{r}"));
    println!("\nworst cosine {:.7}", cmp.worst);
    assert!(cmp.rows.iter().all(|r| r.ends_with("pass |")), "a stage failed its cosine gate");
}
