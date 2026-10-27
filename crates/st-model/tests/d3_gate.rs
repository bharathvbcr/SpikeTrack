//! D3 gate (gp-detector-and-mask-head): image mode end to end against upstream
//! on 20 VISEM frames: upstream's preprocessed image -> our backbone, tokenizer,
//! text encoder and detector -> masks. Gate: mean mask IoU >= 0.95 over the
//! queries upstream keeps at probability 0.5.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... SPIKETRACK_DETECTOR_FIXTURES=fixtures/detector
//!  cargo test -p st-model --release --test d3_gate -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{Autocast, AutocastGuard, AutocastMode, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use ojas_io::SafeTensors;
use st_model::detector::{Detector, DetectorTaps, DetectorWeights, D, MASK, QUERIES};
use st_model::{host, Backbone, Neck, Taps, Text, Tokenizer};
use st_weights::{Checkpoint, Sam31Weights};

const FRAMES: usize = 20;
const PROMPT: &str = "cell";
const GATE: f64 = 0.95;

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

fn iou(a: &[f32], b: &[f32]) -> f64 {
    let (mut i, mut u) = (0u64, 0u64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x > 0.0, y > 0.0);
        i += u64::from(x && y);
        u += u64::from(x || y);
    }
    if u == 0 { 1.0 } else { i as f64 / u as f64 }
}

#[test]
#[ignore = "needs the gated checkpoint, tokenizer files and detector fixtures"]
fn image_mode_masks_match_upstream_on_20_visem_frames() {
    d3(&CpuBackend::with_threads(Budget::new(28 << 30), 8).expect("cpu backend"), "CPU, 8 threads", || None);
}

#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint, tokenizer files and detector fixtures"]
fn image_mode_masks_match_upstream_on_20_visem_frames_on_metal() {
    d3(&MetalBackend::new(Budget::new(28 << 30)).expect("metal backend"), "Metal", || None);
}

/// The same gate with each frame's trunk, detector neck and detector inside an Ojas
/// Autocast bf16 region (matmul-class operands and outputs rounded to bf16; norms,
/// softmax and GELU stay f32). The text prompt is encoded once, outside the region.
#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint, tokenizer files and detector fixtures"]
fn image_mode_masks_match_upstream_on_20_visem_frames_on_metal_bf16() {
    let be = Autocast::new(MetalBackend::new(Budget::new(28 << 30)).expect("metal backend"));
    d3(&be, "Metal, Autocast bf16", || Some(be.autocast_region(AutocastMode::Bf16).expect("bf16 region")));
}

fn d3<B: Backend>(be: &B, label: &str, region: impl Fn() -> Option<AutocastGuard>) {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let backbone = Backbone::new(be, &w.vision).expect("backbone");
    let text = Text::new(be, &w.text).expect("text");
    let det = Detector::new(
        be,
        DetectorWeights { transformer: &w.detector_transformer, geometry: &w.geometry_encoder, segmentation: &w.segmentation_head, scoring: &w.dot_prod_scoring },
    )
    .expect("detector");
    let dir = env_path("SPIKETRACK_MODEL_DIR");
    let tok = Tokenizer::from_hf_files(&dir.join("vocab.json"), &dir.join("merges.txt")).expect("tokenizer");

    // The prompt once: tokenize, encode, keep the valid tokens.
    let ids = tok.tokenize(PROMPT).expect("tokenize");
    let tf = text.encode(&[ids]).expect("text encode");
    let resized = tf.resized.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
    let valid: Vec<f32> = (0..32).filter(|&t| !tf.padding[t]).flat_map(|t| resized[t * D..(t + 1) * D].to_vec()).collect();
    let prompt = Tensor::from_f32(&valid, &[1, valid.len() / D, D], be.budget()).expect("prompt");

    let fx = SafeTensors::open(&env_path("SPIKETRACK_DETECTOR_FIXTURES").join("gate.safetensors")).expect("gate fixtures");
    let f = |name: &str| fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
    let mut rows = Vec::new();
    let mut frame_ious = Vec::new();
    for i in 0..FRAMES {
        let (shape, img) = f(&format!("f{i}.in.image"));
        let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
        let image = be.upload(&Tensor::from_f32(&img, &shape, be.budget()).expect("t")).expect("up");
        let t0 = Instant::now();
        let guard = region();
        let trunk = backbone.trunk(&image, &[], &mut Taps::default()).expect("trunk");
        let fpn = backbone.neck(Neck::Detector, &trunk).expect("neck");
        let out = det.forward(&fpn, &prompt, &mut DetectorTaps::default()).expect("detector");
        drop(guard);
        be.sync().expect("sync");
        let secs = t0.elapsed().as_secs_f64();

        let ref_logits = f(&format!("f{i}.out.pred_logits")).1;
        let ref_presence = f(&format!("f{i}.out.presence_logit_dec")).1[0];
        let ref_prob: Vec<f32> = ref_logits.iter().map(|&l| host::sigmoid(l) * host::sigmoid(ref_presence)).collect();
        let kept: Vec<usize> = fx.read_i64(&format!("f{i}.out.kept_queries")).expect("kept").1.into_iter().map(|q| q as usize).collect();
        let kept_masks = f(&format!("f{i}.out.kept_masks")).1;
        let ours = out.pred_masks.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
        let n = MASK * MASK;
        let mut ious = Vec::new();
        for (j, &q) in kept.iter().enumerate() {
            if ref_prob[q] > 0.5 {
                ious.push(iou(&ours[q * n..(q + 1) * n], &kept_masks[j * n..(j + 1) * n]));
            }
        }
        let mean = if ious.is_empty() { 1.0 } else { ious.iter().sum::<f64>() / ious.len() as f64 };
        let min = ious.iter().copied().fold(1.0f64, f64::min);
        let got_prob = out.probabilities();
        let ref_keep: Vec<usize> = (0..QUERIES).filter(|&q| ref_prob[q] > 0.5).collect();
        let our_keep: Vec<usize> = (0..QUERIES).filter(|&q| got_prob[q] > 0.5).collect();
        let sym_diff = ref_keep.iter().filter(|q| !our_keep.contains(q)).count() + our_keep.iter().filter(|q| !ref_keep.contains(q)).count();
        let max_dp = (0..QUERIES).map(|q| (f64::from(got_prob[q]) - f64::from(ref_prob[q])).abs()).fold(0.0, f64::max);
        rows.push(format!("| {i} | {} | {} | {sym_diff} | {max_dp:.2e} | {mean:.4} | {min:.4} | {secs:.1} |", ref_keep.len(), our_keep.len()));
        frame_ious.push(mean);
    }
    println!("\n| frame | upstream kept @0.5 | ours | kept-set differences | max abs prob diff | mean mask IoU | min mask IoU | s ({label}) |\n|---|---|---|---|---|---|---|---|");
    rows.iter().for_each(|r| println!("{r}"));
    let overall = frame_ious.iter().sum::<f64>() / frame_ious.len() as f64;
    println!("\n[{label}] D3 gate: mean mask IoU {overall:.4} over {FRAMES} frames (gate >= {GATE}), prompt {PROMPT:?}");
    assert!(overall >= GATE, "D3 gate failed: {overall}");
}
