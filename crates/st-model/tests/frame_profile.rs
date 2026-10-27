//! Per-op wall-time breakdown of one frame on a backend (default Metal), for choosing
//! the next kernel (requested by ojas-78 for D6). One frame = what the text-prompted
//! video path runs per frame: the PE ViT trunk, the detector neck + detector with a
//! cached text prompt, and a multiplex tracker step (propagation neck, memory attention,
//! mask decoder, memory encoder). The tracker is measured on DAVIS judo frames 1, 3 and 7
//! (2 objects), where memory attention reads F = 1, 3 and 7 memory frames (the
//! conditioning frame plus up to six recent ones); the detector run is on frame 2.
//!
//! The tracker step runs through `Vos`, which computes its own trunk: each tracker frame's
//! rows therefore include a trunk (identical ops and shapes to the detector run's `trunk`
//! stage), which the production Session computes once per frame. An unprofiled pass is
//! timed first; the profiled pass syncs after every op (wall time, not GPU time). The
//! `_bf16` variant runs every frame inside an Ojas Autocast bf16 region (the text prompt is
//! encoded once, outside it). Single run, machine shared: results are labelled contended
//! and the load average is recorded by tools/jobs/frame_profile.sh.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... SPIKETRACK_TRACKER_FIXTURES=fixtures/tracker/judo.safetensors
//!  cargo test -p st-model --release --test frame_profile -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{Autocast, AutocastMode, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use st_model::detector::{Detector, DetectorTaps, DetectorWeights, D};
use st_model::profile::Profiled;
use st_model::tracker::vos::Vos;
use st_model::{Backbone, Neck, Taps, Text, Tokenizer};
use st_weights::{Checkpoint, Sam31Weights};

const PROMPT: &str = "cell";
/// Tracker frames measured, with the memory depth F each one reads.
const MEASURED: [(usize, usize); 3] = [(1, 1), (3, 3), (7, 7)];
const FRAMES: usize = 8;

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

fn tracker_stage(t: usize) -> Option<String> {
    MEASURED.iter().find(|(f, _)| *f == t).map(|(f, depth)| format!("tracker frame {f} (F={depth}; its trunk + propagation neck + tracker step)"))
}

/// Builds the models on `be`, encodes the prompt, then (inside `mode`'s autocast region)
/// warms up the detector on frame 1, runs it on frame 2, and runs the tracker over frames
/// 0..8. `stage` labels what follows ("warm-up" for unmeasured work). Returns the detector
/// run's ms and each measured tracker frame's (frame, F, ms).
fn measure<B: Backend>(be: &B, w: &Sam31Weights, mode: AutocastMode, mut stage: impl FnMut(&str)) -> (f64, Vec<(usize, usize, f64)>) {
    let backbone = Backbone::new(be, &w.vision).expect("backbone");
    let text = Text::new(be, &w.text).expect("text");
    let det = Detector::new(
        be,
        DetectorWeights { transformer: &w.detector_transformer, geometry: &w.geometry_encoder, segmentation: &w.segmentation_head, scoring: &w.dot_prod_scoring },
    )
    .expect("detector");
    let vos = Vos::new(be, &backbone, &w.tracker).expect("tracker");
    let dir = env_path("SPIKETRACK_MODEL_DIR");
    let tok = Tokenizer::from_hf_files(&dir.join("vocab.json"), &dir.join("merges.txt")).expect("tokenizer");
    stage("text (once per prompt)");
    let tf = text.encode(&[tok.tokenize(PROMPT).expect("tokenize")]).expect("text encode");
    let resized = tf.resized.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
    let valid: Vec<f32> = (0..32).filter(|&t| !tf.padding[t]).flat_map(|t| resized[t * D..(t + 1) * D].to_vec()).collect();
    let prompt = Tensor::from_f32(&valid, &[1, valid.len() / D, D], be.budget()).expect("prompt");

    let fx = SafeTensors::open(&env_path("SPIKETRACK_TRACKER_FIXTURES")).expect("fixtures");
    let h = |name: &str| fx.read_f32_widened(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
    let (mshape, masks0) = h("f0.input_masks");
    let (n, ht, wd) = (mshape[0] as usize, mshape[1] as usize, mshape[2] as usize);
    let frames: Vec<Tensor> = (0..FRAMES)
        .map(|f| be.upload(&Tensor::from_f32(&h(&format!("f{f}.image")).1, &[1, 3, 1008, 1008], be.budget()).expect("t")).expect("up"))
        .collect();

    let _region = be.autocast_region(mode).expect("autocast region");
    let detect = |img: &Tensor| {
        let trunk = backbone.trunk(img, &[], &mut Taps::default()).expect("trunk");
        let fpn = backbone.neck(Neck::Detector, &trunk).expect("neck");
        det.forward(&fpn, &prompt, &mut DetectorTaps::default()).expect("detector");
    };
    stage("warm-up");
    detect(&frames[1]);
    be.sync().expect("sync");
    stage("trunk (detector run)");
    let t0 = Instant::now();
    let trunk = backbone.trunk(&frames[2], &[], &mut Taps::default()).expect("trunk");
    stage("detector neck + detector");
    let fpn = backbone.neck(Neck::Detector, &trunk).expect("neck");
    det.forward(&fpn, &prompt, &mut DetectorTaps::default()).expect("detector");
    be.sync().expect("sync");
    let det_ms = t0.elapsed().as_secs_f64() * 1e3;

    // The sink runs after each frame: label and time the frame that comes next.
    stage("warm-up");
    let mut times = Vec::new();
    let mut t_frame = Instant::now();
    vos.run(&frames, &masks0, n, ht, wd, |out| {
        if let Some((f, depth)) = MEASURED.iter().find(|(f, _)| *f == out.frame) {
            times.push((*f, *depth, t_frame.elapsed().as_secs_f64() * 1e3));
        }
        let next = out.frame + 1;
        stage(&tracker_stage(next).unwrap_or_else(|| "warm-up".into()));
        t_frame = Instant::now();
    })
    .expect("vos run");
    (det_ms, times)
}

fn run<B: Backend>(make: impl Fn() -> B, mode: AutocastMode, label: &str) {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let show = |kind: &str, det_ms: f64, times: &[(usize, usize, f64)]| {
        let tr: Vec<String> = times.iter().map(|(f, d, ms)| format!("frame {f} F={d} {ms:.0} ms")).collect();
        println!("[{label}] {kind}: detector run (trunk + neck + detector) {det_ms:.0} ms; tracker frames (each incl. its trunk + neck + step): {}", tr.join(", "));
    };
    let (det_ms, times) = measure(&make(), &w, mode, |_| {});
    show("unprofiled", det_ms, &times);
    let prof = Profiled::new(make());
    let (pdet, ptimes) = measure(&prof, &w, mode, |s| prof.set_stage(s));
    show("profiled (every op synced)", pdet, &ptimes);
    prof.retain(|c| c.stage != "warm-up");
    println!("\n{}", prof.report());
}

#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint, tokenizer files and the tracker fixture"]
fn frame_profile_metal() {
    run(|| MetalBackend::new(Budget::new(24 << 30)).expect("metal"), AutocastMode::Off, "Metal f32, contended");
}

#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint, tokenizer files and the tracker fixture"]
fn frame_profile_metal_bf16() {
    run(|| Autocast::new(MetalBackend::new(Budget::new(24 << 30)).expect("metal")), AutocastMode::Bf16, "Metal Autocast bf16, contended");
}

#[test]
#[ignore = "needs the gated checkpoint, tokenizer files and the tracker fixture"]
fn frame_profile_cpu() {
    run(|| CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu"), AutocastMode::Off, "CPU f32, contended");
}
