//! Per-op wall-time breakdown of one frame on a backend (default Metal), for choosing
//! the next kernel (requested by ojas-78 for D6). One frame = what the text-prompted
//! video path runs per frame: the PE ViT trunk, the detector neck + detector with a
//! cached text prompt, and a multiplex tracker step (propagation neck, memory attention,
//! mask decoder, memory encoder) on DAVIS judo frame 2 after frames 0-1 (2 objects).
//!
//! The tracker step runs through `Vos`, which computes its own trunk: the tracker frame's
//! rows therefore include a trunk (identical ops and shapes to the detector run's
//! `trunk` stage), which a shared-trunk frame computes once. An unprofiled pass of the
//! same frame is timed first; the profiled pass syncs after every op (wall time, not GPU
//! time), so its total is larger. Single run, machine shared: label results contended.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... SPIKETRACK_TRACKER_FIXTURES=fixtures/tracker/judo.safetensors
//!  cargo test -p st-model --release --test frame_profile -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use st_model::detector::{Detector, DetectorTaps, DetectorWeights, D};
use st_model::profile::Profiled;
use st_model::tracker::vos::Vos;
use st_model::{Backbone, Neck, Taps, Text, Tokenizer};
use st_weights::{Checkpoint, Sam31Weights};

const PROMPT: &str = "cell";
const FRAMES: usize = 3;

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

/// Builds the models on `be`, warms up on frames 0-1, then runs frame 2's detector and
/// tracker step. `between` is called right before the measured frame (to clear and label
/// a profiler). Returns (detector run ms incl. its trunk, tracker frame ms incl. its trunk).
fn one_frame<B: Backend>(be: &B, w: &Sam31Weights, mut stage: impl FnMut(&str)) -> (f64, f64) {
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

    // Detector warm-up on frame 1, then the measured detector run on frame 2.
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

    // Tracker: frames 0-1 warm up (frame 0 is the conditioning frame); frame 2 is measured.
    stage("warm-up");
    let mut t_frame = Instant::now();
    let mut vos_ms = 0.0;
    vos.run(&frames, &masks0, n, ht, wd, |out| {
        if out.frame == FRAMES - 2 {
            stage("tracker frame (its trunk + propagation neck + tracker step)");
            t_frame = Instant::now();
        } else if out.frame == FRAMES - 1 {
            vos_ms = t_frame.elapsed().as_secs_f64() * 1e3;
        }
    })
    .expect("vos run");
    (det_ms, vos_ms)
}

fn run<B: Backend>(make: impl Fn() -> B, label: &str) {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let (det_ms, vos_ms) = one_frame(&make(), &w, |_| {});
    println!("[{label}] unprofiled: detector run (trunk + neck + detector) {det_ms:.0} ms; tracker frame (trunk + neck + step) {vos_ms:.0} ms");
    let prof = Profiled::new(make());
    let (pdet, pvos) = one_frame(&prof, &w, |s| prof.set_stage(s));
    println!("[{label}] profiled (every op synced): detector run {pdet:.0} ms; tracker frame {pvos:.0} ms\n");
    prof.retain(|c| c.stage != "warm-up");
    println!("{}", prof.report());
}

#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint, tokenizer files and the tracker fixture"]
fn frame_profile_metal() {
    run(|| MetalBackend::new(Budget::new(24 << 30)).expect("metal"), "Metal, contended");
}

#[test]
#[ignore = "needs the gated checkpoint, tokenizer files and the tracker fixture"]
fn frame_profile_cpu() {
    run(|| CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu"), "CPU, contended");
}
