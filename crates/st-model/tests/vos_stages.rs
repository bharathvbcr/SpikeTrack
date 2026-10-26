//! The VOS driver end to end on the tracker fixture (DAVIS judo, 8 frames,
//! tools/reference/dump_tracker_fixtures.py): our backbone and tracker run on
//! upstream's frames from the frame-0 masks, carrying our own state forward,
//! and each frame's outputs and stored state are compared with upstream's.
//! Localizes a D5 gate failure to the frame and stage where it starts.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_TRACKER_FIXTURES=fixtures/tracker/judo.safetensors
//!  cargo test -p st-model --release --test vos_stages -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use st_model::tracker::vos::{Vos, D, FEAT, LOW};
use st_model::Backbone;
use st_weights::{Checkpoint, Sam31Weights};

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

/// Fraction of pixels whose foreground decision (logit > 0) agrees.
fn sign_agreement(got: &[f32], want: &[f32]) -> f64 {
    got.iter().zip(want).filter(|(&g, &w)| (g > 0.0) == (w > 0.0)).count() as f64 / got.len() as f64
}

#[test]
#[ignore = "needs the gated checkpoint and tracker fixtures"]
fn vos_driver_tracks_upstream_state_frame_by_frame() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let be = CpuBackend::with_threads(Budget::new(28 << 30), 8).expect("cpu");
    let backbone = Backbone::new(&be, &w.vision).expect("backbone");
    let vos = Vos::new(&be, &backbone, &w.tracker).expect("tracker");
    let fx = SafeTensors::open(&env_path("SPIKETRACK_TRACKER_FIXTURES")).expect("fixtures");
    let h = |name: &str| fx.read_f32_widened(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
    let (mshape, masks0) = h("f0.input_masks");
    let (n, ht, wd) = (mshape[0] as usize, mshape[1] as usize, mshape[2] as usize);
    let frames: Vec<Tensor> = (0..FRAMES)
        .map(|f| be.upload(&Tensor::from_f32(&h(&format!("f{f}.image")).1, &[1, 3, 1008, 1008], be.budget()).expect("t")).expect("up"))
        .collect();
    let mut worst = f64::INFINITY;
    let mut worst_agree = 1f64;
    println!("| frame | stage | cosine | max abs |\n|---|---|---|---|");
    vos.run(&frames, &masks0, n, ht, wd, |out| {
        let f = out.frame;
        let st = if f == 0 { format!("f0.stored.cond_frame_outputs") } else { format!("f{f}.stored.non_cond_frame_outputs") };
        let want_ptr = h(&format!("{st}.obj_ptr")).1;
        for (stage, got, want) in [
            ("low-res masks", out.low_res.clone(), h(&format!("{st}.pred_masks")).1),
            ("object scores", out.object_scores.clone(), h(&format!("{st}.object_score_logits")).1),
            ("object pointers (valid slots)", out.obj_ptr[..n * D].to_vec(), want_ptr[..n * D].to_vec()),
            ("memory features", out.maskmem.clone(), h(&format!("{st}.maskmem_features")).1),
            ("video-res masks", out.video_res.clone(), h(&format!("f{f}.out.video_res_masks")).1),
        ] {
            let (cos, ma) = stats(&got, &want);
            worst = worst.min(cos);
            println!("| {f} | {stage} | {cos:.7} | {ma:.3e} |");
            if stage == "video-res masks" {
                let agree = sign_agreement(&got, &want);
                worst_agree = worst_agree.min(agree);
                println!("| {f} | video-res foreground agreement | {agree:.6} | |");
            }
        }
        assert!(out.obj_ptr[n * D..].iter().all(|&v| v == 0.0), "frame {f}: padding slots carry a pointer");
        assert_eq!(out.low_res.len(), n * LOW * LOW);
        assert_eq!(out.maskmem.len(), D * FEAT * FEAT);
    })
    .expect("vos run");
    println!("\nVOS stages worst cosine {worst:.7}, worst video-res foreground agreement {worst_agree:.6}");
    assert!(worst > 0.999, "VOS stages worst cosine {worst}");
    assert!(worst_agree > 0.999, "video-res foreground agreement {worst_agree}");
}
