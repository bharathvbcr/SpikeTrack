//! The CoreML trunk behind `Backbone::with_external_trunk`, against the fp32 PyTorch fixtures: the
//! trunk output (`trunk.blocks.31.out`) and all nine neck levels, with the necks run by Ojas on the
//! CPU backend, so what is measured is what the detector and tracker would be handed.
//!
//! Needs the gated checkpoint, the git-ignored backbone fixtures and the exported packages
//! (tools/baselines/coreml_trunk.py):
//! `SPIKETRACK_WEIGHTS=.../sam3.1_multiplex.safetensors SPIKETRACK_FIXTURES=fixtures/backbone
//!  SPIKETRACK_MODEL_DIR=.../sam3.1 cargo test -p st-coreml --release --test trunk_parity -- --ignored --nocapture --test-threads=1`
//!
//! Gates, stated before the first run (E10):
//! - fp32: relative L2 < 1e-4 at every stage on all 5 images (the exported graph measured 1-3e-5
//!   at the trunk from Python), and cosine > 0.999 (D2).
//! - fp16: D2's cosine > 0.999 at every stage. fp16 is decided by held-out HOTA, not here; this
//!   test reports its error at the necks and checks it is not broken.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Backend, BackendCore, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use st_coreml::{CoremlTrunk, Units};
use st_gate::{GatePolicy, WindowSet};
use st_model::backbone::TrunkCache;
use st_model::{Backbone, Neck, Taps};
use st_weights::{Checkpoint, VisionBackbone};

const IMAGES: [&str; 5] = [
    "visem_22_f0",
    "visem_47_f300",
    "visem_82_f600",
    "sam3_truck",
    "sam3_groceries",
];
const COS_GATE: f64 = 0.999;
const FP32_REL_L2_GATE: f64 = 1e-4;

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

struct Stat {
    cos: f64,
    rel_l2: f64,
    max_abs: f64,
}

fn compare(got: &[f32], want: &[f32]) -> Stat {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut gg, mut ww, mut dd, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        assert!(g.is_finite(), "non-finite output");
        dot += g * w;
        gg += g * g;
        ww += w * w;
        dd += (g - w) * (g - w);
        max_abs = max_abs.max((g - w).abs());
    }
    Stat {
        cos: dot / (gg.sqrt() * ww.sqrt()),
        rel_l2: (dd / ww).sqrt(),
        max_abs,
    }
}

fn host<B: Backend>(be: &B, t: &Tensor) -> Vec<f32> {
    be.download(t).expect("download").to_f32_vec().expect("f32")
}

fn fixture(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data) = st
        .read_f32(name)
        .unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
    (shape.into_iter().map(|d| d as usize).collect(), data)
}

fn package(precision: &str) -> PathBuf {
    env_path("SPIKETRACK_MODEL_DIR")
        .join("coreml")
        .join(format!("sam3.1_trunk_{precision}.mlpackage"))
}

#[test]
#[ignore = "needs the gated checkpoint, fixtures and the exported CoreML trunk (see the module docs)"]
fn coreml_fp32_trunk_and_ojas_necks_match_upstream() {
    run("fp32", Some(FP32_REL_L2_GATE));
}

#[test]
#[ignore = "needs the gated checkpoint, fixtures and the exported CoreML trunk (see the module docs)"]
fn coreml_fp16_trunk_and_ojas_necks_keep_d2() {
    run("fp16", None);
}

fn run(precision: &str, rel_l2_gate: Option<f64>) {
    let label = format!("coreml-{precision}");
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu backend");
    let fixtures = env_path("SPIKETRACK_FIXTURES");
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("open checkpoint");
    let vision = VisionBackbone::load(&ck).expect("vision weights");
    let trunk =
        CoremlTrunk::load(&package(precision), Units::CpuAndGpu).expect("load CoreML trunk");
    println!(
        "[{label}] loaded in {:.2} s (compiled this run: {}; hash {:.2} s, hashed in full: {}, compile {:.2} s, model load {:.2} s), sha256 {}",
        trunk.load_seconds,
        trunk.compiled,
        trunk.hash_seconds,
        trunk.hashed,
        trunk.compile_seconds,
        trunk.model_seconds,
        trunk.sha256()
    );
    let model = Backbone::with_external_trunk(&be, &vision, Box::new(trunk))
        .expect("backbone with external trunk");
    let described = model
        .external_trunk()
        .expect("the backbone reports its external trunk");
    assert!(described.starts_with("coreml "), "{described}");

    let mut worst_cos = f64::INFINITY;
    let mut worst_rel = 0f64;
    println!("\n| image | stage | cosine | rel L2 | max abs |\n|---|---|---|---|---|");
    for name in IMAGES {
        let st = SafeTensors::open(&Path::new(&fixtures).join(format!("{name}.safetensors")))
            .expect("fixture file");
        let (shape, img) = fixture(&st, "input.image");
        let image = be
            .upload(&Tensor::from_f32(&img, &shape, be.budget()).expect("image"))
            .expect("upload image");
        let mut taps = Taps::default();
        taps.ln_pre = Some(image.clone());
        let t0 = Instant::now();
        let out = model.trunk(&image, &[], &mut taps).expect("trunk");
        let secs = t0.elapsed().as_secs_f64();
        assert!(
            taps.ln_pre.is_none() && taps.patch_embed.is_none() && taps.blocks.is_empty(),
            "an external trunk leaves no Ojas taps"
        );
        assert_eq!(out.shape(), &[1, 72, 72, 1024]);
        let mut stages = vec![(
            "trunk".to_string(),
            host(&be, &out),
            fixture(&st, "trunk.blocks.31.out").1,
        )];
        for (neck, key) in [
            (Neck::Detector, "convs"),
            (Neck::Interactive, "interactive_convs"),
            (Neck::Propagation, "propagation_convs"),
        ] {
            let levels = model.neck(neck, &out).expect("neck");
            for (l, t) in levels.iter().enumerate() {
                let fx = format!("neck.{key}.{l}.out");
                stages.push((format!("{key}[{l}]"), host(&be, t), fixture(&st, &fx).1));
            }
        }
        for (stage, got, want) in &stages {
            let s = compare(got, want);
            println!(
                "| {name} | {stage} | {:.7} | {:.3e} | {:.3e} |",
                s.cos, s.rel_l2, s.max_abs
            );
            worst_cos = worst_cos.min(s.cos);
            worst_rel = worst_rel.max(s.rel_l2);
        }
        println!(
            "[{label}] {name}: trunk call {secs:.3} s (in-process, incl. download and upload)"
        );
    }
    println!("\n[{label}] worst cosine {worst_cos:.7}, worst rel L2 {worst_rel:.3e} over {} images x 10 stages", IMAGES.len());
    assert!(
        worst_cos > COS_GATE,
        "worst cosine {worst_cos} <= {COS_GATE}"
    );
    if let Some(gate) = rel_l2_gate {
        assert!(
            worst_rel < gate,
            "worst rel L2 {worst_rel:.3e} >= {gate:.0e}"
        );
    }

    // The paths that need the Ojas blocks refuse an external trunk instead of running without them.
    let st = SafeTensors::open(&Path::new(&fixtures).join(format!("{}.safetensors", IMAGES[0])))
        .expect("fixture file");
    let (shape, img) = fixture(&st, "input.image");
    let image = be
        .upload(&Tensor::from_f32(&img, &shape, be.budget()).expect("image"))
        .expect("upload image");
    let e = model
        .trunk(&image, &[31], &mut Taps::default())
        .err()
        .expect("block taps refused");
    assert!(e.to_string().contains("block taps"), "{e}");
    let e = model
        .trunk_gated(
            &image,
            &mut TrunkCache::default(),
            &WindowSet::full(3),
            GatePolicy::Off,
            &[],
            &mut Taps::default(),
        )
        .err()
        .expect("gated trunk refused");
    assert!(e.to_string().contains("Ojas trunk blocks"), "{e}");
    let e = model
        .with_trunk_linears_bf16()
        .err()
        .expect("bf16 linears refused");
    assert!(e.to_string().contains("Ojas trunk blocks"), "{e}");
}
