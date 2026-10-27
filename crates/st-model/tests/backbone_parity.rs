//! Backbone parity against the fp32 PyTorch fixtures (tools/reference/dump_fixtures.py).
//!
//! Needs the gated checkpoint and the git-ignored fixtures:
//! `SPIKETRACK_WEIGHTS=.../sam3.1_multiplex.safetensors SPIKETRACK_FIXTURES=fixtures/backbone
//!  cargo test -p st-model --release --test backbone_parity -- --ignored --nocapture --test-threads=1`
//!
//! Gate (gp-vision-backbone, D2): cosine > 0.999 at every recorded stage on all 5 images.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use ojas_io::SafeTensors;
use st_model::{Backbone, Neck, Taps};
use st_weights::{Checkpoint, VisionBackbone};

const IMAGES: [&str; 5] = ["visem_22_f0", "visem_47_f300", "visem_82_f600", "sam3_truck", "sam3_groceries"];
const COS_GATE: f64 = 0.999;

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
    max_abs: f64,
    ref_max: f64,
}

fn compare(got: &[f32], want: &[f32]) -> Stat {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut gg, mut ww, mut max_abs, mut ref_max) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        assert!(g.is_finite(), "non-finite output");
        dot += g * w;
        gg += g * g;
        ww += w * w;
        max_abs = max_abs.max((g - w).abs());
        ref_max = ref_max.max(w.abs());
    }
    Stat { cos: dot / (gg.sqrt() * ww.sqrt()), max_abs, ref_max }
}

fn host(t: &Tensor, budget: &Budget) -> Vec<f32> {
    t.to_host(budget).expect("to host").to_f32_vec().expect("f32")
}

fn fixture(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data) = st.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
    (shape.into_iter().map(|d| d as usize).collect(), data)
}

#[test]
#[ignore = "needs the gated checkpoint and fixtures: set SPIKETRACK_WEIGHTS and SPIKETRACK_FIXTURES"]
fn backbone_matches_pytorch_fp32_on_five_images() {
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu backend");
    run_parity(&be, "cpu");
}

#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint and fixtures: set SPIKETRACK_WEIGHTS and SPIKETRACK_FIXTURES"]
fn backbone_matches_pytorch_fp32_on_five_images_on_metal() {
    let be = MetalBackend::new(Budget::new(24 << 30)).expect("metal backend");
    run_parity(&be, "metal");
}

/// The gate on any backend: same fixtures, same stages, same threshold.
fn run_parity<B: Backend>(be: &B, label: &str) {
    let weights = env_path("SPIKETRACK_WEIGHTS");
    let fixtures = env_path("SPIKETRACK_FIXTURES");
    let t0 = Instant::now();
    let ck = Checkpoint::open(&weights).expect("open checkpoint");
    let vision = VisionBackbone::load(&ck).expect("vision weights");
    let model = Backbone::new(be, &vision).expect("upload backbone");
    drop(vision);
    println!("[{label}] weights loaded and uploaded in {:.1} s", t0.elapsed().as_secs_f64());

    let mut rows = Vec::new();
    let mut worst = f64::INFINITY;
    for (i, name) in IMAGES.iter().enumerate() {
        let st = SafeTensors::open(&Path::new(&fixtures).join(format!("{name}.safetensors"))).expect("fixture file");
        let (shape, img) = fixture(&st, "input.image");
        let image = be.upload(&Tensor::from_f32(&img, &shape, be.budget()).expect("image")).expect("upload image");
        let keep: Vec<usize> = if i == 0 { (0..32).collect() } else { vec![7, 15, 23, 31] };
        let mut taps = Taps::default();
        let t = Instant::now();
        let trunk = match model.trunk(&image, &keep, &mut taps) {
            Ok(t) => t,
            Err(e) => {
                // A backend that refuses an op mid-trunk (Metal before rope lands): report the
                // stages that ran before the refusal, then fail with it.
                for (stage, tap, key) in [("patch_embed", &taps.patch_embed, "trunk.patch_embed.out"), ("ln_pre", &taps.ln_pre, "trunk.ln_pre.out")] {
                    if let Some(got) = tap {
                        let s = compare(&host(got, be.budget()), &fixture(&st, key).1);
                        println!("[{label}] {name} {stage}: cosine {:.7}, max abs {:.3e}, max abs / max ref {:.3e}", s.cos, s.max_abs, s.max_abs / s.ref_max.max(1e-30));
                    }
                }
                panic!("[{label}] {name}: trunk stopped after ln_pre: {e}");
            }
        };
        let necks = [Neck::Detector, Neck::Interactive, Neck::Propagation]
            .map(|n| model.neck(n, &trunk).expect("neck"));
        be.sync().expect("sync");
        let secs = t.elapsed().as_secs_f64();
        let mut check = |stage: String, got: &Tensor, key: &str| {
            let (want_shape, want) = fixture(&st, key);
            assert_eq!(got.shape(), want_shape.as_slice(), "{name} {stage} shape");
            let s = compare(&host(got, be.budget()), &want);
            worst = worst.min(s.cos);
            rows.push(format!(
                "| {name} | {stage} | {:.7} | {:.3e} | {:.3e} | {} |",
                s.cos,
                s.max_abs,
                s.max_abs / s.ref_max.max(1e-30),
                if s.cos > COS_GATE { "pass" } else { "FAIL" }
            ));
        };
        if i == 0 {
            check("patch_embed".into(), taps.patch_embed.as_ref().expect("tap"), "trunk.patch_embed.out");
            check("ln_pre".into(), taps.ln_pre.as_ref().expect("tap"), "trunk.ln_pre.out");
        }
        for &b in &keep {
            check(format!("block {b}"), taps.blocks[b].as_ref().expect("tap"), &format!("trunk.blocks.{b}.out"));
        }
        for (neck, tag) in necks.iter().zip(["convs", "interactive_convs", "propagation_convs"]) {
            for (lvl, t) in neck.iter().enumerate() {
                check(format!("{tag}[{lvl}]"), t, &format!("neck.{tag}.{lvl}.out"));
            }
        }
        println!("[{label}] {name}: trunk + 3 necks {secs:.1} s (one run, not a benchmark)");
    }
    println!("\n| image | stage | cosine | max abs | max abs / max ref | gate |\n|---|---|---|---|---|---|");
    for r in &rows {
        println!("{r}");
    }
    println!("\n[{label}] worst cosine {worst:.7} over {} stages (gate > {COS_GATE})", rows.len());
    assert!(worst > COS_GATE, "worst cosine {worst} <= {COS_GATE}");
}
