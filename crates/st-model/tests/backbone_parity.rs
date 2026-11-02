//! Backbone parity against the fp32 PyTorch fixtures (tools/reference/dump_fixtures.py).
//!
//! Needs the gated checkpoint and the git-ignored fixtures:
//! `SPIKETRACK_WEIGHTS=.../sam3.1_multiplex.safetensors SPIKETRACK_FIXTURES=fixtures/backbone
//!  cargo test -p st-model --release --test backbone_parity -- --ignored --nocapture --test-threads=1`
//!
//! Gate (gp-vision-backbone, D2): cosine > 0.999 at every recorded stage on all 5 images.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Autocast, AutocastMode, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
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
    Stat {
        cos: dot / (gg.sqrt() * ww.sqrt()),
        max_abs,
        ref_max,
    }
}

fn host(t: &Tensor, budget: &Budget) -> Vec<f32> {
    t.to_host(budget)
        .expect("to host")
        .to_f32_vec()
        .expect("f32")
}

fn fixture(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data) = st
        .read_f32(name)
        .unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
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

/// The same gate inside an Ojas Autocast bf16 region (matmul-class operands and outputs
/// rounded to bf16; norms, softmax and GELU stay f32): whether a half-precision trunk keeps
/// D2's cosine > 0.999 at every stage.
#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint and fixtures: set SPIKETRACK_WEIGHTS and SPIKETRACK_FIXTURES"]
fn backbone_matches_pytorch_fp32_on_five_images_on_metal_bf16() {
    let be = Autocast::new(MetalBackend::new(Budget::new(24 << 30)).expect("metal backend"));
    let _region = be.autocast_region(AutocastMode::Bf16).expect("bf16 region");
    run_parity(&be, "metal-bf16");
}

/// The ViT's LayerNorms, op by op: each recorded norm input through `layer_norm_forward`
/// with the checkpoint's affine weights, against upstream's recorded output. The bound sits
/// between what the right epsilon gives (~1e-6 abs) and what the 1e-6 epsilon this port
/// used until 2026-10-27 gives (~1e-5 abs); the old value is checked to fail it, so the test
/// cannot pass with the wrong epsilon.
#[test]
#[ignore = "needs the gated checkpoint and fixtures: set SPIKETRACK_WEIGHTS and SPIKETRACK_FIXTURES"]
fn vit_layer_norm_eps_matches_upstream() {
    const BOUND: f64 = 6e-6;
    const OLD_EPS: f32 = 1e-6;
    let be = CpuBackend::with_threads(Budget::new(8 << 30), 8).expect("cpu backend");
    let ck = SafeTensors::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let st = SafeTensors::open(
        &Path::new(&env_path("SPIKETRACK_FIXTURES")).join(format!("{}.safetensors", IMAGES[0])),
    )
    .expect("fixture");
    let up = |shape: &[usize], data: &[f32]| {
        be.upload(&Tensor::from_f32(data, shape, be.budget()).expect("t"))
            .expect("up")
    };
    // (input, output, weights, whether the old epsilon was used here): ln_pre always had 1e-5.
    let norms: Vec<(String, String, String, bool)> = std::iter::once((
        "trunk.ln_pre.out.in".to_string(),
        "trunk.ln_pre.out".to_string(),
        "trunk.ln_pre".to_string(),
        false,
    ))
    .chain([0usize, 7].into_iter().flat_map(|b| {
        ["norm1", "norm2"].map(move |n| {
            (
                format!("trunk.blocks.{b}.{n}.in"),
                format!("trunk.blocks.{b}.{n}"),
                format!("trunk.blocks.{b}.{n}"),
                true,
            )
        })
    }))
    .collect();
    for (input, output, weights, had_old_eps) in &norms {
        let (shape, x) = fixture(&st, input);
        let (_, want) = fixture(&st, output);
        let w = ck
            .read_f32(&format!(
                "detector.backbone.vision_backbone.{weights}.weight"
            ))
            .expect("weight")
            .1;
        let b = ck
            .read_f32(&format!("detector.backbone.vision_backbone.{weights}.bias"))
            .expect("bias")
            .1;
        let (xt, wt, bt) = (up(&shape, &x), up(&[w.len()], &w), up(&[b.len()], &b));
        let run = |eps: f32| {
            compare(
                &host(
                    &be.layer_norm_forward(&xt, &wt, &bt, eps)
                        .expect("layer_norm"),
                    be.budget(),
                ),
                &want,
            )
            .max_abs
        };
        let (ours, old) = (run(st_model::backbone::VIT_LN_EPS), run(OLD_EPS));
        println!(
            "{weights}: max abs {ours:.3e} at eps {} (old eps {OLD_EPS}: {old:.3e})",
            st_model::backbone::VIT_LN_EPS
        );
        assert!(
            ours < BOUND,
            "{weights}: max abs {ours:.3e} >= {BOUND:.0e} at eps {}",
            st_model::backbone::VIT_LN_EPS
        );
        if *had_old_eps {
            assert!(old >= BOUND, "{weights}: the old eps {OLD_EPS} also passes ({old:.3e}): the bound no longer discriminates");
        }
    }
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
    println!(
        "[{label}] weights loaded and uploaded in {:.1} s",
        t0.elapsed().as_secs_f64()
    );

    let mut rows = Vec::new();
    let mut worst = f64::INFINITY;
    for (i, name) in IMAGES.iter().enumerate() {
        let st = SafeTensors::open(&Path::new(&fixtures).join(format!("{name}.safetensors")))
            .expect("fixture file");
        let (shape, img) = fixture(&st, "input.image");
        let image = be
            .upload(&Tensor::from_f32(&img, &shape, be.budget()).expect("image"))
            .expect("upload image");
        let keep: Vec<usize> = if i == 0 {
            (0..32).collect()
        } else {
            vec![7, 15, 23, 31]
        };
        let mut taps = Taps::default();
        let t = Instant::now();
        let trunk = match model.trunk(&image, &keep, &mut taps) {
            Ok(t) => t,
            Err(e) => {
                // A backend that refuses an op mid-trunk (Metal before rope lands): report the
                // stages that ran before the refusal, then fail with it.
                for (stage, tap, key) in [
                    ("patch_embed", &taps.patch_embed, "trunk.patch_embed.out"),
                    ("ln_pre", &taps.ln_pre, "trunk.ln_pre.out"),
                ] {
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
            check(
                "patch_embed".into(),
                taps.patch_embed.as_ref().expect("tap"),
                "trunk.patch_embed.out",
            );
            check(
                "ln_pre".into(),
                taps.ln_pre.as_ref().expect("tap"),
                "trunk.ln_pre.out",
            );
        }
        for &b in &keep {
            check(
                format!("block {b}"),
                taps.blocks[b].as_ref().expect("tap"),
                &format!("trunk.blocks.{b}.out"),
            );
        }
        for (neck, tag) in necks
            .iter()
            .zip(["convs", "interactive_convs", "propagation_convs"])
        {
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
    println!(
        "\n[{label}] worst cosine {worst:.7} over {} stages (gate > {COS_GATE})",
        rows.len()
    );
    assert!(worst > COS_GATE, "worst cosine {worst} <= {COS_GATE}");
}

/// The Rust RoPE generator against the checkpoint's own tables: `vit_rope_tables(24)` for each of
/// the 28 windowed blocks and `vit_rope_tables(72)` for each of the 4 global ones, element by
/// element. The checkpoint's tables were made by torch in f32 (`compute_axial_cis`: f32 angles,
/// `torch.polar`); ours takes the same f32 angles and rounds an f64 sine and cosine once, so a
/// difference can only be torch's own cosine/sine rounding: at most one f32 ulp of a value in
/// [0.5, 1], 2^-24 = 5.96e-8 (measured 2026-10-28: max 5.96e-8, 99.28% of the 2,359,296 entries
/// bit-identical, the 24 x 24 tables 2.98e-8 and 99.35%). The count of bit-identical entries is printed.
#[test]
#[ignore = "needs the gated checkpoint: set SPIKETRACK_WEIGHTS"]
fn generated_rope_tables_equal_the_checkpoints() {
    const BOUND: f64 = 6.0e-8;
    let budget = Budget::new(8 << 30);
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("open checkpoint");
    let vision = VisionBackbone::load(&ck).expect("vision weights");
    let mut tables = std::collections::BTreeMap::new();
    for grid in [24usize, 72] {
        let (cos, sin) = st_model::backbone::vit_rope_tables(grid, &budget).expect("generate");
        tables.insert(grid, (host(&cos, &budget), host(&sin, &budget)));
    }
    let (mut total, mut identical, mut worst) = (0usize, 0usize, 0f64);
    for (i, blk) in vision.trunk.blocks.iter().enumerate() {
        let grid = if blk.global { 72 } else { 24 };
        let (cos, sin) = &tables[&grid];
        assert_eq!(
            blk.freqs_cis.shape,
            vec![grid * grid, 32, 2],
            "block {i} shape"
        );
        let (mut block_worst, mut block_identical) = (0f64, 0usize);
        for (k, pair) in blk.freqs_cis.data.chunks_exact(2).enumerate() {
            for (got, want) in [(cos[k], pair[0]), (sin[k], pair[1])] {
                block_worst = block_worst.max((f64::from(got) - f64::from(want)).abs());
                block_identical += usize::from(got.to_bits() == want.to_bits());
            }
        }
        let entries = blk.freqs_cis.data.len();
        println!("block {i:2} ({}, grid {grid}): max abs diff {block_worst:.3e}, {block_identical}/{entries} bit-identical", if blk.global { "global  " } else { "windowed" });
        assert!(
            block_worst <= BOUND,
            "block {i}: max abs diff {block_worst:e} > {BOUND:e}"
        );
        (total, identical, worst) = (
            total + entries,
            identical + block_identical,
            worst.max(block_worst),
        );
    }
    println!("all 32 blocks: max abs diff {worst:.3e}, {identical}/{total} entries bit-identical ({:.4}%)", 100.0 * identical as f64 / total as f64);
}

const IMAGE_672: &str = "visem_22_f0";

/// The 672-pixel gate (48 x 48 tokens, 2 x 2 windows): our trunk and three necks on VISEM video
/// 22 frame 0 against upstream's own code built for `img_size=672`
/// (`tools/reference/dump_backbone_size.py`), same stages and threshold as the 1008 gate.
/// `SPIKETRACK_FIXTURES_672` is the folder holding `visem_22_f0.safetensors`.
fn run_parity_672<B: Backend>(be: &B, label: &str) {
    let weights = env_path("SPIKETRACK_WEIGHTS");
    let fixtures = env_path("SPIKETRACK_FIXTURES_672");
    let geo = st_model::Geometry::from_image_side(672).expect("672 is a multiple of 336");
    let ck = Checkpoint::open(&weights).expect("open checkpoint");
    let vision = VisionBackbone::load(&ck).expect("vision weights");
    let model = Backbone::new_at(be, &vision, geo).expect("upload backbone at 672");
    drop(vision);
    let st = SafeTensors::open(&Path::new(&fixtures).join(format!("{IMAGE_672}.safetensors")))
        .expect("fixture file");
    let (shape, img) = fixture(&st, "input.image");
    assert_eq!(shape, vec![1, 3, 672, 672], "the fixture is a 672 input");
    let image = be
        .upload(&Tensor::from_f32(&img, &shape, be.budget()).expect("image"))
        .expect("upload image");
    let mut taps = Taps::default();
    let t = Instant::now();
    let trunk = model
        .trunk(&image, &[7, 15, 23, 31], &mut taps)
        .expect("trunk");
    assert_eq!(trunk.shape(), &[1, 48, 48, 1024], "trunk output shape");
    let necks = [Neck::Detector, Neck::Interactive, Neck::Propagation]
        .map(|n| model.neck(n, &trunk).expect("neck"));
    be.sync().expect("sync");
    println!(
        "[{label}] 672 trunk + 3 necks {:.1} s (one run, not a benchmark)",
        t.elapsed().as_secs_f64()
    );

    let mut rows = Vec::new();
    let mut worst = f64::INFINITY;
    let mut check = |stage: String, got: &Tensor, key: &str| {
        let (want_shape, want) = fixture(&st, key);
        assert_eq!(got.shape(), want_shape.as_slice(), "{stage} shape");
        let s = compare(&host(got, be.budget()), &want);
        worst = worst.min(s.cos);
        rows.push(format!(
            "| {stage} | {:.7} | {:.3e} | {:.3e} | {} |",
            s.cos,
            s.max_abs,
            s.max_abs / s.ref_max.max(1e-30),
            if s.cos > COS_GATE { "pass" } else { "FAIL" }
        ));
    };
    check(
        "ln_pre".into(),
        taps.ln_pre.as_ref().expect("tap"),
        "trunk.ln_pre.out",
    );
    for b in [7usize, 15, 23, 31] {
        check(
            format!("block {b}"),
            taps.blocks[b].as_ref().expect("tap"),
            &format!("trunk.blocks.{b}.out"),
        );
    }
    for (neck, tag) in necks
        .iter()
        .zip(["convs", "interactive_convs", "propagation_convs"])
    {
        for (lvl, t) in neck.iter().enumerate() {
            check(format!("{tag}[{lvl}]"), t, &format!("neck.{tag}.{lvl}.out"));
        }
    }
    // The sine position encodings the detector (72 level -> 48) and the memory encoder use are
    // `host::sine_position_map(side, side)`: upstream's own maps for the three neck levels.
    for (lvl, side) in [(0usize, geo.low_res()), (1, geo.mid_res()), (2, geo.grid)] {
        let (want_shape, want) = fixture(&st, &format!("neck.pos_enc.{lvl}"));
        assert_eq!(want_shape, vec![1, 256, side, side], "pos_enc {lvl} shape");
        let s = compare(&st_model::host::sine_position_map(side, side), &want);
        worst = worst.min(s.cos);
        rows.push(format!(
            "| sine_position_map({side}) vs pos_enc[{lvl}] | {:.7} | {:.3e} | {:.3e} | {} |",
            s.cos,
            s.max_abs,
            s.max_abs / s.ref_max.max(1e-30),
            if s.cos > COS_GATE { "pass" } else { "FAIL" }
        ));
    }
    println!(
        "\n| stage (672) | cosine | max abs | max abs / max ref | gate |\n|---|---|---|---|---|"
    );
    for r in &rows {
        println!("{r}");
    }
    println!(
        "\n[{label}] 672: worst cosine {worst:.7} over {} stages (gate > {COS_GATE})",
        rows.len()
    );
    assert!(worst > COS_GATE, "worst cosine {worst} <= {COS_GATE}");
}

#[test]
#[ignore = "needs the gated checkpoint and the 672 fixture: set SPIKETRACK_WEIGHTS and SPIKETRACK_FIXTURES_672 (tools/reference/dump_backbone_size.py)"]
fn backbone_matches_pytorch_fp32_at_672() {
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu backend");
    run_parity_672(&be, "cpu");
}

#[test]
#[ignore = "needs a Metal GPU, the gated checkpoint and the 672 fixture: set SPIKETRACK_WEIGHTS and SPIKETRACK_FIXTURES_672"]
fn backbone_matches_pytorch_fp32_at_672_on_metal() {
    let be = MetalBackend::new(Budget::new(24 << 30)).expect("metal backend");
    run_parity_672(&be, "metal");
}
