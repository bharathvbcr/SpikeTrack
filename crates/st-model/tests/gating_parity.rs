//! Event gating on the PE ViT trunk: what is exact, what is approximate, and what it saves.
//!
//! Two kinds of test:
//!
//! 1. `synthetic_*` (always run, no weights): a three-block trunk [windowed, global, windowed]
//!    with random weights at the real token geometry, CPU backend under `Numerics::Exact`.
//!    It pins the control flow: the capture path, reuse, exact window recompute and carrying
//!    semantics of the approximate variant.
//! 2. `#[ignore]` (need the checkpoint and the VISEM frames): the real trunk.
//!
//!    `SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors \
//!     cargo test -p st-model --release --test gating_parity -- --ignored --nocapture --test-threads=1`
//!
//! # Numerics
//!
//! Bitwise claims are made on `CpuBackend` under `Numerics::Exact`. Its GEMM bits do not depend on
//! tiling or thread count (ojas-cpu `gemm.rs` module docs), which is what lets a batch of `k`
//! windows give each token the bits the batch of 9 gives. Under `Numerics::Fast` (what
//! `backbone_parity` and the timings use) products go whole to Accelerate, whose blocking depends
//! on the shape, so the same claim is *reported* there, not asserted. Metal is not claimed.
//!
//! What is asserted strictly: (a) the gated trunk with every window recomputed is bitwise the
//! ungated trunk, and a reused frame is bitwise the ungated trunk of the identical input; (b) the
//! exact window policy is bitwise the ungated trunk at every block it exposes and at the output.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Backend, BackendCore, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use st_gate::{Decision, Frame, Gate, GateConfig, GatePolicy, Geometry, WindowSet};
use st_model::backbone::{GatedMode, GatedStats, TrunkCache};
use st_model::{Backbone, Neck, Taps};
use st_weights::names::{VIT_DIM, VIT_GRID, VIT_PATCH, VIT_WINDOW};
use st_weights::{
    Conv, FpnLevel, FpnNeck, LayerNorm, Linear, VisionBackbone, VitBlock, VitTrunk, Weight,
};

const SIDE: usize = VIT_GRID / VIT_WINDOW;
const IMG: usize = VIT_GRID * VIT_PATCH;
const WIN_PX: usize = VIT_WINDOW * VIT_PATCH;

// ---------------------------------------------------------------- helpers

struct Cmp {
    bitwise: bool,
    cos: f64,
    max_abs: f64,
}

fn host<B: Backend>(be: &B, t: &Tensor) -> Vec<f32> {
    be.download(t).expect("download").to_f32_vec().expect("f32")
}

fn compare(got: &[f32], want: &[f32]) -> Cmp {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut gg, mut ww, mut max_abs) = (0f64, 0f64, 0f64, 0f64);
    let mut bitwise = true;
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output");
        bitwise &= g.to_bits() == w.to_bits();
        let (g, w) = (f64::from(g), f64::from(w));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        max_abs = max_abs.max((g - w).abs());
    }
    Cmp {
        bitwise,
        cos: dot / (gg.sqrt() * ww.sqrt()),
        max_abs,
    }
}

fn fnv(data: &[f32]) -> u64 {
    data.iter().fold(0xcbf29ce484222325u64, |h, v| {
        v.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
    })
}

fn tensor<B: Backend>(be: &B, data: &[f32], shape: &[usize]) -> Tensor {
    be.upload(&Tensor::from_f32(data, shape, be.budget()).expect("tensor"))
        .expect("upload")
}

fn assert_same<B: Backend>(be: &B, what: &str, got: &Tensor, want: &Tensor) {
    let c = compare(&host(be, got), &host(be, want));
    assert!(
        c.bitwise,
        "{what}: not bitwise equal (cosine {:.9}, max abs {:.3e})",
        c.cos, c.max_abs
    );
}

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(24 << 30), 8)
        .expect("cpu backend")
        .with_numerics(numerics)
}

// ------------------------------------------------------ synthetic trunk

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 40) as f32) / (1u64 << 24) as f32 // [0, 1)
}

fn rand_weight(name: &str, shape: &[usize], scale: f32, s: &mut u64) -> Weight {
    let n: usize = shape.iter().product();
    Weight {
        name: name.into(),
        shape: shape.to_vec(),
        data: (0..n).map(|_| (lcg(s) - 0.5) * 2.0 * scale).collect(),
    }
}

fn const_weight(name: &str, shape: &[usize], v: f32) -> Weight {
    Weight {
        name: name.into(),
        shape: shape.to_vec(),
        data: vec![v; shape.iter().product()],
    }
}

fn norm(name: &str) -> LayerNorm {
    LayerNorm {
        weight: const_weight(name, &[VIT_DIM], 1.0),
        bias: const_weight(name, &[VIT_DIM], 0.0),
    }
}

fn linear(name: &str, out: usize, inp: usize, s: &mut u64) -> Linear {
    Linear {
        weight: rand_weight(name, &[out, inp], 0.03, s),
        bias: rand_weight(name, &[out], 0.01, s),
    }
}

/// A block with the real attention shapes and a narrow MLP (the MLP is per token, so its width is
/// irrelevant to what gating can get wrong).
fn block(global: bool, mlp: usize, s: &mut u64) -> VitBlock {
    let tokens = if global {
        VIT_GRID * VIT_GRID
    } else {
        VIT_WINDOW * VIT_WINDOW
    };
    let mut rope = Vec::with_capacity(tokens * 32 * 2);
    for _ in 0..tokens * 32 {
        let a = lcg(s) * 6.2831855;
        rope.extend([a.cos(), a.sin()]);
    }
    VitBlock {
        global,
        norm1: norm("norm1"),
        qkv: linear("qkv", 3 * VIT_DIM, VIT_DIM, s),
        proj: linear("proj", VIT_DIM, VIT_DIM, s),
        freqs_cis: Weight {
            name: "freqs_cis".into(),
            shape: vec![tokens, 32, 2],
            data: rope,
        },
        norm2: norm("norm2"),
        fc1: linear("fc1", mlp, VIT_DIM, s),
        fc2: linear("fc2", VIT_DIM, mlp, s),
    }
}

fn dummy_conv() -> Conv {
    Conv {
        weight: const_weight("w", &[1], 0.0),
        bias: const_weight("b", &[1], 0.0),
    }
}

fn dummy_neck() -> FpnNeck {
    FpnNeck {
        levels: [
            FpnLevel::Up4 {
                dconv0: dummy_conv(),
                dconv1: dummy_conv(),
                conv_1x1: dummy_conv(),
                conv_3x3: dummy_conv(),
            },
            FpnLevel::Up2 {
                dconv: dummy_conv(),
                conv_1x1: dummy_conv(),
                conv_3x3: dummy_conv(),
            },
            FpnLevel::Same {
                conv_1x1: dummy_conv(),
                conv_3x3: dummy_conv(),
            },
        ],
    }
}

/// Random trunk weights at the real geometry; `pattern[i]` says whether block `i` is global.
/// The necks are placeholders (these tests never run them).
fn synthetic_vision(pattern: &[bool]) -> VisionBackbone {
    let mut s = 0x5eed_u64;
    let blocks = pattern.iter().map(|&g| block(g, 64, &mut s)).collect();
    VisionBackbone {
        trunk: VitTrunk {
            patch_embed: rand_weight(
                "patch_embed",
                &[VIT_DIM, 3, VIT_PATCH, VIT_PATCH],
                0.05,
                &mut s,
            ),
            pos_embed: rand_weight("pos_embed", &[1, 24 * 24 + 1, VIT_DIM], 0.1, &mut s),
            ln_pre: norm("ln_pre"),
            blocks,
        },
        convs: dummy_neck(),
        interactive_convs: dummy_neck(),
        propagation_convs: dummy_neck(),
    }
}

fn random_image(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..3 * IMG * IMG)
        .map(|_| lcg(&mut s) * 2.0 - 1.0)
        .collect()
}

/// `img` NCHW with every sample of the windows in `set` changed.
fn change_windows(img: &[f32], set: &WindowSet) -> Vec<f32> {
    let mut out = img.to_vec();
    for w in set.iter() {
        let (wy, wx) = (w / SIDE, w % SIDE);
        for c in 0..3 {
            for y in wy * WIN_PX..(wy + 1) * WIN_PX {
                for v in &mut out[c * IMG * IMG + y * IMG + wx * WIN_PX..][..WIN_PX] {
                    *v = (*v + 0.37).clamp(-1.0, 1.0) * 0.99;
                }
            }
        }
    }
    out
}

fn windows_of(t: &Tensor, be: &CpuBackend) -> Vec<Vec<f32>> {
    let parts = ojas_core::window_partition(be, t, VIT_WINDOW).expect("partition");
    host(be, &parts)
        .chunks_exact(VIT_WINDOW * VIT_WINDOW * VIT_DIM)
        .map(<[f32]>::to_vec)
        .collect()
}

#[test]
fn synthetic_gated_trunk_is_exact_where_designed_and_carries_where_approximate() {
    let be = cpu(Numerics::Exact);
    let vision = synthetic_vision(&[false, true, false]); // run 0, global, run 1 (no global after it)
    let model = Backbone::new(&be, &vision).expect("backbone");
    drop(vision);
    let keep = [0usize, 1, 2];

    let img_a = random_image(1);
    let changed = WindowSet::from_indices(SIDE, &[1, 4, 8]);
    let img_b = change_windows(&img_a, &changed);
    let (ta, tb) = (
        tensor(&be, &img_a, &[1, 3, IMG, IMG]),
        tensor(&be, &img_b, &[1, 3, IMG, IMG]),
    );
    let full = WindowSet::full(SIDE);
    let none = WindowSet::new(SIDE);
    let exact = GatePolicy::Windows {
        approximate_after_global: false,
    };
    let approx = GatePolicy::Windows {
        approximate_after_global: true,
    };

    let mut taps_a = Taps::default();
    let ungated_a = model.trunk(&ta, &keep, &mut taps_a).expect("ungated a");
    let mut taps_b = Taps::default();
    let ungated_b = model.trunk(&tb, &keep, &mut taps_b).expect("ungated b");
    assert!(
        !compare(&host(&be, &ungated_a), &host(&be, &ungated_b)).bitwise,
        "the two frames must differ"
    );

    // (a) every window recomputed, cache empty: bitwise the ungated trunk, taps included.
    let mut cache = TrunkCache::new();
    assert!(!cache.is_valid());
    let mut taps = Taps::default();
    let (out, stats) = model
        .trunk_gated(&ta, &mut cache, &full, exact, &keep, &mut taps)
        .expect("capture");
    assert_same(&be, "capture output", &out, &ungated_a);
    assert_same(
        &be,
        "capture patch_embed",
        taps.patch_embed.as_ref().unwrap(),
        taps_a.patch_embed.as_ref().unwrap(),
    );
    assert_same(
        &be,
        "capture ln_pre",
        taps.ln_pre.as_ref().unwrap(),
        taps_a.ln_pre.as_ref().unwrap(),
    );
    for &i in &keep {
        assert_same(
            &be,
            &format!("capture block {i}"),
            taps.blocks[i].as_ref().unwrap(),
            taps_a.blocks[i].as_ref().unwrap(),
        );
    }
    assert_eq!(
        stats,
        GatedStats {
            mode: GatedMode::Full,
            window_blocks_run: 18,
            window_blocks_total: 18,
            global_blocks_run: 1,
            global_blocks_total: 1
        }
    );
    assert!(cache.is_valid());

    // Identical input, nothing to recompute: the cached output, bitwise, and no block runs.
    let (out, stats) = model
        .trunk_gated(&ta, &mut cache, &none, exact, &keep, &mut Taps::default())
        .expect("reuse");
    assert_same(&be, "reused output", &out, &ungated_a);
    assert_eq!(
        (
            stats.mode,
            stats.window_blocks_run,
            stats.global_blocks_run,
            stats.flops()
        ),
        (GatedMode::Reused, 0, 0, 0.0)
    );

    // (b) exact policy on the frame where 3 windows changed: bitwise at every exposed block.
    let cache_a = cache.clone();
    let mut taps = Taps::default();
    let (out, stats) = model
        .trunk_gated(&tb, &mut cache, &changed, exact, &keep, &mut taps)
        .expect("exact partial");
    assert_same(&be, "exact output", &out, &ungated_b);
    for &i in &keep {
        assert_same(
            &be,
            &format!("exact block {i}"),
            taps.blocks[i].as_ref().unwrap(),
            taps_b.blocks[i].as_ref().unwrap(),
        );
    }
    // Run 0 on 3 windows, the global block, run 1 on all 9.
    assert_eq!(
        stats,
        GatedStats {
            mode: GatedMode::Windows,
            window_blocks_run: 3 + 9,
            window_blocks_total: 18,
            global_blocks_run: 1,
            global_blocks_total: 1
        }
    );
    // And the cache now describes frame B: reusing it is bitwise B.
    let (out, _) = model
        .trunk_gated(&tb, &mut cache, &none, exact, &[], &mut Taps::default())
        .expect("reuse b");
    assert_same(&be, "reuse after partial", &out, &ungated_b);

    // The approximate variant: the unchanged windows of the last run carry frame A's values.
    let mut cache = cache_a.clone();
    let mut taps = Taps::default();
    let (out_approx, stats) = model
        .trunk_gated(&tb, &mut cache, &changed, approx, &keep, &mut taps)
        .expect("approx partial");
    assert_eq!(
        stats.window_blocks_run,
        3 + 3,
        "run 0 and run 1 both on 3 windows"
    );
    assert_eq!(
        (stats.mode, stats.global_blocks_run),
        (GatedMode::Windows, 1)
    );
    let carried = windows_of(&out_approx, &be);
    let from_a = windows_of(taps_a.blocks[2].as_ref().unwrap(), &be);
    let from_exact_b = windows_of(taps_b.blocks[2].as_ref().unwrap(), &be);
    for w in 0..9 {
        if changed.contains(w) {
            assert_ne!(
                carried[w], from_a[w],
                "recomputed window {w} must not be frame A's"
            );
        } else {
            assert_eq!(
                carried[w], from_a[w],
                "window {w} must be carried bitwise from the cache"
            );
            assert_ne!(
                carried[w], from_exact_b[w],
                "the exact trunk changes window {w}'s output: carrying is an approximation"
            );
        }
    }
    // Run 0 is exact in the approximate variant too: block 0's windows match the exact trunk.
    assert_same(
        &be,
        "approx block 0",
        taps.blocks[0].as_ref().unwrap(),
        taps_b.blocks[0].as_ref().unwrap(),
    );

    // A full set under the approximate policy is a full run.
    let mut cache = cache_a.clone();
    let (out, stats) = model
        .trunk_gated(&tb, &mut cache, &full, approx, &[], &mut Taps::default())
        .expect("approx full");
    assert_same(&be, "approx with every window changed", &out, &ungated_b);
    assert_eq!(stats.mode, GatedMode::Full);

    // Frame policy: a reuse or a full run, never a partial one.
    let mut cache = cache_a.clone();
    let (out, stats) = model
        .trunk_gated(
            &tb,
            &mut cache,
            &changed,
            GatePolicy::Frame,
            &[],
            &mut Taps::default(),
        )
        .expect("frame full");
    assert_same(&be, "frame policy on a changed frame", &out, &ungated_b);
    assert_eq!(stats.mode, GatedMode::Full);
    let (out, stats) = model
        .trunk_gated(
            &tb,
            &mut cache,
            &none,
            GatePolicy::Frame,
            &[],
            &mut Taps::default(),
        )
        .expect("frame reuse");
    assert_same(&be, "frame policy reuse", &out, &ungated_b);
    assert_eq!(stats.mode, GatedMode::Reused);

    // Off: the ungated trunk, and the cache is dropped.
    let (out, stats) = model
        .trunk_gated(
            &ta,
            &mut cache,
            &none,
            GatePolicy::Off,
            &[],
            &mut Taps::default(),
        )
        .expect("off");
    assert_same(&be, "off", &out, &ungated_a);
    assert_eq!(stats.mode, GatedMode::Full);
    assert!(!cache.is_valid(), "Off drops the cache");

    // A partial request on an invalid cache is a full run, not a read of missing tensors.
    let (out, stats) = model
        .trunk_gated(&tb, &mut cache, &changed, exact, &[], &mut Taps::default())
        .expect("cold partial");
    assert_same(&be, "partial on an empty cache", &out, &ungated_b);
    assert_eq!(stats.mode, GatedMode::Full);
    assert!(cache.is_valid());

    // A window set for another grid is refused.
    assert!(model
        .trunk_gated(
            &tb,
            &mut cache,
            &WindowSet::full(2),
            exact,
            &[],
            &mut Taps::default()
        )
        .is_err());
}

/// The same three-block trunk at 672 pixels (48 x 48 tokens, 2 x 2 windows): the geometry is a
/// runtime value, so the gated path, the window gather and scatter and the stats follow it; the
/// global block's RoPE table, stored for 72 x 72 in the checkpoint, is regenerated for 48 x 48,
/// and the result is bit-identical to a trunk that was handed the generated table as its stored one.
#[test]
fn synthetic_trunk_at_672_follows_the_geometry_and_regenerates_the_global_rope_table() {
    let be = cpu(Numerics::Exact);
    let geo = Geometry::from_image_side(672).expect("672");
    let (grid, side, px) = (geo.grid, geo.side(), geo.pixels());
    assert_eq!((grid, side, px), (48, 2, 672));
    let vision = synthetic_vision(&[false, true, false]);
    assert_eq!(
        vision.trunk.blocks[1].freqs_cis.shape,
        vec![5184, 32, 2],
        "the stored global table is the 72 x 72 one"
    );
    let model = Backbone::new_at(&be, &vision, geo).expect("backbone at 672");
    assert_eq!(model.geometry(), geo);
    let keep = [0usize, 1, 2];

    let mut s = 7u64;
    let img_a: Vec<f32> = (0..3 * px * px).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();
    let changed = WindowSet::from_indices(side, &[1, 2]);
    let mut img_b = img_a.clone();
    let win_px = VIT_WINDOW * VIT_PATCH;
    for w in changed.iter() {
        let (wy, wx) = (w / side, w % side);
        for c in 0..3 {
            for y in wy * win_px..(wy + 1) * win_px {
                for v in &mut img_b[c * px * px + y * px + wx * win_px..][..win_px] {
                    *v = (*v + 0.37).clamp(-1.0, 1.0) * 0.99;
                }
            }
        }
    }
    let (ta, tb) = (
        tensor(&be, &img_a, &[1, 3, px, px]),
        tensor(&be, &img_b, &[1, 3, px, px]),
    );
    let (mut taps_a, mut taps_b) = (Taps::default(), Taps::default());
    let ungated_a = model.trunk(&ta, &keep, &mut taps_a).expect("ungated a");
    let ungated_b = model.trunk(&tb, &keep, &mut taps_b).expect("ungated b");
    assert_eq!(ungated_a.shape(), &[1, grid, grid, VIT_DIM]);
    assert!(
        !compare(&host(&be, &ungated_a), &host(&be, &ungated_b)).bitwise,
        "the two frames must differ"
    );

    // Gated, exactly as at 1008: capture is the ungated trunk, a partial recompute of 2 of 4
    // windows is bitwise the ungated trunk of the new frame, the stats count 4 windows per block.
    let exact = GatePolicy::Windows {
        approximate_after_global: false,
    };
    let mut cache = TrunkCache::new();
    let (out, stats) = model
        .trunk_gated(
            &ta,
            &mut cache,
            &WindowSet::full(side),
            exact,
            &keep,
            &mut Taps::default(),
        )
        .expect("capture");
    assert_same(&be, "capture at 672", &out, &ungated_a);
    assert_eq!(
        stats,
        GatedStats {
            mode: GatedMode::Full,
            window_blocks_run: 8,
            window_blocks_total: 8,
            global_blocks_run: 1,
            global_blocks_total: 1
        }
    );
    let mut taps = Taps::default();
    let (out, stats) = model
        .trunk_gated(&tb, &mut cache, &changed, exact, &keep, &mut taps)
        .expect("exact partial");
    assert_same(&be, "exact partial at 672", &out, &ungated_b);
    for &i in &keep {
        assert_same(
            &be,
            &format!("exact block {i} at 672"),
            taps.blocks[i].as_ref().unwrap(),
            taps_b.blocks[i].as_ref().unwrap(),
        );
    }
    // Run 0 on 2 windows, the global block, run 1 on all 4: 6 of 8 window-blocks.
    assert_eq!(
        stats,
        GatedStats {
            mode: GatedMode::Windows,
            window_blocks_run: 2 + 4,
            window_blocks_total: 8,
            global_blocks_run: 1,
            global_blocks_total: 1
        }
    );
    // A global block weighs `side^2` window-blocks at this grid, nine at the default one.
    assert!((stats.block_fraction_at(geo) - (6.0 + 4.0) / (8.0 + 4.0)).abs() < 1e-12);
    assert!((stats.block_fraction() - (6.0 + 9.0) / (8.0 + 9.0)).abs() < 1e-12);
    assert!(stats.flop_fraction_at(geo) < 1.0 && stats.flops_at(geo) > 0.0);
    // Sets and images for another size are refused, not reinterpreted.
    assert!(model
        .trunk_gated(
            &tb,
            &mut cache,
            &WindowSet::full(3),
            exact,
            &[],
            &mut Taps::default()
        )
        .is_err());
    let native = tensor(&be, &random_image(3), &[1, 3, IMG, IMG]);
    assert!(
        model.trunk(&native, &[], &mut Taps::default()).is_err(),
        "a 1008 image on a 672 backbone"
    );

    // The generated table is what the stored one would be: hand the same weights a stored
    // 48 x 48 table made by the generator and the trunk is bitwise the same.
    let (cos, sin) = st_model::backbone::vit_rope_tables(grid, be.budget()).expect("tables");
    let (cos, sin) = (
        cos.to_host(be.budget()).unwrap().to_f32_vec().unwrap(),
        sin.to_host(be.budget()).unwrap().to_f32_vec().unwrap(),
    );
    let mut stored = synthetic_vision(&[false, true, false]);
    stored.trunk.blocks[1].freqs_cis = Weight {
        name: "freqs_cis".into(),
        shape: vec![grid * grid, 32, 2],
        data: cos.iter().zip(&sin).flat_map(|(&c, &s)| [c, s]).collect(),
    };
    let with_stored =
        Backbone::new_at(&be, &stored, geo).expect("backbone with a stored 48 x 48 table");
    assert_same(
        &be,
        "stored versus generated global table",
        &with_stored.trunk(&ta, &[], &mut Taps::default()).unwrap(),
        &ungated_a,
    );
    // ...and it is not the 72 x 72 table's first 2304 rows: positions are stretched, not cropped.
    let mut cropped = synthetic_vision(&[false, true, false]);
    let first = cropped.trunk.blocks[1].freqs_cis.data[..grid * grid * 64].to_vec();
    cropped.trunk.blocks[1].freqs_cis = Weight {
        name: "freqs_cis".into(),
        shape: vec![grid * grid, 32, 2],
        data: first,
    };
    let with_cropped = Backbone::new_at(&be, &cropped, geo).expect("backbone");
    assert!(
        !compare(
            &host(
                &be,
                &with_cropped.trunk(&ta, &[], &mut Taps::default()).unwrap()
            ),
            &host(&be, &ungated_a)
        )
        .bitwise
    );
}

#[test]
fn a_backbone_refuses_a_grid_that_is_not_whole_windows_and_a_wrong_patch() {
    let be = cpu(Numerics::Exact);
    let vision = synthetic_vision(&[false]);
    for bad in [
        Geometry {
            grid: 50,
            patch: 14,
            window: 24,
        },
        Geometry {
            grid: 48,
            patch: 16,
            window: 24,
        },
        Geometry {
            grid: 48,
            patch: 14,
            window: 12,
        },
        Geometry {
            grid: 0,
            patch: 14,
            window: 24,
        },
    ] {
        assert!(Backbone::new_at(&be, &vision, bad).is_err(), "{bad:?}");
    }
    assert!(Backbone::new_at(&be, &vision, Geometry::from_image_side(336).unwrap()).is_ok());
}

// ------------------------------------------------------------ real data

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

const VISEM_VIDEO: &str = "11";

fn visem_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/visem/VISEM_Tracking_Train_v4/Train")
        .join(VISEM_VIDEO)
        .join("images")
}

/// The model-input frame (Pillow bilinear to 1008x1008, still `u8` RGB) and its normalized
/// NCHW f32 tensor. Decoded with `st_video::decode_image` (for JPEG the Pillow-exact native
/// decoder; results logged before 2026-10-27 used ImageIO); the
/// frame bytes the gate compares are the ones the tensor is built from, and that is checked
/// against `sam_input` so the change map is on exactly the model input.
struct Input {
    u8_frame: st_video::Rgb8,
    nchw: Vec<f32>,
}

#[cfg(target_os = "macos")]
fn load_input(name: &str) -> Input {
    let rgb =
        st_video::decode_image(visem_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let u8_frame = st_video::resize_bilinear_pil(&rgb, IMG, IMG);
    let n = IMG * IMG;
    let mut nchw = vec![0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            nchw[c * n + i] = (f32::from(u8_frame.data[i * 3 + c]) / 255.0 - 0.5) / 0.5;
        }
    }
    assert_eq!(
        nchw,
        st_video::sam_input(&rgb, IMG),
        "{name}: the u8 frame must be what sam_input normalizes"
    );
    Input { u8_frame, nchw }
}

#[cfg(not(target_os = "macos"))]
fn load_input(_: &str) -> Input {
    panic!("frame decoding is macOS-only (st_video::decode_image)")
}

fn gate_frame(i: &Input) -> Frame<'_> {
    Frame::new(&i.u8_frame.data, IMG, IMG, 3).expect("frame")
}

fn model_input<B: Backend>(be: &B, i: &Input) -> Tensor {
    tensor(be, &i.nchw, &[1, 3, IMG, IMG])
}

/// `a` with the pixels of the windows in `set` taken from `b`.
fn mosaic(a: &Input, b: &Input, set: &WindowSet) -> Input {
    let (mut u8_frame, mut nchw) = (a.u8_frame.clone(), a.nchw.clone());
    for w in set.iter() {
        let (wy, wx) = (w / SIDE, w % SIDE);
        for y in wy * WIN_PX..(wy + 1) * WIN_PX {
            let at = (y * IMG + wx * WIN_PX) * 3;
            u8_frame.data[at..at + WIN_PX * 3]
                .copy_from_slice(&b.u8_frame.data[at..at + WIN_PX * 3]);
            for c in 0..3 {
                let at = c * IMG * IMG + y * IMG + wx * WIN_PX;
                nchw[at..at + WIN_PX].copy_from_slice(&b.nchw[at..at + WIN_PX]);
            }
        }
    }
    Input { u8_frame, nchw }
}

fn load_vision() -> VisionBackbone {
    let ck =
        st_weights::Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("open checkpoint");
    VisionBackbone::load(&ck).expect("vision weights")
}

const REAL_BLOCKS: [usize; 8] = [6, 7, 14, 15, 22, 23, 30, 31];

/// (a) and (b) on the real trunk and two real VISEM frames, on one CPU numerics mode.
/// The bitwise claims (a) are asserted whenever the backend repeats itself (which under `Exact`
/// is itself asserted); the partial-recompute claims (b) only when `assert_partial`, else reported.
fn exactness_on_real_frames(vision: &VisionBackbone, numerics: Numerics, assert_partial: bool) {
    let label = format!("{numerics:?}");
    let be = cpu(numerics);
    let model = Backbone::new(&be, vision).expect("backbone");
    let (f0, f1) = (load_input("11_frame_0.jpg"), load_input("11_frame_1.jpg"));

    // Which windows differ between the two frames at tau 0, through the gate itself.
    let geo = Geometry::SAM3;
    let mut probe = Gate::new(GateConfig::exact(true), geo).expect("gate");
    probe.decide(&gate_frame(&f0)).expect("first");
    probe.decide(&gate_frame(&f1)).expect("second");
    let all_differ = probe.last_map().windows().clone();
    println!("[{label}] frames 0 and 1 of video {VISEM_VIDEO}: {}/9 windows differ at tau 0, {:.1}% of patches", all_differ.count(), 100.0 * probe.last_map().patch_fraction());

    // The frame pair for (b): windows 0, 4 and 8 of frame 1 pasted into frame 0.
    let wanted = WindowSet::from_indices(SIDE, &[0, 4, 8]);
    let b = mosaic(&f0, &f1, &wanted);
    let mut gate = Gate::new(GateConfig::exact(true), geo).expect("gate");
    assert_eq!(
        gate.decide(&gate_frame(&f0)).expect("decide a"),
        Decision::Full(st_gate::FullReason::First)
    );
    let recompute = match gate.decide(&gate_frame(&b)).expect("decide b") {
        Decision::Windows(s) => s,
        other => panic!("expected a window subset for the mosaic frame, got {other:?} (windows 0, 4, 8 of frame 1 must differ from frame 0)"),
    };
    assert!(
        recompute.iter().all(|w| wanted.contains(w)),
        "the gate named windows outside the pasted ones: {:?}",
        recompute.iter().collect::<Vec<_>>()
    );
    println!(
        "[{label}] gate at tau 0 on the mosaic frame: recompute windows {:?}",
        recompute.iter().collect::<Vec<_>>()
    );

    let (ta, tb) = (model_input(&be, &f0), model_input(&be, &b));
    let all_blocks: Vec<usize> = (0..32).collect();
    let timed = |f: &mut dyn FnMut()| {
        let t = Instant::now();
        f();
        t.elapsed().as_secs_f64()
    };

    // Ungated references; frame A twice, to know whether the backend repeats itself at all.
    let mut taps_a = Taps::default();
    let mut ungated_a = None;
    let s_ungated = timed(&mut || {
        ungated_a = Some(
            model
                .trunk(&ta, &all_blocks, &mut taps_a)
                .expect("ungated a"),
        )
    });
    let ungated_a = ungated_a.unwrap();
    let again = model
        .trunk(&ta, &[], &mut Taps::default())
        .expect("ungated a again");
    let repeatable = compare(&host(&be, &ungated_a), &host(&be, &again)).bitwise;
    println!("[{label}] ungated trunk {s_ungated:.1} s; the same input run twice is bitwise equal: {repeatable}");
    let mut taps_b = Taps::default();
    let ungated_b = model
        .trunk(&tb, &all_blocks, &mut taps_b)
        .expect("ungated b");
    if numerics == Numerics::Exact {
        assert!(
            repeatable,
            "[{label}] the CPU backend under Numerics::Exact must repeat itself bitwise"
        );
    }

    let exact = GatePolicy::Windows {
        approximate_after_global: false,
    };
    let approx = GatePolicy::Windows {
        approximate_after_global: true,
    };
    let full = WindowSet::full(SIDE);
    let mut rows = Vec::new();
    let check = |name: &str, got: &Tensor, want: &Tensor, must: bool, rows: &mut Vec<String>| {
        let c = compare(&host(&be, got), &host(&be, want));
        rows.push(format!(
            "| {label} | {name} | {} | {:.9} | {:.3e} |",
            if c.bitwise { "yes" } else { "no" },
            c.cos,
            c.max_abs
        ));
        if must {
            assert!(c.bitwise, "[{label}] {name}: not bitwise equal to the ungated trunk (cosine {:.9}, max abs {:.3e})", c.cos, c.max_abs);
        }
        c
    };

    // (a) all windows changed, cold cache: the gated trunk is the ungated trunk.
    let mut cache = TrunkCache::new();
    let mut taps = Taps::default();
    let mut s_cap = 0.0;
    let mut cap = None;
    s_cap += timed(&mut || {
        cap = Some(
            model
                .trunk_gated(&ta, &mut cache, &full, exact, &all_blocks, &mut taps)
                .expect("capture"),
        )
    });
    let (out, stats) = cap.unwrap();
    assert_eq!(stats.mode, GatedMode::Full);
    let must_a = repeatable;
    let must_b = repeatable && assert_partial;
    check(
        "(a) all windows changed: trunk output",
        &out,
        &ungated_a,
        must_a,
        &mut rows,
    );
    for i in [0usize, 6, 7, 15, 23, 31] {
        check(
            &format!("(a) all windows changed: block {i}"),
            taps.blocks[i].as_ref().unwrap(),
            taps_a.blocks[i].as_ref().unwrap(),
            must_a,
            &mut rows,
        );
    }
    println!("[{label}] gated, every window recomputed, cache fill: {s_cap:.1} s");
    // Identical input again: reused, bitwise.
    let (out, stats) = model
        .trunk_gated(
            &ta,
            &mut cache,
            &WindowSet::new(SIDE),
            exact,
            &[],
            &mut Taps::default(),
        )
        .expect("reuse");
    assert_eq!(stats.mode, GatedMode::Reused);
    check(
        "(a) identical input: reused trunk output",
        &out,
        &ungated_a,
        must_a,
        &mut rows,
    );

    // (b) exact policy on the mosaic pair.
    let cache_a = cache.clone();
    let mut taps = Taps::default();
    let mut part = None;
    let s_exact = timed(&mut || {
        part = Some(
            model
                .trunk_gated(&tb, &mut cache, &recompute, exact, &REAL_BLOCKS, &mut taps)
                .expect("exact partial"),
        )
    });
    let (out_exact, stats_exact) = part.unwrap();
    for &i in &REAL_BLOCKS {
        check(
            &format!("(b) exact policy: block {i}"),
            taps.blocks[i].as_ref().unwrap(),
            taps_b.blocks[i].as_ref().unwrap(),
            must_b,
            &mut rows,
        );
    }
    check(
        "(b) exact policy: trunk output",
        &out_exact,
        &ungated_b,
        must_b,
        &mut rows,
    );
    println!("[{label}] exact policy on {} changed windows: {s_exact:.1} s, {:.1}% of the blocks, {:.1}% of the FLOPs", recompute.count(), 100.0 * stats_exact.block_fraction(), 100.0 * stats_exact.flop_fraction());

    // (b) approximate policy: reported, with the neck outputs.
    let mut cache = cache_a;
    let mut taps = Taps::default();
    let mut part = None;
    let s_approx = timed(&mut || {
        part = Some(
            model
                .trunk_gated(&tb, &mut cache, &recompute, approx, &REAL_BLOCKS, &mut taps)
                .expect("approx partial"),
        )
    });
    let (out_approx, stats_approx) = part.unwrap();
    println!("[{label}] approximate policy on {} changed windows: {s_approx:.1} s, {:.1}% of the blocks, {:.1}% of the FLOPs", recompute.count(), 100.0 * stats_approx.block_fraction(), 100.0 * stats_approx.flop_fraction());
    let mut approx_rows = Vec::new();
    let mut worst = 1.0f64;
    for &i in &REAL_BLOCKS {
        let c = compare(
            &host(&be, taps.blocks[i].as_ref().unwrap()),
            &host(&be, taps_b.blocks[i].as_ref().unwrap()),
        );
        worst = worst.min(c.cos);
        approx_rows.push(format!(
            "| {label} | approximate: block {i} | {} | {:.7} | {:.3e} |",
            if c.bitwise { "yes" } else { "no" },
            c.cos,
            c.max_abs
        ));
    }
    let c = compare(&host(&be, &out_approx), &host(&be, &ungated_b));
    worst = worst.min(c.cos);
    approx_rows.push(format!(
        "| {label} | approximate: trunk output | {} | {:.7} | {:.3e} |",
        if c.bitwise { "yes" } else { "no" },
        c.cos,
        c.max_abs
    ));
    for (tag, neck) in [
        ("detector", Neck::Detector),
        ("interactive", Neck::Interactive),
        ("propagation", Neck::Propagation),
    ] {
        let (ours, theirs) = (
            model.neck(neck, &out_approx).expect("neck"),
            model.neck(neck, &ungated_b).expect("neck"),
        );
        for (lvl, (o, t)) in ours.iter().zip(&theirs).enumerate() {
            let c = compare(&host(&be, o), &host(&be, t));
            worst = worst.min(c.cos);
            approx_rows.push(format!(
                "| {label} | approximate: {tag} neck level {lvl} | {} | {:.7} | {:.3e} |",
                if c.bitwise { "yes" } else { "no" },
                c.cos,
                c.max_abs
            ));
        }
    }
    println!("\n| numerics | stage (gated vs ungated, frame B) | bitwise | cosine | max abs |\n|---|---|---|---|---|");
    rows.iter()
        .chain(&approx_rows)
        .for_each(|r| println!("{r}"));
    println!();
    // A floor that catches a mis-scattered window, not a quality bar: the real numbers are in the table.
    assert!(worst > 0.9, "[{label}] approximate policy: worst cosine {worst} is far below anything a stale-window carry explains");
}

#[test]
#[ignore = "needs the gated checkpoint and VISEM frames: set SPIKETRACK_WEIGHTS"]
fn gated_trunk_exactness_on_real_frames_cpu_exact_numerics() {
    exactness_on_real_frames(&load_vision(), Numerics::Exact, true);
}

#[test]
#[ignore = "needs the gated checkpoint and VISEM frames: set SPIKETRACK_WEIGHTS"]
fn gated_trunk_exactness_on_real_frames_cpu_fast_numerics_reported() {
    // Fast: the all-windows-changed and reuse claims hold when the backend repeats itself; the
    // partial-recompute claim is reported (Accelerate's blocking depends on the batch shape).
    exactness_on_real_frames(&load_vision(), Numerics::Fast, false);
}

struct PolicyRun {
    name: String,
    decisions: String,
    secs: f64,
    gate_secs: f64,
    block_frac: f64,
    flop_frac: f64,
    min_cos: f64,
    hash: u64,
}

fn letter(d: &Decision, set_count: usize) -> String {
    match d {
        Decision::Full(_) => "F".into(),
        Decision::Reuse => "R".into(),
        Decision::Windows(_) => format!("W{set_count}"),
    }
}

/// One policy over the frames, in order, with the gate and trunk cache a real stream would have.
fn run_policy<B: Backend>(
    be: &B,
    model: &Backbone<'_, B>,
    name: &str,
    cfg: GateConfig,
    inputs: &[Input],
    tensors: &[Tensor],
    truth: &[Vec<f32>],
) -> PolicyRun {
    let mut gate = Gate::new(cfg, Geometry::SAM3).expect("gate");
    let mut cache = TrunkCache::new();
    let (mut decisions, mut secs, mut gate_secs) = (Vec::new(), 0.0, 0.0);
    let (mut blocks, mut flops) = (0.0, 0.0);
    let (mut min_cos, mut hash) = (1.0f64, 0xcbf29ce484222325u64);
    for (i, (input, tensor)) in inputs.iter().zip(tensors).enumerate() {
        let t = Instant::now();
        let decision = gate.decide(&gate_frame(input)).expect("decide");
        gate_secs += t.elapsed().as_secs_f64();
        let set = decision.recompute_set(SIDE);
        let (out, stats) = model
            .trunk_gated(
                tensor,
                &mut cache,
                &set,
                cfg.policy,
                &[],
                &mut Taps::default(),
            )
            .unwrap_or_else(|e| {
                gate.reset();
                panic!("frame {i}: {e}")
            });
        be.sync().expect("sync");
        secs += t.elapsed().as_secs_f64();
        let got = host(be, &out);
        min_cos = min_cos.min(compare(&got, &truth[i]).cos);
        hash = hash.wrapping_mul(0x100000001b3) ^ fnv(&got);
        blocks += stats.block_fraction();
        flops += stats.flop_fraction();
        decisions.push(letter(&decision, set.count()));
    }
    let n = inputs.len() as f64;
    PolicyRun {
        name: name.into(),
        decisions: decisions.join(" "),
        secs,
        gate_secs,
        block_frac: blocks / n,
        flop_frac: flops / n,
        min_cos,
        hash,
    }
}

/// (c) recompute fraction and wall time per policy on ten consecutive VISEM frames, CPU Fast
/// numerics (the backend the parity gate and the timings use). Quality is the cosine of the
/// gated trunk output to the ungated trunk output of the same frame.
#[test]
#[ignore = "needs the gated checkpoint and VISEM frames: set SPIKETRACK_WEIGHTS"]
fn policy_recompute_fraction_and_wall_time_on_ten_consecutive_visem_frames() {
    const FRAMES: usize = 10;
    let vision = load_vision();
    let be = cpu(Numerics::Fast);
    let model = Backbone::new(&be, &vision).expect("backbone");
    drop(vision);
    let inputs: Vec<Input> = (0..FRAMES)
        .map(|i| load_input(&format!("{VISEM_VIDEO}_frame_{i}.jpg")))
        .collect();
    let tensors: Vec<Tensor> = inputs.iter().map(|i| model_input(&be, i)).collect();

    // Change statistics of the stream, per threshold (what the gate sees, frame t against t-1).
    println!("\nchange rates between consecutive frames of video {VISEM_VIDEO} (frames 0..{FRAMES}), model-input u8 frames");
    println!("| tau | patches changed | windows changed |\n|---|---|---|");
    for tau in [0u32, 4, 8, 16, 32] {
        let mut mapper = st_gate::ChangeMapper::new(Geometry::SAM3).expect("mapper");
        let mut map = st_gate::ChangeMap::new(Geometry::SAM3);
        let (mut p, mut w) = (0.0, 0.0);
        for pair in inputs.windows(2) {
            mapper
                .compute(&gate_frame(&pair[0]), &gate_frame(&pair[1]), tau, &mut map)
                .expect("map");
            p += map.patch_fraction();
            w += map.window_fraction();
        }
        let n = (FRAMES - 1) as f64;
        println!("| {tau} | {:.4} | {:.3} |", p / n, w / n);
    }

    // Ground truth: the ungated trunk on every frame, timed. The first frame is run twice (the
    // repeat is not timed): the determinism check below is only meaningful on a backend that
    // repeats itself.
    let mut truth = Vec::new();
    let mut ungated_secs = 0.0;
    for x in &tensors {
        let t = Instant::now();
        let out = model.trunk(x, &[], &mut Taps::default()).expect("ungated");
        be.sync().expect("sync");
        ungated_secs += t.elapsed().as_secs_f64();
        truth.push(host(&be, &out));
    }
    let repeatable = compare(
        &host(
            &be,
            &model
                .trunk(&tensors[0], &[], &mut Taps::default())
                .expect("ungated again"),
        ),
        &truth[0],
    )
    .bitwise;
    println!("ungated trunk repeats itself bitwise on this backend: {repeatable}");

    let win = |approximate_after_global| GatePolicy::Windows {
        approximate_after_global,
    };
    let cfg = |policy, tau, refresh| GateConfig {
        refresh_every: refresh,
        ..GateConfig::new(policy, tau)
    };
    let configs: Vec<(String, GateConfig)> = vec![
        ("frame, tau 8".into(), cfg(GatePolicy::Frame, 8, 0)),
        ("windows exact, tau 8".into(), cfg(win(false), 8, 0)),
        ("windows exact, tau 16".into(), cfg(win(false), 16, 0)),
        ("windows exact, tau 32".into(), cfg(win(false), 32, 0)),
        (
            "windows approx, tau 8, refresh 5".into(),
            cfg(win(true), 8, 5),
        ),
        (
            "windows approx, tau 16, refresh 5".into(),
            cfg(win(true), 16, 5),
        ),
        (
            "windows approx, tau 32, refresh 5".into(),
            cfg(win(true), 32, 5),
        ),
    ];
    let mut runs: Vec<PolicyRun> = configs
        .iter()
        .map(|(n, c)| run_policy(&be, &model, n, *c, &inputs, &tensors, &truth))
        .collect();

    println!(
        "\nungated: {:.2} s for {FRAMES} frames ({:.2} s/frame)",
        ungated_secs,
        ungated_secs / FRAMES as f64
    );
    println!("\n| policy | decisions (F full, R reuse, Wk k windows) | blocks run | model FLOPs run | s/frame | speedup | gate ms/frame | min cosine to ungated |\n|---|---|---|---|---|---|---|---|");
    println!(
        "| ungated | F x {FRAMES} | 100.0% | 100.0% | {:.2} | 1.00x | 0 | 1.0000000 |",
        ungated_secs / FRAMES as f64
    );
    for r in &runs {
        println!(
            "| {} | {} | {:.1}% | {:.1}% | {:.2} | {:.2}x | {:.3} | {:.7} |",
            r.name,
            r.decisions,
            100.0 * r.block_frac,
            100.0 * r.flop_frac,
            r.secs / FRAMES as f64,
            ungated_secs / r.secs,
            1e3 * r.gate_secs / FRAMES as f64,
            r.min_cos
        );
    }

    // Two runs on the same video give identical output hashes (acceptance criterion), on a backend that repeats itself.
    let again = run_policy(
        &be,
        &model,
        &configs[4].0,
        configs[4].1,
        &inputs,
        &tensors,
        &truth,
    );
    let first = runs.remove(4);
    println!(
        "\ndeterminism: run 1 hash {:016x}, run 2 hash {:016x} ({})",
        first.hash, again.hash, configs[4].0
    );
    if repeatable {
        assert_eq!(
            first.hash, again.hash,
            "two runs over the same frames gave different outputs"
        );
        assert_eq!(first.decisions, again.decisions);
    } else {
        println!(
            "(not asserted: the ungated trunk itself is not bitwise repeatable on this backend)"
        );
    }
}
