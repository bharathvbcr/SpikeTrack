//! Memory attention with the stored frames' projections cached, against the uncached forward.
//!
//! Random weights at a small token grid (8 x 8 = 64 tokens a frame; the real 256-wide, 8-head
//! layers), `CpuBackend` under `Numerics::Exact`, so the claims are bitwise:
//!
//! 1. `fwd_is_unchanged`: the uncached `MemoryAttention::fwd` against a hash of its output bits
//!    taken before the cached path existed, so sharing the layer loop cannot move it unseen.
//! 2. `cached_equals_uncached`: `fwd_frames` over per-frame `frame_kv` projections, bit for bit
//!    against `fwd` on the concatenated memory, at 1, 3 and 7 memory frames, with and without
//!    pointers.
//! 3. `cached_equals_uncached_on_metal` (ignored: needs a Metal GPU): the same on Metal, where
//!    it holds only if Metal's linear gives a row the same bits at any row count.
//!    `cached_equals_uncached_on_metal_at_tracker_size` repeats it at the tracker's 72 x 72 grid.
//! 4. `fwd_frames_refuses_malformed_inputs`.
//!
//! `cargo test -p st-model --release --test memory_attention_cache [-- --ignored]`

use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use st_gate::Geometry;
use st_model::tracker::{FrameKv, FramesInputs, MemoryAttention, MemoryInputs};
use st_weights::{TensorGroup, Weight};

const D: usize = 256;
const PREFIX: &str = "tracker.model.transformer.encoder";
const GEO: Geometry = Geometry {
    grid: 8,
    patch: 14,
    window: 4,
};
/// (memory frames, pointer tokens) per case: no pointers, a few, and the full 7-frame memory.
const CASES: [(usize, usize); 3] = [(1, 0), (3, 16), (7, 32)];

fn cpu() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(4 << 30), 4)
        .expect("cpu backend")
        .with_numerics(Numerics::Exact)
}

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 40) as f32) / (1u64 << 24) as f32 // [0, 1)
}

fn rand_vec(n: usize, scale: f32, s: &mut u64) -> Vec<f32> {
    (0..n).map(|_| (lcg(s) - 0.5) * 2.0 * scale).collect()
}

/// Every tensor `MemoryAttention::load_at` reads, random.
fn synthetic_group() -> TensorGroup {
    let mut s = 0x6d65_6d61_u64;
    let mut tensors = std::collections::BTreeMap::new();
    let mut put = |name: String, shape: Vec<usize>, scale: f32, s: &mut u64| {
        let n = shape.iter().product();
        let data = rand_vec(n, scale, s);
        tensors.insert(name.clone(), Weight { name, shape, data });
    };
    for i in 0..4 {
        let l = format!("{PREFIX}.layers.{i}");
        for p in [
            "self_attn_q_proj",
            "self_attn_k_proj",
            "self_attn_v_proj",
            "self_attn_out_proj",
            "cross_attn_q_proj",
            "cross_attn_k_proj",
            "cross_attn_v_proj",
            "cross_attn_out_proj",
            "image_cross_attn_q_proj",
            "image_cross_attn_k_proj",
        ] {
            put(format!("{l}.{p}.weight"), vec![D, D], 0.06, &mut s);
            put(format!("{l}.{p}.bias"), vec![D], 0.02, &mut s);
        }
        put(format!("{l}.linear1.weight"), vec![2048, D], 0.06, &mut s);
        put(format!("{l}.linear1.bias"), vec![2048], 0.02, &mut s);
        put(format!("{l}.linear2.weight"), vec![D, 2048], 0.02, &mut s);
        put(format!("{l}.linear2.bias"), vec![D], 0.02, &mut s);
        for n in ["norm1", "norm2", "norm3"] {
            put(format!("{l}.{n}.weight"), vec![D], 0.1, &mut s);
            put(format!("{l}.{n}.bias"), vec![D], 0.1, &mut s);
        }
    }
    put(format!("{PREFIX}.norm.weight"), vec![D], 0.1, &mut s);
    put(format!("{PREFIX}.norm.bias"), vec![D], 0.1, &mut s);
    // LayerNorm gains near 1, as trained ones are.
    for (name, w) in tensors.iter_mut() {
        if name.contains("norm") && name.ends_with(".weight") {
            w.data.iter_mut().for_each(|v| *v += 1.0);
        }
    }
    TensorGroup {
        prefix: format!("{PREFIX}."),
        tensors,
    }
}

/// One case's inputs on the host: per stored frame its image features, mask memory and
/// position rows (`[T, D]` each), the pointer block and its positions (`[P, D]`), and the
/// current frame's features and positions.
struct Case {
    frames: Vec<[Vec<f32>; 3]>,
    ptrs: Vec<f32>,
    ptrs_pos: Vec<f32>,
    image: Vec<f32>,
    src_pos: Vec<f32>,
}

fn case(geo: Geometry, frames: usize, ptrs: usize, seed: u64) -> Case {
    let t = geo.tokens();
    let mut s = seed;
    Case {
        frames: (0..frames)
            .map(|_| {
                [
                    rand_vec(t * D, 1.0, &mut s),
                    rand_vec(t * D, 1.0, &mut s),
                    rand_vec(t * D, 0.5, &mut s),
                ]
            })
            .collect(),
        ptrs: rand_vec(ptrs * D, 1.0, &mut s),
        ptrs_pos: rand_vec(ptrs * D, 0.5, &mut s),
        image: rand_vec(t * D, 1.0, &mut s),
        src_pos: rand_vec(t * D, 0.5, &mut s),
    }
}

fn up<B: Backend>(be: &B, data: &[f32], rows: usize) -> Tensor {
    be.upload(&Tensor::from_f32(data, &[1, rows, D], be.budget()).expect("host tensor"))
        .expect("upload")
}

fn host<B: Backend>(be: &B, t: &Tensor) -> Vec<f32> {
    be.download(t).expect("download").to_f32_vec().expect("f32")
}

/// The uncached forward on the concatenated inputs, as `TrackerState::compute_propagation`
/// built them before the cache.
fn uncached<B: Backend>(be: &B, attn: &MemoryAttention, c: &Case) -> Vec<f32> {
    let t = c.image.len() / D;
    let p = c.ptrs.len() / D;
    let cat = |i: usize| -> Vec<f32> { c.frames.iter().flat_map(|f| f[i].clone()).collect() };
    let (mimage, mmem, mpos) = (cat(0), cat(1), cat(2));
    let rows = c.frames.len() * t;
    let memory: Vec<f32> = mmem.iter().chain(&c.ptrs).copied().collect();
    let memory_pos: Vec<f32> = mpos.iter().chain(&c.ptrs_pos).copied().collect();
    let (image, src_pos) = (up(be, &c.image, t), up(be, &c.src_pos, t));
    let (memory_image, memory_image_pos) = (up(be, &mimage, rows), up(be, &mpos, rows));
    let (memory, memory_pos) = (up(be, &memory, rows + p), up(be, &memory_pos, rows + p));
    let out = attn
        .fwd(
            be,
            &MemoryInputs {
                image: &image,
                src: &image,
                src_pos: &src_pos,
                memory_image: &memory_image,
                memory_image_pos: &memory_image_pos,
                memory: &memory,
                memory_pos: &memory_pos,
                num_obj_ptr_tokens: p,
            },
        )
        .expect("fwd");
    host(be, &out)
}

/// The cached forward: each stored frame projected once by `frame_kv`, then `fwd_frames`.
fn cached<B: Backend>(be: &B, attn: &MemoryAttention, c: &Case) -> Vec<f32> {
    let t = c.image.len() / D;
    let p = c.ptrs.len() / D;
    let dev: Vec<[Tensor; 3]> = c
        .frames
        .iter()
        .map(|f| [up(be, &f[0], t), up(be, &f[1], t), up(be, &f[2], t)])
        .collect();
    let kvs: Vec<FrameKv> = dev
        .iter()
        .map(|f| attn.frame_kv(be, &f[0], &f[1]).expect("frame_kv"))
        .collect();
    let frames: Vec<(&FrameKv, &Tensor)> =
        kvs.iter().zip(&dev).map(|(kv, f)| (kv, &f[2])).collect();
    let ptrs = (p > 0).then(|| (up(be, &c.ptrs, p), up(be, &c.ptrs_pos, p)));
    let (image, src_pos) = (up(be, &c.image, t), up(be, &c.src_pos, t));
    let out = attn
        .fwd_frames(
            be,
            &FramesInputs {
                image: &image,
                src_pos: &src_pos,
                frames: &frames,
                ptrs: ptrs.as_ref().map(|(a, b)| (a, b)),
            },
        )
        .expect("fwd_frames");
    host(be, &out)
}

/// Count of elements whose bits differ, and the largest absolute difference.
fn diff(got: &[f32], want: &[f32]) -> (usize, f32) {
    assert_eq!(got.len(), want.len(), "length");
    got.iter().zip(want).fold((0, 0f32), |(n, m), (&g, &w)| {
        assert!(g.is_finite() && w.is_finite(), "non-finite output");
        (
            n + usize::from(g.to_bits() != w.to_bits()),
            m.max((g - w).abs()),
        )
    })
}

fn cached_equals_uncached_on<B: Backend>(
    be: &B,
    label: &str,
    geo: Geometry,
    cases: &[(usize, usize)],
) {
    let attn = MemoryAttention::load_at(be, &synthetic_group(), PREFIX, geo).expect("load");
    for (i, &(f, p)) in cases.iter().enumerate() {
        let c = case(geo, f, p, 100 + i as u64);
        let (n, max_abs) = diff(&cached(be, &attn, &c), &uncached(be, &attn, &c));
        println!(
            "[{label}] {f} frame(s), {p} pointer(s): {n} element(s) differ, max abs {max_abs:e}"
        );
        assert_eq!(
            n, 0,
            "[{label}] case {i}: cached output is not bitwise the uncached one"
        );
    }
}

/// FNV-1a over the output's f32 bits.
fn bits_hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    })
}

/// Output-bit hashes of `fwd` per case of [`CASES`], taken on the code before the cache.
const FWD_GOLDEN: [u64; 3] = [0xf415f4f724b886af, 0x99c081c5f257c551, 0xccbad88a35a130ff];

#[test]
fn fwd_is_unchanged() {
    let be = cpu();
    let attn = MemoryAttention::load_at(&be, &synthetic_group(), PREFIX, GEO).expect("load");
    let got: Vec<u64> = CASES
        .iter()
        .enumerate()
        .map(|(i, &(f, p))| {
            let out = uncached(&be, &attn, &case(GEO, f, p, 100 + i as u64));
            assert!(
                out.iter().all(|v| v.is_finite()),
                "case {i}: non-finite output"
            );
            bits_hash(&out)
        })
        .collect();
    assert_eq!(got, FWD_GOLDEN, "fwd output bits moved: {got:#x?}");
}

#[test]
fn cached_equals_uncached() {
    cached_equals_uncached_on(&cpu(), "CPU exact", GEO, &CASES);
}

#[test]
#[ignore = "needs a Metal GPU"]
fn cached_equals_uncached_on_metal() {
    cached_equals_uncached_on(
        &MetalBackend::new(Budget::new(4 << 30)).expect("metal"),
        "Metal",
        GEO,
        &CASES,
    );
}

/// The same at the tracker's own grid (72 x 72 = 5,184 tokens a frame), where Metal's linear
/// may tile by row count: at one memory frame both paths project 5,184 rows, at two the uncached
/// path projects 10,368 in one call. Production predictions first differ at two frames (E23).
#[test]
#[ignore = "needs a Metal GPU"]
fn cached_equals_uncached_on_metal_at_tracker_size() {
    cached_equals_uncached_on(
        &MetalBackend::new(Budget::new(8 << 30)).expect("metal"),
        "Metal 72x72",
        Geometry::SAM3,
        &[(1, 0), (2, 0), (2, 16)],
    );
}

#[test]
fn fwd_frames_refuses_malformed_inputs() {
    let be = cpu();
    let attn = MemoryAttention::load_at(&be, &synthetic_group(), PREFIX, GEO).expect("load");
    let t = GEO.tokens();
    let c = case(GEO, 1, 4, 7);
    let f = [
        up(&be, &c.frames[0][0], t),
        up(&be, &c.frames[0][1], t),
        up(&be, &c.frames[0][2], t),
    ];
    let kv = attn.frame_kv(&be, &f[0], &f[1]).expect("frame_kv");
    let (image, src_pos) = (up(&be, &c.image, t), up(&be, &c.src_pos, t));
    let short = up(&be, &c.ptrs, 4);
    let refused = |r: Result<Tensor, st_model::ModelError>, what: &str| {
        assert!(r.is_err(), "{what} was accepted");
    };
    // A frame's projections from a tensor of the wrong length.
    assert!(
        attn.frame_kv(&be, &short, &f[1]).is_err(),
        "a 4-token image was accepted"
    );
    // No memory at all.
    let none = FramesInputs {
        image: &image,
        src_pos: &src_pos,
        frames: &[],
        ptrs: None,
    };
    refused(attn.fwd_frames(&be, &none), "an empty memory");
    // Positions of the wrong length for a frame.
    let frames = [(&kv, &short)];
    let bad_pos = FramesInputs {
        image: &image,
        src_pos: &src_pos,
        frames: &frames,
        ptrs: None,
    };
    refused(
        attn.fwd_frames(&be, &bad_pos),
        "4-token positions for a frame",
    );
    // Pointers whose positions disagree in length.
    let frames = [(&kv, &f[2])];
    let bad_ptrs = FramesInputs {
        image: &image,
        src_pos: &src_pos,
        frames: &frames,
        ptrs: Some((&short, &image)),
    };
    refused(
        attn.fwd_frames(&be, &bad_ptrs),
        "pointer positions of another length",
    );
    // The current frame in place of its positions' length.
    let bad_image = FramesInputs {
        image: &short,
        src_pos: &src_pos,
        frames: &frames,
        ptrs: None,
    };
    refused(attn.fwd_frames(&be, &bad_image), "a 4-token current frame");
}
