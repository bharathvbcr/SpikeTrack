//! Gating at SAM 3.1's real geometry (1008x1008 RGB, 72x72 tokens, 3x3 windows):
//! the change-map cost against the frame-time budget, determinism, and whole
//! decision sequences. Needs no weights or data.
//!
//! `cargo test --release -p st-gate -- --nocapture --test-threads=2`
//! (the cost assertion is skipped in debug builds, which are 10-50x slower).

use std::time::Instant;

use st_gate::{
    ChangeMap, ChangeMapper, Decision, Frame, FullReason, Gate, GateConfig, GatePolicy, Geometry,
    WindowSet,
};

const GEO: Geometry = Geometry::SAM3;
const SIDE: usize = 1008;
const CH: usize = 3;
const WINDOW_PX: usize = 24 * 14;

/// A lower bound on the ungated frame time, which makes the 5% budget the
/// strictest reading: the backbone is ~4.7 TFLOP of matmul per frame
/// (HANDOFF.md second audit) and tessl's bf16 GEMM peaks at 26.6 TFLOP/s
/// (README), so no real frame is faster than 4.7 / 26.6 = 0.177 s. Measured
/// CPU trunk passes are ~6-9 s.
const FRAME_FLOOR_SECS: f64 = 4.7 / 26.6;
const BUDGET_SECS: f64 = 0.05 * FRAME_FLOOR_SECS;
const REPS: usize = 40;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

/// A textured frame: smooth gradient plus noise, so no shortcut on flat data helps.
fn texture(seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut d = vec![0u8; SIDE * SIDE * CH];
    for y in 0..SIDE {
        for x in 0..SIDE {
            for c in 0..CH {
                d[(y * SIDE + x) * CH + c] =
                    ((x / 5 + y / 7 + c * 30) as u64 % 200 + lcg(&mut s) % 40) as u8;
            }
        }
    }
    d
}

/// `base` with every sample of the windows in `set` shifted by `delta`.
fn perturb(base: &[u8], set: &[usize], delta: u8) -> Vec<u8> {
    let mut d = base.to_vec();
    for &w in set {
        let (wy, wx) = (w / 3, w % 3);
        for y in wy * WINDOW_PX..(wy + 1) * WINDOW_PX {
            let row = &mut d[(y * SIDE + wx * WINDOW_PX) * CH..][..WINDOW_PX * CH];
            row.iter_mut().for_each(|v| *v = v.wrapping_add(delta));
        }
    }
    d
}

fn frame(d: &[u8]) -> Frame<'_> {
    Frame::new(d, SIDE, SIDE, CH).unwrap()
}

fn min_secs(mut f: impl FnMut()) -> f64 {
    (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn change_map_cost_is_under_five_percent_of_the_fastest_possible_frame() {
    let a = texture(1);
    let b = perturb(&a, &[0, 1, 2, 3, 4, 5, 6, 7, 8], 40); // every window differs: the map's whole work
    let mut mapper = ChangeMapper::new(GEO).unwrap();
    let mut map = ChangeMap::new(GEO);
    let (fa, fb) = (frame(&a), frame(&b));

    let map_s = min_secs(|| mapper.compute(&fa, &fb, 8, &mut map).unwrap());
    assert_eq!(
        map.windows_changed(),
        9,
        "the measured case must be the all-changed one"
    );

    // `decide` = map + reference update. Three streams: windows path (4 of 9
    // changed, copies 4 windows), full path (everything changed, copies the frame), reuse.
    let some = perturb(&a, &[1, 3, 4, 8], 40);
    let all = perturb(&a, &[0, 1, 2, 3, 4, 5, 6, 7, 8], 40);
    let decide = |cfg: GateConfig, next: &[u8]| {
        let mut g = Gate::new(cfg, GEO).unwrap();
        g.decide(&frame(&a)).unwrap();
        // Each repetition re-primes the reference, so only `decide(next)` is timed.
        let mut best = f64::INFINITY;
        let mut last = None;
        for _ in 0..REPS {
            g.reset();
            g.decide(&frame(&a)).unwrap();
            let t = Instant::now();
            last = Some(g.decide(&frame(next)).unwrap());
            best = best.min(t.elapsed().as_secs_f64());
        }
        (best, last.unwrap())
    };
    let win = GateConfig::new(
        GatePolicy::Windows {
            approximate_after_global: true,
        },
        8,
    );
    let (windows_s, d) = decide(win, &some);
    assert_eq!(
        d,
        Decision::Windows(WindowSet::from_indices(3, &[1, 3, 4, 8]))
    );
    let (full_s, d) = decide(win, &all);
    assert_eq!(d, Decision::Full(FullReason::AllChanged));
    let (reuse_s, d) = decide(win, &a);
    assert_eq!(d, Decision::Reuse);

    println!("\nchange-map cost, 1008x1008 RGB, tau 8, CPU single thread, min of {REPS}");
    println!(
        "| step | ms | % of the {:.0} ms frame floor |\n|---|---|---|",
        FRAME_FLOOR_SECS * 1e3
    );
    for (name, s) in [
        ("change map only", map_s),
        ("decide: reuse (map)", reuse_s),
        ("decide: windows (map + copy 4 windows)", windows_s),
        ("decide: full (map + copy frame)", full_s),
    ] {
        println!(
            "| {name} | {:.3} | {:.2} |",
            s * 1e3,
            100.0 * s / FRAME_FLOOR_SECS
        );
    }
    println!("budget: 5% of the floor = {:.2} ms (a real frame is 6-9 s on CPU, so the true share is ~1000x smaller)", BUDGET_SECS * 1e3);

    if !cfg!(debug_assertions) {
        for (name, s) in [
            ("map", map_s),
            ("reuse", reuse_s),
            ("windows", windows_s),
            ("full", full_s),
        ] {
            assert!(
                s < BUDGET_SECS,
                "{name}: {:.2} ms is over the 5% budget of {:.2} ms",
                s * 1e3,
                BUDGET_SECS * 1e3
            );
        }
    }
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
}

/// A 24-frame stream: static, then windows come and go, a noise burst, a cut.
fn stream() -> Vec<Vec<u8>> {
    let base = texture(2);
    let cut = texture(3);
    let mut out = Vec::new();
    let mut cur = base.clone();
    for t in 0..24usize {
        cur = match t {
            0..=3 => cur,
            4..=11 => perturb(&cur, &[t % 9, (t * 4) % 9], (t as u8) * 3 + 2),
            12..=15 => perturb(&cur, &[0, 1, 2, 3, 4, 5, 6, 7, 8], 5),
            16..=19 => cur,
            _ => cut.clone(),
        };
        out.push(cur.clone());
    }
    out
}

fn run(cfg: GateConfig, frames: &[Vec<u8>]) -> (Vec<Decision>, u64) {
    let mut g = Gate::new(cfg, GEO).unwrap();
    let (mut ds, mut h) = (Vec::new(), 0xcbf29ce484222325u64);
    for f in frames {
        ds.push(g.decide(&frame(f)).unwrap());
        for w in g.last_map().patches().words() {
            fnv(&mut h, &w.to_le_bytes());
        }
    }
    (ds, h)
}

#[test]
fn two_runs_over_the_same_stream_give_identical_decisions_and_map_hashes() {
    let frames = stream();
    for cfg in [
        GateConfig::exact(false),
        GateConfig::exact(true),
        GateConfig {
            refresh_every: 6,
            full_above_percent: 50,
            ..GateConfig::new(
                GatePolicy::Windows {
                    approximate_after_global: true,
                },
                8,
            )
        },
    ] {
        let (d1, h1) = run(cfg, &frames);
        let (d2, h2) = run(cfg, &frames);
        assert_eq!((d1, h1), (d2, h2), "{cfg:?}");
    }
}

#[test]
fn exact_window_policy_walks_the_stream_as_designed() {
    let frames = stream();
    let (ds, _) = run(GateConfig::exact(true), &frames);
    assert_eq!(ds[0], Decision::Full(FullReason::First));
    assert!(
        ds[1..4].iter().all(|d| *d == Decision::Reuse),
        "static frames are reused"
    );
    for (t, d) in ds.iter().enumerate().take(12).skip(4) {
        let want = WindowSet::from_indices(3, &[t % 9, (t * 4) % 9]);
        assert_eq!(*d, Decision::Windows(want), "frame {t}");
    }
    assert!(
        ds[12..16]
            .iter()
            .all(|d| *d == Decision::Full(FullReason::AllChanged)),
        "uniform shift moves every window"
    );
    assert!(ds[16..20].iter().all(|d| *d == Decision::Reuse));
    assert_eq!(
        ds[20],
        Decision::Full(FullReason::AllChanged),
        "a cut changes everything"
    );
    assert!(ds[21..].iter().all(|d| *d == Decision::Reuse));
}

#[test]
fn frame_policy_reuses_only_when_nothing_changed() {
    let frames = stream();
    let (ds, _) = run(GateConfig::exact(false), &frames);
    let reused: Vec<usize> = ds
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == Decision::Reuse)
        .map(|(t, _)| t)
        .collect();
    assert_eq!(reused, vec![1, 2, 3, 16, 17, 18, 19, 21, 22, 23]);
    assert!(ds.iter().all(|d| !matches!(d, Decision::Windows(_))));
}
