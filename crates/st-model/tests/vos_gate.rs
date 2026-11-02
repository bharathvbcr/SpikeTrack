//! D5 gate (gp-tracker-multiplex): semi-supervised VOS on DAVIS 2017 val clips.
//! Our backbone + multiplex tracker from upstream's preprocessed frames and the
//! frame-0 ground truth; J&F (st-eval, davis2017-evaluation-exact) for ours and
//! for upstream's own predictions on the same frames. Gate: per clip, our J&F
//! within 2 points of upstream's.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_VOS_FIXTURES=fixtures/vos [SPIKETRACK_VOS_CLIPS=judo,shooting]
//!  cargo test -p st-model --release --test vos_gate -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use st_eval::jf::{boundary_f, region_j, Mask, DAVIS_BOUND_TH};
use st_model::tracker::vos::Vos;
use st_model::Backbone;
use st_weights::{Checkpoint, Sam31Weights};

const GATE_POINTS: f64 = 0.02;

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

/// Mean over objects of mean over frames 1..T-2 (DAVIS protocol: first and last frames excluded).
fn jf(pred: &[Vec<u8>], gt: &[Vec<u8>], objects: &[u8], w: usize, h: usize) -> (f64, f64) {
    let (mut js, mut fs) = (Vec::new(), Vec::new());
    for &o in objects {
        let (mut j, mut f) = (0.0, 0.0);
        let frames = 1..pred.len() - 1;
        let count = frames.len() as f64;
        for t in frames {
            let g = Mask::new(w, h, gt[t].iter().map(|&v| v == o).collect());
            let p = Mask::new(w, h, pred[t].iter().map(|&v| v == o).collect());
            j += region_j(&g, &p, None);
            f += boundary_f(&p, &g, None, DAVIS_BOUND_TH);
        }
        js.push(j / count);
        fs.push(f / count);
    }
    (
        js.iter().sum::<f64>() / js.len() as f64,
        fs.iter().sum::<f64>() / fs.len() as f64,
    )
}

#[test]
#[ignore = "needs the gated checkpoint and VOS fixtures (tools/reference/dump_vos_frames.py)"]
fn vos_jf_within_two_points_of_upstream() {
    vos_jf_within_two_points_of_upstream_on(
        &CpuBackend::with_threads(Budget::new(28 << 30), 8).expect("cpu"),
        "CPU",
    );
}

#[test]
#[ignore = "needs a Metal GPU; needs the gated checkpoint and VOS fixtures (tools/reference/dump_vos_frames.py)"]
fn vos_jf_within_two_points_of_upstream_on_metal() {
    vos_jf_within_two_points_of_upstream_on(
        &MetalBackend::new(Budget::new(28 << 30)).expect("metal"),
        "Metal",
    );
}

fn vos_jf_within_two_points_of_upstream_on<B: Backend>(be: &B, label: &str) {
    println!("[{label}]");
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let backbone = Backbone::new(be, &w.vision).expect("backbone");
    let vos = Vos::new(be, &backbone, &w.tracker).expect("tracker");
    let dir = env_path("SPIKETRACK_VOS_FIXTURES");
    let clips: Vec<String> = std::env::var("SPIKETRACK_VOS_CLIPS")
        .unwrap_or_else(|_| "judo,shooting,gold-fish".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let mut rows = Vec::new();
    let mut failed = Vec::new();
    for clip in &clips {
        let fx = SafeTensors::open(&dir.join(format!("{clip}.safetensors"))).expect("clip fixture");
        let (fshape, frames) = fx.read_f32_widened("frames").expect("frames");
        let (t_n, px) = (fshape[0] as usize, 3 * 1008 * 1008);
        let gt_info = fx.info("gt").expect("gt").clone();
        let (h, wd) = (gt_info.shape[1] as usize, gt_info.shape[2] as usize);
        let gt_raw = fx.read_bytes("gt").expect("gt");
        let up_raw = fx.read_bytes("upstream").expect("upstream");
        let split =
            |raw: &[u8]| -> Vec<Vec<u8>> { raw.chunks_exact(h * wd).map(<[u8]>::to_vec).collect() };
        let (gt, upstream) = (split(&gt_raw), split(&up_raw));
        let mut objects: Vec<u8> = gt[0].iter().copied().filter(|&v| v != 0).collect();
        objects.sort_unstable();
        objects.dedup();
        let n = objects.len();
        let masks0: Vec<f32> = objects
            .iter()
            .flat_map(|&o| gt[0].iter().map(move |&v| f32::from(u8::from(v == o))))
            .collect();
        let frame_tensors: Vec<Tensor> = (0..t_n)
            .map(|t| {
                be.upload(
                    &Tensor::from_f32(
                        &frames[t * px..(t + 1) * px],
                        &[1, 3, 1008, 1008],
                        be.budget(),
                    )
                    .expect("t"),
                )
                .expect("up")
            })
            .collect();
        let mut ours: Vec<Vec<u8>> = Vec::with_capacity(t_n);
        let t0 = Instant::now();
        vos.run(&frame_tensors, &masks0, n, h, wd, |out| {
            // Upstream labels a pixel with the object whose (non-overlapped) logit is positive.
            let mut lab = vec![0u8; h * wd];
            for (k, &o) in objects.iter().enumerate() {
                for i in 0..h * wd {
                    if out.video_res[k * h * wd + i] > 0.0 {
                        lab[i] = o;
                    }
                }
            }
            ours.push(lab);
            if out.frame % 10 == 0 {
                println!(
                    "{clip} frame {}/{t_n} ({:.0} s)",
                    out.frame,
                    t0.elapsed().as_secs_f64()
                );
            }
        })
        .expect("vos run");
        let secs = t0.elapsed().as_secs_f64();
        let (oj, of) = jf(&ours, &gt, &objects, wd, h);
        let (uj, uf) = jf(&upstream, &gt, &objects, wd, h);
        let (o_jf, u_jf) = ((oj + of) / 2.0, (uj + uf) / 2.0);
        // Agreement with upstream's own masks, per object over all frames.
        let agree: f64 = objects
            .iter()
            .map(|&o| {
                (1..t_n)
                    .map(|t| {
                        let a = Mask::new(wd, h, ours[t].iter().map(|&v| v == o).collect());
                        let b = Mask::new(wd, h, upstream[t].iter().map(|&v| v == o).collect());
                        region_j(&b, &a, None)
                    })
                    .sum::<f64>()
                    / (t_n - 1) as f64
            })
            .sum::<f64>()
            / n as f64;
        let pass = (o_jf - u_jf).abs() <= GATE_POINTS;
        if !pass {
            failed.push(clip.clone());
        }
        rows.push(format!("| {clip} | {n} | {t_n} | {oj:.4} | {of:.4} | {o_jf:.4} | {u_jf:.4} | {:+.4} | {agree:.4} | {secs:.0} | {} |", o_jf - u_jf, if pass { "pass" } else { "FAIL" }));
    }
    println!("\n| clip | objects | frames | J (ours) | F (ours) | J&F (ours) | J&F (upstream) | diff | mean IoU vs upstream masks | s ({label}) | gate |\n|---|---|---|---|---|---|---|---|---|---|---|");
    rows.iter().for_each(|r| println!("{r}"));
    assert!(
        failed.is_empty(),
        "J&F more than {GATE_POINTS} from upstream on {failed:?}"
    );
}
