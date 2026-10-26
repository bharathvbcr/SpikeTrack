//! Tracker modules against upstream (tools/reference/dump_tracker_fixtures.py,
//! DAVIS judo, 8 frames), each fed upstream's own recorded inputs.
//!
//! `SPIKETRACK_WEIGHTS=... SPIKETRACK_TRACKER_FIXTURES=fixtures/tracker/judo.safetensors
//!  cargo test -p st-model --release --test tracker_parity -- --ignored --nocapture --test-threads=1`

use std::path::PathBuf;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use st_model::tracker::mask_decoder::{MuxMaskDecoder, D, MUX};
use st_weights::{Checkpoint, Sam31Weights};

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

fn stats(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut gg, mut ww, mut max_abs, mut ref_max) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        max_abs = max_abs.max((g - w).abs());
        ref_max = ref_max.max(w.abs());
    }
    let cos = if gg == 0.0 && ww == 0.0 { 1.0 } else { dot / (gg.sqrt() * ww.sqrt()) };
    (cos, max_abs, max_abs / ref_max.max(1e-30))
}

#[test]
#[ignore = "needs the gated checkpoint and tracker fixtures"]
fn multiplex_mask_decoder_matches_upstream() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu");
    let dec = MuxMaskDecoder::load(&be, &w.tracker, "tracker.model.sam_mask_decoder").expect("decoder");
    let fx = SafeTensors::open(&env_path("SPIKETRACK_TRACKER_FIXTURES")).expect("fixtures");
    let t = |name: &str| -> Tensor {
        let (shape, data) = fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
        let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
        be.upload(&Tensor::from_f32(&data, &shape, be.budget()).expect("t")).expect("up")
    };
    let h = |name: &str| fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail())).1;
    let mut worst = f64::INFINITY;
    println!("| frame | output | cosine | max abs | max abs / max ref |\n|---|---|---|---|---|");
    for f in 1..8 {
        let p = format!("f{f}.mask_decoder.call{}", f - 1);
        let extra = h(&format!("{p}.kw.extra_per_object_embeddings"));
        assert_eq!(extra.len(), MUX * D);
        let out = dec
            .fwd(&be, &t(&format!("{p}.kw.image_embeddings")), &t(&format!("{p}.kw.image_pe")), &t(&format!("{p}.kw.high_res_features.0")), &t(&format!("{p}.kw.high_res_features.1")), &extra)
            .expect("decode");
        let masks = out.masks.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
        for (name, got, key) in [
            ("masks", masks.as_slice(), "masks"),
            ("iou_pred", out.iou_pred.as_slice(), "iou_pred"),
            ("object_score_logits", out.object_score_logits.as_slice(), "object_score_logits"),
            ("sam_tokens", out.sam_tokens.as_slice(), "sam_tokens_out"),
        ] {
            let (cos, ma, rel) = stats(got, &h(&format!("{p}.out.{key}")));
            worst = worst.min(cos);
            println!("| {f} | {name} | {cos:.7} | {ma:.3e} | {rel:.3e} |");
        }
    }
    println!("\nmask decoder worst cosine {worst:.7}");
    assert!(worst > 0.999, "mask decoder worst cosine {worst}");
}

#[test]
#[ignore = "needs the gated checkpoint and tracker fixtures"]
fn memory_encoder_matches_upstream() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu");
    let enc = st_model::tracker::MemoryEncoder::load(&be, &w.tracker, "tracker.model.maskmem_backbone").expect("encoder");
    let fx = SafeTensors::open(&env_path("SPIKETRACK_TRACKER_FIXTURES")).expect("fixtures");
    let t = |name: &str| -> Tensor {
        let (shape, data) = fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
        let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
        be.upload(&Tensor::from_f32(&data, &shape, be.budget()).expect("t")).expect("up")
    };
    let h = |name: &str| fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail())).1;
    let mut worst = f64::INFINITY;
    println!("| frame | output | cosine | max abs | max abs / max ref |\n|---|---|---|---|---|");
    for f in 0..8 {
        let p = format!("f{f}.memenc.call{f}");
        let out = enc.fwd(&be, &t(&format!("{p}.arg.0")), &t(&format!("{p}.arg.1"))).expect("encode");
        for (name, got, key) in [("features", &out.features, "vision_features"), ("pos", &out.pos, "vision_pos_enc.0")] {
            let g = got.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
            let (cos, ma, rel) = stats(&g, &h(&format!("{p}.out.{key}")));
            worst = worst.min(cos);
            println!("| {f} | {name} | {cos:.7} | {ma:.3e} | {rel:.3e} |");
        }
    }
    println!("\nmemory encoder worst cosine {worst:.7}");
    assert!(worst > 0.999, "memory encoder worst cosine {worst}");
}

#[test]
#[ignore = "needs the gated checkpoint and tracker fixtures"]
fn memory_attention_matches_upstream() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu");
    let attn = st_model::tracker::MemoryAttention::load(&be, &w.tracker, "tracker.model.transformer.encoder").expect("attention");
    let fx = SafeTensors::open(&env_path("SPIKETRACK_TRACKER_FIXTURES")).expect("fixtures");
    // Seq-first [T, 1, C] fixtures are [1, T, C] in memory for one bucket.
    let t = |name: &str| -> Tensor {
        let (shape, data) = fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
        assert_eq!(shape[1], 1, "{name}: one bucket expected");
        be.upload(&Tensor::from_f32(&data, &[1, shape[0] as usize, shape[2] as usize], be.budget()).expect("t")).expect("up")
    };
    let h = |name: &str| fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail())).1;
    let mut worst = f64::INFINITY;
    println!("| frame | memory frames | pointer tokens | cosine | max abs | max abs / max ref |\n|---|---|---|---|---|---|");
    for f in 1..8 {
        let p = format!("f{f}.memattn.call{}", f - 1);
        let (memory, memory_image) = (t(&format!("{p}.kw.memory")), t(&format!("{p}.kw.memory_image")));
        let ptrs = memory.shape()[1] - memory_image.shape()[1];
        let (image, src, src_pos) = (t(&format!("{p}.kw.image")), t(&format!("{p}.kw.src")), t(&format!("{p}.kw.src_pos")));
        let (memory_image_pos, memory_pos) = (t(&format!("{p}.kw.memory_image_pos")), t(&format!("{p}.kw.memory_pos")));
        let inputs = st_model::tracker::MemoryInputs {
            image: &image,
            src: &src,
            src_pos: &src_pos,
            memory_image: &memory_image,
            memory_image_pos: &memory_image_pos,
            memory: &memory,
            memory_pos: &memory_pos,
            num_obj_ptr_tokens: ptrs,
        };
        let out = attn.fwd(&be, &inputs).expect("attention");
        let got = out.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
        let (cos, ma, rel) = stats(&got, &h(&format!("{p}.out.memory")));
        worst = worst.min(cos);
        println!("| {f} | {} | {ptrs} | {cos:.7} | {ma:.3e} | {rel:.3e} |", memory_image.shape()[1] / 5184);
    }
    println!("\nmemory attention worst cosine {worst:.7}");
    assert!(worst > 0.999, "memory attention worst cosine {worst}");
}

#[test]
#[ignore = "needs the gated checkpoint and tracker fixtures"]
fn interactive_decoder_matches_upstream_on_the_conditioning_frame() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = Sam31Weights::load(&ck).expect("weights");
    let be = CpuBackend::with_threads(Budget::new(24 << 30), 8).expect("cpu");
    let dec = st_model::tracker::InteractiveDecoder::load(&be, &w.tracker, "tracker.model.interactive_sam_mask_decoder").expect("decoder");
    let fx = SafeTensors::open(&env_path("SPIKETRACK_TRACKER_FIXTURES")).expect("fixtures");
    let t = |name: &str| -> Tensor {
        let (shape, data) = fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail()));
        let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
        be.upload(&Tensor::from_f32(&data, &shape, be.budget()).expect("t")).expect("up")
    };
    let h = |name: &str| fx.read_f32(name).unwrap_or_else(|e| panic!("{name}: {}", e.detail())).1;
    let p = "f0.interactive_mask_decoder.call0";
    let (sshape, sparse) = fx.read_f32(&format!("{p}.kw.sparse_prompt_embeddings")).expect("sparse");
    let out = dec
        .fwd(&be, &t(&format!("{p}.kw.image_embeddings")), &t(&format!("{p}.kw.image_pe")), &sparse, sshape[1] as usize, &t(&format!("{p}.kw.dense_prompt_embeddings")), &t(&format!("{p}.kw.high_res_features.0")), &t(&format!("{p}.kw.high_res_features.1")))
        .expect("decode");
    let n = sshape[0] as usize;
    let px = 288 * 288;
    let masks = out.masks.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
    // multimask_output=False: upstream returns mask/token/IoU 0 of the 4.
    let first = |v: &[f32], per: usize, width: usize| -> Vec<f32> { (0..n).flat_map(|i| v[i * per * width..i * per * width + width].to_vec()).collect() };
    let mut worst = f64::INFINITY;
    for (name, got, key) in [
        ("mask 0", first(&masks, 4, px), "out.0"),
        ("iou 0", first(&out.iou_pred, 4, 1), "out.1"),
        ("token 0", first(&out.mask_tokens, 4, D), "out.2"),
        ("object score", out.object_score_logits.clone(), "out.3"),
    ] {
        let (cos, ma, rel) = stats(&got, &h(&format!("{p}.{key}")));
        worst = worst.min(cos);
        println!("interactive {name}: cosine {cos:.7}, max abs {ma:.3e}, rel {rel:.3e}");
    }
    assert!(worst > 0.999, "interactive decoder worst cosine {worst}");
}
