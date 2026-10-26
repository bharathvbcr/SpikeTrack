//! Tokenizer and text encoder against upstream (tools/reference/dump_text_fixtures.py).
//!
//! `SPIKETRACK_MODEL_DIR=~/Code/research/models/sam3.1` (vocab.json, merges.txt) for the
//! tokenizer; plus `SPIKETRACK_WEIGHTS` and `SPIKETRACK_TEXT_FIXTURES=fixtures/text/prompts.safetensors`
//! for the encoder. `cargo test -p st-model --release --test text_parity -- --ignored --nocapture`

use std::path::PathBuf;

use ojas_core::{Backend, Budget};
use ojas_cpu::CpuBackend;
use ojas_io::{parse_json, SafeTensors};
use st_model::tokenizer::CONTEXT;
use st_model::{Text, Tokenizer};
use st_weights::{Checkpoint, TextEncoder};

const CASES: &str = include_str!("../../../tools/reference/tokenizer_cases.json");

fn env_path(var: &str) -> PathBuf {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("set {var}"));
    let p = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(p.exists(), "{var}={} does not exist", p.display());
    p
}

fn cases() -> Vec<(String, [u32; CONTEXT])> {
    let top = parse_json(CASES).expect("tokenizer_cases.json");
    let top = top.as_object().expect("object");
    let cases = top.iter().find(|(k, _)| k == "cases").expect("cases").1.as_array().expect("array");
    cases
        .iter()
        .map(|c| {
            let c = c.as_object().expect("case");
            let get = |k: &str| &c.iter().find(|(key, _)| key == k).expect(k).1;
            let text = get("text").as_str().expect("text").to_string();
            let ids: Vec<u32> = get("ids").as_array().expect("ids").iter().map(|v| v.as_u64().expect("id") as u32).collect();
            (text, ids.try_into().expect("32 ids"))
        })
        .collect()
}

fn tokenizer() -> Tokenizer {
    let dir = env_path("SPIKETRACK_MODEL_DIR");
    Tokenizer::from_hf_files(&dir.join("vocab.json"), &dir.join("merges.txt")).expect("tokenizer")
}

#[test]
#[ignore = "needs facebook/sam3.1 vocab.json and merges.txt: set SPIKETRACK_MODEL_DIR"]
fn tokenizer_matches_upstream_ids() {
    let tok = tokenizer();
    let cases = cases();
    assert_eq!(cases.len(), 13);
    for (text, want) in &cases {
        let got = tok.tokenize(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
        assert_eq!(&got, want, "{text:?}");
    }
    println!("{} prompts tokenize identically to upstream", cases.len());
}

#[test]
#[ignore = "needs the gated checkpoint and text fixtures: set SPIKETRACK_WEIGHTS and SPIKETRACK_TEXT_FIXTURES"]
fn text_encoder_matches_upstream() {
    let ck = Checkpoint::open(&env_path("SPIKETRACK_WEIGHTS")).expect("checkpoint");
    let w = TextEncoder::load(&ck).expect("text weights");
    let be = CpuBackend::new(Budget::new(16 << 30));
    let text = Text::new(&be, &w).expect("upload");
    let ids: Vec<[u32; CONTEXT]> = cases().into_iter().map(|(_, ids)| ids).collect();
    let out = text.encode(&ids).expect("encode");
    be.sync().expect("sync");
    let fx = SafeTensors::open(&env_path("SPIKETRACK_TEXT_FIXTURES")).expect("fixtures");
    // Stored as U8, 1 = padding (torch's key_padding convention).
    assert_eq!(fx.info("pad_mask").expect("pad_mask").dtype, ojas_io::StDtype::U8);
    let pad: Vec<bool> = fx.read_bytes("pad_mask").expect("pad_mask").into_iter().map(|v| v != 0).collect();
    assert_eq!(out.padding, pad, "padding mask");
    for (name, got) in [("encoder.tokens", &out.tokens), ("resizer.out", &out.resized)] {
        let (shape, want) = fx.read_f32(name).expect(name);
        assert_eq!(got.shape(), shape.iter().map(|&d| d as usize).collect::<Vec<_>>().as_slice(), "{name}");
        let got = got.to_host(be.budget()).expect("host").to_f32_vec().expect("f32");
        let (mut dot, mut gg, mut ww, mut max_abs, mut ref_max) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&g, &r) in got.iter().zip(&want) {
            let (g, r) = (f64::from(g), f64::from(r));
            dot += g * r;
            gg += g * g;
            ww += r * r;
            max_abs = max_abs.max((g - r).abs());
            ref_max = ref_max.max(r.abs());
        }
        let cos = dot / (gg.sqrt() * ww.sqrt());
        println!("{name}: cosine {cos:.9}, max abs {max_abs:.3e}, max abs / max ref {:.3e}", max_abs / ref_max);
        assert!(cos > 0.9999, "{name}: cosine {cos}");
    }
}
