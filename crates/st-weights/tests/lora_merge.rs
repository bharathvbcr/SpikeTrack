//! Merged-weight parity: the Rust LoRA merge (`st_weights::lora`) against PyTorch, on the real checkpoint.
//!
//! Needs the gated checkpoint and the fixture from `tools/reference/lora_merge_fixture.py`:
//! `SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors \
//!  SPIKETRACK_LORA_FIXTURES=fixtures/lora \
//!  cargo test -p st-weights --release --test lora_merge -- --ignored --nocapture --test-threads=1`
//!
//! Parity is to f32 rounding, not bit for bit: torch's `B @ A` (BLAS) and this crate's sequential sum
//! over the rank add in a different order, and the fixture records torch's own f32-vs-f64 error
//! (about 2e-8). The test allows `|d| <= 1e-6 + 1e-6 * |expected|` per element and reports the largest
//! difference and how many elements are bit-identical.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ojas_io::SafeTensors;
use st_weights::lora::{LoraAdapter, LoraError};
use st_weights::{Checkpoint, Weight};

const ABS_TOL: f64 = 1e-6;
const REL_TOL: f64 = 1e-6;

fn weights_path() -> PathBuf {
    PathBuf::from(std::env::var("SPIKETRACK_WEIGHTS").expect("set SPIKETRACK_WEIGHTS to the converted checkpoint"))
}

fn fixtures() -> PathBuf {
    match std::env::var("SPIKETRACK_LORA_FIXTURES") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/lora")),
    }
}

fn read(st: &SafeTensors<'_>, name: &str) -> Weight {
    let (shape, data) = st.read_f32(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    Weight { name: name.to_string(), shape: shape.into_iter().map(|d| d as usize).collect(), data }
}

struct Diff {
    max_abs: f64,
    exact: usize,
    n: usize,
    worst: usize,
}

fn compare(name: &str, got: &Weight, want: &Weight) -> Diff {
    assert_eq!(got.shape, want.shape, "{name}: shape");
    let mut d = Diff { max_abs: 0.0, exact: 0, n: got.data.len(), worst: 0 };
    for (i, (&g, &w)) in got.data.iter().zip(&want.data).enumerate() {
        let diff = (f64::from(g) - f64::from(w)).abs();
        if g.to_bits() == w.to_bits() {
            d.exact += 1;
        }
        if diff > d.max_abs {
            d.max_abs = diff;
            d.worst = i;
        }
        assert!(diff <= ABS_TOL + REL_TOL * f64::from(w).abs(), "{name}[{i}]: rust {g} vs torch {w} (|d| = {diff:e})");
    }
    d
}

#[test]
#[ignore = "needs SPIKETRACK_WEIGHTS and fixtures/lora"]
fn merged_tensors_match_pytorch_on_checkpoint_weights() {
    let fx = fixtures();
    let adapter = LoraAdapter::open(&fx.join("adapter.safetensors")).expect("adapter");
    let expected = SafeTensors::open(&fx.join("expected.safetensors")).expect("expected.safetensors");
    let ck = SafeTensors::open(&weights_path()).expect("checkpoint");

    let mut map: BTreeMap<String, Weight> = BTreeMap::new();
    for t in &adapter.targets {
        map.entry(t.param.clone()).or_insert_with(|| read(&ck, &t.param));
    }
    let before = map.clone();
    let report = adapter.merge_into(&mut [&mut map]).expect("merge");
    assert_eq!(report.targets, adapter.targets.len());
    assert_eq!(report.tensors.len(), map.len());

    for (name, w) in &map {
        let d = compare(name, w, &read(&expected, name));
        println!("{name}: max |d| {:.2e} at {}, {}/{} elements bit-identical to torch", d.max_abs, d.worst, d.exact, d.n);
        assert_ne!(w.data, before[name].data, "{name}: the merge changed nothing");
    }

    // Rows no target covers must be untouched: decoder layer 2's cross_attn adapts only q (rows 0..256).
    let cross = "detector.transformer.decoder.layers.2.cross_attn.in_proj_weight";
    let cols = map[cross].shape[1];
    assert_eq!(map[cross].data[256 * cols..], before[cross].data[256 * cols..], "rows past the adapted slice changed");

    // Tensors the adapter does not name are not in its report, and match the checkpoint.
    for name in ["detector.transformer.encoder.layers.0.self_attn.in_proj_bias", "detector.transformer.encoder.layers.1.self_attn.in_proj_weight"] {
        assert!(!report.tensors.contains(name), "{name} must not be targeted");
        assert_eq!(read(&expected, &format!("untouched.{name}")).data, read(&ck, name).data, "{name}");
    }
}

#[test]
#[ignore = "needs SPIKETRACK_WEIGHTS, fixtures/lora and about 4 GB of RAM"]
fn adapter_applies_to_the_fully_loaded_checkpoint() {
    let fx = fixtures();
    let adapter = LoraAdapter::open(&fx.join("adapter.safetensors")).expect("adapter");
    let expected = SafeTensors::open(&fx.join("expected.safetensors")).expect("expected.safetensors");
    let mut w = Checkpoint::load_all(&weights_path()).expect("load_all");
    let report = adapter.apply(&mut w).expect("apply");
    assert!(!report.tensors.is_empty());
    for name in &report.tensors {
        let got = w.detector_transformer.tensors.get(name).unwrap_or_else(|| panic!("{name} not in the detector transformer group"));
        compare(name, got, &read(&expected, name));
    }
}

#[test]
#[ignore = "needs SPIKETRACK_WEIGHTS"]
fn a_target_the_checkpoint_does_not_have_is_refused() {
    let fx = fixtures();
    let adapter = LoraAdapter::open(&fx.join("adapter.safetensors")).expect("adapter");
    // Hand the merge a map that lacks one targeted tensor: refused, nothing merged.
    let ck = SafeTensors::open(&weights_path()).expect("checkpoint");
    let mut map: BTreeMap<String, Weight> = BTreeMap::new();
    let skip = &adapter.targets[0].param;
    for t in adapter.targets.iter().filter(|t| &t.param != skip) {
        map.entry(t.param.clone()).or_insert_with(|| read(&ck, &t.param));
    }
    let before = map.clone();
    let err = adapter.merge_into(&mut [&mut map]).unwrap_err();
    assert!(matches!(err, LoraError::UnknownTarget { ref param } if param == skip), "{err}");
    assert_eq!(map, before);
}
