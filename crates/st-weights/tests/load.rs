//! The schema test needs only the committed manifest. The full-load test needs
//! the gated checkpoint: run it with
//! `SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors
//! cargo test -p st-weights --release -- --ignored --test-threads=2`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use ojas_io::{parse_json, JsonValue};
use st_weights::{schema, Checkpoint, Expected, Source, WeightsError};

const MANIFEST: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/weights_manifest.json"
);

fn field<'a>(obj: &'a [(String, JsonValue)], key: &str) -> &'a JsonValue {
    &obj.iter()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("manifest entry has no {key:?}"))
        .1
}

/// name -> (shape as stored, source dtype), from tools/reference/weights_manifest.json.
fn manifest() -> BTreeMap<String, (Vec<usize>, String)> {
    let text = std::fs::read_to_string(MANIFEST).expect("tools/reference/weights_manifest.json");
    let top = parse_json(&text).expect("manifest JSON");
    let entries = field(top.as_object().expect("object"), "entries")
        .as_array()
        .expect("entries array");
    let mut out = BTreeMap::new();
    for e in entries {
        let e = e.as_object().expect("entry object");
        let name = field(e, "name").as_str().expect("name").to_string();
        let src = field(e, "source_dtype")
            .as_str()
            .expect("source_dtype")
            .to_string();
        let shape = field(e, "shape")
            .as_array()
            .expect("shape array")
            .iter()
            .map(|d| usize::try_from(d.as_u64().expect("integer dim")).expect("dim fits usize"))
            .collect();
        assert!(
            out.insert(name.clone(), (shape, src)).is_none(),
            "{name} listed twice"
        );
    }
    out
}

#[test]
fn schema_claims_every_manifest_tensor_exactly_once() {
    let man = manifest();
    assert_eq!(man.len(), 1623, "manifest tensor count");
    let mut claimed: BTreeMap<String, &'static str> = BTreeMap::new();
    let mut groups = Vec::new();
    for e in schema() {
        match e {
            Expected::Tensor { name, shape } => {
                let (found, src) = man
                    .get(&name)
                    .unwrap_or_else(|| panic!("schema asks for {name}, not in the manifest"));
                assert_eq!(found, &shape, "{name}");
                assert_eq!(src, "float32", "{name}");
                assert!(
                    claimed.insert(name.clone(), "typed").is_none(),
                    "{name} claimed twice"
                );
            }
            Expected::ComplexPairs { name, mut shape } => {
                let (found, src) = man
                    .get(&name)
                    .unwrap_or_else(|| panic!("schema asks for {name}, not in the manifest"));
                shape.push(2);
                assert_eq!(found, &shape, "{name}");
                assert_eq!(src, "complex64", "{name}");
                assert!(
                    claimed.insert(name.clone(), "typed").is_none(),
                    "{name} claimed twice"
                );
            }
            Expected::Group { prefix } => groups.push(prefix),
        }
    }
    // A group takes whatever under its prefix the typed fields left; prefixes are disjoint.
    for (name, (_shape, src)) in &man {
        if claimed.contains_key(name) {
            continue;
        }
        let owners: Vec<&String> = groups
            .iter()
            .filter(|p| name.starts_with(p.as_str()))
            .collect();
        assert_eq!(
            owners.len(),
            1,
            "{name} is claimed by {} groups and no typed field",
            owners.len()
        );
        assert_eq!(
            src, "float32",
            "{name}: only the typed RoPE tables may be complex"
        );
        claimed.insert(name.clone(), "group");
    }
    for g in &groups {
        assert!(
            man.keys()
                .any(|n| n.starts_with(g.as_str()) && claimed[n] == "group"),
            "group {g} claims nothing"
        );
    }
    let typed = claimed.values().filter(|v| **v == "typed").count();
    assert_eq!(claimed.len(), man.len());
    // Vision trunk + 3 necks + text encoder are typed; the rest waits for its port.
    assert_eq!(
        typed,
        4 + 32 * 13 + 3 * 18 + 7 + 24 * 12,
        "typed tensor count"
    );
}

fn weights_path() -> PathBuf {
    let raw = std::env::var("SPIKETRACK_WEIGHTS")
        .expect("set SPIKETRACK_WEIGHTS to sam3.1_multiplex.safetensors (tools/reference/convert_checkpoint.py)");
    let expanded = match raw.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").expect("HOME")).join(rest),
        None => PathBuf::from(raw),
    };
    assert!(expanded.is_file(), "{} is not a file", expanded.display());
    expanded
}

fn peak_rss_bytes() -> u64 {
    // getrusage ru_maxrss is bytes on macOS.
    let mut ru = std::mem::MaybeUninit::<Rusage>::zeroed();
    // SAFETY: getrusage writes a full `struct rusage` into the pointer we pass,
    // which is sized and aligned for it; RUSAGE_SELF is always valid.
    let rc = unsafe { getrusage(0, ru.as_mut_ptr()) };
    assert_eq!(rc, 0, "getrusage failed");
    // SAFETY: rc == 0, so the kernel initialised the struct.
    let ru = unsafe { ru.assume_init() };
    ru.ru_maxrss as u64
}

#[repr(C)]
struct Timeval {
    tv_sec: i64,
    tv_usec: i32,
}

#[repr(C)]
struct Rusage {
    ru_utime: Timeval,
    ru_stime: Timeval,
    ru_maxrss: i64,
    rest: [i64; 13],
}

extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}

#[test]
#[ignore = "needs the gated checkpoint: set SPIKETRACK_WEIGHTS"]
fn loads_full_checkpoint_and_claims_everything() {
    let path = weights_path();
    let t0 = Instant::now();
    let w = Checkpoint::load_all(&path).expect("full load");
    let secs = t0.elapsed().as_secs_f64();
    let rss = peak_rss_bytes();
    assert_eq!(w.vision.trunk.blocks.len(), 32);
    assert!(w.vision.trunk.blocks[7].global && !w.vision.trunk.blocks[6].global);
    assert_eq!(w.vision.trunk.blocks[7].freqs_cis.shape, vec![5184, 32, 2]);
    assert_eq!(w.vision.trunk.blocks[0].freqs_cis.shape, vec![576, 32, 2]);
    // RoPE row 0 is position (0, 0): every pair is the identity rotation (1, 0).
    let f = &w.vision.trunk.blocks[0].freqs_cis.data;
    assert!(f[..64].chunks(2).all(|c| c[0] == 1.0 && c[1] == 0.0));
    let n: usize = [
        &w.detector_transformer,
        &w.geometry_encoder,
        &w.segmentation_head,
        &w.dot_prod_scoring,
        &w.tracker,
    ]
    .iter()
    .map(|g| g.tensors.len())
    .sum();
    assert_eq!(n + 4 + 32 * 13 + 3 * 18 + 7 + 24 * 12, 1623);
    println!(
        "full load: {secs:.2} s, peak RSS {:.2} GiB",
        rss as f64 / (1u64 << 30) as f64
    );
}

#[test]
#[ignore = "needs the gated checkpoint: set SPIKETRACK_WEIGHTS"]
fn refuses_wrong_shape_and_double_claim_and_leftovers() {
    let ck = Checkpoint::open(&weights_path()).expect("open");
    let name = "detector.backbone.vision_backbone.trunk.ln_pre.weight";
    match ck.tensor(name, &[1023]) {
        Err(WeightsError::Shape { found, .. }) => assert_eq!(found, vec![1024]),
        other => panic!("wrong shape accepted: {other:?}"),
    }
    ck.tensor(name, &[1024]).expect("right shape");
    assert!(matches!(
        ck.tensor(name, &[1024]),
        Err(WeightsError::ClaimedTwice { .. })
    ));
    assert!(matches!(
        ck.tensor("no.such.tensor", &[1]),
        Err(WeightsError::Missing { .. })
    ));
    let rope = "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis";
    assert!(matches!(
        ck.tensor(rope, &[576, 32, 2]),
        Err(WeightsError::ComplexPair { .. })
    ));
    match ck.finish() {
        Err(WeightsError::Unclaimed { names }) => assert_eq!(names.len(), 1622),
        other => panic!("leftovers accepted: {other:?}"),
    }
}
