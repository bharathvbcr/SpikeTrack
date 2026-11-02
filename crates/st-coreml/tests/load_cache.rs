//! The compiled-model cache on the real fp32 package (E18b). The package is cloned into a fresh
//! temporary directory (`cp -c`, an APFS clone), so the first load there compiles and hashes in full,
//! and the second reuses both the compiled model and the recorded hash. The two loads must report the
//! same SHA-256 and predict bit-identical trunks on the same image.
//!
//! `SPIKETRACK_MODEL_DIR=.../sam3.1 cargo test -p st-coreml --release --test load_cache -- --ignored --nocapture`

use std::path::PathBuf;
use std::process::Command;

use st_coreml::{CoremlTrunk, Units};

#[test]
#[ignore = "needs the exported fp32 trunk package and a Metal GPU"]
fn a_fresh_package_compiles_and_hashes_once_then_reuses_both() {
    let model_dir =
        PathBuf::from(std::env::var("SPIKETRACK_MODEL_DIR").expect("set SPIKETRACK_MODEL_DIR"));
    let src = model_dir.join("coreml/sam3.1_trunk_fp32.mlpackage");
    assert!(src.is_dir(), "{} is not exported", src.display());
    let root = std::env::temp_dir().join(format!("st-coreml-load-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let pkg = root.join("t.mlpackage");
    let st = Command::new("cp")
        .arg("-c")
        .arg("-R")
        .arg(&src)
        .arg(&pkg)
        .status()
        .expect("run cp");
    assert!(st.success(), "cloning the package failed");

    let first = CoremlTrunk::load(&pkg, Units::CpuAndGpu).expect("first load");
    println!(
        "first: {:.2} s (hash {:.2}, compile {:.2}, model {:.2}); compiled {}, hashed {}",
        first.load_seconds,
        first.hash_seconds,
        first.compile_seconds,
        first.model_seconds,
        first.compiled,
        first.hashed
    );
    assert!(
        first.compiled && first.hashed,
        "a fresh package compiles and hashes in full"
    );
    let second = CoremlTrunk::load(&pkg, Units::CpuAndGpu).expect("second load");
    println!(
        "second: {:.2} s (hash {:.2}, compile {:.2}, model {:.2}); compiled {}, hashed {}",
        second.load_seconds,
        second.hash_seconds,
        second.compile_seconds,
        second.model_seconds,
        second.compiled,
        second.hashed
    );
    assert!(
        !second.compiled && !second.hashed,
        "the second load reuses the compiled model and the recorded hash"
    );
    assert_eq!(first.sha256(), second.sha256());

    let image: Vec<f32> = (0..3 * 1008 * 1008)
        .map(|i| ((i % 977) as f32 / 488.0) - 1.0)
        .collect();
    let a = first.predict(&image).expect("first predict");
    let b = second.predict(&image).expect("second predict");
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "the two loads predict different trunks"
    );
    drop((first, second));
    std::fs::remove_dir_all(&root).unwrap();
}
