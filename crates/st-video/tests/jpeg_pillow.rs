//! The native JPEG decoder against Pillow 13.0.0 (libjpeg-turbo 3.2.0) on small
//! committed cases, at zero tolerance. Always run, std only, any platform.
//!
//! `tests/jpeg_cases/cases/<name>_<W>x<H>.jpg` has Pillow's decode beside it as
//! `.rgb` (raw 8-bit RGB, row-major). The cases cover what the 640x480 VISEM
//! frames cannot: partial MCUs at the right and bottom edges, odd sizes, chroma
//! planes of one to three samples, every upsampler libjpeg-turbo has (h2v1,
//! h1v2, h2v2, whole-factor replication), grayscale, restart intervals,
//! optimised Huffman tables, extreme quantisation, and multi-scan sequential
//! files. `tools/reference/jpeg_cases.py` makes them, and only writes a case
//! when `djpeg` (libjpeg-turbo 3.2.0) agrees with Pillow on it.
//!
//! The rest of the file is the decoder's behaviour on damaged input, on real
//! files: every prefix of each case, and single-byte corruption, must return a
//! value (a picture or an error) and never panic or hang.

use std::path::{Path, PathBuf};

use st_video::{decode_jpeg, JpegError};

fn cases_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/jpeg_cases/cases")
}

/// (name, size, JPEG bytes, Pillow's RGB bytes) for every case.
fn cases() -> Vec<(String, (usize, usize), Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut names: Vec<PathBuf> = std::fs::read_dir(cases_dir()).expect("tests/jpeg_cases/cases").map(|e| e.expect("entry").path()).filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
    names.sort();
    for jpg in names {
        let stem = jpg.file_stem().and_then(|s| s.to_str()).expect("utf-8 name").to_string();
        let dims = stem.rsplit('_').next().expect("name_WxH");
        let (w, h) = dims.split_once('x').expect("WxH");
        let (w, h): (usize, usize) = (w.parse().expect("width"), h.parse().expect("height"));
        let rgb = std::fs::read(jpg.with_extension("rgb")).unwrap_or_else(|e| panic!("{stem}.rgb: {e}"));
        assert_eq!(rgb.len(), w * h * 3, "{stem}: the .rgb is not {w}x{h}x3");
        out.push((stem, (w, h), std::fs::read(&jpg).expect("jpeg"), rgb));
    }
    out
}

#[test]
fn every_case_is_byte_identical_to_pillow() {
    let cases = cases();
    assert!(cases.len() >= 37, "only {} cases found", cases.len());
    let mut failures = Vec::new();
    for (name, (w, h), jpeg, want) in &cases {
        match decode_jpeg(jpeg) {
            Ok(img) => {
                assert_eq!((img.width, img.height), (*w, *h), "{name}: size");
                let bad = img.data.iter().zip(want).filter(|(a, b)| a != b).count();
                if bad != 0 {
                    let first = img.data.iter().zip(want).position(|(a, b)| a != b).expect("a difference");
                    failures.push(format!("{name}: {bad} of {} bytes differ (first at byte {first}: ours {} vs Pillow {})", want.len(), img.data[first], want[first]));
                }
            }
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(failures.is_empty(), "not byte-identical to Pillow:\n{}", failures.join("\n"));
}

#[test]
fn every_proper_prefix_of_every_case_is_an_error() {
    for (name, _, jpeg, _) in cases() {
        for cut in 0..jpeg.len() {
            assert!(decode_jpeg(&jpeg[..cut]).is_err(), "{name}: the first {cut} of {} bytes decoded", jpeg.len());
        }
    }
}

#[test]
fn single_byte_corruption_never_panics() {
    // Every byte of the smaller cases replaced by a few values (including 0x00, 0xff and
    // the next marker-like value). The result may be a picture or an error; the test is
    // that the call returns. Pictures must have the declared size.
    let mut decoded = 0usize;
    let mut refused = 0usize;
    for (name, (w, h), jpeg, _) in cases().into_iter().filter(|c| c.2.len() < 1500) {
        for at in 0..jpeg.len() {
            for v in [0x00u8, 0xff, 0x80, jpeg[at] ^ 0x01, jpeg[at].wrapping_add(0x40)] {
                let mut bad = jpeg.clone();
                bad[at] = v;
                match decode_jpeg(&bad) {
                    Ok(img) => {
                        decoded += 1;
                        assert_eq!(img.data.len(), img.width * img.height * 3, "{name} byte {at}={v:#x}");
                        let _ = (w, h);
                    }
                    Err(_) => refused += 1,
                }
            }
        }
    }
    println!("{decoded} corrupted files decoded, {refused} refused");
    assert!(refused > 0 && decoded > 0);
}

#[test]
fn truncated_error_kinds_are_not_unsupported() {
    // A cut file is damaged, not "a variant to hand to another decoder".
    let (_, _, jpeg, _) = cases().into_iter().next().expect("a case");
    for cut in [1usize, 2, 10, jpeg.len() / 2, jpeg.len() - 1] {
        match decode_jpeg(&jpeg[..cut]) {
            Err(JpegError::Unsupported(u)) => panic!("cut at {cut} reported as unsupported: {u}"),
            Err(_) => {}
            Ok(_) => panic!("cut at {cut} decoded"),
        }
    }
}
