//! st-video's bilinear resize against Pillow (tools/reference/resize_oracle.py):
//! bit-exact on every case. The real-frame cases need the git-ignored
//! fixtures/resize/*.rgb and are skipped, with a message, when they are absent.

use std::path::PathBuf;

use ojas_io::sha256::sha256_hex;
use ojas_io::{parse_json, JsonValue};
use st_video::{resize_bilinear_pil, Rgb8};

const CASES: &str = include_str!("../../../tools/reference/resize_cases.json");

fn field<'a>(o: &'a [(String, JsonValue)], k: &str) -> &'a JsonValue {
    &o.iter()
        .find(|(key, _)| key == k)
        .unwrap_or_else(|| panic!("no field {k:?}"))
        .1
}

fn usize_of(v: &JsonValue) -> usize {
    v.as_u64().expect("integer") as usize
}

fn b64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("bad base64 byte {c}"),
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            acc |= val(c) << (18 - 6 * i);
        }
        out.extend(acc.to_be_bytes()[1..chunk.len()].iter());
    }
    out
}

#[test]
fn matches_pillow_bilinear_bit_for_bit() {
    let top = parse_json(CASES).expect("resize_cases.json");
    let top = top.as_object().expect("object");
    for c in field(top, "small").as_array().expect("small") {
        let c = c.as_object().expect("case");
        let (w, h, ow, oh) = (
            usize_of(field(c, "w")),
            usize_of(field(c, "h")),
            usize_of(field(c, "ow")),
            usize_of(field(c, "oh")),
        );
        let src = Rgb8::new(w, h, b64(field(c, "src").as_str().expect("src")));
        let want = b64(field(c, "out").as_str().expect("out"));
        let got = resize_bilinear_pil(&src, ow, oh);
        let diff = got.data.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff, 0,
            "{w}x{h} -> {ow}x{oh}: {diff} bytes differ from Pillow"
        );
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for c in field(top, "real").as_array().expect("real") {
        let c = c.as_object().expect("case");
        let path = root.join(field(c, "file").as_str().expect("file"));
        let Ok(raw) = std::fs::read(&path) else {
            println!(
                "NOT RUN: {} missing (run tools/reference/resize_oracle.py)",
                path.display()
            );
            continue;
        };
        assert_eq!(
            sha256_hex(&raw),
            field(c, "src_sha256").as_str().expect("sha"),
            "{}: input bytes",
            path.display()
        );
        let src = Rgb8::new(usize_of(field(c, "w")), usize_of(field(c, "h")), raw);
        let got = resize_bilinear_pil(&src, 1008, 1008);
        assert_eq!(
            sha256_hex(&got.data),
            field(c, "out_sha256").as_str().expect("sha"),
            "{}: 1008x1008 output differs from Pillow",
            path.display()
        );
        println!(
            "{}: bit-identical to Pillow at 1008x1008",
            field(c, "name").as_str().unwrap_or("?")
        );
    }
}
