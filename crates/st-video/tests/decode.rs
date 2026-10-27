//! st-video's native decode (ImageIO, AVFoundation) in three groups.
//!
//! * Always run: a synthetic JPEG decodes upright and in RGB order, and every
//!   failure is an error (missing file, not an image, empty, truncated JPEG,
//!   garbage with a JPEG signature, not a video).
//! * `--ignored`, needs `data/visem` and `fixtures/decode` (git-ignored; make the
//!   fixtures with `tools/reference/.venv/bin/python -s tools/reference/decode_oracle.py`):
//!   ImageIO against Pillow on VISEM JPEG frames, and AVFoundation on `11.mp4`
//!   against the dataset's JPEGs.
//! * `--ignored`, needs `fixtures/decode/rot90.mp4`: a video with rotation
//!   metadata comes out upright. Make it from VISEM video 11 (`-display_rotation`
//!   is counter-clockwise degrees, input option; stream copy keeps the bitstream):
//!   `ffmpeg -display_rotation 90 -i data/visem/VISEM_Tracking_Train_v4/Train/11/11.mp4
//!    -t 2 -c copy fixtures/decode/rot90.mp4`
//!
//! The ignored tests print their measurements; run them with `--nocapture`.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_io::parse_json;
use st_video::decode::DecodeError;
use st_video::{decode_image, sam_input, Rgb8, VideoReader};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("st-video-decode");
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir.join(name)
}

/// A 32x32 4:2:0 JPEG (Pillow, quality 95): red = 8 * x, green = 8 * y, blue = 128.
const GRADIENT_JPEG: &[u8] = &[
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
    0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
    0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06, 0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0b, 0x08,
    0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x06, 0x08, 0x0b, 0x0c, 0x0b, 0x0a, 0x0c, 0x09, 0x0a, 0x0a, 0x0a, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x05, 0x03, 0x03, 0x05, 0x0a, 0x07, 0x06, 0x07, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x20, 0x00, 0x20, 0x03,
    0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xc4, 0x00, 0x1f, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10, 0x00,
    0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
    0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24,
    0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
    0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6,
    0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
    0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xc4, 0x00, 0x1f, 0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x11, 0x00,
    0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
    0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15,
    0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4,
    0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0xfc,
    0x9d, 0xf0, 0x9f, 0xc2, 0x9f, 0xbb, 0xfe, 0x8d, 0xfa, 0x57, 0xa6, 0x78, 0x53, 0xe1, 0x4f, 0xdd, 0xff, 0x00, 0x46, 0xfd, 0x2b, 0xd2, 0xbc, 0x27,
    0xf0, 0xab, 0xee, 0xff, 0x00, 0xa3, 0x7e, 0x95, 0xe9, 0x9e, 0x13, 0xf8, 0x53, 0xf7, 0x7f, 0xd1, 0xbf, 0x4a, 0xfe, 0x88, 0xce, 0x38, 0xc7, 0x7f,
    0x78, 0xf2, 0x7c, 0x36, 0xe3, 0xdf, 0x83, 0xdf, 0xec, 0x79, 0xb7, 0x84, 0xfe, 0x14, 0xfd, 0xdf, 0xf4, 0x6f, 0xd2, 0xbd, 0x33, 0xc2, 0x7f, 0x0a,
    0x7e, 0xef, 0xfa, 0x37, 0xe9, 0x5e, 0x95, 0xe1, 0x4f, 0x85, 0x3f, 0x77, 0xfd, 0x1b, 0xf4, 0xaf, 0x4d, 0xf0, 0x9f, 0xc2, 0xaf, 0xbb, 0xfe, 0x8d,
    0xfa, 0x57, 0xe5, 0x39, 0xcf, 0x18, 0x6f, 0xef, 0x1f, 0xe8, 0x07, 0x86, 0xdc, 0x7b, 0xf0, 0x7b, 0xfd, 0x8f, 0x2f, 0xf0, 0xa7, 0xc2, 0x9f, 0xbb,
    0xfe, 0x8d, 0xfa, 0x57, 0xa6, 0xf8, 0x4f, 0xe1, 0x4f, 0xdd, 0xff, 0x00, 0x46, 0xfd, 0x2b, 0xd2, 0x7c, 0x27, 0xf0, 0xa7, 0xee, 0xff, 0x00, 0xa3,
    0x7e, 0x95, 0xe9, 0xbe, 0x14, 0xf8, 0x53, 0xf7, 0x7f, 0xd1, 0xbf, 0x4a, 0xfc, 0xaf, 0x39, 0xe3, 0x0d, 0xfd, 0xe3, 0xfc, 0x29, 0xf0, 0xdb, 0x8f,
    0x7e, 0x0f, 0x7f, 0xb1, 0xe6, 0xbe, 0x13, 0xf8, 0x53, 0xf7, 0x7f, 0xd1, 0xbf, 0x4a, 0xf4, 0xdf, 0x09, 0xfc, 0x29, 0xfb, 0xbf, 0xe8, 0xdf, 0xa5,
    0x7a, 0x57, 0x84, 0xfe, 0x15, 0x7d, 0xdf, 0xf4, 0x6f, 0xd2, 0xbd, 0x33, 0xc2, 0x7f, 0x0a, 0x7e, 0xef, 0xfa, 0x37, 0xe9, 0x5f, 0x94, 0xe7, 0x1c,
    0x63, 0xbf, 0xbc, 0x7f, 0xa0, 0x1e, 0x1b, 0x71, 0xef, 0xc1, 0xef, 0xf6, 0x3f, 0xff, 0xd9,
];

fn write(name: &str, bytes: &[u8]) -> PathBuf {
    let p = scratch(name);
    std::fs::write(&p, bytes).expect("write temp file");
    p
}

// ---------------------------------------------------------------------------
// Always run
// ---------------------------------------------------------------------------

#[test]
fn a_synthetic_jpeg_decodes_upright_and_in_rgb_order() {
    assert_eq!(GRADIENT_JPEG.len(), 855);
    let img = decode_image(write("gradient.jpg", GRADIENT_JPEG)).expect("decode");
    assert_eq!((img.width, img.height), (32, 32));
    assert_eq!(img.data.len(), 32 * 32 * 3);
    let mut worst = 0i32;
    for y in 0..32usize {
        for x in 0..32usize {
            let px = &img.data[(y * 32 + x) * 3..][..3];
            let want = [(8 * x) as i32, (8 * y) as i32, 128];
            for c in 0..3 {
                worst = worst.max((i32::from(px[c]) - want[c]).abs());
            }
        }
    }
    // Red grows to the right, green downward, blue is flat: a channel swap or a
    // vertical flip would be off by up to 248. 4:2:0 and quantisation leave a few levels.
    assert!(worst <= 12, "worst deviation from the gradient is {worst} levels");
}

#[test]
fn a_missing_file_is_an_io_error() {
    let missing = scratch("does-not-exist.jpg");
    let _ = std::fs::remove_file(&missing);
    match decode_image(&missing) {
        Err(DecodeError::Io { err, .. }) => assert_eq!(err.kind(), std::io::ErrorKind::NotFound),
        other => panic!("expected an Io error, got {other:?}"),
    }
    assert!(matches!(VideoReader::open(&missing), Err(DecodeError::Io { .. })));
}

#[test]
fn a_file_that_is_not_an_image_is_an_error() {
    let text = write("not-an-image.jpg", b"This is not an image, whatever its extension says.\n");
    assert!(matches!(decode_image(&text), Err(DecodeError::Unrecognized { .. })), "text file");
    let empty = write("empty.jpg", b"");
    assert!(matches!(decode_image(&empty), Err(DecodeError::Empty { .. })), "empty file");
    let dir = scratch("a-directory");
    std::fs::create_dir_all(&dir).expect("dir");
    assert!(matches!(decode_image(&dir), Err(DecodeError::Unrecognized { .. })), "directory");
    // A JPEG signature on nothing: not an image either (an error, never an empty picture).
    let garbage = write("garbage.jpg", &[0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    assert!(decode_image(&garbage).is_err(), "garbage with a JPEG signature");
}

#[test]
fn every_truncated_prefix_of_a_jpeg_is_an_error() {
    // Every proper prefix, from 1 byte to all but the last: in the headers, at the
    // start of the scan (where ImageIO alone reports a black picture as complete),
    // inside it, and with only the end-of-image marker missing.
    let p = scratch("prefix.jpg");
    let mut decoded = Vec::new();
    for cut in 1..GRADIENT_JPEG.len() {
        std::fs::write(&p, &GRADIENT_JPEG[..cut]).expect("write");
        if decode_image(&p).is_ok() {
            decoded.push(cut);
        }
    }
    assert!(decoded.is_empty(), "prefixes of {} bytes that decoded as pictures: {decoded:?}", GRADIENT_JPEG.len());
    assert!(decode_image(write("whole.jpg", GRADIENT_JPEG)).is_ok());
}

#[test]
fn a_file_that_is_not_a_video_is_an_error() {
    let text = write("not-a-video.mp4", b"not a video\n");
    assert!(VideoReader::open(&text).is_err());
    let jpeg = write("image-as-video.mp4", GRADIENT_JPEG);
    assert!(VideoReader::open(&jpeg).is_err());
    assert!(matches!(VideoReader::open(write("empty.mp4", b"")), Err(DecodeError::Empty { .. })));
}

// ---------------------------------------------------------------------------
// Measurements against the dataset (ignored: need data/ and fixtures/)
// ---------------------------------------------------------------------------

/// How far two same-sized RGB images are apart, over all samples.
struct Diff {
    max: u32,
    mean: f64,
    /// Pixels where any channel differs, and the fraction of all pixels.
    pixels: usize,
    fraction: f64,
    /// Samples whose absolute difference is 1, 2, 3 or 4, and 5 or more.
    hist: [usize; 5],
    /// Mean of `a - b` per channel (a bias, if any).
    bias: [f64; 3],
}

impl Diff {
    fn of(a: &Rgb8, b: &Rgb8) -> Diff {
        assert_eq!((a.width, a.height), (b.width, b.height), "sizes differ");
        let (mut max, mut sum, mut pixels) = (0u32, 0u64, 0usize);
        let mut hist = [0usize; 5];
        let mut bias = [0i64; 3];
        for (pa, pb) in a.data.chunks_exact(3).zip(b.data.chunks_exact(3)) {
            let mut any = false;
            for c in 0..3 {
                let d = i32::from(pa[c]) - i32::from(pb[c]);
                bias[c] += i64::from(d);
                let ad = d.unsigned_abs();
                max = max.max(ad);
                sum += u64::from(ad);
                if ad > 0 {
                    any = true;
                    hist[(ad as usize - 1).min(4)] += 1;
                }
            }
            pixels += usize::from(any);
        }
        let n = (a.width * a.height) as f64;
        Diff { max, mean: sum as f64 / (n * 3.0), pixels, fraction: pixels as f64 / n, hist, bias: bias.map(|s| s as f64 / n) }
    }

    fn line(&self) -> String {
        format!(
            "max {} mean {:.4} differing pixels {} ({:.2}%) |d|=1:{} 2:{} 3:{} 4:{} >=5:{} bias RGB [{:+.3} {:+.3} {:+.3}]",
            self.max, self.mean, self.pixels, 100.0 * self.fraction, self.hist[0], self.hist[1], self.hist[2], self.hist[3], self.hist[4], self.bias[0], self.bias[1], self.bias[2]
        )
    }
}

fn read_rgb(path: &Path, w: usize, h: usize) -> Rgb8 {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e} (run tools/reference/decode_oracle.py)", path.display()));
    assert_eq!(data.len(), w * h * 3, "{}: not {w}x{h} RGB", path.display());
    Rgb8::new(w, h, data)
}

/// Largest and mean absolute difference of two model inputs, in normalised units.
fn input_diff(a: &Rgb8, b: &Rgb8) -> (f32, f64) {
    let (ia, ib) = (sam_input(a, 1008), sam_input(b, 1008));
    let max = ia.iter().zip(&ib).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mean = ia.iter().zip(&ib).map(|(x, y)| f64::from((x - y).abs())).sum::<f64>() / ia.len() as f64;
    (max, mean)
}

#[test]
#[ignore = "needs data/visem and fixtures/decode (tools/reference/decode_oracle.py)"]
fn imageio_decode_against_pillow_on_visem_jpegs() {
    let root = repo();
    let manifest = std::fs::read_to_string(root.join("fixtures/decode/manifest.json")).expect("fixtures/decode/manifest.json: run tools/reference/decode_oracle.py");
    let manifest = parse_json(&manifest).expect("manifest json");
    println!("oracle: Pillow {}, libjpeg {}", manifest.get_str("pillow").unwrap(), manifest.get_str("libjpeg").unwrap());
    let frames = manifest.get_array("frames").expect("frames");
    assert!(frames.len() >= 4, "{} frames in the manifest", frames.len());
    let (mut worst_max, mut worst_mean, mut worst_input) = (0u32, 0f64, 0f32);
    for f in frames {
        let (w, h) = (f.get_u64("width").unwrap() as usize, f.get_u64("height").unwrap() as usize);
        let jpeg = root.join(f.get_str("jpeg").unwrap());
        let ours = decode_image(&jpeg).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((ours.width, ours.height), (w, h));
        let theirs = read_rgb(&root.join("fixtures/decode").join(f.get_str("file").unwrap()), w, h);
        let d = Diff::of(&ours, &theirs);
        let (imax, imean) = input_diff(&ours, &theirs);
        println!(
            "{} [{}, icc {}]: {}\n    sam_input (1008, normalised [-1, 1]): max abs diff {imax:.5} ({:.2} levels), mean {imean:.3e}",
            f.get_str("file").unwrap(),
            f.get_str("subsampling").unwrap(),
            f.get_bool("icc_profile").unwrap(),
            d.line(),
            f64::from(imax) * 127.5,
        );
        worst_max = worst_max.max(d.max);
        worst_mean = worst_mean.max(d.mean);
        worst_input = worst_input.max(imax);
    }
    println!("worst over {} frames: max {worst_max} levels, mean {worst_mean:.4}, sam_input max {worst_input:.5}", frames.len());

    // Truncation on a real 79 KB frame: a stride of cuts through the file, every cut
    // in and just after the scan header (ImageIO alone reports those complete), and
    // the last bytes.
    let bytes = std::fs::read(root.join(frames[0].get_str("jpeg").unwrap())).expect("jpeg bytes");
    let p = scratch("real-cut.jpg");
    let mut cuts: Vec<usize> = (1..bytes.len()).step_by(499).collect();
    cuts.extend(600..720);
    cuts.extend(bytes.len() - 40..bytes.len());
    let decoded: Vec<usize> = cuts
        .iter()
        .copied()
        .filter(|&cut| {
            std::fs::write(&p, &bytes[..cut]).expect("write");
            decode_image(&p).is_ok()
        })
        .collect();
    println!("{} real-frame cuts tried, {} decoded", cuts.len(), decoded.len());
    assert!(decoded.is_empty(), "cuts of the {}-byte frame that decoded as pictures: {decoded:?}", bytes.len());
}

#[test]
#[ignore = "needs data/visem (VISEM-Tracking video 11)"]
fn avfoundation_decodes_visem_video_11_and_is_compared_with_its_jpegs() {
    let dir = repo().join("data/visem/VISEM_Tracking_Train_v4/Train/11");
    let jpegs = std::fs::read_dir(dir.join("images")).expect("images dir").filter(|e| e.as_ref().is_ok_and(|e| e.path().extension().is_some_and(|x| x == "jpg"))).count();
    let mut reader = VideoReader::open(dir.join("11.mp4")).expect("open 11.mp4");
    let info = reader.info().clone();
    println!("{info:?}\n{jpegs} JPEG frames in images/");
    assert_eq!((info.width, info.height), (640, 480));
    assert_eq!(info.transform, [1, 0, 0, 1]);
    assert_eq!(info.frame_count, jpegs, "frame count against the dataset's extracted frames");
    assert!((info.fps - info.nominal_fps).abs() < 0.5, "fps {} vs nominal {}", info.fps, info.nominal_fps);
    assert!((info.duration * info.fps - info.frame_count as f64).abs() < 1e-6);

    // Decode everything: indices are consecutive, presentation times strictly increase
    // and follow the frame rate, sizes are constant.
    let probe = [0usize, 1, 700];
    let mut kept: Vec<(usize, Rgb8)> = Vec::new();
    let (mut last_pts, mut max_dev, mut count) = (f64::NEG_INFINITY, 0f64, 0usize);
    while let Some((i, pts, rgb)) = reader.next_frame().expect("decode") {
        assert_eq!(i, count);
        assert!(pts > last_pts, "frame {i}: pts {pts} after {last_pts}");
        assert_eq!((rgb.width, rgb.height), (640, 480));
        max_dev = max_dev.max((pts - i as f64 / info.fps).abs());
        last_pts = pts;
        count += 1;
        if probe.contains(&i) || probe.contains(&(i + 1)) || probe.contains(&(i.wrapping_sub(1))) {
            kept.push((i, rgb));
        }
    }
    assert_eq!(count, info.frame_count);
    println!("decoded {count} frames; last pts {last_pts:.4} s; max |pts - i/fps| = {max_dev:.5} s");
    assert!(reader.next_frame().expect("after the end").is_none(), "the reader stays finished");
    assert!(reader.next().is_none());

    for &i in &probe {
        for j in [i.saturating_sub(1), i, i + 1] {
            let jpeg = decode_image(dir.join(format!("images/11_frame_{j}.jpg"))).expect("jpeg");
            let video = &kept.iter().find(|(k, _)| *k == i).expect("kept frame").1;
            let d = Diff::of(video, &jpeg);
            let (imax, imean) = input_diff(video, &jpeg);
            println!("video frame {i} vs 11_frame_{j}.jpg: {}\n    sam_input max {imax:.4} mean {imean:.3e}", d.line());
        }
    }
}

#[test]
#[ignore = "needs fixtures/decode/rot90.mp4 (see this file's header for the ffmpeg command)"]
fn a_video_with_rotation_metadata_is_delivered_upright() {
    let root = repo();
    let rot = root.join("fixtures/decode/rot90.mp4");
    assert!(rot.is_file(), "{} is missing", rot.display());
    let mut r = VideoReader::open(&rot).expect("open");
    let info = r.info().clone();
    println!("{info:?}");
    assert_eq!((info.stored_width, info.stored_height), (640, 480));
    assert_eq!((info.width, info.height), (480, 640), "a quarter turn swaps the delivered size");
    let (_, _, got) = r.next_frame().expect("frame").expect("a first frame");
    assert_eq!((got.width, got.height), (480, 640));
    // The same bitstream without the metadata: the unrotated frame 0.
    let mut plain = VideoReader::open(root.join("data/visem/VISEM_Tracking_Train_v4/Train/11/11.mp4")).expect("open plain");
    let (_, _, flat) = plain.next_frame().expect("frame").expect("first frame");
    // Independent of the module's own rotation code: the two quarter turns by formula.
    let (w, h) = (flat.width, flat.height);
    let px = |img: &Rgb8, x: usize, y: usize| -> [u8; 3] { img.data[(y * img.width + x) * 3..][..3].try_into().unwrap() };
    let cw = |x: usize, y: usize| px(&flat, y, h - 1 - x); // out[y][x] = in[h-1-x][y]
    let ccw = |x: usize, y: usize| px(&flat, w - 1 - y, x); // out[y][x] = in[x][w-1-y]
    let (mut cw_bad, mut ccw_bad) = (0usize, 0usize);
    for y in 0..got.height {
        for x in 0..got.width {
            let g = px(&got, x, y);
            cw_bad += usize::from(g != cw(x, y));
            ccw_bad += usize::from(g != ccw(x, y));
        }
    }
    println!("transform {:?}: pixels differing from a clockwise turn {cw_bad}, from a counter-clockwise turn {ccw_bad} of {}", info.transform, got.width * got.height);
    match info.transform {
        [0, 1, -1, 0] => assert_eq!(cw_bad, 0),
        [0, -1, 1, 0] => assert_eq!(ccw_bad, 0),
        t => panic!("not a quarter turn: {t:?}"),
    }
}
