//! st-video's H.264 `.mp4` writer ([`st_video::VideoWriter`]), always run (macOS only, no data
//! needed): a synthetic clip is written, read back with [`st_video::VideoReader`] and compared, and
//! every refusal and failure path returns a typed error.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use st_video::encode::{self, EncodeError};
use st_video::{Rgb8, VideoReader, VideoWriter, WriterOptions};

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("st-video-encode");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let p = dir.join(name);
    let _ = std::fs::remove_file(&p);
    p
}

const W: usize = 320;
const H: usize = 240;
const SQUARE: usize = 40;
const SQUARE_COLOUR: [u8; 3] = [230, 230, 40];

/// The background of frame `i`: a mid-range colour that differs from every neighbour's by tens of
/// levels in each channel (full-range primaries clip in video-range Y'CbCr, and the test is about
/// the frames being told apart, not about the extremes).
fn background(i: usize) -> [u8; 3] {
    [
        (40 + i * 37 % 160) as u8,
        (40 + i * 71 % 160) as u8,
        (40 + i * 113 % 160) as u8,
    ]
}

/// Where the square's top-left corner is in frame `i`: it moves right and down, and never touches
/// the probe points `(310, 10)` and `(10, 235)`.
fn square_at(i: usize) -> (usize, usize) {
    (4 + 5 * i, 20 + 3 * i)
}

fn clip_frame(i: usize) -> Rgb8 {
    let bg = background(i);
    let (x0, y0) = square_at(i);
    let mut data = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            let inside = (x0..x0 + SQUARE).contains(&x) && (y0..y0 + SQUARE).contains(&y);
            data.extend_from_slice(if inside { &SQUARE_COLOUR } else { &bg });
        }
    }
    Rgb8::new(W, H, data)
}

fn px(img: &Rgb8, x: usize, y: usize) -> [u8; 3] {
    let o = (y * img.width + x) * 3;
    [img.data[o], img.data[o + 1], img.data[o + 2]]
}

fn max_diff(a: [u8; 3], b: [u8; 3]) -> u8 {
    (0..3).map(|k| a[k].abs_diff(b[k])).max().unwrap()
}

/// Read every frame of `path`.
fn read_all(path: &Path) -> (st_video::VideoInfo, Vec<(usize, f64, Rgb8)>) {
    let mut reader = VideoReader::open(path).expect("open");
    let info = reader.info().clone();
    let frames: Vec<_> = reader.by_ref().map(|f| f.expect("decode")).collect();
    (info, frames)
}

fn write_clip(path: &Path, frames: usize, fps: f64) -> st_video::VideoFile {
    let mut w = VideoWriter::create(path, W, H, fps).expect("create");
    for i in 0..frames {
        w.push(&clip_frame(i)).expect("push");
    }
    w.finish().expect("finish")
}

/// One H.264 level of error per channel in the interior of flat areas is the encoder's quantiser;
/// the rest is 8-bit BT.709 Y'CbCr quantisation (a step of 1/219 in Y' and 1/224 in Cb and Cr scales
/// to about 1.2 to 2 RGB levels, and the conversion back adds the same) and the chroma upsampling of
/// 4:2:0, which only matters at the square's edge (the probes are 15 pixels inside it). The bound is
/// measured (printed by `a_synthetic_clip_reads_back_with_its_frame_count_size_times_and_colours`;
/// 4 levels on this clip, twice that as the margin).
const COLOUR_TOLERANCE: u8 = 8;

#[test]
fn a_synthetic_clip_reads_back_with_its_frame_count_size_times_and_colours() {
    let path = scratch("clip.mp4");
    let file = write_clip(&path, 48, 24.0);
    assert_eq!(
        (file.frames, &file.path),
        (
            48,
            &std::fs::canonicalize(path.parent().unwrap())
                .unwrap()
                .join("clip.mp4")
        )
    );
    assert_eq!(file.bytes, std::fs::metadata(&path).unwrap().len());
    assert!(file.bytes > 1000);

    // It is an MP4 with an H.264 ('avc1') track.
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[4..8], b"ftyp");
    assert!(
        bytes.windows(4).any(|w| w == b"avc1") && bytes.windows(4).any(|w| w == b"avcC"),
        "no H.264 sample entry"
    );

    let (info, frames) = read_all(&path);
    assert_eq!((info.width, info.height), (W, H));
    assert_eq!(info.frame_count, 48);
    assert_eq!(frames.len(), 48);
    assert!(
        (info.fps - 24.0).abs() < 0.24,
        "fps {} (nominal {})",
        info.fps,
        info.nominal_fps
    );
    assert!(
        (info.duration - 2.0).abs() < 0.02,
        "duration {}",
        info.duration
    );

    let mut worst = 0u8;
    for (k, (i, pts, img)) in frames.iter().enumerate() {
        assert_eq!(*i, k);
        assert!(
            (pts - k as f64 / 24.0).abs() < 0.5 / 24.0,
            "frame {k} is presented at {pts} s, expected {}",
            k as f64 / 24.0
        );
        assert_eq!((img.width, img.height), (W, H));
        let (x0, y0) = square_at(k);
        let probes = [
            ((310, 10), background(k)),
            ((10, 235), background(k)),
            ((160, 235), background(k)),
            ((x0 + SQUARE / 2, y0 + SQUARE / 2), SQUARE_COLOUR),
            ((x0 + 8, y0 + 8), SQUARE_COLOUR),
        ];
        for ((x, y), want) in probes {
            let d = max_diff(px(img, x, y), want);
            worst = worst.max(d);
            assert!(
                d <= COLOUR_TOLERANCE,
                "frame {k} at ({x}, {y}): got {:?}, want {want:?}",
                px(img, x, y)
            );
        }
    }
    println!("worst per-channel colour error over 48 frames x 5 probes: {worst} levels (tolerance {COLOUR_TOLERANCE}); {} bytes", file.bytes);
}

#[test]
fn the_picture_is_upright_and_in_rgb_order() {
    // Red at the top left, blue at the bottom right, 40-pixel blocks on mid grey, so a flip, a
    // transpose or an R/B swap each changes a probe.
    let path = scratch("orientation.mp4");
    let mut img = Rgb8::new(
        W,
        H,
        [128u8, 128, 128]
            .iter()
            .copied()
            .cycle()
            .take(W * H * 3)
            .collect(),
    );
    for y in 0..60 {
        for x in 0..60 {
            img.data[(y * W + x) * 3..][..3].copy_from_slice(&[220, 40, 40]);
            img.data[((H - 1 - y) * W + (W - 1 - x)) * 3..][..3].copy_from_slice(&[40, 40, 220]);
        }
    }
    let mut w = VideoWriter::create(&path, W, H, 10.0).unwrap();
    for _ in 0..10 {
        w.push(&img).unwrap();
    }
    w.finish().unwrap();
    let (_, frames) = read_all(&path);
    assert_eq!(frames.len(), 10);
    let last = &frames[9].2;
    assert!(
        max_diff(px(last, 30, 30), [220, 40, 40]) <= COLOUR_TOLERANCE,
        "top left: {:?}",
        px(last, 30, 30)
    );
    assert!(
        max_diff(px(last, W - 31, H - 31), [40, 40, 220]) <= COLOUR_TOLERANCE,
        "bottom right: {:?}",
        px(last, W - 31, H - 31)
    );
    assert!(max_diff(px(last, W - 31, 30), [128, 128, 128]) <= COLOUR_TOLERANCE);
}

#[test]
fn a_fractional_frame_rate_is_stored_exactly_to_three_decimals() {
    let path = scratch("ntsc.mp4");
    let mut w = VideoWriter::create(&path, W, H, 29.97).unwrap();
    assert_eq!(w.fps(), 29.97);
    assert_eq!(w.size(), (W, H));
    for i in 0..30 {
        w.push(&clip_frame(i)).unwrap();
    }
    assert_eq!(w.frames(), 30);
    w.finish().unwrap();
    let (info, frames) = read_all(&path);
    assert_eq!(frames.len(), 30);
    assert!((info.fps - 29.97).abs() < 0.3, "fps {}", info.fps);
    for (k, (_, pts, _)) in frames.iter().enumerate() {
        assert!(
            (pts - k as f64 / 29.97).abs() < 0.5 / 29.97,
            "frame {k} at {pts}"
        );
    }
}

#[test]
fn a_clip_the_encoder_cannot_keep_up_with_is_written_in_full() {
    // Noise is the worst case for the encoder: frames are produced faster than they are encoded,
    // so the writer's input stops being ready and `push` has to wait, not throw.
    let path = scratch("noise.mp4");
    let (w_px, h_px, n) = (640usize, 480usize, 240usize);
    let mut w = VideoWriter::create(&path, w_px, h_px, 30.0).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let started = Instant::now();
    for _ in 0..n {
        let data: Vec<u8> = (0..w_px * h_px * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect();
        w.push(&Rgb8::new(w_px, h_px, data)).unwrap();
    }
    let file = w.finish().unwrap();
    println!(
        "{n} noise frames of {w_px}x{h_px}: {:.1} s, {} bytes",
        started.elapsed().as_secs_f64(),
        file.bytes
    );
    let (info, frames) = read_all(&path);
    assert_eq!((info.frame_count, frames.len()), (n, n));
}

#[test]
fn the_smallest_accepted_size_is_written() {
    let path = scratch("tiny.mp4");
    let min = encode::MIN_DIMENSION;
    let mut w = VideoWriter::create(&path, min, min, 5.0).unwrap();
    for i in 0..5u8 {
        w.push(&Rgb8::new(min, min, vec![60 + 30 * i; min * min * 3]))
            .unwrap();
    }
    w.finish().unwrap();
    let (info, frames) = read_all(&path);
    assert_eq!((info.width, info.height, frames.len()), (min, min, 5));
}

// ---------------------------------------------------------------------------
// Refusals and failures
// ---------------------------------------------------------------------------

#[test]
fn an_odd_size_is_refused_and_nothing_is_created() {
    let path = scratch("odd.mp4");
    for (w, h) in [(321, 240), (320, 241), (321, 241)] {
        assert!(
            matches!(
                VideoWriter::create(&path, w, h, 24.0),
                Err(EncodeError::InvalidSize { .. })
            ),
            "{w}x{h}"
        );
    }
    assert!(matches!(
        VideoWriter::create(&path, 0, 0, 24.0),
        Err(EncodeError::InvalidSize { .. })
    ));
    assert!(matches!(
        VideoWriter::create(&path, 20_000, 16, 24.0),
        Err(EncodeError::InvalidSize { .. })
    ));
    for fps in [0.0, f64::NAN, 1000.0] {
        assert!(
            matches!(
                VideoWriter::create(&path, W, H, fps),
                Err(EncodeError::InvalidFps { .. })
            ),
            "{fps}"
        );
    }
    assert!(!path.exists());
}

#[test]
fn a_frame_of_another_size_or_a_short_pixel_vector_is_refused_and_the_writer_carries_on() {
    let path = scratch("mismatch.mp4");
    let mut w = VideoWriter::create(&path, W, H, 10.0).unwrap();
    w.push(&clip_frame(0)).unwrap();
    let other = Rgb8::new(W + 2, H, vec![0; (W + 2) * H * 3]);
    assert!(matches!(
        w.push(&other),
        Err(EncodeError::BadFrame { index: 1, .. })
    ));
    let transposed = Rgb8::new(H, W, vec![0; W * H * 3]);
    assert!(matches!(
        w.push(&transposed),
        Err(EncodeError::BadFrame { .. })
    ));
    // The fields are public: a frame whose pixel vector does not match its stated size.
    let short = Rgb8 {
        width: W,
        height: H,
        data: vec![0; W * H * 3 - 1],
    };
    assert!(matches!(w.push(&short), Err(EncodeError::BadFrame { .. })));
    assert_eq!(w.frames(), 1, "a refused frame is not counted");
    w.push(&clip_frame(1)).unwrap();
    assert_eq!(w.finish().unwrap().frames, 2);
    assert_eq!(read_all(&path).1.len(), 2);
}

#[test]
fn a_bad_output_path_is_refused() {
    let dir = scratch("dir.mp4");
    assert!(matches!(
        VideoWriter::create("/nonexistent-st-video-dir/x.mp4", W, H, 24.0),
        Err(EncodeError::Io { .. })
    ));
    assert!(matches!(
        VideoWriter::create(scratch("clip.mov"), W, H, 24.0),
        Err(EncodeError::InvalidPath { .. })
    ));
    assert!(matches!(
        VideoWriter::create(scratch("noext"), W, H, 24.0),
        Err(EncodeError::InvalidPath { .. })
    ));
    assert!(matches!(
        VideoWriter::create("/", W, H, 24.0),
        Err(EncodeError::InvalidPath { .. })
    ));
    std::fs::create_dir(&dir).unwrap();
    let r = VideoWriter::create(&dir, W, H, 24.0);
    assert!(matches!(r, Err(EncodeError::InvalidPath { .. })), "{r:?}");
    assert!(
        dir.is_dir(),
        "a directory in the way is left alone, even with overwrite"
    );
    let r = VideoWriter::create_with(&dir, W, H, 24.0, WriterOptions { overwrite: true });
    assert!(matches!(r, Err(EncodeError::InvalidPath { .. })), "{r:?}");
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn an_unwritable_directory_is_an_error_not_a_panic() {
    use std::os::unix::fs::PermissionsExt;
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("st-video-encode")
        .join("readonly");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    let probe = std::fs::write(dir.join("probe"), b"x");
    let result = if probe.is_ok() {
        None // the process may write anyway (root): nothing to test
    } else {
        Some(
            VideoWriter::create(dir.join("x.mp4"), W, H, 24.0).and_then(|mut w| {
                for i in 0..4 {
                    w.push(&clip_frame(i))?;
                }
                w.finish()
            }),
        )
    };
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    if let Some(r) = result {
        let e = r.expect_err("writing into a read-only directory succeeded");
        println!("read-only directory: {e}");
        assert!(
            matches!(e, EncodeError::Failed { .. } | EncodeError::Io { .. }),
            "{e:?}"
        );
        assert!(
            !dir.join("x.mp4").exists(),
            "no partial file is left behind"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_existing_file_is_refused_unless_overwrite_is_asked_for() {
    let path = scratch("exists.mp4");
    std::fs::write(&path, b"precious").unwrap();
    assert!(matches!(
        VideoWriter::create(&path, W, H, 24.0),
        Err(EncodeError::Exists { .. })
    ));
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"precious",
        "a refusal leaves the file as it was"
    );
    // The early check says the same and changes nothing, with or without overwrite.
    assert!(matches!(
        encode::check_output(&path, false),
        Err(EncodeError::Exists { .. })
    ));
    assert_eq!(
        encode::check_output(&path, true).unwrap(),
        std::fs::canonicalize(&path).unwrap()
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"precious",
        "check_output removes nothing"
    );
    // Overwrite replaces it with a real video.
    let mut w =
        VideoWriter::create_with(&path, W, H, 24.0, WriterOptions { overwrite: true }).unwrap();
    for i in 0..6 {
        w.push(&clip_frame(i)).unwrap();
    }
    w.finish().unwrap();
    assert_eq!(read_all(&path).1.len(), 6);
    // And a finished video is an existing file in its turn.
    assert!(matches!(
        VideoWriter::create(&path, W, H, 24.0),
        Err(EncodeError::Exists { .. })
    ));
}

#[test]
fn a_writer_with_no_frame_has_no_video() {
    let path = scratch("empty.mp4");
    let w = VideoWriter::create(&path, W, H, 24.0).unwrap();
    assert!(matches!(w.finish(), Err(EncodeError::Empty { .. })));
    assert!(!path.exists(), "the cancelled session leaves no file");
}

#[test]
fn a_dropped_unfinished_writer_leaves_no_file_and_no_hang() {
    let path = scratch("dropped.mp4");
    let started = Instant::now();
    {
        let mut w = VideoWriter::create(&path, W, H, 24.0).unwrap();
        for i in 0..5 {
            w.push(&clip_frame(i)).unwrap();
        }
    }
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "dropping took {:?}",
        started.elapsed()
    );
    assert!(!path.exists(), "the half-written file is removed");
    // The path is free again, and the cancelled session left nothing running.
    let file = write_clip(&path, 4, 24.0);
    assert_eq!(read_all(&file.path).1.len(), 4);
    // A writer that never got a frame drops the same way.
    let path2 = scratch("dropped-empty.mp4");
    drop(VideoWriter::create(&path2, W, H, 24.0).unwrap());
    assert!(!path2.exists());
}

#[test]
fn a_writer_whose_output_disappears_reports_the_failure() {
    // Remove the output directory under a running writer: AVAssetWriter cannot finish the file, and
    // the error must come out of `push` or `finish`, not be swallowed.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("st-video-encode")
        .join("vanishing");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("v.mp4");
    let mut w = VideoWriter::create(&path, W, H, 24.0).unwrap();
    for i in 0..8 {
        w.push(&clip_frame(i)).unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
    let outcome = (|| {
        for i in 8..40 {
            w.push(&clip_frame(i))?;
        }
        w.finish().map(|_| ())
    })();
    println!("output removed mid-write: {outcome:?}");
    assert!(
        outcome.is_err(),
        "a video whose directory vanished was reported as written"
    );
    assert!(!path.exists());
}
