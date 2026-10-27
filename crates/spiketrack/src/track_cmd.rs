//! `spiketrack track <input> --prompt <text>`: text-prompted multi-object tracking of a
//! video with SAM 3.1's Object Multiplex tracker (`st_model::tracker::session::Session`:
//! detector, multiplex tracker and upstream's per-frame bookkeeping).
//!
//! `<input>` is either
//!
//! * a **folder of frames** named `<video>_frame_<n>.jpg` (VISEM-Tracking's `images/`
//!   folders; `.jpeg` and `.png` are accepted too), taken in numeric order of `n`, which
//!   must be consecutive; or
//! * an **mp4** (also `.mov`, `.m4v`), decoded by AVFoundation (`st_video::VideoReader`).
//!
//! `--start F` begins at frame `F` (the folder's frame number `n = F`; for a video the
//! `F`-th frame, the earlier ones decoded and discarded) and `--max-frames N` stops after
//! `N` frames. The text prompt is applied on the first frame processed and tracking runs
//! forward only: the one mode the session supports (upstream's `add_prompt(frame 0)` then
//! `propagate_in_video`). The session is told the clip has exactly the frames processed,
//! so `--max-frames` tracks "a video of N frames", as the parity tests do.
//!
//! # Preprocessing, per input kind
//!
//! Every frame is resized to 1008 x 1008 with `st_video::resize_bilinear_pil` (Pillow-exact
//! bilinear) and normalised **in float16 as upstream's loader does** (`io_utils.py`: the
//! frame is stored as `torch.float16`; `/255`, `- 0.5` and `/ 0.5` each round to f16, see
//! [`upstream_input`]), then widened to f32 for the model, as upstream's `.float()` does.
//! (`spiketrack image` keeps f32: upstream's single-image path is torchvision in f32.)
//!
//! * **Frame folder**: upstream's JPEG-folder path (`load_video_frames`) resizes with Pillow
//!   bilinear, the same resize. The difference left is ImageIO's JPEG decode instead of
//!   Pillow's libjpeg-turbo (`st-video/tests/decode.rs` measures it).
//! * **mp4**: upstream decodes with OpenCV and resizes with `cv2.resize(.., INTER_CUBIC)`
//!   after BGR to RGB (`io_utils.py:313-333`). Here AVFoundation decodes and the resize is
//!   bilinear, so **mp4 input does not match upstream's mp4 preprocessing**
//!   (GAP-MP4-PREPROCESSING). Frame-folder results are the comparable ones.
//!
//! # Outputs, under `--out DIR` (default `./spiketrack-track`)
//!
//! * `DIR/<video>.json`: the predictions in exactly the schema `spiketrack eval-visem`
//!   reads (documented in `eval_visem.rs`): `{"video", "width", "height", "frames": [{"frame":
//!   n, "objects": [{"id", "box_xyxy_norm": [x0, y0, x1, y1], "score"}]}]}`, one entry per
//!   processed frame, `n` being the frame number (the folder's `n`, or the video's frame
//!   index). Boxes are the session's own (`masks_to_boxes` extents of the track's mask,
//!   normalised by the frame size; upstream's convention, whose width is `x1 - x0` of pixel
//!   indices); a box that comes out with no width or height (a mask one pixel wide or high)
//!   is widened to one pixel so the file stays valid, and the count is reported. `score` is
//!   the id's birth probability, as upstream reports it.
//! * `DIR/masks/<video>_frame_<n>.json`: `{"frame": n, "size": [H, W], "objects": [{"id",
//!   "area", "counts"}]}`, each mask a COCO uncompressed RLE (column-major, zeros first) at
//!   the frame's own resolution.
//! * `DIR/<video>.run.json`: the run's provenance (input, prompt, backend, window,
//!   timings, preprocessing note, frame precision).
//! * with `--overlay K`: `DIR/overlay/<video>_frame_<n>.png` for every frame whose number is
//!   a multiple of K, masks and boxes coloured by track id.
//!
//! The session releases a frame only after 15 later ones (upstream's hot-start delay, which
//! decides which ids are real), so the first outputs appear 15 frames in; the run ends by
//! flushing the rest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use st_model::tracker::session::{Session, SessionConfig, YieldedFrame};
use st_model::tracker::vos::Vos;
use st_model::{Backbone, Detector, DetectorWeights, Text, Tokenizer};
use st_video::{decode_image, resize_bilinear_pil, Rgb8, VideoReader};
use st_weights::{Checkpoint, Sam31Weights};

use crate::cli::{existing_path, Args, CliError};
use crate::image_cmd::BackendChoice;
use crate::json;
use crate::mask::Rle;
use crate::overlay::{self, Drawn};
use crate::png;

pub const USAGE: &str = "\
spiketrack track <input> --prompt <text> [--out DIR] [--backend metal|cpu] [--max-frames N]
                 [--start F] [--overlay K] [--video ID] [--weights FILE] [--model-dir DIR]
                 [--threads N]

  Tracks every instance of <text> through a video with SAM 3.1's multiplex tracker, forward
  from the first frame processed. <input> is a folder of <video>_frame_<n>.jpg frames
  (numeric order, consecutive) or an .mp4/.mov/.m4v file.

  --out DIR         output directory (default ./spiketrack-track): <video>.json (predictions
                    in the schema `spiketrack eval-visem` reads), masks/, <video>.run.json
  --backend B       metal (Ojas on tessl, default) or cpu
  --max-frames N    stop after N frames (default: all from --start)
  --start F         first frame: the folder's frame number F, or the F-th video frame (default 0
                    for a video, the folder's lowest number for a folder)
  --overlay K       also write overlay/<video>_frame_<n>.png for every frame n divisible by K
  --video ID        the video id used in file names and \"video\" (default: from the frame
                    names, or the mp4's file name without extension)
  --weights FILE    the converted checkpoint (default: $SPIKETRACK_WEIGHTS)
  --model-dir DIR   folder with the tokenizer's vocab.json and merges.txt (default: $SPIKETRACK_MODEL_DIR)
  --threads N       CPU worker threads for --backend cpu, 1 to 256 (default 8)

  Preprocessing: every frame is resized with Pillow-style bilinear and normalised in
  float16, as upstream's video loader stores frames (float16 -> f32 for the model;
  `image` keeps f32). That is what upstream does for a folder of JPEG frames. For mp4 input it is NOT what
  upstream does: upstream decodes mp4 with OpenCV and resizes with INTER_CUBIC, so mp4
  results do not match upstream's mp4 preprocessing (GAP-MP4-PREPROCESSING).";

const PREPROCESSING_FOLDER: &str = "ImageIO decode; Pillow-exact bilinear resize to 1008x1008, as upstream's JPEG-folder path; /255, -0.5, /0.5 each rounded to float16 as upstream's loader stores frames (widened to f32 for the model)";
const PREPROCESSING_VIDEO: &str = "AVFoundation decode (32BGRA); Pillow-exact bilinear resize to 1008x1008; float16 normalisation as upstream. NOT upstream's mp4 path (OpenCV decode, cv2.INTER_CUBIC): GAP-MP4-PREPROCESSING";
const BUDGET_BYTES: u64 = 28 << 30;

// ---------------------------------------------------------------------------
// The model input, as upstream's video loader makes it
// ---------------------------------------------------------------------------

/// `value` rounded to the nearest float16 (ties to even), as an f32.
fn f16_round(value: f32) -> f32 {
    ojas_io::f16_to_f32(ojas_io::f32_to_f16(value))
}

/// Upstream's normalisation of one 8-bit sample (`io_utils.py`): `img / 255.0` (float64) is
/// stored as float16, then `img -= 0.5` and `img /= 0.5` run on float16 tensors, each result
/// rounded to float16. The subtraction of two float16 values and the doubling are exact in
/// f32, so rounding after each step reproduces the float16 arithmetic.
fn upstream_normalise(v: u8) -> f32 {
    let stored = f16_round((f64::from(v) / 255.0) as f32);
    let centred = f16_round(stored - 0.5);
    f16_round(centred / 0.5)
}

/// `[1, 3, size, size]` (NCHW) model input from 8-bit RGB: Pillow bilinear resize, then
/// [`upstream_normalise`] per sample, widened to f32.
pub fn upstream_input(src: &Rgb8, size: usize) -> Vec<f32> {
    let resized = resize_bilinear_pil(src, size, size);
    let lut: Vec<f32> = (0..=255u8).map(upstream_normalise).collect();
    let n = size * size;
    let mut out = vec![0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            out[c * n + i] = lut[usize::from(resized.data[i * 3 + c])];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Frame sources
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameFile {
    pub n: u32,
    pub path: PathBuf,
}

/// The frames of a folder, in numeric order.
#[derive(Debug, PartialEq, Eq)]
pub struct Folder {
    pub video: String,
    pub frames: Vec<FrameFile>,
}

const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];

/// `<prefix>_frame_<n>` from a file name stem.
fn parse_stem(stem: &str) -> Option<(&str, u32)> {
    let (prefix, n) = stem.rsplit_once("_frame_")?;
    if prefix.is_empty() || n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((prefix, n.parse().ok()?))
}

/// List `dir`: every image file must be named `<video>_frame_<n>.<jpg|jpeg|png>` with one
/// `<video>` and no repeated `n`; other files are ignored.
pub fn discover_frames(dir: &Path) -> Result<Folder, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut video: Option<String> = None;
    let mut frames: BTreeMap<u32, PathBuf> = BTreeMap::new();
    for entry in entries {
        let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
        if !IMAGE_EXTENSIONS.contains(&ext.as_str()) || !path.is_file() {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let (prefix, n) = parse_stem(stem).ok_or_else(|| format!("{}: an image whose name is not <video>_frame_<n>.{ext}", path.display()))?;
        match &video {
            None => video = Some(prefix.to_string()),
            Some(v) if v == prefix => {}
            Some(v) => return Err(format!("{}: frames of two videos ({v:?} and {prefix:?})", dir.display())),
        }
        if let Some(prev) = frames.insert(n, path.clone()) {
            return Err(format!("frame {n} appears twice: {} and {}", prev.display(), path.display()));
        }
    }
    let video = video.ok_or_else(|| format!("{}: no <video>_frame_<n> images", dir.display()))?;
    Ok(Folder { video, frames: frames.into_iter().map(|(n, path)| FrameFile { n, path }).collect() })
}

/// The frames from number `start` (default: the first), at most `max` of them; they must be
/// consecutive numbers.
pub fn select_window(frames: &[FrameFile], start: Option<u32>, max: Option<usize>) -> Result<&[FrameFile], String> {
    let first = frames.first().ok_or("the folder has no frames")?.n;
    let start = start.unwrap_or(first);
    let at = frames.iter().position(|f| f.n == start).ok_or_else(|| format!("no frame {start} in the folder (frames {first}..{})", frames[frames.len() - 1].n))?;
    let rest = &frames[at..];
    let take = max.map_or(rest.len(), |m| m.min(rest.len()));
    let window = &rest[..take];
    for pair in window.windows(2) {
        if pair[1].n != pair[0].n + 1 {
            return Err(format!("frames {} and {} are not consecutive (the tracker needs every frame)", pair[0].n, pair[1].n));
        }
    }
    Ok(window)
}

enum Kind {
    Folder { files: Vec<FrameFile> },
    Video { reader: VideoReader },
}

/// The frames to track, one at a time, all of one size.
pub struct FrameStream {
    kind: Kind,
    pub video: String,
    /// The number of the first frame (`n` for a folder, the frame index for a video).
    pub first_n: u32,
    pub len: usize,
    pub height: usize,
    pub width: usize,
    pub from_video: bool,
    produced: usize,
    pending: Option<Rgb8>,
}

fn is_video_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(), Some("mp4" | "mov" | "m4v"))
}

impl FrameStream {
    pub fn open(input: &Path, start: Option<u32>, max: Option<usize>, video_override: Option<&str>) -> Result<FrameStream, CliError> {
        if input.is_dir() {
            let folder = discover_frames(input).map_err(CliError::failed)?;
            let files = select_window(&folder.frames, start, max).map_err(CliError::failed)?.to_vec();
            let first = decode_image(&files[0].path).map_err(CliError::from_err)?;
            Ok(FrameStream {
                first_n: files[0].n,
                len: files.len(),
                height: first.height,
                width: first.width,
                video: video_override.map_or(folder.video, str::to_string),
                kind: Kind::Folder { files },
                from_video: false,
                produced: 0,
                pending: Some(first),
            })
        } else if input.is_file() && is_video_file(input) {
            let mut reader = VideoReader::open(input).map_err(CliError::from_err)?;
            let info = reader.info().clone();
            let start = start.unwrap_or(0);
            if start as usize >= info.frame_count {
                return Err(CliError::failed(format!("--start {start} is past the video's {} frames", info.frame_count)));
            }
            for skipped in 0..start as usize {
                match reader.next_frame().map_err(CliError::from_err)? {
                    Some(_) => {}
                    None => return Err(CliError::failed(format!("the video ended after {skipped} frames while skipping to --start {start}"))),
                }
            }
            let len = max.map_or(info.frame_count - start as usize, |m| m.min(info.frame_count - start as usize));
            let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("video").to_string();
            Ok(FrameStream {
                first_n: start,
                len,
                height: info.height,
                width: info.width,
                video: video_override.map_or(stem, str::to_string),
                kind: Kind::Video { reader },
                from_video: true,
                produced: 0,
                pending: None,
            })
        } else if input.exists() {
            Err(CliError::failed(format!("{}: not a folder of frames or an .mp4/.mov/.m4v file", input.display())))
        } else {
            Err(CliError::failed(format!("{}: does not exist", input.display())))
        }
    }

    /// The next frame and its number, or `None` after `len` frames.
    pub fn next_frame(&mut self) -> Result<Option<(u32, Rgb8)>, CliError> {
        if self.produced >= self.len {
            return Ok(None);
        }
        let n = self.first_n + self.produced as u32;
        let img = match &mut self.kind {
            Kind::Folder { files } => match self.pending.take() {
                Some(img) => img,
                None => decode_image(&files[self.produced].path).map_err(CliError::from_err)?,
            },
            Kind::Video { reader } => match reader.next_frame().map_err(CliError::from_err)? {
                Some((i, _, rgb)) if i == n as usize => rgb,
                Some((i, _, _)) => return Err(CliError::failed(format!("the video returned frame {i} where {n} was expected"))),
                None => return Err(CliError::failed(format!("the video ended at frame {n}, before the {} frames asked for", self.len))),
            },
        };
        if (img.height, img.width) != (self.height, self.width) {
            return Err(CliError::failed(format!("frame {n} is {}x{} but the video is {}x{}", img.width, img.height, self.width, self.height)));
        }
        self.produced += 1;
        Ok(Some((n, img)))
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct TrackedObject {
    pub id: i64,
    pub score: f32,
    pub box_xyxy_norm: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameResult {
    pub n: u32,
    pub objects: Vec<TrackedObject>,
}

/// A released frame's objects with boxes as normalised xyxy, and how many boxes were
/// widened to one pixel because the mask is a single pixel wide or high.
pub fn objects_of(y: &YieldedFrame, height: usize, width: usize) -> Result<(Vec<TrackedObject>, usize), String> {
    let n = y.obj_ids.len();
    if y.probs.len() != n || y.boxes_xywh.len() != n * 4 || y.masks.len() != n {
        return Err(format!("frame {}: {n} ids, {} scores, {} box values, {} masks", y.frame, y.probs.len(), y.boxes_xywh.len(), y.masks.len()));
    }
    let mut widened = 0;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let b = &y.boxes_xywh[k * 4..k * 4 + 4];
        if !b.iter().all(|v| v.is_finite()) || !y.probs[k].is_finite() {
            return Err(format!("frame {}: id {} has a non-finite box or score", y.frame, y.obj_ids[k]));
        }
        let (x0, y0) = (b[0], b[1]);
        let (mut x1, mut y1) = (b[0] + b[2], b[1] + b[3]);
        let mut changed = false;
        if x1 <= x0 {
            x1 = x0 + 1.0 / width as f32;
            changed = true;
        }
        if y1 <= y0 {
            y1 = y0 + 1.0 / height as f32;
            changed = true;
        }
        widened += usize::from(changed);
        out.push(TrackedObject { id: y.obj_ids[k], score: y.probs[k], box_xyxy_norm: [x0, y0, x1, y1] });
    }
    Ok((out, widened))
}

/// `DIR/<video>.json`: the `spiketrack eval-visem` prediction schema.
pub fn predictions_json(video: &str, width: usize, height: usize, frames: &[FrameResult]) -> Result<String, String> {
    let mut out = format!("{{\n  \"video\": {},\n  \"width\": {width},\n  \"height\": {height},\n  \"frames\": [", json::string(video));
    for (i, f) in frames.iter().enumerate() {
        out.push_str(&format!("{}\n    {{\"frame\": {}, \"objects\": [", if i == 0 { "" } else { "," }, f.n));
        for (k, o) in f.objects.iter().enumerate() {
            let b: Result<Vec<String>, String> = o.box_xyxy_norm.iter().map(|&v| json::number32(v, "box")).collect();
            out.push_str(&format!(
                "{}{{\"id\": {}, \"box_xyxy_norm\": [{}], \"score\": {}}}",
                if k == 0 { "" } else { ", " },
                o.id,
                b?.join(", "),
                json::number32(o.score, "score")?
            ));
        }
        out.push_str("]}");
    }
    out.push_str(if frames.is_empty() { "]\n}\n" } else { "\n  ]\n}\n" });
    Ok(out)
}

/// `DIR/masks/<video>_frame_<n>.json`.
pub fn masks_json(n: u32, height: usize, width: usize, items: &[(i64, Rle)]) -> String {
    let objs: Vec<String> = items.iter().map(|(id, r)| format!("{{\"id\": {id}, \"area\": {}, \"counts\": [{}]}}", r.area, r.counts_json())).collect();
    format!("{{\"frame\": {n}, \"size\": [{height}, {width}], \"objects\": [{}]}}\n", objs.join(", "))
}

/// Receives released frames: keeps their results and writes the per-frame files.
pub struct Collector {
    out: PathBuf,
    video: String,
    first_n: u32,
    height: usize,
    width: usize,
    overlay_every: Option<u32>,
    images: BTreeMap<usize, Rgb8>,
    pub results: Vec<FrameResult>,
    pub widened: usize,
    pub ids: BTreeSet<i64>,
}

impl Collector {
    pub fn new(out: &Path, video: &str, first_n: u32, height: usize, width: usize, overlay_every: Option<u32>) -> Result<Collector, CliError> {
        std::fs::create_dir_all(out.join("masks")).map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
        if overlay_every.is_some() {
            std::fs::create_dir_all(out.join("overlay")).map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
        }
        Ok(Collector { out: out.to_path_buf(), video: video.to_string(), first_n, height, width, overlay_every, images: BTreeMap::new(), results: Vec::new(), widened: 0, ids: BTreeSet::new() })
    }

    /// Frame `index` (0-based within the run) was pushed: keep it if an overlay will want it.
    pub fn keep_image(&mut self, index: usize, image: &Rgb8) {
        if self.overlay_every.is_some_and(|k| (self.first_n + index as u32) % k == 0) {
            self.images.insert(index, image.clone());
        }
    }

    pub fn accept(&mut self, y: YieldedFrame) -> Result<(), CliError> {
        let (h, w) = (self.height, self.width);
        let n = self.first_n + y.frame as u32;
        let (objects, widened) = objects_of(&y, h, w).map_err(CliError::failed)?;
        self.widened += widened;
        self.ids.extend(objects.iter().map(|o| o.id));
        let rles: Vec<(i64, Rle)> = y.obj_ids.iter().zip(&y.masks).map(|(&id, m)| (id, Rle::from_fn(h, w, |r, c| m.get(r * w + c)))).collect();
        let path = self.out.join("masks").join(format!("{}_frame_{n}.json", self.video));
        std::fs::write(&path, masks_json(n, h, w, &rles)).map_err(|e| CliError::failed(format!("{}: {e}", path.display())))?;
        if let Some(image) = self.images.remove(&y.frame) {
            let bits: Vec<Vec<bool>> = rles.iter().map(|(_, r)| r.decode()).collect();
            let drawn: Vec<Drawn<'_>> = objects
                .iter()
                .zip(&bits)
                .map(|(o, m)| {
                    let b = o.box_xyxy_norm;
                    let (fw, fh) = (w as f64, h as f64);
                    Drawn { mask: m, box_px: [f64::from(b[0]) * fw, f64::from(b[1]) * fh, f64::from(b[2]) * fw, f64::from(b[3]) * fh], colour: o.id.rem_euclid(1 << 20) as usize }
                })
                .collect();
            let bytes = png::encode_rgb(w, h, &overlay::render(&image, &drawn).data).map_err(CliError::from_err)?;
            let path = self.out.join("overlay").join(format!("{}_frame_{n}.png", self.video));
            std::fs::write(&path, bytes).map_err(|e| CliError::failed(format!("{}: {e}", path.display())))?;
        }
        self.results.push(FrameResult { n, objects });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

struct Options {
    prompt: String,
    out: PathBuf,
    overlay_every: Option<u32>,
    backend: BackendChoice,
    threads: usize,
    weights: PathBuf,
    model_dir: PathBuf,
}

struct Timing {
    load_seconds: f64,
    frame_seconds: Vec<f64>,
}

fn track_on<B: Backend>(be: &B, w: &Sam31Weights, stream: &mut FrameStream, opts: &Options, collector: &mut Collector) -> Result<Timing, CliError> {
    let t_load = Instant::now();
    let backbone = Backbone::new(be, &w.vision).map_err(CliError::from_err)?;
    let text = Text::new(be, &w.text).map_err(CliError::from_err)?;
    let detector = Detector::new(
        be,
        DetectorWeights { transformer: &w.detector_transformer, geometry: &w.geometry_encoder, segmentation: &w.segmentation_head, scoring: &w.dot_prod_scoring },
    )
    .map_err(CliError::from_err)?;
    let vos = Vos::new(be, &backbone, &w.tracker).map_err(CliError::from_err)?;
    let tok = Tokenizer::from_hf_files(&opts.model_dir.join("vocab.json"), &opts.model_dir.join("merges.txt"))
        .map_err(|e| CliError::failed(format!("tokenizer files in {}: {e}", opts.model_dir.display())))?;
    let prompt = Session::encode_prompt(be, &text, &tok, &opts.prompt).map_err(|e| CliError::failed(format!("prompt {:?}: {e}", opts.prompt)))?;
    let mut session = Session::new(&vos, &detector, prompt, SessionConfig::new(stream.len, stream.height, stream.width));
    let load_seconds = t_load.elapsed().as_secs_f64();
    println!("model loaded in {load_seconds:.1} s on {}; tracking {} frame(s) of {:?}", opts.backend.name(), stream.len, stream.video);

    let side = st_model::IMAGE_SIZE;
    let mut frame_seconds = Vec::with_capacity(stream.len);
    let mut index = 0usize;
    while let Some((n, image)) = stream.next_frame()? {
        let input = upstream_input(&image, side);
        let tensor = Tensor::from_f32(&input, &[1, 3, side, side], be.budget()).map_err(CliError::from_err)?;
        let device = be.upload(&tensor).map_err(CliError::from_err)?;
        collector.keep_image(index, &image);
        let t0 = Instant::now();
        let released = session.push_frame(&device).map_err(|e| CliError::failed(format!("frame {n}: {e}")))?;
        be.sync().map_err(CliError::from_err)?;
        let secs = t0.elapsed().as_secs_f64();
        frame_seconds.push(secs);
        println!("frame {n} ({}/{}): {secs:.2} s, {} active track(s), {} frame(s) released", index + 1, stream.len, session.book().obj_ids().len(), released.len());
        for y in released {
            collector.accept(y)?;
        }
        index += 1;
    }
    for y in session.finish() {
        collector.accept(y)?;
    }
    Ok(Timing { load_seconds, frame_seconds })
}

pub fn run(raw: &[String]) -> Result<(), CliError> {
    let args = Args::parse(raw, &["prompt", "out", "backend", "max-frames", "start", "overlay", "video", "weights", "model-dir", "threads"], 1)?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }
    let input = PathBuf::from(&args.positional[0]);
    let prompt = args.require("prompt")?.trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("--prompt is blank"));
    }
    let backend = BackendChoice::parse(args.get("backend").unwrap_or("metal"))?;
    let threads: usize = args.parsed("threads", 8)?;
    if !(1..=256).contains(&threads) {
        return Err(CliError::usage(format!("--threads {threads} is outside 1..=256")));
    }
    let max_frames = args.get("max-frames").map(|_| args.parsed("max-frames", 0usize)).transpose()?;
    if max_frames == Some(0) {
        return Err(CliError::usage("--max-frames must be at least 1"));
    }
    let start = args.get("start").map(|_| args.parsed("start", 0u32)).transpose()?;
    let overlay_every = args.get("overlay").map(|_| args.parsed("overlay", 0u32)).transpose()?;
    if overlay_every == Some(0) {
        return Err(CliError::usage("--overlay must be at least 1"));
    }
    if let Some(v) = args.get("video") {
        if v.contains(['/', '\\']) || v == "." || v == ".." {
            return Err(CliError::usage(format!("--video {v:?} is not a plain id")));
        }
    }
    let out = PathBuf::from(args.get("out").unwrap_or("spiketrack-track"));

    // The input is opened (and its first frame decoded) before the model is touched, so a bad
    // path or a gap in the frames fails in a second, not after the weights are loaded.
    let mut stream = FrameStream::open(&input, start, max_frames, args.get("video"))?;
    let weights = existing_path(args.get("weights"), "SPIKETRACK_WEIGHTS", "weights", "weights")?;
    let model_dir = existing_path(args.get("model-dir"), "SPIKETRACK_MODEL_DIR", "tokenizer folder", "model-dir")?;
    let opts = Options { prompt, out, overlay_every, backend, threads, weights, model_dir };

    let ck = Checkpoint::open(&opts.weights).map_err(|e| CliError::failed(format!("checkpoint {}: {e}", opts.weights.display())))?;
    let w = Sam31Weights::load(&ck).map_err(|e| CliError::failed(format!("checkpoint {}: {e}", opts.weights.display())))?;
    let mut collector = Collector::new(&opts.out, &stream.video, stream.first_n, stream.height, stream.width, opts.overlay_every)?;
    let t_all = Instant::now();
    let timing = match opts.backend {
        BackendChoice::Cpu => track_on(&CpuBackend::with_threads(Budget::new(BUDGET_BYTES), opts.threads).map_err(CliError::from_err)?, &w, &mut stream, &opts, &mut collector)?,
        BackendChoice::Metal => track_on(&MetalBackend::new(Budget::new(BUDGET_BYTES)).map_err(CliError::from_err)?, &w, &mut stream, &opts, &mut collector)?,
    };
    let total_seconds = t_all.elapsed().as_secs_f64();

    let mut results = std::mem::take(&mut collector.results);
    results.sort_by_key(|r| r.n);
    let expect: Vec<u32> = (stream.first_n..stream.first_n + stream.len as u32).collect();
    if results.iter().map(|r| r.n).collect::<Vec<_>>() != expect {
        return Err(CliError::failed(format!("the session released {} frame(s) for {} processed", results.len(), stream.len)));
    }
    let text = predictions_json(&stream.video, stream.width, stream.height, &results).map_err(CliError::failed)?;
    let pred_path = opts.out.join(format!("{}.json", stream.video));
    std::fs::write(&pred_path, text).map_err(|e| CliError::failed(format!("{}: {e}", pred_path.display())))?;

    let mean = timing.frame_seconds.iter().sum::<f64>() / timing.frame_seconds.len().max(1) as f64;
    let run_text = format!(
        "{{\n  \"video\": {},\n  \"input\": {},\n  \"prompt\": {},\n  \"backend\": {},\n  \"start\": {},\n  \"frames\": {},\n  \"width\": {},\n  \"height\": {},\n  \"tracks\": {},\n  \"boxes_widened_to_one_pixel\": {},\n  \"model_load_seconds\": {},\n  \"mean_seconds_per_frame\": {},\n  \"total_seconds\": {},\n  \"frame_precision\": \"float16 normalisation as upstream's loader, widened to f32\",\n  \"preprocessing\": {}\n}}\n",
        json::string(&stream.video),
        json::string(&input.to_string_lossy()),
        json::string(&opts.prompt),
        json::string(opts.backend.name()),
        stream.first_n,
        stream.len,
        stream.width,
        stream.height,
        collector.ids.len(),
        collector.widened,
        json::number(timing.load_seconds, "load time").map_err(CliError::failed)?,
        json::number(mean, "mean time").map_err(CliError::failed)?,
        json::number(total_seconds, "total time").map_err(CliError::failed)?,
        json::string(if stream.from_video { PREPROCESSING_VIDEO } else { PREPROCESSING_FOLDER }),
    );
    let run_path = opts.out.join(format!("{}.run.json", stream.video));
    std::fs::write(&run_path, run_text).map_err(|e| CliError::failed(format!("{}: {e}", run_path.display())))?;

    println!(
        "{} frame(s) of {:?} tracked on {}: {} track(s) born, {:.2} s/frame (model, mean), {:.1} s total\n  {}\n  {}",
        stream.len,
        stream.video,
        opts.backend.name(),
        collector.ids.len(),
        mean,
        total_seconds,
        pred_path.display(),
        run_path.display()
    );
    if collector.widened > 0 {
        println!("  note: {} box(es) widened to one pixel (a mask one pixel wide or high)", collector.widened);
    }
    if stream.from_video {
        println!("  note: mp4 input does not match upstream's mp4 preprocessing (GAP-MP4-PREPROCESSING)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_visem::{evaluate_window, parse_predictions, Window};
    use st_model::tracker::associate::Bitmask;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("spiketrack-track-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"x").unwrap();
    }

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn f16_rounding_is_nearest_even() {
        assert_eq!(f16_round(1.0), 1.0);
        assert_eq!(f16_round(0.0), 0.0);
        assert_eq!(f16_round(-1.0), -1.0);
        // Between 1 and 1 + 2^-10 the spacing is 2^-10: the halfway point rounds to the even 1.0, ...
        assert_eq!(f16_round(1.0 + 2f32.powi(-11)), 1.0);
        // ... and 1 + 3 * 2^-11 (halfway between 1 + 2^-10 and 1 + 2^-9) to the even 1 + 2^-9.
        assert_eq!(f16_round(1.0 + 3.0 * 2f32.powi(-11)), 1.0 + 2f32.powi(-9));
        // Just above the halfway point rounds up; just below rounds down.
        assert_eq!(f16_round(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 1.0 + 2f32.powi(-10));
        assert_eq!(f16_round(1.0 + 2f32.powi(-11) - 2f32.powi(-20)), 1.0);
        // 0.1 is not representable: the float16 neighbour is 0.0999755859375 (= 1638 / 2^14).
        assert_eq!(f16_round(0.1), 1638.0 / 16384.0);
        assert_eq!(f16_round(65519.0), 65504.0);
    }

    #[test]
    fn the_upstream_normalisation_is_hand_checked_at_a_few_samples() {
        // 0 and 255 map to -1 and 1 exactly.
        assert_eq!(upstream_normalise(0), -1.0);
        assert_eq!(upstream_normalise(255), 1.0);
        // 128/255 = 0.50196..: float16 spacing there is 2^-11, so it is stored as 1028 / 2048;
        // minus 0.5 is exactly 2^-9, doubled 2^-8.
        assert_eq!(upstream_normalise(128), 2f32.powi(-8));
        // 127/255 = 0.49803..: spacing 2^-12, stored as 2040 / 4096; minus 0.5 is -2^-9, doubled -2^-8.
        assert_eq!(upstream_normalise(127), -(2f32.powi(-8)));
        // 1/255 = 0.0039215..: spacing 2^-18, stored as 1028 / 2^18; minus 0.5 is -0.49607849..,
        // which in [0.25, 0.5) rounds to 2032 / 4096 = 0.4961 (-0.49609375); doubled -0.9921875.
        assert_eq!(upstream_normalise(1), -0.9921875);
        // Plain f32 would give (1/255 - 0.5) / 0.5 = -0.99215686: the f16 steps change it.
        assert!(((1.0f32 / 255.0 - 0.5) / 0.5 - upstream_normalise(1)).abs() > 1e-5);
        // Every value is a float16 number in [-1, 1], and the map is monotone.
        let all: Vec<f32> = (0..=255u8).map(upstream_normalise).collect();
        assert!(all.iter().all(|&v| (-1.0..=1.0).contains(&v) && f16_round(v) == v));
        assert!(all.windows(2).all(|w| w[0] <= w[1]));
    }

    /// Upstream's own input frames (`fixtures/detect_track/11.safetensors`, `frames`, float16,
    /// made by its folder loader from `11_frame_<n>.jpg`) against `upstream_input` of the same
    /// JPEG decoded by Pillow (`fixtures/decode/11_frame_0.rgb`, tools/reference/decode_oracle.py):
    /// the resize and the float16 normalisation must agree bit for bit. Then the same frame
    /// decoded by ImageIO: the difference the decoder alone makes.
    #[test]
    #[ignore = "needs fixtures/detect_track/11.safetensors, fixtures/decode/11_frame_0.rgb and data/visem"]
    fn upstream_input_matches_upstreams_loader_on_a_visem_frame() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let st = ojas_io::SafeTensors::open(&root.join("fixtures/detect_track/11.safetensors")).unwrap_or_else(|e| panic!("{}", e.detail()));
        let (shape, frames) = st.read_f32_widened("frames").unwrap_or_else(|e| panic!("{}", e.detail()));
        assert_eq!(&shape[1..], &[3, 1008, 1008]);
        let want = &frames[..3 * 1008 * 1008];
        let rgb = std::fs::read(root.join("fixtures/decode/11_frame_0.rgb")).expect("run tools/reference/decode_oracle.py");
        let pillow = Rgb8::new(640, 480, rgb);
        let ours = upstream_input(&pillow, 1008);
        let differing = ours.iter().zip(want).filter(|(a, b)| a != b).count();
        let max = ours.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("Pillow-decoded frame 0: {differing} of {} samples differ from upstream's float16 frame (max {max})", want.len());
        assert_eq!(differing, 0, "the resize and float16 normalisation are not bit-identical to upstream's loader");
        let imageio = decode_image(root.join("data/visem/VISEM_Tracking_Train_v4/Train/11/images/11_frame_0.jpg")).expect("jpeg");
        let ours = upstream_input(&imageio, 1008);
        let differing = ours.iter().zip(want).filter(|(a, b)| a != b).count();
        let max = ours.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("ImageIO-decoded frame 0: {differing} of {} samples differ (max {max}); plain f32 sam_input differs from upstream by", want.len());
        let plain = st_video::sam_input(&Rgb8::new(640, 480, std::fs::read(root.join("fixtures/decode/11_frame_0.rgb")).unwrap()), 1008);
        let max_plain = plain.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let diff_plain = plain.iter().zip(want).filter(|(a, b)| a != b).count();
        println!("  (Pillow-decoded, f32 sam_input: {diff_plain} samples differ, max {max_plain})");
    }

    #[test]
    fn the_model_input_is_resized_then_normalised_planar() {
        // A constant image stays constant under bilinear resize; 3 planes, each one value.
        let img = Rgb8::new(5, 3, [10u8, 128, 250].iter().copied().cycle().take(5 * 3 * 3).collect());
        let x = upstream_input(&img, 8);
        assert_eq!(x.len(), 3 * 64);
        for (c, v) in [10u8, 128, 250].into_iter().enumerate() {
            assert!(x[c * 64..(c + 1) * 64].iter().all(|&s| s == upstream_normalise(v)), "plane {c}");
        }
    }

    #[test]
    fn frames_are_ordered_numerically_not_alphabetically() {
        let d = temp("order");
        for n in [10, 2, 1, 100, 21] {
            touch(&d, &format!("22_frame_{n}.jpg"));
        }
        touch(&d, ".DS_Store");
        touch(&d, "notes.txt");
        let f = discover_frames(&d).unwrap();
        assert_eq!(f.video, "22");
        assert_eq!(f.frames.iter().map(|f| f.n).collect::<Vec<_>>(), vec![1, 2, 10, 21, 100]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_stem_must_be_video_frame_number() {
        assert_eq!(parse_stem("22_frame_640"), Some(("22", 640)));
        assert_eq!(parse_stem("my_clip_frame_7"), Some(("my_clip", 7)));
        assert_eq!(parse_stem("_frame_7"), None);
        assert_eq!(parse_stem("22_frame_"), None);
        assert_eq!(parse_stem("22_frame_-1"), None);
        assert_eq!(parse_stem("22_frame_1a"), None);
        assert_eq!(parse_stem("22.frame.1"), None);
    }

    #[test]
    fn folders_that_cannot_be_tracked_are_refused() {
        let d = temp("bad");
        assert!(discover_frames(&d).unwrap_err().contains("no <video>_frame_<n>"));
        touch(&d, "22_frame_1.jpg");
        touch(&d, "23_frame_2.jpg");
        assert!(discover_frames(&d).unwrap_err().contains("two videos"));
        std::fs::remove_file(d.join("23_frame_2.jpg")).unwrap();
        touch(&d, "22_frame_01.png");
        assert!(discover_frames(&d).unwrap_err().contains("twice"), "frame 1 as .jpg and as 01.png");
        std::fs::remove_file(d.join("22_frame_01.png")).unwrap();
        touch(&d, "stray.JPG");
        assert!(discover_frames(&d).unwrap_err().contains("not <video>_frame_<n>"));
        let _ = std::fs::remove_dir_all(&d);
    }

    fn files(ns: &[u32]) -> Vec<FrameFile> {
        ns.iter().map(|&n| FrameFile { n, path: PathBuf::from(format!("{n}.jpg")) }).collect()
    }

    #[test]
    fn the_window_starts_at_a_frame_number_and_stops_at_max() {
        let f = files(&[5, 6, 7, 8, 9]);
        assert_eq!(select_window(&f, None, None).unwrap().len(), 5);
        assert_eq!(select_window(&f, Some(7), None).unwrap().iter().map(|f| f.n).collect::<Vec<_>>(), vec![7, 8, 9]);
        assert_eq!(select_window(&f, Some(6), Some(2)).unwrap().iter().map(|f| f.n).collect::<Vec<_>>(), vec![6, 7]);
        assert_eq!(select_window(&f, None, Some(99)).unwrap().len(), 5);
        assert!(select_window(&f, Some(4), None).unwrap_err().contains("no frame 4"));
        assert!(select_window(&[], None, None).is_err());
    }

    #[test]
    fn a_gap_inside_the_window_is_an_error_but_one_outside_is_not() {
        let f = files(&[0, 1, 2, 5, 6]);
        assert!(select_window(&f, None, None).unwrap_err().contains("not consecutive"));
        assert_eq!(select_window(&f, None, Some(3)).unwrap().len(), 3);
        assert_eq!(select_window(&f, Some(5), None).unwrap().len(), 2);
    }

    fn yielded(frame: usize, ids: &[i64], boxes_xywh: &[f32], masks: Vec<Bitmask>) -> YieldedFrame {
        YieldedFrame { frame, obj_ids: ids.to_vec(), probs: ids.iter().map(|&i| 0.5 + i as f32 / 100.0).collect(), boxes_xywh: boxes_xywh.to_vec(), masks }
    }

    #[test]
    fn boxes_become_xyxy_and_degenerate_ones_are_widened_and_counted() {
        let m = || Bitmask::from_fn(4 * 8, |_| false);
        // 8 wide, 4 high: id 1 normal, id 2 zero width, id 3 zero height.
        let y = yielded(0, &[1, 2, 3], &[0.25, 0.5, 0.25, 0.25, 0.5, 0.0, 0.0, 0.25, 0.0, 0.5, 0.5, 0.0], vec![m(), m(), m()]);
        let (o, widened) = objects_of(&y, 4, 8).unwrap();
        assert_eq!(o[0].box_xyxy_norm, [0.25, 0.5, 0.5, 0.75]);
        assert_eq!(o[1].box_xyxy_norm, [0.5, 0.0, 0.625, 0.25], "one pixel wide = 1/8");
        assert_eq!(o[2].box_xyxy_norm, [0.0, 0.5, 0.5, 0.75], "one pixel high = 1/4");
        assert_eq!(widened, 2);
        assert_eq!(o[1].score, 0.52);
    }

    #[test]
    fn inconsistent_or_non_finite_frames_are_errors() {
        let m = || Bitmask::from_fn(4, |_| false);
        assert!(objects_of(&yielded(0, &[1], &[0.1, 0.1], vec![m()]), 2, 2).is_err());
        assert!(objects_of(&yielded(0, &[1], &[0.1, 0.1, f32::NAN, 0.1], vec![m()]), 2, 2).is_err());
        assert!(objects_of(&yielded(0, &[1, 2], &[0.1, 0.1, 0.1, 0.1], vec![m()]), 2, 2).is_err());
    }

    #[test]
    fn the_predictions_file_is_what_eval_visem_reads() {
        let frames = vec![
            FrameResult { n: 5, objects: vec![TrackedObject { id: 1, score: 0.9, box_xyxy_norm: [0.1, 0.2, 0.3, 0.4] }, TrackedObject { id: 7, score: 0.6, box_xyxy_norm: [0.5, 0.5, 0.75, 0.625] }] },
            FrameResult { n: 6, objects: vec![] },
            FrameResult { n: 7, objects: vec![TrackedObject { id: 1, score: 0.9, box_xyxy_norm: [0.125, 0.25, 0.3, 0.4] }] },
        ];
        let text = predictions_json("22", 640, 480, &frames).unwrap();
        let p = parse_predictions(&text, "22").expect("eval-visem's reader accepts it");
        assert_eq!(p.frames.keys().copied().collect::<Vec<_>>(), vec![5, 6, 7]);
        assert_eq!(p.frames[&5].len(), 2);
        assert_eq!(p.frames[&5][1].0, "7");
        assert!(p.frames[&6].is_empty());
        let b = p.frames[&7][0].1;
        assert_eq!((b.x, b.y), (0.125, 0.25));
        assert!((b.w - 0.175).abs() < 1e-6 && (b.h - 0.15).abs() < 1e-6);
        assert!(parse_predictions(&text, "23").is_err(), "the video id is checked");
        // No frames at all is still a valid file.
        assert!(parse_predictions(&predictions_json("22", 640, 480, &[]).unwrap(), "22").unwrap().frames.is_empty());
        assert!(predictions_json("22", 640, 480, &[FrameResult { n: 0, objects: vec![TrackedObject { id: 1, score: f32::NAN, box_xyxy_norm: [0.0; 4] }] }]).is_err());
    }

    #[test]
    fn the_collector_writes_masks_and_overlays_and_numbers_frames_from_the_start() {
        let d = temp("collect");
        let (h, w) = (4usize, 6usize);
        let mut c = Collector::new(&d, "9", 100, h, w, Some(2)).unwrap();
        let image = Rgb8::new(w, h, vec![50; w * h * 3]);
        for i in 0..3 {
            c.keep_image(i, &image);
        }
        assert_eq!(c.images.len(), 2, "frames 100 and 102 are multiples of 2");
        let mask = Bitmask::from_fn(h * w, |p| p % w < 2); // the two left columns
        c.accept(yielded(0, &[3], &[0.0, 0.0, 0.25, 0.75], vec![mask])).unwrap();
        c.accept(yielded(1, &[], &[], vec![])).unwrap();
        assert!(d.join("masks/9_frame_100.json").is_file() && d.join("masks/9_frame_101.json").is_file());
        assert!(d.join("overlay/9_frame_100.png").is_file());
        assert!(!d.join("overlay/9_frame_101.png").exists(), "an odd frame has no overlay");
        let m = ojas_io::parse_json(&std::fs::read_to_string(d.join("masks/9_frame_100.json")).unwrap()).unwrap();
        assert_eq!(m.get_u64("frame").unwrap(), 100);
        let obj = &m.get_array("objects").unwrap()[0];
        assert_eq!(obj.get_u64("id").unwrap(), 3);
        assert_eq!(obj.get_u64("area").unwrap(), 8);
        // Column-major: two full columns of 4 set, then 16 clear.
        let counts: Vec<u64> = obj.get_array("counts").unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(counts, vec![0, 8, 16]);
        assert_eq!(c.results.iter().map(|r| r.n).collect::<Vec<_>>(), vec![100, 101]);
        assert_eq!(c.ids.iter().copied().collect::<Vec<_>>(), vec![3]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_collector_prediction_file_scores_one_against_matching_ground_truth() {
        // Ground truth with one track; predictions made through the collector's own types.
        let d = temp("e2e");
        let labels = d.join("gt/5/labels_ftid");
        std::fs::create_dir_all(&labels).unwrap();
        let mut frames = Vec::new();
        for t in 0..6u32 {
            let x = 0.1 + 0.05 * f64::from(t);
            std::fs::write(labels.join(format!("5_frame_{t}_with_ftid.txt")), format!("a 0 {} 0.5 0.1 0.2\n", x + 0.05)).unwrap();
            frames.push(FrameResult { n: t, objects: vec![TrackedObject { id: 42, score: 0.9, box_xyxy_norm: [x as f32, 0.4, (x + 0.1) as f32, 0.6] }] });
        }
        std::fs::create_dir_all(d.join("pred")).unwrap();
        std::fs::write(d.join("pred/5.json"), predictions_json("5", 640, 480, &frames).unwrap()).unwrap();
        let e = evaluate_window(&d.join("pred"), &["5".to_string()], &d.join("gt"), Window { start: Some(0), max_frames: Some(6) }).unwrap();
        let hota = e.videos[0].hota.summary().hota;
        assert!(hota > 0.999, "HOTA {hota}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn bad_command_lines_fail_before_any_model_work() {
        let usage = |a: &[&str]| matches!(run(&args(a)), Err(CliError::Usage(_)));
        assert!(usage(&[]), "no input");
        assert!(usage(&["in", "x"]), "two positionals");
        assert!(usage(&["in"]), "--prompt is required");
        assert!(usage(&["in", "--prompt", " "]), "blank prompt");
        assert!(usage(&["in", "--prompt", "p", "--backend", "cuda"]));
        assert!(usage(&["in", "--prompt", "p", "--max-frames", "0"]));
        assert!(usage(&["in", "--prompt", "p", "--max-frames", "-3"]));
        assert!(usage(&["in", "--prompt", "p", "--start", "x"]));
        assert!(usage(&["in", "--prompt", "p", "--overlay", "0"]));
        assert!(usage(&["in", "--prompt", "p", "--threads", "0"]));
        assert!(usage(&["in", "--prompt", "p", "--video", "a/b"]));
        assert!(usage(&["in", "--prompt", "p", "--bogus", "1"]));
    }

    #[test]
    fn a_bad_input_fails_before_the_weights_are_looked_for() {
        // Weights are not given and may not exist: the input error must come first.
        let missing = run(&args(&["/nonexistent/spiketrack-input", "--prompt", "p"]));
        assert!(matches!(&missing, Err(CliError::Failed(m)) if m.contains("does not exist")), "{missing:?}");
        let d = temp("notframes");
        touch(&d, "readme.txt");
        let empty = run(&args(&[d.to_str().unwrap(), "--prompt", "p"]));
        assert!(matches!(&empty, Err(CliError::Failed(m)) if m.contains("no <video>_frame_<n>")), "{empty:?}");
        let file = d.join("readme.txt");
        let notvideo = run(&args(&[file.to_str().unwrap(), "--prompt", "p"]));
        assert!(matches!(&notvideo, Err(CliError::Failed(m)) if m.contains("not a folder of frames")), "{notvideo:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn help_mentions_the_mp4_preprocessing_gap() {
        assert!(USAGE.contains("GAP-MP4-PREPROCESSING") && USAGE.contains("INTER_CUBIC"));
        assert!(run(&args(&["--help"])).is_ok());
    }

    /// VISEM video 22, frames 0..8, on Metal: the output parses with eval-visem's reader,
    /// frames and ids are what a run of 8 frames must give, and a track keeps its id while it
    /// moves a little. Needs the gated checkpoint, the tokenizer files and `data/visem`.
    ///
    /// `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... tools/locked.sh track-e2e cargo test --release -p spiketrack track_cmd::tests::visem_22 -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the gated checkpoint, tokenizer files, data/visem and a Metal GPU"]
    fn visem_22_frames_0_to_7_on_metal() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let visem = root.join("data/visem/VISEM_Tracking_Train_v4/Train");
        let out = temp("visem22");
        let a = args(&[
            visem.join("22/images").to_str().unwrap(),
            "--prompt",
            "cell",
            "--out",
            out.to_str().unwrap(),
            "--backend",
            "metal",
            "--start",
            "0",
            "--max-frames",
            "8",
            "--overlay",
            "4",
        ]);
        run(&a).expect("track");
        let text = std::fs::read_to_string(out.join("22.json")).expect("22.json");
        let p = parse_predictions(&text, "22").expect("eval-visem reads the file");
        assert_eq!(p.frames.keys().copied().collect::<Vec<_>>(), (0..8).collect::<Vec<u32>>());
        for n in 0..8u32 {
            assert!(out.join(format!("masks/22_frame_{n}.json")).is_file(), "masks for frame {n}");
        }
        assert!(out.join("overlay/22_frame_0.png").is_file() && out.join("overlay/22_frame_4.png").is_file());
        let run_json = ojas_io::parse_json(&std::fs::read_to_string(out.join("22.run.json")).unwrap()).unwrap();
        assert_eq!(run_json.get_str("backend").unwrap(), "metal");
        assert_eq!(run_json.get_u64("frames").unwrap(), 8);

        // Stability: print the id sets, then require that the ids of frame 0 are carried to
        // frame 7 and that a track's box centre moves less than 0.1 of the frame per frame.
        let ids = |n: u32| -> BTreeSet<String> { p.frames[&n].iter().map(|(id, _)| id.clone()).collect() };
        for n in 0..8 {
            println!("frame {n}: {} objects", p.frames[&n].len());
        }
        assert!(!ids(0).is_empty(), "no object in frame 0: the prompt found nothing");
        let kept = ids(0).intersection(&ids(7)).count();
        println!("ids in frame 0: {}, still present in frame 7: {kept}", ids(0).len());
        assert!(kept * 10 >= ids(0).len() * 9, "only {kept} of {} ids of frame 0 reach frame 7", ids(0).len());
        let centre = |b: &st_eval::Xywh| (b.x + b.w / 2.0, b.y + b.h / 2.0);
        for n in 0..7u32 {
            for (id, b0) in &p.frames[&n] {
                if let Some((_, b1)) = p.frames[&(n + 1)].iter().find(|(i, _)| i == id) {
                    let (c0, c1) = (centre(b0), centre(b1));
                    assert!((c0.0 - c1.0).abs() < 0.1 && (c0.1 - c1.1).abs() < 0.1, "id {id} jumps between frames {n} and {}: {c0:?} -> {c1:?}", n + 1);
                }
            }
        }
        let visem_str = visem.to_str().unwrap();
        let e = evaluate_window(&out, &["22".to_string()], Path::new(visem_str), Window { start: Some(0), max_frames: Some(8) }).expect("eval-visem");
        println!("HOTA on frames 0..8 of video 22: {:.4}", e.videos[0].hota.summary().hota);
        let _ = std::fs::remove_dir_all(&out);
    }
}
