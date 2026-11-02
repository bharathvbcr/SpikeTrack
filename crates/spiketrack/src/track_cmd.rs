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
//! Every frame is resized to `--input-size` x `--input-size` (1008, the checkpoint's native
//! size, unless given) with `st_video::resize_bilinear_pil` (Pillow-exact
//! bilinear) and normalised **in float16 as upstream's loader does** (`io_utils.py`: the
//! frame is stored as `torch.float16`; `/255`, `- 0.5` and `/ 0.5` each round to f16, see
//! [`upstream_input`]), then widened to f32 for the model, as upstream's `.float()` does.
//! (`spiketrack image` keeps f32: upstream's single-image path is torchvision in f32.)
//!
//! * **Frame folder**: upstream's JPEG-folder path (`load_video_frames`) resizes with Pillow
//!   bilinear, the same resize, and the JPEG decode is byte-identical to Pillow's
//!   (`st_video::jpeg`, `st-video/tests/decode.rs`), so the model input is upstream's bit for
//!   bit. Frames in other formats, or JPEG variants that decoder refuses, go through ImageIO;
//!   `<video>.run.json` counts the frames per decoder.
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
//!   timings, preprocessing note, frame precision, the `--lora` adapter or null).
//! * with `--overlay K`: `DIR/overlay/<video>_frame_<n>.png` for every frame whose number is
//!   a multiple of K, masks and boxes coloured by track id.
//! * with `--overlay-video FILE`: an H.264 `.mp4` (`st_video::VideoWriter`, AVFoundation) of
//!   every processed frame with the same overlay, rendered by the same `overlay::render` as
//!   the PNGs (so the same colours by id and the same boxes), in frame order, at `--overlay-fps`
//!   or else the input video's own rate or, for a folder of frames, [`FOLDER_OVERLAY_FPS`]
//!   (49: VISEM-Tracking's videos are 48 to 50 fps). It is independent of `--overlay K`: either,
//!   both or neither. Frames are rendered and encoded as the session releases them, and
//!   [`Collector::keep_image`] holds a decoded frame only until its release (15 frames late), so
//!   memory does not grow with the clip. The file must not exist and the frame size must be
//!   even (checked before the model loads); a run that fails drops the writer, which cancels the
//!   session and removes the partial file. `<video>.run.json` records `overlay_video` (path),
//!   `overlay_video_fps`, `overlay_video_fps_source` and `overlay_video_frames` (all `null`
//!   without the option).
//!
//! The session releases a frame only after 15 later ones (upstream's hot-start delay, which
//! decides which ids are real), so the first outputs appear 15 frames in; the run ends by
//! flushing the rest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Autocast, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::{MetalBackend, WaitCounts};
use st_coreml::{CoremlTrunk, Units};
use st_model::tracker::session::{Session, SessionConfig, YieldedFrame};
use st_model::tracker::vos::Vos;
use st_model::Geometry;
use st_model::{Backbone, Detector, DetectorWeights, Text, Tokenizer};
use st_video::encode;
use st_video::{
    decode_image_with_source, resize_bilinear_pil, Rgb8, VideoFile, VideoReader, VideoWriter,
};
use st_weights::Sam31Weights;

use crate::cli::{existing_path, Args, CliError};
use crate::image_cmd::BackendChoice;
use crate::json;
use crate::mask::Rle;
use crate::overlay::{self, Drawn};
use crate::png;
use crate::procinfo::{phys_footprint_bytes, vm_pageouts_swapins};
use crate::track_profile::{self, OpsLabels, ProfileMode, StageClock, StageSink};
use crate::weights::{self, lora_json};
use st_model::profile::Profiled;

/// `--max-objects` when not given. Upstream's builder caps tracking at 16, which bounds the dense
/// VISEM videos (27-46 sperm a frame): at 64 the held-out HOTA is 0.4868 against 0.4695
/// (bench/results/track/held300-lora600-sperm-detector-max64).
pub const DEFAULT_MAX_OBJECTS: usize = 64;

pub const USAGE: &str = "\
spiketrack track <input> --prompt <text> [--out DIR] [--backend metal|cpu] [--max-frames N]
                 [--start F] [--overlay K] [--overlay-video FILE] [--overlay-fps F]
                 [--video ID] [--boxes mask|detector] [--weights FILE]
                 [--lora FILE] [--model-dir DIR] [--threads N] [--trim-pool-every K]
                 [--trunk-precision f32|bf16-linears|coreml-fp32|coreml-fp16] [--max-objects N]
                 [--input-size N] [--profile stages|ops]

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
  --overlay-video FILE  also write every frame with its masks and boxes, coloured by track id as
                    in --overlay, as an H.264 .mp4 (the system encoder, AVFoundation; independent
                    of --overlay K). The file name must end in .mp4 and must not exist; the frame
                    size must be even; a run that fails leaves no partial file.
  --overlay-fps F   the video's frame rate, 1 to 240 (default: the input video's own rate; for a
                    folder of frames 49, VISEM-Tracking's: its videos are 48 to 50 fps)
  --video ID        the video id used in file names and \"video\" (default: from the frame
                    names, or the mp4's file name without extension)
  --boxes SOURCE    mask (default, upstream's out_boxes_xywh: the extent of the track's mask
                    before the non-overlap rule) or detector (not upstream: the box of the
                    detection this frame's association assigned to the track, the mask box
                    when none was; for datasets whose boxes are not mask extents, such as
                    VISEM-Tracking's sperm-head boxes). Masks are the same either way.
  --weights FILE    the converted checkpoint (default: $SPIKETRACK_WEIGHTS)
  --lora FILE       fold this detector LoRA adapter (spiketrack-lora-v1) into the weights
  --model-dir DIR   folder with the tokenizer's vocab.json and merges.txt (default: $SPIKETRACK_MODEL_DIR)
  --threads N       CPU worker threads for --backend cpu, 1 to 256 (default 8)
  --trim-pool-every K  diagnostic, --backend metal only: hand Ojas's cached free buffers back to
                    the device after every K-th frame (MetalBackend::trim_pool)
  --profile MODE    diagnostic, --backend metal only (not with bf16-linears): break each frame
                    down by stage (trunk, detector, each tracker state, memory, births, outputs,
                    decode, collector) into <out>/<video>.profile-<MODE>.md. stages: wall time
                    per stage, the device synced as each begins (frame times grow a little);
                    ops: every backend op timed and synced (much slower). Frame 0 is apart.
  --max-objects N   most objects tracked at once, 1 to 128 (default 64; upstream's builder uses
                    16 and its class allows 128). Above 16 new objects fill extra tracker states
                    16 at a time, which costs one more tracker step per state per frame and is not
                    parity-checked against upstream; --max-objects 16 is upstream's setting.
  --input-size N    the square side every frame is resized to before the model, a multiple of 336
                    (the ViT's 14-pixel patches in 24-token windows): 1008 (default, the
                    checkpoint's native size), 672 or 336. Smaller is much faster, with an
                    accuracy cost on small objects: at 672 the held-out VISEM HOTA falls below
                    0.47 (bench/results/track/held300-lora600-sperm-detector-672-max64). No weight
                    changes: the position embedding is tiled and the global blocks' RoPE
                    regenerated for the grid (see st_model::backbone::vit_rope_tables). The CoreML
                    trunks are exported at 1008, so any other size needs --trunk-precision f32 or
                    bf16-linears. Recorded in run.json.
  --trunk-precision P  what runs the ViT trunk (default: coreml-fp32 on --backend metal, f32 on
                    --backend cpu). coreml-fp32 / coreml-fp16: the trunk runs as a CoreML ML
                    Program (CPU and GPU) from <model-dir>/coreml/sam3.1_trunk_<fp32|fp16>.mlpackage,
                    exported by tools/baselines/coreml_trunk.py (a missing package is an error,
                    never a fallback); the necks, detector and tracker stay on --backend.
                    coreml-fp32 matches upstream's trunk to relative L2 ~1e-5, coreml-fp16 to ~1e-2
                    (bench/results/coreml-trunk). The first run of a package compiles it beside
                    itself, and loading it adds ~3-13 s to the model load. f32: the Ojas trunk,
                    upstream's numerics. bf16-linears (an experiment: the trunk's linears in bf16
                    through an Ojas Autocast region, everything else f32; not upstream, see
                    Backbone::with_trunk_linears_bf16).

  Each frame's progress line ends with this process's physical footprint, the machine's pages
  paged out and swapped in during the frame and, on Metal, the frame's change in Ojas's wait
  counts (by cause) and dispatches and the device memory Metal reports (allocated after the
  frame, peak during it): Ojas telemetry, not a stable interface.

  Preprocessing: every frame is resized with Pillow-style bilinear and normalised in
  float16, as upstream's video loader stores frames (float16 -> f32 for the model;
  `image` keeps f32). That is what upstream does for a folder of JPEG frames. For mp4 input it is NOT what
  upstream does: upstream decodes mp4 with OpenCV and resizes with INTER_CUBIC, so mp4
  results do not match upstream's mp4 preprocessing (GAP-MP4-PREPROCESSING).";

/// `--input-size` when not given: the checkpoint's native size.
pub const DEFAULT_INPUT_SIZE: usize = st_model::IMAGE_SIZE;

/// The run.json `preprocessing` text for a `size` x `size` model input.
fn preprocessing_text(from_video: bool, size: usize) -> String {
    if from_video {
        PREPROCESSING_VIDEO.replace("{size}", &size.to_string())
    } else {
        PREPROCESSING_FOLDER.replace("{size}", &size.to_string())
    }
}

const PREPROCESSING_FOLDER: &str = "decode per \"decoders\" (pillow-exact-jpeg: byte-identical to Pillow 13 / libjpeg-turbo 3.2.0); Pillow-exact bilinear resize to {size}x{size}, as upstream's JPEG-folder path; /255, -0.5, /0.5 each rounded to float16 as upstream's loader stores frames (widened to f32 for the model)";
const PREPROCESSING_VIDEO: &str = "AVFoundation decode (32BGRA); Pillow-exact bilinear resize to {size}x{size}; float16 normalisation as upstream. NOT upstream's mp4 path (OpenCV decode, cv2.INTER_CUBIC): GAP-MP4-PREPROCESSING";
const BUDGET_BYTES: u64 = 28 << 30;
/// Metal's memory cap for a track run: the bytes allocated (or freed) since the last waited
/// commit before the device waits for one. ojas opens at min(1 GiB, budget / 4); at that default
/// the cap forced 231 of 387 waits per 4-frame profile pass. 4 GiB was the knee of the sweep:
/// ~0.87x frame time for +3 GiB peak footprint (11.0 -> 14.0 GiB;
/// bench/results/speed-baseline/memcap-footprint). It changes when commits wait, not what runs.
const METAL_MEM_CAP: u64 = 4 << 30;

/// The Metal backend a track run uses: the budget above, the memory cap raised to
/// [`METAL_MEM_CAP`]. Returns the cap the device reports, for run.json.
fn open_metal() -> Result<(MetalBackend, u64), CliError> {
    let be = MetalBackend::new(Budget::new(BUDGET_BYTES)).map_err(CliError::from_err)?;
    be.set_mem_cap(METAL_MEM_CAP).map_err(CliError::from_err)?;
    let cap = be.mem_cap().map_err(CliError::from_err)?;
    Ok((be, cap))
}

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
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        if !IMAGE_EXTENSIONS.contains(&ext.as_str()) || !path.is_file() {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let (prefix, n) = parse_stem(stem).ok_or_else(|| {
            format!(
                "{}: an image whose name is not <video>_frame_<n>.{ext}",
                path.display()
            )
        })?;
        match &video {
            None => video = Some(prefix.to_string()),
            Some(v) if v == prefix => {}
            Some(v) => {
                return Err(format!(
                    "{}: frames of two videos ({v:?} and {prefix:?})",
                    dir.display()
                ))
            }
        }
        if let Some(prev) = frames.insert(n, path.clone()) {
            return Err(format!(
                "frame {n} appears twice: {} and {}",
                prev.display(),
                path.display()
            ));
        }
    }
    let video = video.ok_or_else(|| format!("{}: no <video>_frame_<n> images", dir.display()))?;
    Ok(Folder {
        video,
        frames: frames
            .into_iter()
            .map(|(n, path)| FrameFile { n, path })
            .collect(),
    })
}

/// The frames from number `start` (default: the first), at most `max` of them; they must be
/// consecutive numbers.
pub fn select_window(
    frames: &[FrameFile],
    start: Option<u32>,
    max: Option<usize>,
) -> Result<&[FrameFile], String> {
    let first = frames.first().ok_or("the folder has no frames")?.n;
    let start = start.unwrap_or(first);
    let at = frames.iter().position(|f| f.n == start).ok_or_else(|| {
        format!(
            "no frame {start} in the folder (frames {first}..{})",
            frames[frames.len() - 1].n
        )
    })?;
    let rest = &frames[at..];
    let take = max.map_or(rest.len(), |m| m.min(rest.len()));
    let window = &rest[..take];
    for pair in window.windows(2) {
        if pair[1].n != pair[0].n + 1 {
            return Err(format!(
                "frames {} and {} are not consecutive (the tracker needs every frame)",
                pair[0].n, pair[1].n
            ));
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
    /// The input video's frame rate (`VideoInfo::fps`, frames over duration), `None` for a folder of frames.
    pub fps: Option<f64>,
    /// Frames delivered so far, by decoder (`pillow-exact-jpeg`, `imageio`, `avfoundation`).
    pub decoders: BTreeMap<&'static str, usize>,
    produced: usize,
    /// A folder's first frame, decoded by `open` to learn the size, and its decoder.
    pending: Option<(Rgb8, &'static str)>,
}

fn is_video_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("mp4" | "mov" | "m4v")
    )
}

impl FrameStream {
    pub fn open(
        input: &Path,
        start: Option<u32>,
        max: Option<usize>,
        video_override: Option<&str>,
    ) -> Result<FrameStream, CliError> {
        if input.is_dir() {
            let folder = discover_frames(input).map_err(CliError::failed)?;
            let files = select_window(&folder.frames, start, max)
                .map_err(CliError::failed)?
                .to_vec();
            let (first, decoder) =
                decode_image_with_source(&files[0].path).map_err(CliError::from_err)?;
            Ok(FrameStream {
                decoders: BTreeMap::new(),
                first_n: files[0].n,
                len: files.len(),
                height: first.height,
                width: first.width,
                video: video_override.map_or(folder.video, str::to_string),
                kind: Kind::Folder { files },
                from_video: false,
                fps: None,
                produced: 0,
                pending: Some((first, decoder.name())),
            })
        } else if input.is_file() && is_video_file(input) {
            let mut reader = VideoReader::open(input).map_err(CliError::from_err)?;
            let info = reader.info().clone();
            let start = start.unwrap_or(0);
            if start as usize >= info.frame_count {
                return Err(CliError::failed(format!(
                    "--start {start} is past the video's {} frames",
                    info.frame_count
                )));
            }
            for skipped in 0..start as usize {
                match reader.next_frame().map_err(CliError::from_err)? {
                    Some(_) => {}
                    None => {
                        return Err(CliError::failed(format!(
                        "the video ended after {skipped} frames while skipping to --start {start}"
                    )))
                    }
                }
            }
            let len = max.map_or(info.frame_count - start as usize, |m| {
                m.min(info.frame_count - start as usize)
            });
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video")
                .to_string();
            Ok(FrameStream {
                first_n: start,
                len,
                height: info.height,
                width: info.width,
                video: video_override.map_or(stem, str::to_string),
                kind: Kind::Video { reader },
                from_video: true,
                fps: Some(info.fps),
                decoders: BTreeMap::new(),
                produced: 0,
                pending: None,
            })
        } else if input.exists() {
            Err(CliError::failed(format!(
                "{}: not a folder of frames or an .mp4/.mov/.m4v file",
                input.display()
            )))
        } else {
            Err(CliError::failed(format!(
                "{}: does not exist",
                input.display()
            )))
        }
    }

    /// The next frame and its number, or `None` after `len` frames.
    pub fn next_frame(&mut self) -> Result<Option<(u32, Rgb8)>, CliError> {
        if self.produced >= self.len {
            return Ok(None);
        }
        let n = self.first_n + self.produced as u32;
        let img = match &mut self.kind {
            Kind::Folder { files } => {
                let (img, decoder) = match self.pending.take() {
                    Some(first) => first,
                    None => decode_image_with_source(&files[self.produced].path)
                        .map(|(img, d)| (img, d.name()))
                        .map_err(CliError::from_err)?,
                };
                *self.decoders.entry(decoder).or_insert(0) += 1;
                img
            }
            Kind::Video { reader } => match reader.next_frame().map_err(CliError::from_err)? {
                Some((i, _, rgb)) if i == n as usize => {
                    *self.decoders.entry("avfoundation").or_insert(0) += 1;
                    rgb
                }
                Some((i, _, _)) => {
                    return Err(CliError::failed(format!(
                        "the video returned frame {i} where {n} was expected"
                    )))
                }
                None => {
                    return Err(CliError::failed(format!(
                        "the video ended at frame {n}, before the {} frames asked for",
                        self.len
                    )))
                }
            },
        };
        if (img.height, img.width) != (self.height, self.width) {
            return Err(CliError::failed(format!(
                "frame {n} is {}x{} but the video is {}x{}",
                img.width, img.height, self.width, self.height
            )));
        }
        self.produced += 1;
        Ok(Some((n, img)))
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// Where a reported box comes from (`--boxes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxSource {
    /// Upstream's `out_boxes_xywh`: the mask's extent.
    Mask,
    /// Not upstream: the assigned detection's box, else the mask's extent.
    Detector,
}

impl BoxSource {
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s {
            "mask" => Ok(Self::Mask),
            "detector" => Ok(Self::Detector),
            other => Err(CliError::usage(format!(
                "--boxes: {other:?} is not a box source (choices: mask, detector)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mask => "mask",
            Self::Detector => "detector",
        }
    }
}

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

/// A released frame's objects with boxes as normalised xyxy from `source`, how many boxes were
/// widened to one pixel (a mask a single pixel wide or high, or a degenerate detection box), and
/// how many came from a detection. Detection boxes are clipped to the frame.
pub fn objects_of(
    y: &YieldedFrame,
    height: usize,
    width: usize,
    source: BoxSource,
) -> Result<(Vec<TrackedObject>, usize, usize), String> {
    let n = y.obj_ids.len();
    if y.probs.len() != n
        || y.boxes_xywh.len() != n * 4
        || y.masks.len() != n
        || y.det_boxes_xyxy.len() != n
    {
        return Err(format!(
            "frame {}: {n} ids, {} scores, {} box values, {} masks, {} detection boxes",
            y.frame,
            y.probs.len(),
            y.boxes_xywh.len(),
            y.masks.len(),
            y.det_boxes_xyxy.len()
        ));
    }
    let (mut widened, mut from_detector) = (0, 0);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let b = &y.boxes_xywh[k * 4..k * 4 + 4];
        if !b.iter().all(|v| v.is_finite())
            || !y.probs[k].is_finite()
            || y.det_boxes_xyxy[k].is_some_and(|d| !d.iter().all(|v| v.is_finite()))
        {
            return Err(format!(
                "frame {}: id {} has a non-finite box or score",
                y.frame, y.obj_ids[k]
            ));
        }
        let ([x0, y0], [mut x1, mut y1]) = match (source, y.det_boxes_xyxy[k]) {
            (BoxSource::Detector, Some(d)) => {
                from_detector += 1;
                let c = |v: f32| v.clamp(0.0, 1.0);
                ([c(d[0]), c(d[1])], [c(d[2]), c(d[3])])
            }
            _ => ([b[0], b[1]], [b[0] + b[2], b[1] + b[3]]),
        };
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
        out.push(TrackedObject {
            id: y.obj_ids[k],
            score: y.probs[k],
            box_xyxy_norm: [x0, y0, x1, y1],
        });
    }
    Ok((out, widened, from_detector))
}

/// `{"<decoder>": frames, ...}` for `<video>.run.json`.
pub fn decoders_json(d: &BTreeMap<&'static str, usize>) -> String {
    let parts: Vec<String> = d
        .iter()
        .map(|(k, v)| format!("{}: {v}", json::string(k)))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// `DIR/<video>.json`: the `spiketrack eval-visem` prediction schema.
pub fn predictions_json(
    video: &str,
    width: usize,
    height: usize,
    frames: &[FrameResult],
) -> Result<String, String> {
    let mut out = format!(
        "{{\n  \"video\": {},\n  \"width\": {width},\n  \"height\": {height},\n  \"frames\": [",
        json::string(video)
    );
    for (i, f) in frames.iter().enumerate() {
        out.push_str(&format!(
            "{}\n    {{\"frame\": {}, \"objects\": [",
            if i == 0 { "" } else { "," },
            f.n
        ));
        for (k, o) in f.objects.iter().enumerate() {
            let b: Result<Vec<String>, String> = o
                .box_xyxy_norm
                .iter()
                .map(|&v| json::number32(v, "box"))
                .collect();
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
    out.push_str(if frames.is_empty() {
        "]\n}\n"
    } else {
        "\n  ]\n}\n"
    });
    Ok(out)
}

/// `DIR/masks/<video>_frame_<n>.json`.
pub fn masks_json(n: u32, height: usize, width: usize, items: &[(i64, Rle)]) -> String {
    let objs: Vec<String> = items
        .iter()
        .map(|(id, r)| {
            format!(
                "{{\"id\": {id}, \"area\": {}, \"counts\": [{}]}}",
                r.area,
                r.counts_json()
            )
        })
        .collect();
    format!(
        "{{\"frame\": {n}, \"size\": [{height}, {width}], \"objects\": [{}]}}\n",
        objs.join(", ")
    )
}

/// The `--overlay-video` writer, and the index of the frame it expects next (frames are released
/// in order and must be written in order).
struct OverlayVideo {
    writer: VideoWriter,
    next: usize,
}

/// Receives released frames: keeps their results and writes the per-frame files.
pub struct Collector {
    out: PathBuf,
    video: String,
    first_n: u32,
    height: usize,
    width: usize,
    overlay_every: Option<u32>,
    overlay_video: Option<OverlayVideo>,
    boxes: BoxSource,
    /// The decoded frames still awaiting release (the session releases a frame 15 frames late)
    /// that an overlay will want: every one while a video is being written, else the `--overlay K`
    /// frames. `accept` removes a frame as it is rendered, so this holds at most the lag's worth.
    images: BTreeMap<usize, Rgb8>,
    pub results: Vec<FrameResult>,
    pub widened: usize,
    /// Boxes reported (all frames), and how many of them came from a detection.
    pub reported: usize,
    pub from_detector: usize,
    pub ids: BTreeSet<i64>,
}

impl Collector {
    pub fn new(
        out: &Path,
        video: &str,
        first_n: u32,
        height: usize,
        width: usize,
        overlay_every: Option<u32>,
        boxes: BoxSource,
    ) -> Result<Collector, CliError> {
        std::fs::create_dir_all(out.join("masks"))
            .map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
        if overlay_every.is_some() {
            std::fs::create_dir_all(out.join("overlay"))
                .map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
        }
        Ok(Collector {
            out: out.to_path_buf(),
            video: video.to_string(),
            first_n,
            height,
            width,
            overlay_every,
            overlay_video: None,
            boxes,
            images: BTreeMap::new(),
            results: Vec::new(),
            widened: 0,
            reported: 0,
            from_detector: 0,
            ids: BTreeSet::new(),
        })
    }

    /// Write the overlay of every released frame to `writer`, in frame order (`--overlay-video`),
    /// whatever `--overlay K` says. It must be a writer for this collector's frame size, with no
    /// frame written yet, and be set before the first [`Collector::keep_image`].
    pub fn set_overlay_video(&mut self, writer: VideoWriter) -> Result<(), CliError> {
        if writer.size() != (self.width, self.height)
            || writer.frames() != 0
            || !self.images.is_empty()
        {
            return Err(CliError::failed(format!(
                "the overlay video is {}x{} with {} frame(s) written; the frames are {}x{}",
                writer.size().0,
                writer.size().1,
                writer.frames(),
                self.width,
                self.height
            )));
        }
        self.overlay_video = Some(OverlayVideo { writer, next: 0 });
        Ok(())
    }

    /// Complete the overlay video, if one is being written, and check that it holds `expected` frames.
    pub fn finish_overlay_video(&mut self, expected: usize) -> Result<Option<VideoFile>, CliError> {
        let Some(v) = self.overlay_video.take() else {
            return Ok(None);
        };
        let file = v
            .writer
            .finish()
            .map_err(|e| CliError::failed(format!("--overlay-video: {e}")))?;
        if file.frames != expected {
            return Err(CliError::failed(format!(
                "--overlay-video: {} frame(s) written for {expected} processed",
                file.frames
            )));
        }
        Ok(Some(file))
    }

    /// Whether frame `index` (0-based within the run) gets a PNG for `--overlay K`.
    fn wants_png(&self, index: usize) -> bool {
        self.overlay_every
            .is_some_and(|k| (self.first_n + index as u32) % k == 0)
    }

    /// Frame `index` (0-based within the run) was pushed: keep it if an overlay will want it.
    pub fn keep_image(&mut self, index: usize, image: &Rgb8) {
        if self.overlay_video.is_some() || self.wants_png(index) {
            self.images.insert(index, image.clone());
        }
    }

    pub fn accept(&mut self, y: YieldedFrame) -> Result<(), CliError> {
        let (h, w) = (self.height, self.width);
        let n = self.first_n + y.frame as u32;
        let (objects, widened, from_detector) =
            objects_of(&y, h, w, self.boxes).map_err(CliError::failed)?;
        self.widened += widened;
        self.reported += objects.len();
        self.from_detector += from_detector;
        self.ids.extend(objects.iter().map(|o| o.id));
        let rles: Vec<(i64, Rle)> = y
            .obj_ids
            .iter()
            .zip(&y.masks)
            .map(|(&id, m)| (id, Rle::from_fn(h, w, |r, c| m.get(r * w + c))))
            .collect();
        let path = self
            .out
            .join("masks")
            .join(format!("{}_frame_{n}.json", self.video));
        std::fs::write(&path, masks_json(n, h, w, &rles))
            .map_err(|e| CliError::failed(format!("{}: {e}", path.display())))?;
        let png_wanted = self.wants_png(y.frame);
        if png_wanted || self.overlay_video.is_some() {
            let image = self.images.remove(&y.frame).ok_or_else(|| {
                CliError::failed(format!(
                    "frame {n} was released but its image was not kept for the overlay"
                ))
            })?;
            let bits: Vec<Vec<bool>> = rles.iter().map(|(_, r)| r.decode()).collect();
            let drawn: Vec<Drawn<'_>> = objects
                .iter()
                .zip(&bits)
                .map(|(o, m)| {
                    let b = o.box_xyxy_norm;
                    let (fw, fh) = (w as f64, h as f64);
                    Drawn {
                        mask: m,
                        box_px: [
                            f64::from(b[0]) * fw,
                            f64::from(b[1]) * fh,
                            f64::from(b[2]) * fw,
                            f64::from(b[3]) * fh,
                        ],
                        colour: o.id.rem_euclid(1 << 20) as usize,
                        label: u64::try_from(o.id).ok(),
                    }
                })
                .collect();
            // One rendering feeds both outputs, so the video shows exactly the PNG's pixels.
            let rendered = overlay::render(&image, &drawn);
            if let Some(v) = &mut self.overlay_video {
                if y.frame != v.next {
                    return Err(CliError::failed(format!("--overlay-video: frame {n} released out of order (the video is at its frame {})", v.next)));
                }
                v.writer
                    .push(&rendered)
                    .map_err(|e| CliError::failed(format!("--overlay-video, frame {n}: {e}")))?;
                v.next += 1;
            }
            if png_wanted {
                let bytes = png::encode_rgb(w, h, &rendered.data).map_err(CliError::from_err)?;
                let path = self
                    .out
                    .join("overlay")
                    .join(format!("{}_frame_{n}.png", self.video));
                std::fs::write(&path, bytes)
                    .map_err(|e| CliError::failed(format!("{}: {e}", path.display())))?;
            }
        }
        self.results.push(FrameResult { n, objects });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// The frame rate of an overlay video made from a folder of frames, which records none:
/// VISEM-Tracking's 20 videos are 49 fps (17 of them), 48 (2) and 50 (1) by `ffprobe`'s
/// `avg_frame_rate` of the mp4s in `data/visem`, and the dataset paper (arXiv 2212.02842) says
/// 45 to 50. Pass `--overlay-fps` for a folder recorded at another rate.
pub const FOLDER_OVERLAY_FPS: f64 = 49.0;

/// Where an overlay video's frame rate came from.
const FPS_FROM_OPTION: &str = "--overlay-fps";
const FPS_FROM_INPUT: &str = "input video";
const FPS_FROM_DEFAULT: &str = "default for a folder of frames (VISEM-Tracking: 49 fps)";

/// The overlay video's frame rate: `--overlay-fps`, else the input video's own, else
/// [`FOLDER_OVERLAY_FPS`], with the source of the number.
pub fn overlay_fps_choice(option: Option<f64>, input_video: Option<f64>) -> (f64, &'static str) {
    match (option, input_video) {
        (Some(f), _) => (f, FPS_FROM_OPTION),
        (None, Some(f)) => (f, FPS_FROM_INPUT),
        (None, None) => (FOLDER_OVERLAY_FPS, FPS_FROM_DEFAULT),
    }
}

/// `--overlay-video`, checked: the file to write and the frame rate to write it at.
#[derive(Debug)]
struct OverlayVideoSpec {
    path: PathBuf,
    fps: f64,
    fps_source: &'static str,
}

/// What `<video>.run.json` records of the overlay video.
struct OverlayVideoRun {
    /// The frame rate as stored (the request rounded to 0.001).
    fps: f64,
    fps_source: &'static str,
    file: Option<VideoFile>,
}

/// Check `--overlay-video` against the frames (even size, a frame rate the encoder takes, a free
/// `.mp4` path) before any model work, so a refusal costs a second and not a model load.
fn plan_overlay_video(
    path: PathBuf,
    fps_option: Option<f64>,
    stream: &FrameStream,
) -> Result<OverlayVideoSpec, CliError> {
    let (fps, fps_source) = overlay_fps_choice(fps_option, stream.fps);
    let refuse = |e: encode::EncodeError| {
        let hint = match (&e, fps_source) {
            (encode::EncodeError::InvalidFps { .. }, FPS_FROM_INPUT) => {
                " (the input video's own rate; pass --overlay-fps)"
            }
            (encode::EncodeError::InvalidSize { .. }, _) => {
                " (the video is written at the frames' size, which H.264 4:2:0 needs even)"
            }
            _ => "",
        };
        CliError::failed(format!("--overlay-video: {e}{hint}"))
    };
    encode::check_format(stream.width, stream.height, fps).map_err(refuse)?;
    encode::check_output(&path, false).map_err(refuse)?;
    Ok(OverlayVideoSpec {
        path,
        fps,
        fps_source,
    })
}

/// `overlay_video`, `overlay_video_fps`, `overlay_video_fps_source`, `overlay_video_frames` for
/// `<video>.run.json`, as JSON values (all `null` without `--overlay-video`).
fn overlay_video_json(run: Option<&OverlayVideoRun>) -> Result<[String; 4], String> {
    let Some(run) = run else {
        return Ok([
            "null".to_string(),
            "null".to_string(),
            "null".to_string(),
            "null".to_string(),
        ]);
    };
    let file = run
        .file
        .as_ref()
        .ok_or("the overlay video was not finished")?;
    Ok([
        json::string(&file.path.to_string_lossy()),
        json::number(run.fps, "overlay video fps")?,
        json::string(run.fps_source),
        file.frames.to_string(),
    ])
}

struct Options {
    prompt: String,
    out: PathBuf,
    overlay_every: Option<u32>,
    boxes: BoxSource,
    backend: BackendChoice,
    threads: usize,
    weights: PathBuf,
    model_dir: PathBuf,
    trim_pool_every: Option<u32>,
    /// `--trunk-precision`.
    trunk: TrunkPrecision,
    /// `--max-objects`.
    max_objects: usize,
    /// `--input-size`, validated: the model's token grid and every size derived from it.
    geometry: Geometry,
    /// `--profile`.
    profile: Option<ProfileMode>,
}

/// `--trunk-precision`: what computes the ViT trunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrunkPrecision {
    F32,
    Bf16Linears,
    /// The CoreML package of this precision ("fp32" or "fp16").
    Coreml(&'static str),
}

impl TrunkPrecision {
    fn parse(s: &str) -> Result<Self, CliError> {
        Ok(match s {
            "f32" => Self::F32,
            "bf16-linears" => Self::Bf16Linears,
            "coreml-fp32" => Self::Coreml("fp32"),
            "coreml-fp16" => Self::Coreml("fp16"),
            other => {
                return Err(CliError::usage(format!(
                "--trunk-precision: {other:?} is not f32, bf16-linears, coreml-fp32 or coreml-fp16"
            )))
            }
        })
    }

    /// What `--trunk-precision` defaults to. On Metal, the CoreML fp32 trunk: E10 measured it at
    /// 0.80x the Ojas trunk's frame, with the same id sets on all 1500 held-out VISEM frames and the
    /// same HOTA (bench/results/coreml-trunk). On CPU, the Ojas f32 trunk: that path is the
    /// upstream-parity oracle, and CoreML would run on the GPU anyway.
    fn default_for(backend: BackendChoice) -> Self {
        match backend {
            BackendChoice::Metal => Self::Coreml("fp32"),
            BackendChoice::Cpu => Self::F32,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16Linears => "bf16-linears",
            Self::Coreml("fp16") => "coreml-fp16",
            Self::Coreml(_) => "coreml-fp32",
        }
    }
}

/// `<model-dir>/coreml/sam3.1_trunk_<precision>.mlpackage`, where tools/baselines/coreml_trunk.py writes it.
fn coreml_package(model_dir: &Path, precision: &str) -> PathBuf {
    model_dir
        .join("coreml")
        .join(format!("sam3.1_trunk_{precision}.mlpackage"))
}

struct Timing {
    load_seconds: f64,
    frame_seconds: Vec<f64>,
    /// `Backbone::external_trunk`: `None` when the Ojas blocks computed the trunk.
    external_trunk: Option<String>,
}

const MIB: f64 = (1u64 << 20) as f64;

/// For every backend's frame line: this process's footprint, and the machine-wide pages paged
/// out and swapped in since the previous call (memory pressure; Metal's shared buffers are
/// pageable).
struct HostProbe {
    vm: Option<(u64, u64)>,
}

impl HostProbe {
    fn new() -> Self {
        Self {
            vm: vm_pageouts_swapins(),
        }
    }

    fn field(&mut self) -> String {
        let fp = match phys_footprint_bytes() {
            Some(b) => format!("footprint {:.0} MiB", b as f64 / MIB),
            None => "footprint unreadable".to_string(),
        };
        let now = vm_pageouts_swapins();
        let vm = match (self.vm, now) {
            (Some((po0, si0)), Some((po, si))) => format!(
                "pageouts +{}, swapins +{}",
                po.saturating_sub(po0),
                si.saturating_sub(si0)
            ),
            _ => "vm counters unreadable".to_string(),
        };
        self.vm = now;
        format!("{fp}, {vm}")
    }
}

/// Per-frame Metal telemetry (Ojas's doc-hidden counters): the change in each wait cause and in
/// dispatches since the previous call, and Metal's allocated bytes now and their peak since the
/// previous call (`memory_and_reset_peak`), then `pool` when `trim` says the pool was trimmed.
struct MetalProbe {
    waits: WaitCounts,
    dispatches: u64,
}

impl MetalProbe {
    fn new(be: &MetalBackend) -> Result<Self, CliError> {
        be.memory_and_reset_peak().map_err(CliError::from_err)?;
        Ok(Self {
            waits: be.wait_counts(),
            dispatches: be.dispatches(),
        })
    }

    fn frame(&mut self, be: &MetalBackend) -> Result<String, CliError> {
        let (w, d) = (be.wait_counts(), be.dispatches());
        let p = self.waits;
        let causes = [
            ("upload", w.upload - p.upload),
            ("read", w.read - p.read),
            ("sync", w.sync - p.sync),
            ("clip_norm", w.clip_norm - p.clip_norm),
            ("mem_cap", w.mem_cap - p.mem_cap),
            ("working_set", w.working_set - p.working_set),
            ("slab_full", w.slab_full - p.slab_full),
            ("recycle", w.recycle - p.recycle),
            ("freed", w.freed - p.freed),
        ];
        let nonzero: Vec<String> = causes
            .iter()
            .filter(|c| c.1 > 0)
            .map(|(k, v)| format!("{k} {v}"))
            .collect();
        let m = be.memory_and_reset_peak().map_err(CliError::from_err)?;
        let out = format!(
            "waits +{} ({}), dispatches +{}, metal allocated {:.0} MiB (frame peak {:.0} MiB)",
            w.total() - p.total(),
            if nonzero.is_empty() {
                "none".to_string()
            } else {
                nonzero.join(", ")
            },
            d - self.dispatches,
            m.allocated as f64 / MIB,
            m.peak_allocated as f64 / MIB
        );
        self.waits = w;
        self.dispatches = d;
        Ok(out)
    }
}

/// What a backend adds to each frame's progress line, and the optional pool trim.
trait FrameTelemetry<B> {
    fn after_frame(&mut self, be: &B, index: usize) -> Result<String, CliError>;
}

/// CPU: the host fields only.
impl<B: Backend> FrameTelemetry<B> for HostProbe {
    fn after_frame(&mut self, _: &B, _: usize) -> Result<String, CliError> {
        Ok(self.field())
    }
}

/// Metal telemetry through a clone of the backend the frames run on (`MetalBackend` clones share
/// one device link and its counters), so it reads the same device whether the frames run on the
/// plain backend or on an `Autocast` wrapper around it.
struct MetalTelemetry {
    metal: MetalBackend,
    probe: MetalProbe,
    host: HostProbe,
    trim_every: Option<u32>,
}

impl MetalTelemetry {
    fn new(metal: &MetalBackend, trim_every: Option<u32>) -> Result<Self, CliError> {
        Ok(Self {
            metal: metal.clone(),
            probe: MetalProbe::new(metal)?,
            host: HostProbe::new(),
            trim_every,
        })
    }
}

impl<B: Backend> FrameTelemetry<B> for MetalTelemetry {
    fn after_frame(&mut self, _: &B, index: usize) -> Result<String, CliError> {
        let be = &self.metal;
        let mut line = format!("{}, {}", self.probe.frame(be)?, self.host.field());
        if self
            .trim_every
            .is_some_and(|k| (index + 1) % k as usize == 0)
        {
            be.trim_pool().map_err(CliError::from_err)?;
            line.push_str(&format!(
                ", pool trimmed (metal allocated {:.0} MiB)",
                be.memory().map_err(CliError::from_err)?.allocated as f64 / MIB
            ));
        }
        Ok(line)
    }
}

/// `--input-size N` as the model geometry. The one check: a multiple of 336, since the grid
/// must be a whole number of 24-token windows (`Geometry::from_image_side` names the valid sizes).
fn parse_input_size(n: usize) -> Result<Geometry, CliError> {
    Geometry::from_image_side(n).map_err(|e| CliError::usage(format!("--input-size {n}: {e}")))
}

fn track_on<B: Backend>(
    be: &B,
    w: &Sam31Weights,
    stream: &mut FrameStream,
    opts: &Options,
    collector: &mut Collector,
    telemetry: &mut dyn FrameTelemetry<B>,
    profile: Option<&dyn StageSink>,
) -> Result<Timing, CliError> {
    let t_load = Instant::now();
    let backbone = match opts.trunk {
        TrunkPrecision::F32 => {
            Backbone::new_at(be, &w.vision, opts.geometry).map_err(CliError::from_err)?
        }
        TrunkPrecision::Bf16Linears => Backbone::new_at(be, &w.vision, opts.geometry)
            .and_then(|b| b.with_trunk_linears_bf16())
            .map_err(CliError::from_err)?,
        TrunkPrecision::Coreml(precision) => {
            let package = coreml_package(&opts.model_dir, precision);
            if !package.is_dir() {
                return Err(CliError::failed(format!(
                    "--trunk-precision {}: {} does not exist; export it with tools/baselines/coreml_trunk.py export --precision {precision}, or pass --trunk-precision f32 for the Ojas trunk",
                    opts.trunk.name(),
                    package.display()
                )));
            }
            let trunk =
                CoremlTrunk::load(&package, Units::CpuAndGpu).map_err(CliError::from_err)?;
            println!(
                "CoreML trunk {} loaded in {:.1} s ({}; hash {:.2} s {}, compile {:.2} s, model load {:.2} s)",
                package.display(),
                trunk.load_seconds,
                if trunk.compiled { "compiled this run" } else { "compiled model reused" },
                trunk.hash_seconds,
                if trunk.hashed { "full" } else { "recorded" },
                trunk.compile_seconds,
                trunk.model_seconds
            );
            Backbone::with_external_trunk(be, &w.vision, Box::new(trunk))
                .map_err(CliError::from_err)?
        }
    };
    let external_trunk = backbone.external_trunk();
    let text = Text::new(be, &w.text).map_err(CliError::from_err)?;
    let detector = Detector::new_at(
        be,
        DetectorWeights {
            transformer: &w.detector_transformer,
            geometry: &w.geometry_encoder,
            segmentation: &w.segmentation_head,
            scoring: &w.dot_prod_scoring,
        },
        opts.geometry,
    )
    .map_err(CliError::from_err)?;
    let vos = Vos::new(be, &backbone, &w.tracker).map_err(CliError::from_err)?;
    let tok = Tokenizer::from_hf_files(
        &opts.model_dir.join("vocab.json"),
        &opts.model_dir.join("merges.txt"),
    )
    .map_err(|e| {
        CliError::failed(format!(
            "tokenizer files in {}: {e}",
            opts.model_dir.display()
        ))
    })?;
    let prompt = Session::encode_prompt(be, &text, &tok, &opts.prompt)
        .map_err(|e| CliError::failed(format!("prompt {:?}: {e}", opts.prompt)))?;
    let mut cfg = SessionConfig::new(stream.len, stream.height, stream.width);
    cfg.book.max_num_objects = opts.max_objects;
    let mut session = Session::new(&vos, &detector, prompt, cfg);
    if let Some(p) = profile {
        session.set_stage_marker(move |stage| p.enter(stage));
    }
    let load_seconds = t_load.elapsed().as_secs_f64();
    println!(
        "model loaded in {load_seconds:.1} s on {}; tracking {} frame(s) of {:?}",
        opts.backend.name(),
        stream.len,
        stream.video
    );

    let side = opts.geometry.pixels();
    let mut frame_seconds = Vec::with_capacity(stream.len);
    let mut index = 0usize;
    loop {
        if let Some(p) = profile {
            p.enter(track_profile::DECODE);
        }
        let Some((n, image)) = stream.next_frame()? else {
            break;
        };
        let input = upstream_input(&image, side);
        let tensor = Tensor::from_f32(&input, &[1, 3, side, side], be.budget())
            .map_err(CliError::from_err)?;
        let device = be.upload(&tensor).map_err(CliError::from_err)?;
        collector.keep_image(index, &image);
        let t0 = Instant::now();
        let released = session
            .push_frame(&device)
            .map_err(|e| CliError::failed(format!("frame {n}: {e}")))?;
        be.sync().map_err(CliError::from_err)?;
        let secs = t0.elapsed().as_secs_f64();
        frame_seconds.push(secs);
        if let Some(p) = profile {
            p.enter(track_profile::COLLECTOR);
        }
        let extra = telemetry.after_frame(be, index)?;
        println!(
            "frame {n} ({}/{}): {secs:.2} s, {} active track(s), {} frame(s) released; {extra}",
            index + 1,
            stream.len,
            session.book().obj_ids().len(),
            released.len()
        );
        for y in released {
            collector.accept(y)?;
        }
        if let Some(p) = profile {
            p.end_frame()?;
        }
        index += 1;
    }
    for y in session.finish() {
        collector.accept(y)?;
    }
    Ok(Timing {
        load_seconds,
        frame_seconds,
        external_trunk,
    })
}

pub fn run(raw: &[String]) -> Result<(), CliError> {
    let args = Args::parse(
        raw,
        &[
            "prompt",
            "out",
            "backend",
            "max-frames",
            "start",
            "overlay",
            "overlay-video",
            "overlay-fps",
            "video",
            "boxes",
            "weights",
            "lora",
            "model-dir",
            "threads",
            "trim-pool-every",
            "trunk-precision",
            "max-objects",
            "input-size",
            "profile",
        ],
        1,
    )?;
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
    let boxes = BoxSource::parse(args.get("boxes").unwrap_or("mask"))?;
    let trim_pool_every = args
        .get("trim-pool-every")
        .map(|_| args.parsed("trim-pool-every", 0u32))
        .transpose()?;
    if trim_pool_every == Some(0) {
        return Err(CliError::usage("--trim-pool-every must be at least 1"));
    }
    if trim_pool_every.is_some() && backend != BackendChoice::Metal {
        return Err(CliError::usage("--trim-pool-every is for --backend metal"));
    }
    let max_objects: usize = args.parsed("max-objects", DEFAULT_MAX_OBJECTS)?;
    if !(1..=128).contains(&max_objects) {
        return Err(CliError::usage(format!(
            "--max-objects {max_objects} is outside 1..=128"
        )));
    }
    let geometry = parse_input_size(args.parsed("input-size", DEFAULT_INPUT_SIZE)?)?;
    let trunk = match args.get("trunk-precision") {
        Some(p) => TrunkPrecision::parse(p)?,
        None => TrunkPrecision::default_for(backend),
    };
    let profile = args.get("profile").map(ProfileMode::parse).transpose()?;
    if profile.is_some()
        && (backend != BackendChoice::Metal || trunk == TrunkPrecision::Bf16Linears)
    {
        return Err(CliError::usage(
            "--profile is for --backend metal, with any --trunk-precision but bf16-linears",
        ));
    }
    if matches!(trunk, TrunkPrecision::Coreml(_)) && geometry != Geometry::SAM3 {
        return Err(CliError::usage(format!(
            "--input-size {}: the CoreML trunk ({}) is exported at {}; pass --trunk-precision f32 or bf16-linears",
            geometry.pixels(),
            trunk.name(),
            Geometry::SAM3.pixels()
        )));
    }
    let threads: usize = args.parsed("threads", 8)?;
    if !(1..=256).contains(&threads) {
        return Err(CliError::usage(format!(
            "--threads {threads} is outside 1..=256"
        )));
    }
    let max_frames = args
        .get("max-frames")
        .map(|_| args.parsed("max-frames", 0usize))
        .transpose()?;
    if max_frames == Some(0) {
        return Err(CliError::usage("--max-frames must be at least 1"));
    }
    let start = args
        .get("start")
        .map(|_| args.parsed("start", 0u32))
        .transpose()?;
    let overlay_every = args
        .get("overlay")
        .map(|_| args.parsed("overlay", 0u32))
        .transpose()?;
    if overlay_every == Some(0) {
        return Err(CliError::usage("--overlay must be at least 1"));
    }
    let overlay_video = args.get("overlay-video").map(PathBuf::from);
    let overlay_fps = args
        .get("overlay-fps")
        .map(|_| args.parsed("overlay-fps", 0f64))
        .transpose()?;
    if overlay_fps.is_some() && overlay_video.is_none() {
        return Err(CliError::usage("--overlay-fps needs --overlay-video"));
    }
    if let Some(f) = overlay_fps {
        if !(f.is_finite() && (encode::MIN_FPS..=encode::MAX_FPS).contains(&f)) {
            return Err(CliError::usage(format!(
                "--overlay-fps {f} is outside {}..={} fps",
                encode::MIN_FPS,
                encode::MAX_FPS
            )));
        }
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
    // So is the overlay video: its size, frame rate and file name are checked before the weights load.
    // The video usually goes under `--out`, which may not exist yet: its parent must, to be checked.
    if overlay_video.is_some() {
        std::fs::create_dir_all(&out)
            .map_err(|e| CliError::failed(format!("{}: {e}", out.display())))?;
    }
    let overlay_spec = overlay_video
        .map(|path| plan_overlay_video(path, overlay_fps, &stream))
        .transpose()?;
    let weights = existing_path(
        args.get("weights"),
        "SPIKETRACK_WEIGHTS",
        "weights",
        "weights",
    )?;
    let lora = weights::lora_path(args.get("lora"))?;
    let model_dir = existing_path(
        args.get("model-dir"),
        "SPIKETRACK_MODEL_DIR",
        "tokenizer folder",
        "model-dir",
    )?;
    let opts = Options {
        prompt,
        out,
        overlay_every,
        boxes,
        backend,
        threads,
        weights,
        model_dir,
        trim_pool_every,
        trunk,
        max_objects,
        geometry,
        profile,
    };

    let (w, applied) = weights::load(&opts.weights, lora.as_deref())?;
    let mut collector = Collector::new(
        &opts.out,
        &stream.video,
        stream.first_n,
        stream.height,
        stream.width,
        opts.overlay_every,
        opts.boxes,
    )?;
    let mut overlay_run = None;
    if let Some(spec) = overlay_spec {
        let writer = VideoWriter::create(&spec.path, stream.width, stream.height, spec.fps)
            .map_err(|e| CliError::failed(format!("--overlay-video: {e}")))?;
        overlay_run = Some(OverlayVideoRun {
            fps: writer.fps(),
            fps_source: spec.fps_source,
            file: None,
        });
        collector.set_overlay_video(writer)?;
    }
    let t_all = Instant::now();
    let (timing, metal_mem_cap, profile_text) = match opts.backend {
        BackendChoice::Cpu if opts.trunk == TrunkPrecision::Bf16Linears => {
            let be = Autocast::new(
                CpuBackend::with_threads(Budget::new(BUDGET_BYTES), opts.threads)
                    .map_err(CliError::from_err)?,
            );
            (
                track_on(
                    &be,
                    &w,
                    &mut stream,
                    &opts,
                    &mut collector,
                    &mut HostProbe::new(),
                    None,
                )?,
                None,
                None,
            )
        }
        BackendChoice::Cpu => {
            let be = CpuBackend::with_threads(Budget::new(BUDGET_BYTES), opts.threads)
                .map_err(CliError::from_err)?;
            (
                track_on(
                    &be,
                    &w,
                    &mut stream,
                    &opts,
                    &mut collector,
                    &mut HostProbe::new(),
                    None,
                )?,
                None,
                None,
            )
        }
        BackendChoice::Metal if opts.trunk == TrunkPrecision::Bf16Linears => {
            let (metal, cap) = open_metal()?;
            let mut telemetry = MetalTelemetry::new(&metal, opts.trim_pool_every)?;
            (
                track_on(
                    &Autocast::new(metal),
                    &w,
                    &mut stream,
                    &opts,
                    &mut collector,
                    &mut telemetry,
                    None,
                )?,
                Some(cap),
                None,
            )
        }
        BackendChoice::Metal => {
            let (be, cap) = open_metal()?;
            let mut telemetry = MetalTelemetry::new(&be, opts.trim_pool_every)?;
            match opts.profile {
                None => (
                    track_on(
                        &be,
                        &w,
                        &mut stream,
                        &opts,
                        &mut collector,
                        &mut telemetry,
                        None,
                    )?,
                    Some(cap),
                    None,
                ),
                Some(ProfileMode::Stages) => {
                    let clock = StageClock::new(&be);
                    let timing = track_on(
                        &be,
                        &w,
                        &mut stream,
                        &opts,
                        &mut collector,
                        &mut telemetry,
                        Some(&clock),
                    )?;
                    let text = track_profile::stage_report(&clock.rows(), &timing.frame_seconds);
                    (timing, Some(cap), Some(text))
                }
                Some(ProfileMode::Ops) => {
                    let profiled =
                        Profiled::with_counters(be.clone(), track_profile::metal_counters);
                    let labels = OpsLabels::new(&profiled);
                    let timing = track_on(
                        &profiled,
                        &w,
                        &mut stream,
                        &opts,
                        &mut collector,
                        &mut telemetry,
                        Some(&labels),
                    )?;
                    // Before the first frame's first stage: the model load's uploads.
                    profiled.retain(|c| c.stage != "unlabelled");
                    (timing, Some(cap), Some(profiled.report()))
                }
            }
        }
    };
    let total_seconds = t_all.elapsed().as_secs_f64();
    if let (Some(mode), Some(text)) = (opts.profile, profile_text) {
        let path = opts
            .out
            .join(format!("{}.profile-{}.md", stream.video, mode.name()));
        std::fs::write(&path, text)
            .map_err(|e| CliError::failed(format!("{}: {e}", path.display())))?;
        println!("profile ({}) written to {}", mode.name(), path.display());
    }

    let mut results = std::mem::take(&mut collector.results);
    results.sort_by_key(|r| r.n);
    let expect: Vec<u32> = (stream.first_n..stream.first_n + stream.len as u32).collect();
    if results.iter().map(|r| r.n).collect::<Vec<_>>() != expect {
        return Err(CliError::failed(format!(
            "the session released {} frame(s) for {} processed",
            results.len(),
            stream.len
        )));
    }
    let text = predictions_json(&stream.video, stream.width, stream.height, &results)
        .map_err(CliError::failed)?;
    let pred_path = opts.out.join(format!("{}.json", stream.video));
    std::fs::write(&pred_path, text)
        .map_err(|e| CliError::failed(format!("{}: {e}", pred_path.display())))?;
    if let Some(run) = overlay_run.as_mut() {
        run.file = collector.finish_overlay_video(stream.len)?;
    }
    let [video_path_json, video_fps_json, video_fps_source_json, video_frames_json] =
        overlay_video_json(overlay_run.as_ref()).map_err(CliError::failed)?;

    let mean = timing.frame_seconds.iter().sum::<f64>() / timing.frame_seconds.len().max(1) as f64;
    let run_text = format!(
        "{{\n  \"video\": {},\n  \"input\": {},\n  \"prompt\": {},\n  \"backend\": {},\n  \"start\": {},\n  \"frames\": {},\n  \"width\": {},\n  \"height\": {},\n  \"tracks\": {},\n  \"boxes_widened_to_one_pixel\": {},\n  \"model_load_seconds\": {},\n  \"mean_seconds_per_frame\": {},\n  \"total_seconds\": {},\n  \"frame_precision\": \"float16 normalisation as upstream's loader, widened to f32\",\n  \"preprocessing\": {},\n  \"decoders\": {},\n  \"lora\": {},\n  \"boxes\": {},\n  \"boxes_reported\": {},\n  \"boxes_from_detector\": {},\n  \"overlay_video\": {},\n  \"overlay_video_fps\": {},\n  \"overlay_video_fps_source\": {},\n  \"overlay_video_frames\": {},\n  \"trunk_precision\": {},\n  \"external_trunk\": {},\n  \"max_objects\": {},\n  \"metal_mem_cap_bytes\": {},\n  \"input_size\": {},\n  \"profile\": {}\n}}\n",
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
        json::string(&preprocessing_text(stream.from_video, opts.geometry.pixels())),
        decoders_json(&stream.decoders),
        lora_json(applied.as_ref()),
        json::string(opts.boxes.name()),
        collector.reported,
        collector.from_detector,
        video_path_json,
        video_fps_json,
        video_fps_source_json,
        video_frames_json,
        json::string(opts.trunk.name()),
        timing.external_trunk.as_deref().map_or_else(|| "null".to_string(), json::string),
        opts.max_objects,
        metal_mem_cap.map_or_else(|| "null".to_string(), |b| b.to_string()),
        opts.geometry.pixels(),
        opts.profile.map_or_else(|| "null".to_string(), |m| json::string(m.name())),
    );
    let run_path = opts.out.join(format!("{}.run.json", stream.video));
    std::fs::write(&run_path, run_text)
        .map_err(|e| CliError::failed(format!("{}: {e}", run_path.display())))?;

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
    if let Some(run) = &overlay_run {
        if let Some(file) = &run.file {
            println!(
                "  {} (overlay video: {} frame(s) at {} fps from {}, {:.1} MiB)",
                file.path.display(),
                file.frames,
                run.fps,
                run.fps_source,
                file.bytes as f64 / MIB
            );
        }
    }
    if collector.widened > 0 {
        println!(
            "  note: {} box(es) widened to one pixel (a mask one pixel wide or high)",
            collector.widened
        );
    }
    if stream.from_video {
        println!(
            "  note: mp4 input does not match upstream's mp4 preprocessing (GAP-MP4-PREPROCESSING)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_visem::{evaluate_window, parse_predictions, Window};
    use st_model::tracker::associate::Bitmask;

    fn temp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("spiketrack-track-{name}-{}", std::process::id()));
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
        assert_eq!(
            f16_round(1.0 + 2f32.powi(-11) + 2f32.powi(-20)),
            1.0 + 2f32.powi(-10)
        );
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
        assert!(all
            .iter()
            .all(|&v| (-1.0..=1.0).contains(&v) && f16_round(v) == v));
        assert!(all.windows(2).all(|w| w[0] <= w[1]));
    }

    /// Upstream's own input frames (`fixtures/detect_track/11.safetensors`, `frames`, float16,
    /// made by its folder loader from `11_frame_<n>.jpg`) against `upstream_input` of the same
    /// JPEG decoded by Pillow (`fixtures/decode/11_frame_0.rgb`, tools/reference/decode_oracle.py):
    /// the resize and the float16 normalisation must agree bit for bit. Then the same JPEG through
    /// `decode_image_with_source`, which the CLI uses: the Pillow-exact decoder, so also bit for
    /// bit.
    #[test]
    #[ignore = "needs fixtures/detect_track/11.safetensors, fixtures/decode/11_frame_0.rgb and data/visem"]
    fn upstream_input_matches_upstreams_loader_on_a_visem_frame() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let st = ojas_io::SafeTensors::open(&root.join("fixtures/detect_track/11.safetensors"))
            .unwrap_or_else(|e| panic!("{}", e.detail()));
        let (shape, frames) = st
            .read_f32_widened("frames")
            .unwrap_or_else(|e| panic!("{}", e.detail()));
        assert_eq!(&shape[1..], &[3, 1008, 1008]);
        let want = &frames[..3 * 1008 * 1008];
        let rgb = std::fs::read(root.join("fixtures/decode/11_frame_0.rgb"))
            .expect("run tools/reference/decode_oracle.py");
        let pillow = Rgb8::new(640, 480, rgb);
        let ours = upstream_input(&pillow, 1008);
        let differing = ours.iter().zip(want).filter(|(a, b)| a != b).count();
        let max = ours
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("Pillow-decoded frame 0: {differing} of {} samples differ from upstream's float16 frame (max {max})", want.len());
        assert_eq!(
            differing, 0,
            "the resize and float16 normalisation are not bit-identical to upstream's loader"
        );
        let (decoded, decoder) = decode_image_with_source(
            root.join("data/visem/VISEM_Tracking_Train_v4/Train/11/images/11_frame_0.jpg"),
        )
        .expect("jpeg");
        assert_eq!(decoder.name(), "pillow-exact-jpeg");
        let ours = upstream_input(&decoded, 1008);
        let differing = ours.iter().zip(want).filter(|(a, b)| a != b).count();
        let max = ours
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("CLI-decoded frame 0: {differing} of {} samples differ (max {max}); plain f32 sam_input differs from upstream by", want.len());
        assert_eq!(
            differing, 0,
            "the CLI's decode + upstream_input is not bit-identical to upstream's loader"
        );
        let plain = st_video::sam_input(
            &Rgb8::new(
                640,
                480,
                std::fs::read(root.join("fixtures/decode/11_frame_0.rgb")).unwrap(),
            ),
            1008,
        );
        let max_plain = plain
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let diff_plain = plain.iter().zip(want).filter(|(a, b)| a != b).count();
        println!("  (Pillow-decoded, f32 sam_input: {diff_plain} samples differ, max {max_plain})");
    }

    #[test]
    fn the_model_input_is_resized_then_normalised_planar() {
        // A constant image stays constant under bilinear resize; 3 planes, each one value.
        let img = Rgb8::new(
            5,
            3,
            [10u8, 128, 250]
                .iter()
                .copied()
                .cycle()
                .take(5 * 3 * 3)
                .collect(),
        );
        let x = upstream_input(&img, 8);
        assert_eq!(x.len(), 3 * 64);
        for (c, v) in [10u8, 128, 250].into_iter().enumerate() {
            assert!(
                x[c * 64..(c + 1) * 64]
                    .iter()
                    .all(|&s| s == upstream_normalise(v)),
                "plane {c}"
            );
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
        assert_eq!(
            f.frames.iter().map(|f| f.n).collect::<Vec<_>>(),
            vec![1, 2, 10, 21, 100]
        );
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
        assert!(discover_frames(&d)
            .unwrap_err()
            .contains("no <video>_frame_<n>"));
        touch(&d, "22_frame_1.jpg");
        touch(&d, "23_frame_2.jpg");
        assert!(discover_frames(&d).unwrap_err().contains("two videos"));
        std::fs::remove_file(d.join("23_frame_2.jpg")).unwrap();
        touch(&d, "22_frame_01.png");
        assert!(
            discover_frames(&d).unwrap_err().contains("twice"),
            "frame 1 as .jpg and as 01.png"
        );
        std::fs::remove_file(d.join("22_frame_01.png")).unwrap();
        touch(&d, "stray.JPG");
        assert!(discover_frames(&d)
            .unwrap_err()
            .contains("not <video>_frame_<n>"));
        let _ = std::fs::remove_dir_all(&d);
    }

    fn files(ns: &[u32]) -> Vec<FrameFile> {
        ns.iter()
            .map(|&n| FrameFile {
                n,
                path: PathBuf::from(format!("{n}.jpg")),
            })
            .collect()
    }

    #[test]
    fn the_window_starts_at_a_frame_number_and_stops_at_max() {
        let f = files(&[5, 6, 7, 8, 9]);
        assert_eq!(select_window(&f, None, None).unwrap().len(), 5);
        assert_eq!(
            select_window(&f, Some(7), None)
                .unwrap()
                .iter()
                .map(|f| f.n)
                .collect::<Vec<_>>(),
            vec![7, 8, 9]
        );
        assert_eq!(
            select_window(&f, Some(6), Some(2))
                .unwrap()
                .iter()
                .map(|f| f.n)
                .collect::<Vec<_>>(),
            vec![6, 7]
        );
        assert_eq!(select_window(&f, None, Some(99)).unwrap().len(), 5);
        assert!(select_window(&f, Some(4), None)
            .unwrap_err()
            .contains("no frame 4"));
        assert!(select_window(&[], None, None).is_err());
    }

    #[test]
    fn a_gap_inside_the_window_is_an_error_but_one_outside_is_not() {
        let f = files(&[0, 1, 2, 5, 6]);
        assert!(select_window(&f, None, None)
            .unwrap_err()
            .contains("not consecutive"));
        assert_eq!(select_window(&f, None, Some(3)).unwrap().len(), 3);
        assert_eq!(select_window(&f, Some(5), None).unwrap().len(), 2);
    }

    fn yielded(frame: usize, ids: &[i64], boxes_xywh: &[f32], masks: Vec<Bitmask>) -> YieldedFrame {
        YieldedFrame {
            frame,
            obj_ids: ids.to_vec(),
            probs: ids.iter().map(|&i| 0.5 + i as f32 / 100.0).collect(),
            boxes_xywh: boxes_xywh.to_vec(),
            det_boxes_xyxy: vec![None; ids.len()],
            masks,
        }
    }

    #[test]
    fn boxes_become_xyxy_and_degenerate_ones_are_widened_and_counted() {
        let m = || Bitmask::from_fn(4 * 8, |_| false);
        // 8 wide, 4 high: id 1 normal, id 2 zero width, id 3 zero height.
        let y = yielded(
            0,
            &[1, 2, 3],
            &[
                0.25, 0.5, 0.25, 0.25, 0.5, 0.0, 0.0, 0.25, 0.0, 0.5, 0.5, 0.0,
            ],
            vec![m(), m(), m()],
        );
        let (o, widened, from_detector) = objects_of(&y, 4, 8, BoxSource::Mask).unwrap();
        assert_eq!(from_detector, 0);
        assert_eq!(o[0].box_xyxy_norm, [0.25, 0.5, 0.5, 0.75]);
        assert_eq!(
            o[1].box_xyxy_norm,
            [0.5, 0.0, 0.625, 0.25],
            "one pixel wide = 1/8"
        );
        assert_eq!(
            o[2].box_xyxy_norm,
            [0.0, 0.5, 0.5, 0.75],
            "one pixel high = 1/4"
        );
        assert_eq!(widened, 2);
        assert_eq!(o[1].score, 0.52);
    }

    #[test]
    fn inconsistent_or_non_finite_frames_are_errors() {
        let m = || Bitmask::from_fn(4, |_| false);
        assert!(objects_of(
            &yielded(0, &[1], &[0.1, 0.1], vec![m()]),
            2,
            2,
            BoxSource::Mask
        )
        .is_err());
        assert!(objects_of(
            &yielded(0, &[1], &[0.1, 0.1, f32::NAN, 0.1], vec![m()]),
            2,
            2,
            BoxSource::Mask
        )
        .is_err());
        assert!(objects_of(
            &yielded(0, &[1, 2], &[0.1, 0.1, 0.1, 0.1], vec![m()]),
            2,
            2,
            BoxSource::Mask
        )
        .is_err());
        let mut y = yielded(0, &[1], &[0.1, 0.1, 0.1, 0.1], vec![m()]);
        y.det_boxes_xyxy = vec![];
        assert!(
            objects_of(&y, 2, 2, BoxSource::Mask).is_err(),
            "a missing detection-box slot is an error under either source"
        );
        y.det_boxes_xyxy = vec![Some([0.1, f32::NAN, 0.2, 0.2])];
        assert!(objects_of(&y, 2, 2, BoxSource::Detector).is_err());
    }

    #[test]
    fn detector_boxes_replace_mask_boxes_where_assigned_and_are_clipped() {
        let m = || Bitmask::from_fn(4 * 8, |_| false);
        // id 1 has a detection (partly outside the frame), id 2 has none, id 3 a degenerate one.
        let mut y = yielded(
            0,
            &[1, 2, 3],
            &[
                0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 0.5, 0.5,
            ],
            vec![m(), m(), m()],
        );
        y.det_boxes_xyxy = vec![
            Some([-0.1, 0.25, 0.375, 1.2]),
            None,
            Some([0.5, 0.5, 0.5, 0.75]),
        ];
        let (o, widened, from_detector) = objects_of(&y, 4, 8, BoxSource::Detector).unwrap();
        assert_eq!(
            o[0].box_xyxy_norm,
            [0.0, 0.25, 0.375, 1.0],
            "clipped to the frame"
        );
        assert_eq!(
            o[1].box_xyxy_norm,
            [0.5, 0.25, 0.75, 0.5],
            "no detection: the mask box"
        );
        assert_eq!(
            o[2].box_xyxy_norm,
            [0.5, 0.5, 0.625, 0.75],
            "zero width widened to one pixel"
        );
        assert_eq!((widened, from_detector), (1, 2));
        // The same frame under --boxes mask ignores the detections.
        let (o, _, from_detector) = objects_of(&y, 4, 8, BoxSource::Mask).unwrap();
        assert_eq!(
            (o[0].box_xyxy_norm, from_detector),
            ([0.25, 0.5, 0.5, 0.75], 0)
        );
        assert!(BoxSource::parse("detector").is_ok() && BoxSource::parse("box").is_err());
    }

    #[test]
    fn the_predictions_file_is_what_eval_visem_reads() {
        let frames = vec![
            FrameResult {
                n: 5,
                objects: vec![
                    TrackedObject {
                        id: 1,
                        score: 0.9,
                        box_xyxy_norm: [0.1, 0.2, 0.3, 0.4],
                    },
                    TrackedObject {
                        id: 7,
                        score: 0.6,
                        box_xyxy_norm: [0.5, 0.5, 0.75, 0.625],
                    },
                ],
            },
            FrameResult {
                n: 6,
                objects: vec![],
            },
            FrameResult {
                n: 7,
                objects: vec![TrackedObject {
                    id: 1,
                    score: 0.9,
                    box_xyxy_norm: [0.125, 0.25, 0.3, 0.4],
                }],
            },
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
        assert!(
            parse_predictions(&text, "23").is_err(),
            "the video id is checked"
        );
        // No frames at all is still a valid file.
        assert!(
            parse_predictions(&predictions_json("22", 640, 480, &[]).unwrap(), "22")
                .unwrap()
                .frames
                .is_empty()
        );
        assert!(predictions_json(
            "22",
            640,
            480,
            &[FrameResult {
                n: 0,
                objects: vec![TrackedObject {
                    id: 1,
                    score: f32::NAN,
                    box_xyxy_norm: [0.0; 4]
                }]
            }]
        )
        .is_err());
    }

    #[test]
    fn the_collector_writes_masks_and_overlays_and_numbers_frames_from_the_start() {
        let d = temp("collect");
        let (h, w) = (4usize, 6usize);
        let mut c = Collector::new(&d, "9", 100, h, w, Some(2), BoxSource::Mask).unwrap();
        let image = Rgb8::new(w, h, vec![50; w * h * 3]);
        for i in 0..3 {
            c.keep_image(i, &image);
        }
        assert_eq!(c.images.len(), 2, "frames 100 and 102 are multiples of 2");
        let mask = Bitmask::from_fn(h * w, |p| p % w < 2); // the two left columns
        c.accept(yielded(0, &[3], &[0.0, 0.0, 0.25, 0.75], vec![mask]))
            .unwrap();
        c.accept(yielded(1, &[], &[], vec![])).unwrap();
        assert!(
            d.join("masks/9_frame_100.json").is_file()
                && d.join("masks/9_frame_101.json").is_file()
        );
        assert!(d.join("overlay/9_frame_100.png").is_file());
        assert!(
            !d.join("overlay/9_frame_101.png").exists(),
            "an odd frame has no overlay"
        );
        let m = ojas_io::parse_json(
            &std::fs::read_to_string(d.join("masks/9_frame_100.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(m.get_u64("frame").unwrap(), 100);
        let obj = &m.get_array("objects").unwrap()[0];
        assert_eq!(obj.get_u64("id").unwrap(), 3);
        assert_eq!(obj.get_u64("area").unwrap(), 8);
        // Column-major: two full columns of 4 set, then 16 clear.
        let counts: Vec<u64> = obj
            .get_array("counts")
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(counts, vec![0, 8, 16]);
        assert_eq!(
            c.results.iter().map(|r| r.n).collect::<Vec<_>>(),
            vec![100, 101]
        );
        assert_eq!(c.ids.iter().copied().collect::<Vec<_>>(), vec![3]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A collector over `w x h` grey frames numbered from 100, with `overlay_every` and, when
    /// `video` is a path, an overlay video at 10 fps; every frame's mask is the left half.
    fn video_collector(dir: &Path, overlay_every: Option<u32>, video: Option<&Path>) -> Collector {
        let mut c = Collector::new(dir, "9", 100, VH, VW, overlay_every, BoxSource::Mask).unwrap();
        if let Some(p) = video {
            c.set_overlay_video(VideoWriter::create(p, VW, VH, 10.0).unwrap())
                .unwrap();
        }
        c
    }

    const VW: usize = 64;
    const VH: usize = 48;

    fn grey_frame() -> Rgb8 {
        Rgb8::new(VW, VH, vec![50; VW * VH * 3])
    }

    /// The released frame `k`: id 3 with the left half masked and boxed.
    fn left_half(k: usize) -> YieldedFrame {
        yielded(
            k,
            &[3],
            &[0.0, 0.0, 0.5, 1.0],
            vec![Bitmask::from_fn(VH * VW, |p| p % VW < VW / 2)],
        )
    }

    #[test]
    fn the_overlay_video_has_every_frame_in_order_and_the_collector_holds_only_unreleased_frames() {
        let d = temp("video");
        let path = d.join("overlay.mp4");
        let mut c = video_collector(&d, None, Some(&path));
        // The session releases frame i - LAG when frame i is pushed; the collector must hold the
        // frames in between, and no more.
        const LAG: usize = 14;
        let total = 30;
        let mut peak = 0;
        for i in 0..total {
            c.keep_image(i, &grey_frame());
            peak = peak.max(c.images.len());
            if i >= LAG {
                c.accept(left_half(i - LAG)).unwrap();
            }
        }
        for k in total - LAG..total {
            c.accept(left_half(k)).unwrap();
        }
        assert_eq!(
            peak,
            LAG + 1,
            "one frame awaiting release for each frame of lag, plus the one just pushed"
        );
        assert!(c.images.is_empty(), "every kept frame was released");
        let file = c
            .finish_overlay_video(total)
            .unwrap()
            .expect("a video was asked for");
        assert_eq!(file.frames, total);
        assert!(!d.join("overlay").exists(), "no --overlay K, no PNGs");
        assert!(
            c.finish_overlay_video(total).unwrap().is_none(),
            "finished once"
        );

        let mut reader = VideoReader::open(&file.path).unwrap();
        assert_eq!(
            (
                reader.info().width,
                reader.info().height,
                reader.info().frame_count
            ),
            (VW, VH, total)
        );
        assert!(
            (reader.info().fps - 10.0).abs() < 0.1,
            "{}",
            reader.info().fps
        );
        let mut n = 0;
        while let Some((i, _, img)) = reader.next_frame().unwrap() {
            assert_eq!(i, n);
            let at = |x: usize, y: usize| -> [u8; 3] {
                img.data[(y * VW + x) * 3..][..3].try_into().unwrap()
            };
            // Id 3 is the palette's fourth colour (0, 130, 200) blended 100/256 over grey 50, away
            // from the box outline; the right half is the untouched frame.
            let tinted = (0..3)
                .map(|k| ((50 * 156 + u32::from(overlay::colour(3)[k]) * 100) / 256) as u8)
                .collect::<Vec<_>>();
            for k in 0..3 {
                assert!(
                    at(10, 24)[k].abs_diff(tinted[k]) <= 8,
                    "frame {i} masked pixel {:?}, expected about {tinted:?}",
                    at(10, 24)
                );
                assert!(
                    at(50, 24)[k].abs_diff(50) <= 8,
                    "frame {i} unmasked pixel {:?}",
                    at(50, 24)
                );
            }
            n += 1;
        }
        assert_eq!(n, total);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_video_and_the_pngs_are_independent_and_render_the_same_pixels() {
        let (a, b, c_dir) = (temp("png-only"), temp("png-and-video"), temp("video-only"));
        let mut png_only = video_collector(&a, Some(2), None);
        let mut both = video_collector(&b, Some(2), Some(&b.join("o.mp4")));
        let mut video_only = video_collector(&c_dir, None, Some(&c_dir.join("o.mp4")));
        for c in [&mut png_only, &mut both, &mut video_only] {
            for i in 0..4 {
                c.keep_image(i, &grey_frame());
            }
            for k in 0..4 {
                c.accept(left_half(k)).unwrap();
            }
        }
        assert_eq!(both.finish_overlay_video(4).unwrap().unwrap().frames, 4);
        assert_eq!(
            video_only.finish_overlay_video(4).unwrap().unwrap().frames,
            4
        );
        assert!(png_only.finish_overlay_video(4).unwrap().is_none());
        for n in [100, 102] {
            let name = format!("overlay/9_frame_{n}.png");
            assert_eq!(
                std::fs::read(a.join(&name)).unwrap(),
                std::fs::read(b.join(&name)).unwrap(),
                "{name}: the PNG is the same with or without a video"
            );
        }
        for odd in [101, 103] {
            assert!(!b.join(format!("overlay/9_frame_{odd}.png")).exists());
        }
        assert!(!a.join("o.mp4").exists() && !c_dir.join("overlay").exists());
        for d in [a, b, c_dir] {
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn a_frame_released_out_of_order_or_without_its_image_is_an_error() {
        let d = temp("video-order");
        let mut c = video_collector(&d, None, Some(&d.join("o.mp4")));
        for i in 0..3 {
            c.keep_image(i, &grey_frame());
        }
        let e = c.accept(left_half(1)).unwrap_err().to_string();
        assert!(e.contains("out of order"), "{e}");
        c.accept(left_half(0)).unwrap();
        let e = c.accept(left_half(5)).unwrap_err().to_string();
        assert!(e.contains("image was not kept"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_video_for_other_frames_is_refused() {
        let d = temp("video-size");
        let mut c = Collector::new(&d, "9", 0, VH, VW, None, BoxSource::Mask).unwrap();
        let wrong = VideoWriter::create(d.join("o.mp4"), VW + 2, VH, 10.0).unwrap();
        assert!(c.set_overlay_video(wrong).is_err());
        assert!(
            !d.join("o.mp4").exists(),
            "the refused writer was dropped and cleaned up"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_collector_dropped_mid_run_leaves_no_partial_video() {
        let d = temp("video-drop");
        let path = d.join("o.mp4");
        {
            let mut c = video_collector(&d, None, Some(&path));
            for i in 0..3 {
                c.keep_image(i, &grey_frame());
            }
            c.accept(left_half(0)).unwrap();
            // The run fails here: the collector is dropped with the video unfinished.
        }
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_overlay_frame_rate_comes_from_the_option_then_the_input_then_the_folder_default() {
        assert_eq!(
            overlay_fps_choice(Some(30.0), Some(49.0)),
            (30.0, "--overlay-fps")
        );
        assert_eq!(overlay_fps_choice(None, Some(25.0)), (25.0, "input video"));
        assert_eq!(overlay_fps_choice(None, None).0, FOLDER_OVERLAY_FPS);
        assert_eq!(FOLDER_OVERLAY_FPS, 49.0);
    }

    fn stream_of(width: usize, height: usize, fps: Option<f64>) -> FrameStream {
        FrameStream {
            kind: Kind::Folder { files: vec![] },
            video: "v".into(),
            first_n: 0,
            len: 1,
            height,
            width,
            from_video: fps.is_some(),
            fps,
            decoders: BTreeMap::new(),
            produced: 0,
            pending: None,
        }
    }

    #[test]
    fn the_overlay_video_is_checked_before_any_model_work() {
        let d = temp("video-plan");
        let ok = plan_overlay_video(d.join("o.mp4"), None, &stream_of(640, 480, None)).unwrap();
        assert_eq!(
            (ok.fps, ok.fps_source),
            (FOLDER_OVERLAY_FPS, FPS_FROM_DEFAULT)
        );
        let ok =
            plan_overlay_video(d.join("o.mp4"), None, &stream_of(640, 480, Some(47.95))).unwrap();
        assert_eq!((ok.fps, ok.fps_source), (47.95, FPS_FROM_INPUT));
        let e = plan_overlay_video(d.join("o.mp4"), Some(30.0), &stream_of(641, 480, None))
            .unwrap_err()
            .to_string();
        assert!(e.contains("--overlay-video") && e.contains("even"), "{e}");
        let e = plan_overlay_video(d.join("o.mov"), None, &stream_of(640, 480, None))
            .unwrap_err()
            .to_string();
        assert!(e.contains(".mp4"), "{e}");
        let e = plan_overlay_video(d.join("o.mp4"), None, &stream_of(640, 480, Some(0.5)))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("--overlay-fps"),
            "an input rate the encoder refuses points at the option: {e}"
        );
        std::fs::write(d.join("taken.mp4"), b"x").unwrap();
        let e = plan_overlay_video(d.join("taken.mp4"), None, &stream_of(640, 480, None))
            .unwrap_err()
            .to_string();
        assert!(e.contains("already exists"), "{e}");
        assert_eq!(std::fs::read(d.join("taken.mp4")).unwrap(), b"x");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_overlay_video_may_go_under_an_out_directory_that_does_not_exist_yet() {
        // The README's example: `--out out/track --overlay-video out/track/22-overlay.mp4` on a first run.
        let d = temp("video-newout");
        let frames = d.join("frames");
        std::fs::create_dir_all(&frames).unwrap();
        std::fs::write(
            frames.join("v_frame_0.png"),
            png::encode_rgb(16, 16, &[90; 16 * 16 * 3]).unwrap(),
        )
        .unwrap();
        let (out, video) = (d.join("newout"), d.join("newout/o.mp4"));
        // The run stops at the missing weights, after the video has been checked and before it is created.
        let r = run(&args(&[
            frames.to_str().unwrap(),
            "--prompt",
            "p",
            "--out",
            out.to_str().unwrap(),
            "--weights",
            "/nonexistent/weights",
            "--overlay-video",
            video.to_str().unwrap(),
        ]));
        assert!(
            matches!(&r, Err(CliError::Failed(m)) if m.contains("weights") && m.contains("does not exist")),
            "{r:?}"
        );
        assert!(out.is_dir(), "--out was created for the video's sake");
        assert!(!video.exists());
        // A bad video request is refused at the same point, ahead of the weights.
        let r = run(&args(&[
            frames.to_str().unwrap(),
            "--prompt",
            "p",
            "--out",
            out.to_str().unwrap(),
            "--weights",
            "/nonexistent/weights",
            "--overlay-video",
            out.join("o.mov").to_str().unwrap(),
        ]));
        assert!(
            matches!(&r, Err(CliError::Failed(m)) if m.contains("--overlay-video") && m.contains(".mp4")),
            "{r:?}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn run_json_records_the_overlay_video_or_null() {
        assert_eq!(
            overlay_video_json(None).unwrap(),
            ["null", "null", "null", "null"].map(String::from)
        );
        let run = OverlayVideoRun {
            fps: 49.0,
            fps_source: FPS_FROM_DEFAULT,
            file: Some(VideoFile {
                path: PathBuf::from("/x/o.mp4"),
                frames: 24,
                bytes: 10,
            }),
        };
        let [path, fps, source, frames] = overlay_video_json(Some(&run)).unwrap();
        assert_eq!((path.as_str(), frames.as_str()), ("\"/x/o.mp4\"", "24"));
        assert_eq!(fps.parse::<f64>().unwrap(), 49.0);
        assert!(source.contains("49 fps"));
        let unfinished = OverlayVideoRun { file: None, ..run };
        assert!(overlay_video_json(Some(&unfinished)).is_err());
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
            std::fs::write(
                labels.join(format!("5_frame_{t}_with_ftid.txt")),
                format!("a 0 {} 0.5 0.1 0.2\n", x + 0.05),
            )
            .unwrap();
            frames.push(FrameResult {
                n: t,
                objects: vec![TrackedObject {
                    id: 42,
                    score: 0.9,
                    box_xyxy_norm: [x as f32, 0.4, (x + 0.1) as f32, 0.6],
                }],
            });
        }
        std::fs::create_dir_all(d.join("pred")).unwrap();
        std::fs::write(
            d.join("pred/5.json"),
            predictions_json("5", 640, 480, &frames).unwrap(),
        )
        .unwrap();
        let e = evaluate_window(
            &d.join("pred"),
            &["5".to_string()],
            &d.join("gt"),
            Window {
                start: Some(0),
                max_frames: Some(6),
            },
        )
        .unwrap();
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
        assert!(
            usage(&["in", "--prompt", "p", "--overlay-fps", "49"]),
            "a frame rate without a video"
        );
        for fps in ["0", "0.5", "241", "abc", "NaN", "inf"] {
            assert!(
                usage(&[
                    "in",
                    "--prompt",
                    "p",
                    "--overlay-video",
                    "o.mp4",
                    "--overlay-fps",
                    fps
                ]),
                "--overlay-fps {fps}"
            );
        }
        assert!(
            usage(&["in", "--prompt", "p", "--overlay-video"]),
            "no file name"
        );
        assert!(usage(&["in", "--prompt", "p", "--threads", "0"]));
        assert!(usage(&["in", "--prompt", "p", "--video", "a/b"]));
        assert!(usage(&["in", "--prompt", "p", "--bogus", "1"]));
        // The CoreML trunks exist at 1008 only, and coreml-fp32 is the Metal default.
        assert!(
            usage(&["in", "--prompt", "p", "--input-size", "672"]),
            "default Metal trunk at 672"
        );
        assert!(usage(&[
            "in",
            "--prompt",
            "p",
            "--input-size",
            "672",
            "--trunk-precision",
            "coreml-fp16"
        ]));
        assert!(usage(&[
            "in",
            "--prompt",
            "p",
            "--input-size",
            "672",
            "--backend",
            "cpu",
            "--trunk-precision",
            "coreml-fp32"
        ]));
        assert!(
            !usage(&[
                "in",
                "--prompt",
                "p",
                "--input-size",
                "672",
                "--trunk-precision",
                "f32"
            ]),
            "the Ojas trunk takes 672"
        );
        assert!(
            !usage(&[
                "in",
                "--prompt",
                "p",
                "--input-size",
                "672",
                "--backend",
                "cpu"
            ]),
            "the CPU default is the Ojas trunk"
        );
        assert!(usage(&["in", "--prompt", "p", "--profile", "all"]));
        assert!(usage(&[
            "in",
            "--prompt",
            "p",
            "--profile",
            "stages",
            "--backend",
            "cpu"
        ]));
        assert!(usage(&[
            "in",
            "--prompt",
            "p",
            "--profile",
            "ops",
            "--trunk-precision",
            "bf16-linears"
        ]));
        assert!(
            !usage(&["in", "--prompt", "p", "--profile", "stages"]),
            "the default Metal trunk profiles"
        );
        assert!(!usage(&[
            "in",
            "--prompt",
            "p",
            "--profile",
            "ops",
            "--trunk-precision",
            "f32"
        ]));
    }

    #[test]
    fn a_bad_input_fails_before_the_weights_are_looked_for() {
        // Weights are not given and may not exist: the input error must come first.
        let missing = run(&args(&["/nonexistent/spiketrack-input", "--prompt", "p"]));
        assert!(
            matches!(&missing, Err(CliError::Failed(m)) if m.contains("does not exist")),
            "{missing:?}"
        );
        let d = temp("notframes");
        touch(&d, "readme.txt");
        let empty = run(&args(&[d.to_str().unwrap(), "--prompt", "p"]));
        assert!(
            matches!(&empty, Err(CliError::Failed(m)) if m.contains("no <video>_frame_<n>")),
            "{empty:?}"
        );
        let file = d.join("readme.txt");
        let notvideo = run(&args(&[file.to_str().unwrap(), "--prompt", "p"]));
        assert!(
            matches!(&notvideo, Err(CliError::Failed(m)) if m.contains("not a folder of frames")),
            "{notvideo:?}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn help_mentions_the_mp4_preprocessing_gap() {
        assert!(USAGE.contains("GAP-MP4-PREPROCESSING") && USAGE.contains("INTER_CUBIC"));
    }

    #[test]
    fn max_objects_defaults_to_64_and_the_help_says_so() {
        assert_eq!(DEFAULT_MAX_OBJECTS, 64);
        assert!(
            USAGE.contains(&format!("1 to 128 (default {DEFAULT_MAX_OBJECTS};")),
            "--max-objects help names another default"
        );
        assert!(run(&args(&["--help"])).is_ok());
    }

    #[test]
    fn input_size_defaults_to_1008_accepts_whole_windows_only_and_the_help_says_so() {
        assert_eq!(DEFAULT_INPUT_SIZE, 1008);
        assert!(USAGE.contains("--input-size N") && USAGE.contains("multiple of 336"));
        for (n, grid) in [(1008usize, 72usize), (672, 48), (336, 24), (1344, 96)] {
            let g = parse_input_size(n).unwrap_or_else(|e| panic!("{n}: {e}"));
            assert_eq!((g.pixels(), g.grid), (n, grid));
        }
        // Multiples of the patch that tile no window, and sizes that are not even patches.
        for n in [0usize, 14, 350, 448, 504, 1000, 1009] {
            let Err(CliError::Usage(m)) = parse_input_size(n) else {
                panic!("{n} was accepted")
            };
            assert!(m.contains("--input-size") && m.contains("336"), "{n}: {m}");
        }
        // The flag is checked before the input is opened: a missing folder does not mask it.
        assert!(
            matches!(run(&args(&["/nonexistent", "--prompt", "p", "--input-size", "500"])), Err(CliError::Usage(m)) if m.contains("--input-size 500"))
        );
        assert!(matches!(
            run(&args(&[
                "/nonexistent",
                "--prompt",
                "p",
                "--input-size",
                "abc"
            ])),
            Err(CliError::Usage(_))
        ));
        // The run's preprocessing text names the size that ran.
        assert!(
            preprocessing_text(false, 672).contains("resize to 672x672")
                && !preprocessing_text(false, 672).contains("1008")
        );
        assert!(preprocessing_text(true, 1008).contains("resize to 1008x1008"));
    }

    /// VISEM video 22, frames 0..8, on Metal: the output parses with eval-visem's reader,
    /// frames and ids are what a run of 8 frames must give, and a track keeps its id while it
    /// moves a little. Needs the gated checkpoint, the tokenizer files and `data/visem`.
    ///
    /// `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... tools/locked.sh track-e2e cargo test --release -p spiketrack track_cmd::tests::visem_22 -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a Metal GPU"]
    fn a_track_runs_metal_backend_has_the_raised_memory_cap() {
        let default = MetalBackend::new(Budget::new(BUDGET_BYTES))
            .expect("metal")
            .mem_cap()
            .expect("mem_cap");
        let (_be, cap) = open_metal().expect("open_metal");
        assert_eq!(default, 1 << 30, "ojas's default is min(1 GiB, budget / 4)");
        assert_eq!(cap, METAL_MEM_CAP);
    }

    #[test]
    fn trunk_precision_parses_its_four_names_and_nothing_else() {
        for name in ["f32", "bf16-linears", "coreml-fp32", "coreml-fp16"] {
            assert_eq!(TrunkPrecision::parse(name).expect(name).name(), name);
        }
        assert_eq!(
            TrunkPrecision::parse("coreml-fp16").unwrap(),
            TrunkPrecision::Coreml("fp16")
        );
        for bad in ["", "fp32", "coreml", "coreml-bf16", "F32"] {
            assert!(TrunkPrecision::parse(bad).is_err(), "{bad:?} accepted");
        }
        assert_eq!(
            coreml_package(Path::new("/m"), "fp32"),
            PathBuf::from("/m/coreml/sam3.1_trunk_fp32.mlpackage")
        );
    }

    #[test]
    fn the_default_trunk_is_coreml_fp32_on_metal_and_the_ojas_trunk_on_cpu() {
        assert_eq!(
            TrunkPrecision::default_for(BackendChoice::Metal),
            TrunkPrecision::Coreml("fp32")
        );
        assert_eq!(
            TrunkPrecision::default_for(BackendChoice::Cpu),
            TrunkPrecision::F32
        );
    }

    /// With no --trunk-precision, a Metal run whose model dir has no CoreML package fails loud and
    /// names both ways out; it does not quietly fall back to the Ojas trunk.
    #[test]
    #[ignore = "needs the gated checkpoint, tokenizer files and data/visem"]
    fn the_default_metal_trunk_without_its_package_is_an_error_not_a_fallback() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let images = root.join("data/visem/VISEM_Tracking_Train_v4/Train/22/images");
        let model_dir = temp("no-coreml-model-dir");
        let tokenizer =
            PathBuf::from(std::env::var("SPIKETRACK_MODEL_DIR").expect("set SPIKETRACK_MODEL_DIR"));
        for f in ["vocab.json", "merges.txt"] {
            std::fs::copy(tokenizer.join(f), model_dir.join(f)).unwrap();
        }
        let out = temp("default-trunk-missing");
        let r = run(&args(&[
            images.to_str().unwrap(),
            "--prompt",
            "cell",
            "--out",
            out.to_str().unwrap(),
            "--max-frames",
            "1",
            "--model-dir",
            model_dir.to_str().unwrap(),
        ]));
        let Err(CliError::Failed(msg)) = r else {
            panic!("expected a failure, got {r:?}")
        };
        assert!(msg.starts_with("--trunk-precision coreml-fp32: "), "{msg}");
        assert!(
            msg.contains("coreml_trunk.py export --precision fp32")
                && msg.contains("--trunk-precision f32"),
            "{msg}"
        );
        assert!(
            !out.join("22.json").exists(),
            "a failed run wrote predictions"
        );
    }

    /// The CoreML fp32 trunk on the first 8 frames of VISEM 22 against the Ojas f32 trunk: run.json
    /// names the trunk that ran, and the tracks agree (the two trunks differ by ~1e-5 relative).
    #[test]
    #[ignore = "needs the gated checkpoint, tokenizer files, data/visem, a Metal GPU and the exported CoreML fp32 trunk"]
    fn visem_22_coreml_fp32_trunk_tracks_as_the_ojas_trunk() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let images = root.join("data/visem/VISEM_Tracking_Train_v4/Train/22/images");
        let mut preds = Vec::new();
        for precision in ["f32", "coreml-fp32"] {
            let out = temp(&format!("visem22-{precision}"));
            run(&args(&[
                images.to_str().unwrap(),
                "--prompt",
                "cell",
                "--out",
                out.to_str().unwrap(),
                "--backend",
                "metal",
                "--max-frames",
                "8",
                "--trunk-precision",
                precision,
            ]))
            .expect("track");
            let run_json =
                ojas_io::parse_json(&std::fs::read_to_string(out.join("22.run.json")).unwrap())
                    .unwrap();
            assert_eq!(run_json.get_str("trunk_precision").unwrap(), precision);
            match precision {
                "f32" => assert!(
                    run_json.field("external_trunk").unwrap().is_null(),
                    "the Ojas trunk reports no external trunk"
                ),
                _ => {
                    let ext = run_json.get_str("external_trunk").unwrap();
                    assert!(
                        ext.starts_with("coreml sam3.1_trunk_fp32.mlpackage (cpu_and_gpu) sha256 "),
                        "{ext}"
                    );
                }
            }
            let text = std::fs::read_to_string(out.join("22.json")).unwrap();
            preds.push(parse_predictions(&text, "22").expect("predictions"));
        }
        for n in 0..8u32 {
            let ids = |i: usize| -> BTreeSet<String> {
                preds[i].frames[&n]
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect()
            };
            println!(
                "frame {n}: ojas {} ids, coreml {} ids",
                ids(0).len(),
                ids(1).len()
            );
            assert_eq!(ids(0), ids(1), "frame {n}: the id sets differ");
        }
        assert!(
            !preds[0].frames[&0].is_empty(),
            "no object in frame 0: the comparison would be vacuous"
        );
    }

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
        assert_eq!(
            p.frames.keys().copied().collect::<Vec<_>>(),
            (0..8).collect::<Vec<u32>>()
        );
        for n in 0..8u32 {
            assert!(
                out.join(format!("masks/22_frame_{n}.json")).is_file(),
                "masks for frame {n}"
            );
        }
        assert!(
            out.join("overlay/22_frame_0.png").is_file()
                && out.join("overlay/22_frame_4.png").is_file()
        );
        let run_json =
            ojas_io::parse_json(&std::fs::read_to_string(out.join("22.run.json")).unwrap())
                .unwrap();
        assert_eq!(run_json.get_str("backend").unwrap(), "metal");
        assert_eq!(run_json.get_u64("frames").unwrap(), 8);
        assert_eq!(run_json.get_str("boxes").unwrap(), "mask");
        assert_eq!(
            run_json.get_u64("metal_mem_cap_bytes").unwrap(),
            METAL_MEM_CAP
        );

        // Stability: print the id sets, then require that the ids of frame 0 are carried to
        // frame 7 and that a track's written mask (its centroid) moves less than 0.1 of the frame
        // per frame. Not the box: upstream's box is the extent of the mask *before* the
        // non-overlap rule (sam3_multiplex_tracking.py:729-746), so a few raw-mask pixels on a
        // neighbouring cell that the rule then gives to that cell stretch the box while the
        // track's own mask stays put (VISEM 22 id 3, frame 2: bench/results/track/box-convention).
        let ids = |n: u32| -> BTreeSet<String> {
            p.frames[&n].iter().map(|(id, _)| id.clone()).collect()
        };
        for n in 0..8 {
            println!("frame {n}: {} objects", p.frames[&n].len());
        }
        assert!(
            !ids(0).is_empty(),
            "no object in frame 0: the prompt found nothing"
        );
        let kept = ids(0).intersection(&ids(7)).count();
        println!(
            "ids in frame 0: {}, still present in frame 7: {kept}",
            ids(0).len()
        );
        assert!(
            kept * 10 >= ids(0).len() * 9,
            "only {kept} of {} ids of frame 0 reach frame 7",
            ids(0).len()
        );
        // id -> normalised centroid of its written mask, per frame.
        let centroids = |n: u32| -> BTreeMap<i64, (f64, f64)> {
            let v = ojas_io::parse_json(
                &std::fs::read_to_string(out.join(format!("masks/22_frame_{n}.json"))).unwrap(),
            )
            .unwrap();
            let size = v.get_array("size").unwrap();
            let (h, w) = (
                size[0].as_u64().unwrap() as usize,
                size[1].as_u64().unwrap() as usize,
            );
            let mut c = BTreeMap::new();
            for o in v.get_array("objects").unwrap() {
                let counts: Vec<u64> = o
                    .get_array("counts")
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap())
                    .collect();
                let bits = Rle {
                    height: h,
                    width: w,
                    area: o.get_u64("area").unwrap(),
                    counts,
                }
                .decode();
                let (mut sx, mut sy, mut k) = (0f64, 0f64, 0f64);
                for (i, _) in bits.iter().enumerate().filter(|(_, &b)| b) {
                    sx += (i % w) as f64;
                    sy += (i / w) as f64;
                    k += 1.0;
                }
                if k > 0.0 {
                    c.insert(
                        o.get_i64("id").unwrap(),
                        (sx / k / w as f64, sy / k / h as f64),
                    );
                }
            }
            c
        };
        for n in 0..7u32 {
            let (a, b) = (centroids(n), centroids(n + 1));
            for (id, c0) in &a {
                if let Some(c1) = b.get(id) {
                    assert!(
                        (c0.0 - c1.0).abs() < 0.1 && (c0.1 - c1.1).abs() < 0.1,
                        "id {id}'s mask jumps between frames {n} and {}: {c0:?} -> {c1:?}",
                        n + 1
                    );
                }
            }
        }
        let visem_str = visem.to_str().unwrap();
        let e = evaluate_window(
            &out,
            &["22".to_string()],
            Path::new(visem_str),
            Window {
                start: Some(0),
                max_frames: Some(8),
            },
        )
        .expect("eval-visem");
        println!(
            "HOTA on frames 0..8 of video 22: {:.4}",
            e.videos[0].hota.summary().hota
        );
        let _ = std::fs::remove_dir_all(&out);
    }

    /// VISEM video 22, frames 0..24, on Metal, with `--overlay-video` and `--overlay 8`: the video
    /// has all 24 frames (more than the 15-frame release delay, so frames are released both while
    /// the run goes and by the final flush) at 640x480 and the folder default of 49 fps, is
    /// readable by AVFoundation, shows the overlay (frame 0 differs from the bare JPEG where a mask
    /// is) and agrees with the PNG overlay of frame 8 up to H.264's loss; `<video>.run.json`
    /// records it and the other outputs are still written. Needs the gated checkpoint, the
    /// tokenizer files, `data/visem` and a Metal GPU; heavy, so under the lock.
    ///
    /// `SPIKETRACK_WEIGHTS=... SPIKETRACK_MODEL_DIR=... tools/locked.sh track-overlay-video cargo test --release -j 2 -p spiketrack track_cmd::tests::visem_22_overlay_video -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "needs the gated checkpoint, tokenizer files, data/visem and a Metal GPU"]
    fn visem_22_overlay_video_on_metal() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let visem = root.join("data/visem/VISEM_Tracking_Train_v4/Train");
        let out = temp("visem22-video");
        let video = out.join("22-overlay.mp4");
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
            "24",
            "--overlay",
            "8",
            "--overlay-video",
            video.to_str().unwrap(),
        ]);
        run(&a).expect("track");

        let mut reader =
            VideoReader::open(&video).expect("the overlay video opens in AVFoundation");
        let info = reader.info().clone();
        assert_eq!((info.width, info.height, info.frame_count), (640, 480, 24));
        assert!(
            (info.fps - FOLDER_OVERLAY_FPS).abs() < 0.5,
            "fps {}",
            info.fps
        );
        let frames: Vec<Rgb8> =
            std::iter::from_fn(|| reader.next_frame().expect("decode").map(|f| f.2)).collect();
        assert_eq!(frames.len(), 24);

        // The overlay is in the video: frame 0 differs from the bare JPEG by more than codec noise somewhere.
        let bare = decode_image_with_source(visem.join("22/images/22_frame_0.jpg"))
            .expect("jpeg")
            .0;
        let changed = frames[0]
            .data
            .iter()
            .zip(&bare.data)
            .filter(|(a, b)| a.abs_diff(**b) > 30)
            .count();
        println!(
            "frame 0: {changed} of {} samples differ from the bare frame by more than 30 levels",
            bare.data.len()
        );
        assert!(
            changed > 100,
            "frame 0 of the video shows no overlay ({changed} samples changed)"
        );

        // The video's frame 8 is the PNG overlay of frame 8, up to H.264 loss: the mean absolute difference is small.
        let png = decode_image_with_source(out.join("overlay/22_frame_8.png"))
            .expect("png overlay")
            .0;
        let mean = frames[8]
            .data
            .iter()
            .zip(&png.data)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / png.data.len() as f64;
        println!(
            "video frame 8 against overlay/22_frame_8.png: mean abs difference {mean:.2} levels"
        );
        assert!(mean < 4.0, "mean abs difference {mean}");

        let run_json =
            ojas_io::parse_json(&std::fs::read_to_string(out.join("22.run.json")).unwrap())
                .unwrap();
        assert_eq!(
            run_json.get_str("overlay_video").unwrap(),
            std::fs::canonicalize(&video).unwrap().to_str().unwrap()
        );
        assert_eq!(run_json.get_u64("overlay_video_frames").unwrap(), 24);
        assert_eq!(run_json.get_f64("overlay_video_fps").unwrap(), 49.0);
        assert!(run_json
            .get_str("overlay_video_fps_source")
            .unwrap()
            .contains("folder of frames"));
        for n in 0..24u32 {
            assert!(
                out.join(format!("masks/22_frame_{n}.json")).is_file(),
                "masks for frame {n}"
            );
        }
        assert!(out.join("22.json").is_file());
        let _ = std::fs::remove_dir_all(&out);
    }

    /// The same 8-frame Session on {upstream's f16 frames from the oracle, frames the CLI decodes
    /// (`decode_image_with_source`) through `upstream_input`} x {CPU, Metal}, printing every
    /// yielded id's mask area per frame and how many input samples the two frame sources differ
    /// on. With the Pillow-exact JPEG decoder the two sources must be identical (0 samples) and give
    /// the same births; the committed log predates it (ImageIO decode, 7-9% of samples differed). Written to bisect the VISEM 22
    /// "balloon" (a track whose box grows to 0.42 x 0.70 in the CLI run). Its result
    /// (bench/results/track/balloon-bisect.log): no mask grows in any configuration, CPU and Metal
    /// agree per frame source, and the frame sources give different frame-0 births. The balloon
    /// was the box, not the mask: upstream boxes the mask before the non-overlap rule, so raw-mask
    /// pixels on a neighbouring cell stretch it (bench/results/track/box-convention).
    #[test]
    #[ignore = "needs a Metal GPU, the gated checkpoint, tokenizer files and fixtures/detect_track/22.safetensors"]
    fn bisect_visem_22_mask_balloon() {
        const N: usize = 8;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let weights = std::env::var("SPIKETRACK_WEIGHTS").expect("set SPIKETRACK_WEIGHTS");
        let model_dir =
            PathBuf::from(std::env::var("SPIKETRACK_MODEL_DIR").expect("set SPIKETRACK_MODEL_DIR"));
        let (w, _) = weights::load(Path::new(&weights), None).expect("weights");
        let fx = ojas_io::SafeTensors::open(&root.join("fixtures/detect_track/22.safetensors"))
            .expect("oracle fixture");
        let (_, oracle) = fx.read_f32_widened("frames").expect("frames");
        let px = 3 * 1008 * 1008;
        let imageio: Vec<Vec<f32>> = (0..N)
            .map(|t| {
                upstream_input(
                    &decode_image_with_source(root.join(format!(
                        "data/visem/VISEM_Tracking_Train_v4/Train/22/images/22_frame_{t}.jpg"
                    )))
                    .map(|d| d.0)
                    .expect("decode"),
                    1008,
                )
            })
            .collect();
        for t in 0..N {
            let d = oracle[t * px..(t + 1) * px]
                .iter()
                .zip(&imageio[t])
                .filter(|(a, b)| a != b)
                .count();
            println!("frame {t}: CLI-decoded input differs from upstream's on {d} of {px} samples");
        }
        let (h, wd) = (480, 640);

        fn areas<B: Backend>(
            be: &B,
            w: &Sam31Weights,
            model_dir: &Path,
            frames: &dyn Fn(usize) -> Vec<f32>,
            label: &str,
        ) -> Vec<Vec<(i64, u32)>> {
            let backbone = Backbone::new(be, &w.vision).expect("backbone");
            let text = Text::new(be, &w.text).expect("text");
            let det = Detector::new(
                be,
                DetectorWeights {
                    transformer: &w.detector_transformer,
                    geometry: &w.geometry_encoder,
                    segmentation: &w.segmentation_head,
                    scoring: &w.dot_prod_scoring,
                },
            )
            .expect("detector");
            let vos = Vos::new(be, &backbone, &w.tracker).expect("tracker");
            let tok = Tokenizer::from_hf_files(
                &model_dir.join("vocab.json"),
                &model_dir.join("merges.txt"),
            )
            .expect("tokenizer");
            let prompt = Session::encode_prompt(be, &text, &tok, "cell").expect("prompt");
            let mut session = Session::new(&vos, &det, prompt, SessionConfig::new(N, 480, 640));
            let mut out = vec![Vec::new(); N];
            let mut take = |ys: Vec<YieldedFrame>| {
                for y in ys {
                    out[y.frame] = y
                        .obj_ids
                        .iter()
                        .zip(&y.masks)
                        .map(|(&id, m)| (id, m.area()))
                        .collect();
                }
            };
            for t in 0..N {
                let x = be
                    .upload(
                        &Tensor::from_f32(&frames(t), &[1, 3, 1008, 1008], be.budget()).expect("t"),
                    )
                    .expect("up");
                take(session.push_frame(&x).expect("push"));
            }
            take(session.finish());
            println!("[{label}] done");
            out
        }
        let oracle_frame = |t: usize| oracle[t * px..(t + 1) * px].to_vec();
        let imageio_frame = |t: usize| imageio[t].clone();
        let cpu = || CpuBackend::with_threads(Budget::new(28 << 30), 8).expect("cpu");
        let metal = || MetalBackend::new(Budget::new(28 << 30)).expect("metal");
        let runs = [
            (
                "upstream frames, CPU",
                areas(
                    &cpu(),
                    &w,
                    &model_dir,
                    &oracle_frame,
                    "upstream frames, CPU",
                ),
            ),
            (
                "upstream frames, Metal",
                areas(
                    &metal(),
                    &w,
                    &model_dir,
                    &oracle_frame,
                    "upstream frames, Metal",
                ),
            ),
            (
                "CLI-decoded frames, Metal",
                areas(
                    &metal(),
                    &w,
                    &model_dir,
                    &imageio_frame,
                    "CLI-decoded frames, Metal",
                ),
            ),
            (
                "CLI-decoded frames, CPU",
                areas(
                    &cpu(),
                    &w,
                    &model_dir,
                    &imageio_frame,
                    "CLI-decoded frames, CPU",
                ),
            ),
        ];
        let px_frame = (h * wd) as f64;
        for (label, a) in &runs {
            println!(
                "\n[{label}] per frame: id:area (largest area / frame {:.0} px)",
                px_frame
            );
            for (t, row) in a.iter().enumerate() {
                let max = row.iter().map(|&(_, ar)| ar).max().unwrap_or(0);
                println!(
                    "frame {t}: {} objects, max area {max} ({:.2}% of frame): {:?}",
                    row.len(),
                    100.0 * max as f64 / px_frame,
                    row
                );
            }
        }
    }
}
