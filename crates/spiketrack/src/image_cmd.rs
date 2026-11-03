//! `spiketrack image <img> --prompt <text>`: one image, one text prompt, the SAM 3.1
//! detector, and the result on disk.
//!
//! Pipeline (the same wiring as `crates/st-model/tests/d3_gate.rs`, which is the
//! gated end-to-end image test): decode (`st_video::decode_image_with_source`: JPEG by the
//! Pillow-exact native decoder, other images by ImageIO) ->
//! `st_video::sam_input` (Pillow-exact bilinear resize to 1008 x 1008, /255,
//! mean 0.5, std 0.5) -> `Backbone::trunk` -> `Backbone::neck(Neck::Detector)`;
//! the prompt goes through `Tokenizer` and `Text`, and `Detector::forward`
//! yields 200 query boxes, scores and 288 x 288 mask logits. A query is kept when
//! `sigmoid(logit) * sigmoid(presence) > threshold`, as upstream's `Sam3Processor`
//! keeps it.
//!
//! # Preprocessing, per input kind (read this before comparing numbers)
//!
//! * **Image files (this command)**: `sam_input` uses Pillow's bilinear resize.
//!   Upstream's image path (`Sam3Processor`) resizes a uint8 tensor with
//!   `torchvision.transforms.v2.Resize`, which is antialiased bilinear; the two
//!   differ by at most one grey level in 0.1 to 0.4 % of pixels
//!   (`tools/reference/README.md`). A JPEG decodes byte-identically to Pillow (`st_video::jpeg`);
//!   other formats, and the JPEG variants that decoder refuses, go through ImageIO, not
//!   Pillow's libjpeg-turbo; `crates/st-video/tests/decode.rs` measures that.
//! * **A folder of JPEG frames (the future `track` command)**: upstream's
//!   `load_video_frames` resizes with Pillow bilinear, which is what `sam_input` is.
//! * **An mp4 file (the future `track` command)**: upstream decodes with OpenCV
//!   and resizes with `cv2.resize(..., INTER_CUBIC)` after BGR to RGB
//!   (`io_utils.py:313-333`). `st_video::VideoReader` decodes with AVFoundation and
//!   `sam_input` resizes bilinearly, so mp4 input does **not** match upstream's mp4
//!   preprocessing. Tracked as a gap; there is no mp4 command yet.
//!
//! # Output
//!
//! `detections.json`:
//!
//! ```text
//! { "image": <path as given>, "width": W, "height": H, "prompt": <text>,
//!   "threshold": t, "backend": "cpu", "presence_probability": p,
//!   "preprocessing": <one-line description>,
//!   "lora": null | {"adapter", "targets", "tensors", "base_sha256"},   // what --lora folded in
//!   "weight_edit": null | {"manifest", "description", "edits", "tensors", "values_changed", "total_values", "fraction_changed"},
//!   "detections": [            // kept queries, highest score first
//!     { "query": 0..199,       // index into the detector's 200 queries
//!       "score": s,            // sigmoid(logit) * sigmoid(presence)
//!       "box_xyxy_norm": [x0, y0, x1, y1],   // fractions of the image, as predicted (not clipped)
//!       "box_xyxy_px":   [x0, y0, x1, y1],   // the same times (W, H, W, H)
//!       "mask_area": n,                      // set pixels at W x H
//!       "mask": { "size": [H, W], "counts": [..] } }  // COCO uncompressed RLE, column-major
//!   ] }
//! ```
//!
//! The mask is the 288 x 288 logit map resized to H x W with plain bilinear
//! interpolation (`align_corners=False`, no antialiasing: what upstream does, see
//! `mask.rs`) and thresholded at logit 0. `overlay.png` is the image with the
//! masks tinted and the boxes outlined in the order of the `detections` array.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use st_model::detector::{Detector, DetectorOut, DetectorWeights, D, MASK, QUERIES};
use st_model::tile::{collect_tile_candidates, TileSpec, DEFAULT_SEAM_MARGIN, DEFAULT_TILE_SPAN};
use st_model::tracker::associate::{nms_box_and_iom, Bitmask};
use st_model::{Backbone, Neck, Text, Tokenizer};
use st_video::{decode_image_with_source, sam_input, Rgb8};
use st_weights::Sam31Weights;

use crate::cli::{existing_path, Args, CliError};
use crate::json;
use crate::mask::{ResizedMask, Rle};
use crate::overlay::{self, Drawn};
use crate::png;
use crate::weights::{self, edit_json, lora_json, Applied, AppliedEdit};

/// Where the model runs: Ojas's CPU backend, or its Metal backend (on tessl). Both pass
/// the same upstream parity gates (bench/results/module-gates).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Cpu,
    Metal,
}

impl BackendChoice {
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            other => Err(CliError::usage(format!(
                "--backend: {other:?} is not available (choices: cpu, metal)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileMode {
    None,
    Grid2x2,
}

impl TileMode {
    pub fn parse(s: &str) -> Result<Self, CliError> {
        match s {
            "none" => Ok(Self::None),
            "2x2" => Ok(Self::Grid2x2),
            other => Err(CliError::usage(format!(
                "--tile: {other:?} is not available (choices: none, 2x2)"
            ))),
        }
    }
}

/// Parse `--post-box-delta dx,dy,dw,dh` as four finite logit shifts in `[-4.0, 4.0]`.
pub fn parse_post_box_delta(s: &str) -> Result<[f32; 4], CliError> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err(CliError::usage(format!(
            "--post-box-delta {s:?} must be four comma-separated numbers dx,dy,dw,dh"
        )));
    }
    let mut out = [0.0f32; 4];
    for (dst, p) in out.iter_mut().zip(parts) {
        let v: f32 = p.trim().parse().map_err(|_| {
            CliError::usage(format!("--post-box-delta {s:?}: {p:?} is not a float"))
        })?;
        if !v.is_finite() || !(-4.0..=4.0).contains(&v) {
            return Err(CliError::usage(format!(
                "--post-box-delta {s:?}: {v} must be finite in [-4.0, 4.0]"
            )));
        }
        *dst = v;
    }
    Ok(out)
}

/// Parse `--aux-score-weight W` as a finite blend weight in `[0.0, 1.0]`.
pub fn parse_aux_score_weight(s: &str) -> Result<f32, CliError> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| CliError::usage(format!("--aux-score-weight {s:?} is not a float")))?;
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(CliError::usage(format!(
            "--aux-score-weight {v} is outside [0.0, 1.0]"
        )));
    }
    Ok(v)
}

/// Crop an `Rgb8` sub-rectangle matching `spec`'s exact pixel coordinates.
pub fn crop_tile_rgb8(src: &Rgb8, spec: &TileSpec) -> Result<Rgb8, String> {
    if src.width != spec.full_w || src.height != spec.full_h {
        return Err(format!(
            "tile spec {}x{} does not match image {}x{}",
            spec.full_w, spec.full_h, src.width, src.height
        ));
    }
    if spec.pw == 0
        || spec.ph == 0
        || spec.px0 + spec.pw > src.width
        || spec.py0 + spec.ph > src.height
    {
        return Err(format!(
            "tile [{}..{}, {}..{}] out of bounds for {}x{}",
            spec.px0,
            spec.px0 + spec.pw,
            spec.py0,
            spec.py0 + spec.ph,
            src.width,
            src.height
        ));
    }
    let mut data = Vec::with_capacity(spec.pw * spec.ph * 3);
    for r in spec.py0..(spec.py0 + spec.ph) {
        let row_start = (r * src.width + spec.px0) * 3;
        data.extend_from_slice(&src.data[row_start..row_start + spec.pw * 3]);
    }
    Ok(Rgb8::new(spec.pw, spec.ph, data))
}

/// The footprint the backend may use, as in the gated tests.
const BUDGET_BYTES: u64 = 28 << 30;
const PREPROCESSING: &str = "decode to RGB8 (see \"decoder\": pillow-exact-jpeg is byte-identical to Pillow 13 / libjpeg-turbo 3.2.0, imageio otherwise); Pillow-exact bilinear resize to 1008x1008 (st_video::sam_input); mean 0.5, std 0.5";

pub const USAGE: &str = "\
spiketrack image <img> --prompt <text> [--out DIR] [--threshold 0.5] [--backend metal|cpu]
                 [--weights FILE] [--lora FILE] [--weight-edit FILE] [--model-dir DIR] [--threads N]
                 [--post-box-delta dx,dy,dw,dh] [--aux-score-weight W] [--tile none|2x2]

  Detects every instance of <text> in <img> with SAM 3.1 and writes
  DIR/detections.json (score, box, COCO RLE mask per kept query) and DIR/overlay.png.

  --out DIR         output directory, created if missing (default ./spiketrack-out)
  --threshold T     keep queries with probability > T, 0 <= T < 1 (default 0.5)
  --backend B       metal (Ojas on tessl, default) or cpu
  --weights FILE    the converted checkpoint (default: $SPIKETRACK_WEIGHTS)
  --lora FILE       fold this detector LoRA adapter (spiketrack-lora-v1) into the weights
  --weight-edit FILE  apply this surgical weight-edit manifest (spiketrack-weight-edit-v1)
  --model-dir DIR   folder with the tokenizer's vocab.json and merges.txt (default: $SPIKETRACK_MODEL_DIR)
  --threads N       CPU worker threads for --backend cpu, 1 to 256 (default 8)
  --post-box-delta D  post-decoder [dx,dy,dw,dh] logit shift on pred_boxes, decoupled from boxRPB
  --aux-score-weight W  blend weight in [0, 1] mixing decoder layer 4 scores into layer 5 (default 0)
  --tile MODE       none (default) or 2x2 overlapping 1008x1008 sub-tiles merged by cross-scale NMS

  Preprocessing: JPEGs are decoded byte-identically to Pillow (other images by ImageIO) and
  resized with Pillow-style
  bilinear, as upstream's JPEG-folder path does (upstream's single-image path uses
  torchvision v2.Resize, within one grey level). mp4 input (not in this command) would not
  match upstream, which decodes mp4 with OpenCV and resizes with INTER_CUBIC.";

/// One kept query.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub query: usize,
    pub score: f32,
    pub box_norm: [f32; 4],
    pub box_px: [f64; 4],
    pub rle: Rle,
}

/// The detector outputs this command reads, on the host.
pub struct RawDetections<'a> {
    /// `[QUERIES]`: `sigmoid(logit) * sigmoid(presence)`.
    pub probabilities: &'a [f32],
    /// `[QUERIES, 4]` cxcywh in [0, 1].
    pub boxes_cxcywh: &'a [f32],
    /// `[QUERIES, mask * mask]` logits.
    pub masks: &'a [f32],
    pub mask_side: usize,
}

/// Keep the queries above `threshold`, highest score first, with boxes in
/// normalised and pixel xyxy and masks resized to `height x width` and run-length coded.
pub fn select(
    raw: &RawDetections<'_>,
    threshold: f32,
    height: usize,
    width: usize,
) -> Result<Vec<Detection>, String> {
    let n = raw.probabilities.len();
    let plane = raw.mask_side * raw.mask_side;
    if raw.boxes_cxcywh.len() != n * 4 || raw.masks.len() != n * plane {
        return Err(format!(
            "detector outputs disagree: {n} probabilities, {} box values, {} mask values",
            raw.boxes_cxcywh.len(),
            raw.masks.len()
        ));
    }
    // A NaN probability compares false and is never kept.
    let mut kept: Vec<usize> = (0..n)
        .filter(|&q| raw.probabilities[q] > threshold)
        .collect();
    kept.sort_by(|&a, &b| {
        raw.probabilities[b]
            .total_cmp(&raw.probabilities[a])
            .then(a.cmp(&b))
    });
    Ok(kept
        .into_iter()
        .map(|q| {
            let b = &raw.boxes_cxcywh[q * 4..q * 4 + 4];
            let norm = [
                b[0] - b[2] / 2.0,
                b[1] - b[3] / 2.0,
                b[0] + b[2] / 2.0,
                b[1] + b[3] / 2.0,
            ];
            let (w, h) = (width as f64, height as f64);
            let px = [
                f64::from(norm[0]) * w,
                f64::from(norm[1]) * h,
                f64::from(norm[2]) * w,
                f64::from(norm[3]) * h,
            ];
            let rle = ResizedMask::new(
                &raw.masks[q * plane..(q + 1) * plane],
                raw.mask_side,
                raw.mask_side,
                height,
                width,
            )
            .rle();
            Detection {
                query: q,
                score: raw.probabilities[q],
                box_norm: norm,
                box_px: px,
                rle,
            }
        })
        .collect())
}

pub struct Report<'a> {
    pub image: &'a str,
    pub width: usize,
    pub height: usize,
    pub prompt: &'a str,
    pub threshold: f32,
    pub backend: BackendChoice,
    pub presence: f32,
    pub decoder: &'a str,
    pub lora: Option<&'a Applied>,
    pub weight_edit: Option<&'a AppliedEdit>,
    pub detections: &'a [Detection],
}

/// The `detections.json` text for a report; an error for any non-finite number.
pub fn detections_json(r: &Report<'_>) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(&format!(
        "{{\n  \"image\": {},\n  \"width\": {},\n  \"height\": {},\n  \"prompt\": {},\n  \"threshold\": {},\n  \"backend\": {},\n  \"presence_probability\": {},\n  \"preprocessing\": {},\n  \"decoder\": {},\n  \"lora\": {},\n  \"weight_edit\": {},\n  \"detections\": [",
        json::string(r.image),
        r.width,
        r.height,
        json::string(r.prompt),
        json::number32(r.threshold, "threshold")?,
        json::string(r.backend.name()),
        json::number32(r.presence, "presence probability")?,
        json::string(PREPROCESSING),
        json::string(r.decoder),
        lora_json(r.lora),
        edit_json(r.weight_edit)?,
    ));
    for (i, d) in r.detections.iter().enumerate() {
        let norm_text: Result<Vec<String>, String> = d
            .box_norm
            .iter()
            .map(|&v| json::number32(v, "box"))
            .collect();
        let counts: Vec<String> = d.rle.counts.iter().map(u64::to_string).collect();
        out.push_str(&format!(
            "{}\n    {{\"query\": {}, \"score\": {}, \"box_xyxy_norm\": [{}], \"box_xyxy_px\": {}, \"mask_area\": {}, \"mask\": {{\"size\": [{}, {}], \"counts\": [{}]}}}}",
            if i == 0 { "" } else { "," },
            d.query,
            json::number32(d.score, "score")?,
            norm_text?.join(", "),
            json::number_array(&d.box_px, "box")?,
            d.rle.area,
            d.rle.height,
            d.rle.width,
            counts.join(", "),
        ));
    }
    out.push_str(if r.detections.is_empty() {
        "]\n}\n"
    } else {
        "\n  ]\n}\n"
    });
    Ok(out)
}

/// Run the model on one decoded image: the detector's raw outputs.
struct Inference {
    probabilities: Vec<f32>,
    presence: f32,
    boxes: Vec<f32>,
    masks: Vec<f32>,
    seconds: f64,
}

#[derive(Clone, Copy)]
struct DetectorOpts {
    post_box_delta: [f32; 4],
    aux_score_weight: f32,
    tile: TileMode,
    threshold: f32,
}

fn infer(
    image: &Rgb8,
    prompt: &str,
    w: Sam31Weights,
    model_dir: &Path,
    backend: BackendChoice,
    threads: usize,
    det_opts: DetectorOpts,
) -> Result<Inference, CliError> {
    match backend {
        BackendChoice::Cpu => infer_on(
            &CpuBackend::with_threads(Budget::new(BUDGET_BYTES), threads)
                .map_err(CliError::from_err)?,
            w,
            image,
            prompt,
            model_dir,
            det_opts,
        ),
        BackendChoice::Metal => infer_on(
            &MetalBackend::new(Budget::new(BUDGET_BYTES)).map_err(CliError::from_err)?,
            w,
            image,
            prompt,
            model_dir,
            det_opts,
        ),
    }
}

fn infer_on<B: Backend>(
    be: &B,
    w: Sam31Weights,
    image: &Rgb8,
    prompt: &str,
    model_dir: &Path,
    det_opts: DetectorOpts,
) -> Result<Inference, CliError> {
    let Sam31Weights {
        vision,
        text: text_w,
        geometry_encoder,
        detector_transformer,
        segmentation_head,
        dot_prod_scoring,
        tracker,
    } = w;
    drop(tracker);
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let tok = Tokenizer::from_hf_files(&vocab, &merges).map_err(|e| {
        CliError::failed(format!("tokenizer files in {}: {e}", model_dir.display()))
    })?;
    let ids = tok
        .tokenize(prompt)
        .map_err(|e| CliError::failed(format!("prompt {prompt:?}: {e}")))?;
    drop(tok);
    let text = Text::new(be, &text_w).map_err(CliError::from_err)?;
    drop(text_w);
    let tf = text.encode(&[ids]).map_err(CliError::from_err)?;
    drop(text);
    let resized = tf
        .resized
        .to_host(be.budget())
        .map_err(CliError::from_err)?;
    let resized_slice = resized.f32_slice().map_err(CliError::from_err)?;
    let valid: Vec<f32> = (0..tf.padding.len())
        .filter(|&t| !tf.padding[t])
        .flat_map(|t| resized_slice[t * D..(t + 1) * D].iter().copied())
        .collect();
    drop((resized, tf));
    if valid.is_empty() {
        return Err(CliError::failed(format!("prompt {prompt:?} has no tokens")));
    }
    let prompt_host = Tensor::from_f32(&valid, &[1, valid.len() / D, D], be.budget())
        .map_err(CliError::from_err)?;
    drop(valid);
    let prompt_t = be.upload(&prompt_host).map_err(CliError::from_err)?;
    drop(prompt_host);

    let backbone = Backbone::new(be, &vision).map_err(CliError::from_err)?;
    drop(vision);
    let mut det = Detector::new(
        be,
        DetectorWeights {
            transformer: &detector_transformer,
            geometry: &geometry_encoder,
            segmentation: &segmentation_head,
            scoring: &dot_prod_scoring,
        },
    )
    .map_err(CliError::from_err)?;
    det.set_post_box_delta(det_opts.post_box_delta)
        .map_err(CliError::usage)?;
    det.set_aux_score_weight(det_opts.aux_score_weight)
        .map_err(CliError::usage)?;
    drop((
        detector_transformer,
        geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
    ));

    let side = st_model::IMAGE_SIZE;
    let input = sam_input(image, side);
    let t =
        Tensor::from_f32(&input, &[1, 3, side, side], be.budget()).map_err(CliError::from_err)?;
    drop(input);
    let device = be.upload(&t).map_err(CliError::from_err)?;
    drop(t);
    let t0 = Instant::now();
    let trunk = backbone
        .trunk_inference(&device)
        .map_err(CliError::from_err)?;
    drop(device);
    let fpn = backbone
        .neck(Neck::Detector, &trunk)
        .map_err(CliError::from_err)?;
    drop(trunk);
    let out = det
        .forward_inference(&fpn, &prompt_t, &mut |_| {})
        .map_err(CliError::from_err)?;
    drop(fpn);

    let mut probabilities = out.probabilities();
    let mut max_presence = st_model::host::sigmoid(out.presence_logit);
    let DetectorOut {
        pred_logits: mut logits,
        pred_boxes: mut boxes,
        pred_masks,
        ..
    } = out;
    let masks_host = pred_masks
        .to_host(be.budget())
        .map_err(CliError::from_err)?;
    drop(pred_masks);
    let mut masks = masks_host.to_f32_vec().map_err(CliError::from_err)?;
    drop(masks_host);

    if det_opts.tile == TileMode::Grid2x2 {
        let specs = TileSpec::grid_2x2(image.width, image.height, DEFAULT_TILE_SPAN)
            .map_err(CliError::failed)?;
        let min_tile_prob = det_opts.threshold.min(0.05);
        for spec in &specs {
            let crop = crop_tile_rgb8(image, spec).map_err(CliError::failed)?;
            let t_in = sam_input(&crop, side);
            let t_host = Tensor::from_f32(&t_in, &[1, 3, side, side], be.budget())
                .map_err(CliError::from_err)?;
            drop((crop, t_in));
            let t_dev = be.upload(&t_host).map_err(CliError::from_err)?;
            drop(t_host);
            let t_trunk = backbone
                .trunk_inference(&t_dev)
                .map_err(CliError::from_err)?;
            drop(t_dev);
            let t_fpn = backbone
                .neck(Neck::Detector, &t_trunk)
                .map_err(CliError::from_err)?;
            drop(t_trunk);
            let t_out = det
                .forward_inference(&t_fpn, &prompt_t, &mut |_| {})
                .map_err(CliError::from_err)?;
            drop(t_fpn);
            max_presence = max_presence.max(st_model::host::sigmoid(t_out.presence_logit));
            let t_masks_host = t_out
                .pred_masks
                .to_host(be.budget())
                .map_err(CliError::from_err)?;
            let t_masks = t_masks_host.f32_slice().map_err(CliError::from_err)?;
            let before = logits.len();
            collect_tile_candidates(
                spec,
                &t_out.pred_logits,
                &t_out.pred_boxes,
                t_masks,
                MASK,
                min_tile_prob,
                DEFAULT_SEAM_MARGIN,
                &mut logits,
                &mut boxes,
                &mut masks,
            );
            drop((t_out.pred_masks, t_masks_host));
            for &l in &logits[before..] {
                probabilities.push(st_model::host::sigmoid(l));
            }
        }
        // Cross-scale duplicate suppression between full-frame and zoomed 2x2 sub-tiles.
        let corners: Vec<[f32; 4]> = boxes
            .chunks_exact(4)
            .map(|b| {
                [
                    b[0] - 0.5 * b[2],
                    b[1] - 0.5 * b[3],
                    b[0] + 0.5 * b[2],
                    b[1] + 0.5 * b[3],
                ]
            })
            .collect();
        let px = MASK * MASK;
        let bits: Vec<Bitmask> = (0..probabilities.len())
            .map(|q| Bitmask::from_logits(&masks[q * px..(q + 1) * px]))
            .collect();
        let nms_keep = nms_box_and_iom(&probabilities, &corners, &bits, min_tile_prob, 0.35, 0.65);
        for (p, keep) in probabilities.iter_mut().zip(nms_keep) {
            if !keep {
                *p = 0.0;
            }
        }
    }

    be.sync().map_err(CliError::from_err)?;
    let seconds = t0.elapsed().as_secs_f64();
    drop((backbone, det, prompt_t));
    Ok(Inference {
        probabilities,
        presence: max_presence,
        boxes,
        masks,
        seconds,
    })
}

pub fn run(raw: &[String]) -> Result<(), CliError> {
    let args = Args::parse(
        raw,
        &[
            "prompt",
            "out",
            "threshold",
            "backend",
            "weights",
            "lora",
            "weight-edit",
            "model-dir",
            "threads",
            "post-box-delta",
            "aux-score-weight",
            "tile",
        ],
        1,
    )?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }
    let image_path = PathBuf::from(&args.positional[0]);
    let prompt = args.require("prompt")?.trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("--prompt is blank"));
    }
    let threshold: f32 = args.parsed("threshold", 0.5)?;
    if !(0.0..1.0).contains(&threshold) {
        return Err(CliError::usage(format!(
            "--threshold {threshold} is outside [0, 1)"
        )));
    }
    let backend = BackendChoice::parse(args.get("backend").unwrap_or("metal"))?;
    let threads: usize = args.parsed("threads", 8)?;
    if !(1..=256).contains(&threads) {
        return Err(CliError::usage(format!(
            "--threads {threads} is outside 1..=256"
        )));
    }
    let post_box_delta = args
        .get("post-box-delta")
        .map(parse_post_box_delta)
        .transpose()?
        .unwrap_or([0.0; 4]);
    let aux_score_weight = args
        .get("aux-score-weight")
        .map(parse_aux_score_weight)
        .transpose()?
        .unwrap_or(0.0);
    let tile = TileMode::parse(args.get("tile").unwrap_or("none"))?;
    let out_dir = PathBuf::from(args.get("out").unwrap_or("spiketrack-out"));
    let weights = existing_path(
        args.get("weights"),
        "SPIKETRACK_WEIGHTS",
        "weights",
        "weights",
    )?;
    let lora = weights::lora_path(args.get("lora"))?;
    let weight_edit = weights::edit_path(args.get("weight-edit"))?;
    let model_dir = existing_path(
        args.get("model-dir"),
        "SPIKETRACK_MODEL_DIR",
        "tokenizer folder",
        "model-dir",
    )?;

    let (image, decoder) = decode_image_with_source(&image_path).map_err(CliError::from_err)?;
    let (w, applied, applied_edit) =
        weights::load(&weights, lora.as_deref(), weight_edit.as_deref())?;
    let det_opts = DetectorOpts {
        post_box_delta,
        aux_score_weight,
        tile,
        threshold,
    };
    let inf = infer(&image, &prompt, w, &model_dir, backend, threads, det_opts)?;
    let n = inf.probabilities.len();
    let expected_ok = match tile {
        TileMode::None => n == QUERIES,
        TileMode::Grid2x2 => n >= QUERIES,
    };
    if !expected_ok || inf.masks.len() != n * MASK * MASK {
        return Err(CliError::failed(format!(
            "the detector returned {n} queries and {} mask values, expected >= {QUERIES} and {}",
            inf.masks.len(),
            n * MASK * MASK
        )));
    }
    let raw = RawDetections {
        probabilities: &inf.probabilities,
        boxes_cxcywh: &inf.boxes,
        masks: &inf.masks,
        mask_side: MASK,
    };
    let detections =
        select(&raw, threshold, image.height, image.width).map_err(CliError::failed)?;

    let report = Report {
        image: &image_path.to_string_lossy(),
        width: image.width,
        height: image.height,
        prompt: &prompt,
        threshold,
        backend,
        presence: inf.presence,
        decoder: decoder.name(),
        lora: applied.as_ref(),
        weight_edit: applied_edit.as_ref(),
        detections: &detections,
    };
    let text = detections_json(&report).map_err(CliError::failed)?;
    let masks: Vec<Vec<bool>> = detections.iter().map(|d| d.rle.decode()).collect();
    let drawn: Vec<Drawn<'_>> = detections
        .iter()
        .zip(&masks)
        .enumerate()
        .map(|(i, (d, m))| Drawn {
            mask: m,
            box_px: d.box_px,
            colour: i,
            label: None,
        })
        .collect();
    let overlay_png = png::encode_rgb(
        image.width,
        image.height,
        &overlay::render(&image, &drawn).data,
    )
    .map_err(CliError::from_err)?;

    std::fs::create_dir_all(&out_dir)
        .map_err(|e| CliError::failed(format!("{}: {e}", out_dir.display())))?;
    let json_path = out_dir.join("detections.json");
    let png_path = out_dir.join("overlay.png");
    std::fs::write(&json_path, text)
        .map_err(|e| CliError::failed(format!("{}: {e}", json_path.display())))?;
    std::fs::write(&png_path, overlay_png)
        .map_err(|e| CliError::failed(format!("{}: {e}", png_path.display())))?;
    println!(
        "{} detection(s) for {prompt:?} above {threshold} in {} ({}x{}), {:.1} s on {}\n  {}\n  {}",
        detections.len(),
        image_path.display(),
        image.width,
        image.height,
        inf.seconds,
        backend.name(),
        json_path.display(),
        png_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_io::parse_json;

    /// Three queries on a 2x2 mask plane: probabilities 0.2, 0.9, 0.7.
    fn raw_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let probs = vec![0.2, 0.9, 0.7];
        let boxes = vec![0.5, 0.5, 0.2, 0.2, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 1.0];
        let masks = vec![
            -1.0, -1.0, -1.0, -1.0, // q0: empty (not kept)
            1.0, -1.0, -1.0, -1.0, // q1: top-left pixel
            1.0, 1.0, 1.0, 1.0, // q2: full
        ];
        (probs, boxes, masks)
    }

    #[test]
    fn select_keeps_above_threshold_sorted_by_score_and_scales_boxes() {
        let (probs, boxes, masks) = raw_fixture();
        let raw = RawDetections {
            probabilities: &probs,
            boxes_cxcywh: &boxes,
            masks: &masks,
            mask_side: 2,
        };
        let d = select(&raw, 0.5, 4, 8).unwrap();
        assert_eq!(d.iter().map(|d| d.query).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(d[0].box_norm, [0.25, 0.0, 0.75, 0.5]);
        assert_eq!(d[0].box_px, [2.0, 0.0, 6.0, 2.0]);
        assert_eq!(d[1].box_px, [0.0, 0.0, 4.0, 4.0]);
        assert_eq!(d[1].rle.counts, vec![0, 32], "a full mask at 4 x 8");
        assert_eq!((d[0].rle.height, d[0].rle.width), (4, 8));
        // Strictly above: a query exactly at the threshold is dropped.
        assert_eq!(select(&raw, 0.7, 4, 8).unwrap().len(), 1);
    }

    #[test]
    fn select_refuses_inconsistent_outputs() {
        let (probs, boxes, masks) = raw_fixture();
        let short = RawDetections {
            probabilities: &probs,
            boxes_cxcywh: &boxes[..8],
            masks: &masks,
            mask_side: 2,
        };
        assert!(select(&short, 0.5, 4, 4).is_err());
        let nan = [f32::NAN, 0.9, 0.7];
        let bad = RawDetections {
            probabilities: &nan,
            boxes_cxcywh: &boxes,
            masks: &masks,
            mask_side: 2,
        };
        // NaN > threshold is false, so it is not kept and cannot poison the output.
        assert_eq!(select(&bad, 0.5, 4, 4).unwrap().len(), 2);
    }

    #[test]
    fn detections_json_parses_and_carries_every_field() {
        let (probs, boxes, masks) = raw_fixture();
        let raw = RawDetections {
            probabilities: &probs,
            boxes_cxcywh: &boxes,
            masks: &masks,
            mask_side: 2,
        };
        let d = select(&raw, 0.5, 4, 8).unwrap();
        let edit = AppliedEdit {
            path: PathBuf::from("edit.json"),
            description: Some("surgical".into()),
            edits: 1,
            tensors: 1,
            values_changed: 1024,
            total_values: 850_000_000,
        };
        let text = detections_json(&Report {
            image: "a \"b\".jpg",
            width: 8,
            height: 4,
            prompt: "cell",
            threshold: 0.5,
            backend: BackendChoice::Cpu,
            presence: 0.99,
            decoder: "pillow-exact-jpeg",
            lora: None,
            weight_edit: Some(&edit),
            detections: &d,
        })
        .unwrap();
        let v = parse_json(&text).expect("valid JSON");
        assert_eq!(v.get_str("image").unwrap(), "a \"b\".jpg");
        assert_eq!(v.get_str("backend").unwrap(), "cpu");
        assert_eq!(v.get_str("decoder").unwrap(), "pillow-exact-jpeg");
        assert_eq!(v.get_u64("width").unwrap(), 8);
        assert_eq!(
            v.field("weight_edit")
                .unwrap()
                .get_u64("values_changed")
                .unwrap(),
            1024
        );
        let dets = v.get_array("detections").unwrap();
        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].get_u64("query").unwrap(), 1);
        assert!((dets[0].get_f64("score").unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(dets[0].get_array("box_xyxy_px").unwrap().len(), 4);
        let mask = dets[1].get_object("mask").unwrap();
        let counts: Vec<u64> = mask
            .get_array("counts")
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap())
            .collect();
        assert_eq!(counts, vec![0, 32]);
        assert_eq!(mask.get_array("size").unwrap().len(), 2);
        assert_eq!(dets[1].get_u64("mask_area").unwrap(), 32);
        // No detections is still valid JSON with an empty list.
        let empty = detections_json(&Report {
            image: "x",
            width: 1,
            height: 1,
            prompt: "p",
            threshold: 0.5,
            backend: BackendChoice::Cpu,
            presence: 0.0,
            decoder: "imageio",
            lora: None,
            weight_edit: None,
            detections: &[],
        })
        .unwrap();
        let empty_v = parse_json(&empty).unwrap();
        assert!(empty_v.field("weight_edit").unwrap().is_null());
        assert_eq!(empty_v.get_array("detections").unwrap().len(), 0);
    }

    #[test]
    fn a_non_finite_number_is_an_error_not_a_null() {
        let (probs, boxes, masks) = raw_fixture();
        let raw = RawDetections {
            probabilities: &probs,
            boxes_cxcywh: &boxes,
            masks: &masks,
            mask_side: 2,
        };
        let d = select(&raw, 0.5, 4, 8).unwrap();
        let r = Report {
            image: "x",
            width: 8,
            height: 4,
            prompt: "p",
            threshold: 0.5,
            backend: BackendChoice::Cpu,
            presence: f32::NAN,
            decoder: "imageio",
            lora: None,
            weight_edit: None,
            detections: &d,
        };
        assert!(detections_json(&r).unwrap_err().contains("presence"));
    }

    #[test]
    fn backends_are_cpu_and_metal() {
        assert_eq!(BackendChoice::parse("cpu").unwrap(), BackendChoice::Cpu);
        assert_eq!(BackendChoice::parse("metal").unwrap(), BackendChoice::Metal);
        assert!(BackendChoice::parse("cuda").is_err());
    }

    #[test]
    fn detector_flag_parsers_and_tile_cropping_validate_inputs() {
        assert_eq!(
            parse_post_box_delta("0, -0.05, 0.1, 0.15").unwrap(),
            [0.0, -0.05, 0.1, 0.15]
        );
        assert!(parse_post_box_delta("0,0,0").is_err());
        assert!(parse_post_box_delta("0,0,0,NaN").is_err());
        assert!(parse_post_box_delta("0,0,5.0,0").is_err());

        assert_eq!(parse_aux_score_weight("0.25").unwrap(), 0.25);
        assert!(parse_aux_score_weight("-0.1").is_err());
        assert!(parse_aux_score_weight("1.1").is_err());
        assert!(parse_aux_score_weight("NaN").is_err());

        assert_eq!(TileMode::parse("none").unwrap(), TileMode::None);
        assert_eq!(TileMode::parse("2x2").unwrap(), TileMode::Grid2x2);
        assert!(TileMode::parse("3x3").is_err());

        // Odd-sized RGB image (7x5): verify pixel-quantized TileSpec matches cropped sub-image exactly.
        let mut data = Vec::with_capacity(7 * 5 * 3);
        for y in 0..5u8 {
            for x in 0..7u8 {
                data.extend_from_slice(&[x, y, x + y]);
            }
        }
        let img = Rgb8::new(7, 5, data);
        let grid = TileSpec::grid_2x2(7, 5, DEFAULT_TILE_SPAN).unwrap();
        let sub_br = crop_tile_rgb8(&img, &grid[3]).unwrap();
        assert_eq!((sub_br.width, sub_br.height), (4, 3));
        assert_eq!((grid[3].px0, grid[3].py0), (3, 2));
        assert!((grid[3].norm_xyxy[0] - 3.0 / 7.0).abs() < 1e-6);
        assert!((grid[3].norm_xyxy[1] - 2.0 / 5.0).abs() < 1e-6);
        assert_eq!(&sub_br.data[0..3], &[3, 2, 5]);
    }
}
