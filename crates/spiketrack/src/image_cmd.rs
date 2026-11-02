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
use st_model::detector::{Detector, DetectorTaps, DetectorWeights, D, MASK, QUERIES};
use st_model::{Backbone, Neck, Taps, Text, Tokenizer};
use st_video::{decode_image_with_source, sam_input, Rgb8};
use st_weights::Sam31Weights;

use crate::cli::{existing_path, Args, CliError};
use crate::json;
use crate::mask::{ResizedMask, Rle};
use crate::overlay::{self, Drawn};
use crate::png;
use crate::weights::{self, lora_json, Applied};

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

/// The footprint the backend may use, as in the gated tests.
const BUDGET_BYTES: u64 = 28 << 30;
const PREPROCESSING: &str = "decode to RGB8 (see \"decoder\": pillow-exact-jpeg is byte-identical to Pillow 13 / libjpeg-turbo 3.2.0, imageio otherwise); Pillow-exact bilinear resize to 1008x1008 (st_video::sam_input); mean 0.5, std 0.5";

pub const USAGE: &str = "\
spiketrack image <img> --prompt <text> [--out DIR] [--threshold 0.5] [--backend metal|cpu]
                 [--weights FILE] [--lora FILE] [--model-dir DIR] [--threads N]

  Detects every instance of <text> in <img> with SAM 3.1 and writes
  DIR/detections.json (score, box, COCO RLE mask per kept query) and DIR/overlay.png.

  --out DIR         output directory, created if missing (default ./spiketrack-out)
  --threshold T     keep queries with probability > T, 0 <= T < 1 (default 0.5)
  --backend B       metal (Ojas on tessl, default) or cpu
  --weights FILE    the converted checkpoint (default: $SPIKETRACK_WEIGHTS)
  --lora FILE       fold this detector LoRA adapter (spiketrack-lora-v1) into the weights
  --model-dir DIR   folder with the tokenizer's vocab.json and merges.txt (default: $SPIKETRACK_MODEL_DIR)
  --threads N       CPU worker threads for --backend cpu, 1 to 256 (default 8)

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
    pub detections: &'a [Detection],
}

/// The `detections.json` text for a report; an error for any non-finite number.
pub fn detections_json(r: &Report<'_>) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(&format!(
        "{{\n  \"image\": {},\n  \"width\": {},\n  \"height\": {},\n  \"prompt\": {},\n  \"threshold\": {},\n  \"backend\": {},\n  \"presence_probability\": {},\n  \"preprocessing\": {},\n  \"decoder\": {},\n  \"lora\": {},\n  \"detections\": [",
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

fn host_f32(t: &Tensor, budget: &Budget) -> Result<Vec<f32>, CliError> {
    t.to_host(budget)
        .and_then(|h| h.to_f32_vec())
        .map_err(CliError::from_err)
}

/// Run the model on one decoded image: the detector's raw outputs.
struct Inference {
    probabilities: Vec<f32>,
    presence: f32,
    boxes: Vec<f32>,
    masks: Vec<f32>,
    seconds: f64,
}

fn infer(
    image: &Rgb8,
    prompt: &str,
    w: &Sam31Weights,
    model_dir: &Path,
    backend: BackendChoice,
    threads: usize,
) -> Result<Inference, CliError> {
    match backend {
        BackendChoice::Cpu => infer_on(
            &CpuBackend::with_threads(Budget::new(BUDGET_BYTES), threads)
                .map_err(CliError::from_err)?,
            w,
            image,
            prompt,
            model_dir,
        ),
        BackendChoice::Metal => infer_on(
            &MetalBackend::new(Budget::new(BUDGET_BYTES)).map_err(CliError::from_err)?,
            w,
            image,
            prompt,
            model_dir,
        ),
    }
}

fn infer_on<B: Backend>(
    be: &B,
    w: &Sam31Weights,
    image: &Rgb8,
    prompt: &str,
    model_dir: &Path,
) -> Result<Inference, CliError> {
    let backbone = Backbone::new(be, &w.vision).map_err(CliError::from_err)?;
    let text = Text::new(be, &w.text).map_err(CliError::from_err)?;
    let det = Detector::new(
        be,
        DetectorWeights {
            transformer: &w.detector_transformer,
            geometry: &w.geometry_encoder,
            segmentation: &w.segmentation_head,
            scoring: &w.dot_prod_scoring,
        },
    )
    .map_err(CliError::from_err)?;
    let vocab = model_dir.join("vocab.json");
    let merges = model_dir.join("merges.txt");
    let tok = Tokenizer::from_hf_files(&vocab, &merges).map_err(|e| {
        CliError::failed(format!("tokenizer files in {}: {e}", model_dir.display()))
    })?;

    let ids = tok
        .tokenize(prompt)
        .map_err(|e| CliError::failed(format!("prompt {prompt:?}: {e}")))?;
    let tf = text.encode(&[ids]).map_err(CliError::from_err)?;
    let resized = host_f32(&tf.resized, be.budget())?;
    let valid: Vec<f32> = (0..tf.padding.len())
        .filter(|&t| !tf.padding[t])
        .flat_map(|t| resized[t * D..(t + 1) * D].to_vec())
        .collect();
    if valid.is_empty() {
        return Err(CliError::failed(format!("prompt {prompt:?} has no tokens")));
    }
    let prompt_t = Tensor::from_f32(&valid, &[1, valid.len() / D, D], be.budget())
        .map_err(CliError::from_err)?;

    let side = st_model::IMAGE_SIZE;
    let input = sam_input(image, side);
    let t =
        Tensor::from_f32(&input, &[1, 3, side, side], be.budget()).map_err(CliError::from_err)?;
    let device = be.upload(&t).map_err(CliError::from_err)?;
    let t0 = Instant::now();
    let trunk = backbone
        .trunk(&device, &[], &mut Taps::default())
        .map_err(CliError::from_err)?;
    let fpn = backbone
        .neck(Neck::Detector, &trunk)
        .map_err(CliError::from_err)?;
    let out = det
        .forward(&fpn, &prompt_t, &mut DetectorTaps::default())
        .map_err(CliError::from_err)?;
    be.sync().map_err(CliError::from_err)?;
    let seconds = t0.elapsed().as_secs_f64();
    Ok(Inference {
        probabilities: out.probabilities(),
        presence: st_model::host::sigmoid(out.presence_logit),
        boxes: out.pred_boxes.clone(),
        masks: host_f32(&out.pred_masks, be.budget())?,
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
            "model-dir",
            "threads",
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
    let out_dir = PathBuf::from(args.get("out").unwrap_or("spiketrack-out"));
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

    let (image, decoder) = decode_image_with_source(&image_path).map_err(CliError::from_err)?;
    let (w, applied) = weights::load(&weights, lora.as_deref())?;
    let inf = infer(&image, &prompt, &w, &model_dir, backend, threads)?;
    let n = inf.probabilities.len();
    if n != QUERIES || inf.masks.len() != QUERIES * MASK * MASK {
        return Err(CliError::failed(format!(
            "the detector returned {n} queries and {} mask values, expected {QUERIES} and {}",
            inf.masks.len(),
            QUERIES * MASK * MASK
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
            detections: &d,
        })
        .unwrap();
        let v = parse_json(&text).expect("valid JSON");
        assert_eq!(v.get_str("image").unwrap(), "a \"b\".jpg");
        assert_eq!(v.get_str("backend").unwrap(), "cpu");
        assert_eq!(v.get_str("decoder").unwrap(), "pillow-exact-jpeg");
        assert_eq!(v.get_u64("width").unwrap(), 8);
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
            detections: &[],
        })
        .unwrap();
        assert_eq!(
            parse_json(&empty)
                .unwrap()
                .get_array("detections")
                .unwrap()
                .len(),
            0
        );
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
}
