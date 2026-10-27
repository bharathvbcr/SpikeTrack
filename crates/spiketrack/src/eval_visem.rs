//! `spiketrack eval-visem --pred DIR --split FILE`: HOTA of predicted tracks
//! against the VISEM-Tracking ground truth, with `st_eval` (TrackEval's
//! definition: 19 alpha thresholds 0.05..0.95, IoU similarity, per-frame
//! Hungarian matching on alignment * IoU; the reported numbers are means over
//! alpha).
//!
//! # Inputs
//!
//! **Split file**: one VISEM video id per line (`11`, `22`, ...); blank lines and
//! `#` comments are ignored; an id must not repeat. Every listed video must have
//! ground truth under the VISEM folder (`--visem DIR`, else `$SPIKETRACK_VISEM`,
//! else `data/visem/VISEM_Tracking_Train_v4/Train`) and a prediction file.
//!
//! **Prediction file** `DIR/<video id>.json` (the schema the future `track`
//! command writes):
//!
//! ```text
//! { "video": "11",            // optional; when present it must equal the file name's id
//!   "width": 640, "height": 480,   // optional provenance, not used
//!   "frames": [
//!     { "frame": 0,           // the n of the ground truth's <id>_frame_<n>_with_ftid.txt
//!       "objects": [
//!         { "id": 7,          // track id: an integer or a string, unique within the frame
//!           "box_xyxy_norm": [x0, y0, x1, y1],  // fractions of the frame; x1 > x0, y1 > y0
//!           "score": 0.93 }   // optional, ignored
//!       ] }
//!   ] }
//! ```
//!
//! Frames with no entry have no predictions. A frame listed twice, an id repeated
//! within a frame, a box that is not finite or has no area, or an unknown key is
//! an error. Prediction frames outside the ground truth's labelled range are not
//! scored (HOTA is defined on the ground truth's frames); their count is reported
//! as `pred_frames_outside_gt` so they cannot pass unnoticed. A video listed in
//! the split with no prediction file is an error, not a zero.
//!
//! # What is scored
//!
//! Class 0 (sperm) ground-truth boxes only, as `crates/st-eval/tests/visem_data.rs`
//! does. Boxes are compared as fractions of the frame; IoU does not change when
//! both axes are scaled, so no pixel size is needed. The overall figure combines
//! the videos as TrackEval's `combine_sequences` does.
//!
//! # Output
//!
//! A table on stdout and, with `--out FILE`, a JSON file:
//! `{"metric", "classes", "alphas", "videos": [{id, frames, gt_boxes, pred_boxes,
//! pred_frames_outside_gt, hota, det_a, ass_a, loc_a}], "combined": {...}}`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ojas_io::{parse_json, JsonValue};
use st_eval::visem::{load_video, VisemClass};
use st_eval::{combine, hota_sequence, Frame, Hota, HotaSummary, Xywh};

use crate::cli::{Args, CliError};
use crate::json;

pub const USAGE: &str = "\
spiketrack eval-visem --pred DIR --split FILE [--visem DIR] [--out FILE]

  HOTA (TrackEval definition) of predicted per-frame boxes and track ids against
  the VISEM-Tracking ground truth, sperm class only.

  --pred DIR     folder with one <video id>.json per video in the split
  --split FILE   video ids, one per line (# comments allowed)
  --visem DIR    VISEM 'Train' folder (default: $SPIKETRACK_VISEM, else
                 data/visem/VISEM_Tracking_Train_v4/Train)
  --out FILE     also write the results as JSON

  Prediction file schema: see the documentation of crates/spiketrack/src/eval_visem.rs.";

const DEFAULT_VISEM: &str = "data/visem/VISEM_Tracking_Train_v4/Train";

/// The ids in a split file.
pub fn parse_split(text: &str) -> Result<Vec<String>, String> {
    let mut ids = Vec::new();
    let mut seen = BTreeSet::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.split_whitespace().count() != 1 || line.contains(['/', '\\']) || line == "." || line == ".." {
            return Err(format!("line {}: {line:?} is not a single video id", n + 1));
        }
        if !seen.insert(line.to_string()) {
            return Err(format!("line {}: video {line:?} is listed twice", n + 1));
        }
        ids.push(line.to_string());
    }
    if ids.is_empty() {
        return Err("the split lists no videos".to_string());
    }
    Ok(ids)
}

/// Predictions of one video: boxes by frame index.
#[derive(Debug, Default)]
pub struct Predictions {
    pub frames: std::collections::BTreeMap<u32, Vec<(String, Xywh)>>,
}

fn id_text(v: &JsonValue) -> Result<String, String> {
    if let Ok(s) = v.as_str() {
        if s.is_empty() {
            return Err("an empty track id".to_string());
        }
        return Ok(s.to_string());
    }
    v.as_i64().map(|i| i.to_string()).map_err(|_| format!("a track id must be an integer or a string, not {}", v.kind()))
}

/// Parse one prediction file's text; `expect_video` is the id from the file name.
pub fn parse_predictions(text: &str, expect_video: &str) -> Result<Predictions, String> {
    let top = parse_json(text).map_err(|e| format!("not valid JSON: {}", e.detail()))?;
    top.deny_unknown_keys_at("prediction file", &["video", "width", "height", "frames"]).map_err(|e| e.detail().to_string())?;
    if let Some(v) = top.field_opt("video").map_err(|e| e.detail().to_string())? {
        let named = match v.as_str() {
            Ok(s) => s.to_string(),
            Err(_) => v.as_i64().map_err(|_| "\"video\" must be a string or an integer".to_string())?.to_string(),
        };
        if named != expect_video {
            return Err(format!("\"video\" is {named:?} but the file is for video {expect_video:?}"));
        }
    }
    let frames = top.get_array("frames").map_err(|e| e.detail().to_string())?;
    let mut out = Predictions::default();
    for (fi, f) in frames.iter().enumerate() {
        let at = |what: &str| format!("frames[{fi}]: {what}");
        f.deny_unknown_keys_at(&format!("frames[{fi}]"), &["frame", "objects"]).map_err(|e| e.detail().to_string())?;
        let n = f.get_u64("frame").map_err(|e| at(e.detail()))?;
        let n = u32::try_from(n).map_err(|_| at(&format!("frame {n} is too large")))?;
        let mut ids = BTreeSet::new();
        let mut boxes = Vec::new();
        for (oi, o) in f.get_array("objects").map_err(|e| at(e.detail()))?.iter().enumerate() {
            let at = |what: &str| format!("frames[{fi}].objects[{oi}]: {what}");
            o.deny_unknown_keys_at("object", &["id", "box_xyxy_norm", "score"]).map_err(|e| at(e.detail()))?;
            let id = id_text(o.field("id").map_err(|e| at(e.detail()))?).map_err(|e| at(&e))?;
            if !ids.insert(id.clone()) {
                return Err(at(&format!("track id {id:?} appears twice in frame {n}")));
            }
            let b = o.get_array("box_xyxy_norm").map_err(|e| at(e.detail()))?;
            let [x0, y0, x1, y1] = b.iter().map(|v| v.as_f64().map_err(|e| at(e.detail()))).collect::<Result<Vec<f64>, String>>()?[..] else {
                return Err(at(&format!("box_xyxy_norm has {} numbers, expected 4", b.len())));
            };
            if ![x0, y0, x1, y1].iter().all(|v| v.is_finite()) || x1 <= x0 || y1 <= y0 {
                return Err(at(&format!("box [{x0}, {y0}, {x1}, {y1}] is not a finite box with positive width and height")));
            }
            boxes.push((id, Xywh { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }));
        }
        if out.frames.insert(n, boxes).is_some() {
            return Err(at(&format!("frame {n} is listed twice")));
        }
    }
    Ok(out)
}

/// One video's result.
#[derive(Clone, Debug)]
pub struct VideoResult {
    pub id: String,
    pub frames: usize,
    pub gt_boxes: usize,
    pub pred_boxes: usize,
    pub pred_frames_outside_gt: usize,
    pub hota: Hota,
}

#[derive(Debug)]
pub struct Evaluation {
    pub videos: Vec<VideoResult>,
    pub combined: Hota,
}

/// Score every video in `ids`: ground truth under `visem`, predictions under `pred`.
pub fn evaluate(pred: &Path, ids: &[String], visem: &Path) -> Result<Evaluation, String> {
    let mut videos = Vec::new();
    for id in ids {
        let gt_dir = visem.join(id);
        let gt = load_video(&gt_dir).map_err(|e| format!("video {id}: ground truth: {e}"))?;
        let mut frames: Vec<Frame<String>> = gt.hota_frames(&[VisemClass::Sperm]).map_err(|e| format!("video {id}: ground truth: {e}"))?;
        let first = *gt.frames.keys().next().ok_or_else(|| format!("video {id}: no ground-truth frames"))?;
        let pred_path = pred.join(format!("{id}.json"));
        let text = std::fs::read_to_string(&pred_path).map_err(|e| format!("video {id}: {}: {e}", pred_path.display()))?;
        let preds = parse_predictions(&text, id).map_err(|e| format!("video {id}: {}: {e}", pred_path.display()))?;
        let (mut outside, mut pred_boxes) = (0usize, 0usize);
        for (&n, boxes) in &preds.frames {
            match n.checked_sub(first).map(|k| k as usize).filter(|&k| k < frames.len()) {
                Some(k) => {
                    pred_boxes += boxes.len();
                    frames[k].pred = boxes.clone();
                }
                None => outside += 1,
            }
        }
        videos.push(VideoResult {
            id: id.clone(),
            frames: frames.len(),
            gt_boxes: frames.iter().map(|f| f.gt.len()).sum(),
            pred_boxes,
            pred_frames_outside_gt: outside,
            hota: hota_sequence(&frames),
        });
    }
    let combined = combine(&videos.iter().map(|v| v.hota.clone()).collect::<Vec<_>>());
    Ok(Evaluation { videos, combined })
}

fn summary_json(s: &HotaSummary) -> Result<String, String> {
    Ok(format!(
        "\"hota\": {}, \"det_a\": {}, \"ass_a\": {}, \"loc_a\": {}",
        json::number(s.hota, "HOTA")?,
        json::number(s.det_a, "DetA")?,
        json::number(s.ass_a, "AssA")?,
        json::number(s.loc_a, "LocA")?
    ))
}

pub fn results_json(e: &Evaluation) -> Result<String, String> {
    let mut out = String::from("{\n  \"metric\": \"HOTA, TrackEval definition (st_eval): mean over alpha 0.05..0.95 step 0.05\",\n  \"classes\": [\"sperm\"],\n  \"videos\": [");
    for (i, v) in e.videos.iter().enumerate() {
        out.push_str(&format!(
            "{}\n    {{\"id\": {}, \"frames\": {}, \"gt_boxes\": {}, \"pred_boxes\": {}, \"pred_frames_outside_gt\": {}, {}}}",
            if i == 0 { "" } else { "," },
            json::string(&v.id),
            v.frames,
            v.gt_boxes,
            v.pred_boxes,
            v.pred_frames_outside_gt,
            summary_json(&v.hota.summary())?
        ));
    }
    out.push_str(&format!("\n  ],\n  \"combined\": {{{}}}\n}}\n", summary_json(&e.combined.summary())?));
    Ok(out)
}

pub fn table(e: &Evaluation) -> String {
    let mut out = String::from("video   frames  gt boxes  pred boxes  outside   HOTA   DetA   AssA   LocA\n");
    let row = |name: &str, frames: String, gt: String, pr: String, outside: String, s: HotaSummary| {
        format!("{name:<7} {frames:>6}  {gt:>8}  {pr:>10}  {outside:>7}  {:.4} {:.4} {:.4} {:.4}\n", s.hota, s.det_a, s.ass_a, s.loc_a)
    };
    for v in &e.videos {
        out.push_str(&row(&v.id, v.frames.to_string(), v.gt_boxes.to_string(), v.pred_boxes.to_string(), v.pred_frames_outside_gt.to_string(), v.hota.summary()));
    }
    let sum = |f: fn(&VideoResult) -> usize| e.videos.iter().map(f).sum::<usize>().to_string();
    out.push_str(&row("all", sum(|v| v.frames), sum(|v| v.gt_boxes), sum(|v| v.pred_boxes), sum(|v| v.pred_frames_outside_gt), e.combined.summary()));
    out
}

pub fn run(raw: &[String]) -> Result<(), CliError> {
    let args = Args::parse(raw, &["pred", "split", "visem", "out"], 0)?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }
    let pred = PathBuf::from(args.require("pred")?);
    let split = PathBuf::from(args.require("split")?);
    let visem = match args.get("visem") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(std::env::var("SPIKETRACK_VISEM").unwrap_or_else(|_| DEFAULT_VISEM.to_string())),
    };
    if !pred.is_dir() {
        return Err(CliError::failed(format!("--pred {} is not a folder", pred.display())));
    }
    if !visem.is_dir() {
        return Err(CliError::failed(format!("the VISEM folder {} does not exist (pass --visem or set SPIKETRACK_VISEM)", visem.display())));
    }
    let text = std::fs::read_to_string(&split).map_err(|e| CliError::failed(format!("{}: {e}", split.display())))?;
    let ids = parse_split(&text).map_err(|e| CliError::failed(format!("{}: {e}", split.display())))?;
    let result = evaluate(&pred, &ids, &visem).map_err(CliError::failed)?;
    let json_text = results_json(&result).map_err(CliError::failed)?;
    print!("{}", table(&result));
    if let Some(out) = args.get("out") {
        std::fs::write(out, json_text).map_err(|e| CliError::failed(format!("{out}: {e}")))?;
        println!("wrote {out}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dirs {
        root: PathBuf,
    }

    impl Dirs {
        fn new(name: &str) -> Dirs {
            let root = std::env::temp_dir().join(format!("spiketrack-eval-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(root.join("gt")).unwrap();
            std::fs::create_dir_all(root.join("pred")).unwrap();
            Dirs { root }
        }

        /// Video `id` with `frames` frames; track "a" moves right along y = 0.2, track "b" along y = 0.6.
        fn gt(&self, id: &str, frames: u32) {
            let labels = self.root.join("gt").join(id).join("labels_ftid");
            std::fs::create_dir_all(&labels).unwrap();
            for t in 0..frames {
                let x = 0.1 + 0.02 * f64::from(t);
                let text = format!("a 0 {x} 0.2 0.1 0.1\nb 0 {x} 0.6 0.1 0.1\nc 1 0.9 0.9 0.05 0.05\n");
                std::fs::write(labels.join(format!("{id}_frame_{t}_with_ftid.txt")), text).unwrap();
            }
        }

        /// A prediction file with the same two tracks; `swap_from`: from this frame the ids are exchanged.
        fn pred(&self, id: &str, frames: u32, swap_from: Option<u32>) {
            let mut fr = Vec::new();
            for t in 0..frames {
                let x = 0.1 + 0.02 * f64::from(t);
                let swapped = swap_from.is_some_and(|s| t >= s);
                let (ida, idb) = if swapped { (2, 1) } else { (1, 2) };
                let b = |y: f64| format!("[{}, {}, {}, {}]", x - 0.05, y - 0.05, x + 0.05, y + 0.05);
                fr.push(format!("{{\"frame\": {t}, \"objects\": [{{\"id\": {ida}, \"box_xyxy_norm\": {}}}, {{\"id\": \"{idb}\", \"box_xyxy_norm\": {}, \"score\": 0.9}}]}}", b(0.2), b(0.6)));
            }
            self.write_pred(id, &format!("{{\"video\": \"{id}\", \"frames\": [{}]}}", fr.join(", ")));
        }

        fn write_pred(&self, id: &str, text: &str) {
            std::fs::write(self.root.join("pred").join(format!("{id}.json")), text).unwrap();
        }

        fn eval(&self, ids: &[&str]) -> Result<Evaluation, String> {
            let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
            evaluate(&self.root.join("pred"), &ids, &self.root.join("gt"))
        }
    }

    impl Drop for Dirs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn ground_truth_as_predictions_scores_one() {
        let d = Dirs::new("perfect");
        d.gt("7", 12);
        d.pred("7", 12, None);
        let e = d.eval(&["7"]).unwrap();
        let s = e.videos[0].hota.summary();
        assert!((s.hota - 1.0).abs() < 1e-9 && (s.det_a - 1.0).abs() < 1e-9 && (s.ass_a - 1.0).abs() < 1e-9, "{s:?}");
        // Class 1 (cluster) ground truth is not scored: 12 frames x 2 sperm tracks.
        assert_eq!((e.videos[0].frames, e.videos[0].gt_boxes, e.videos[0].pred_boxes), (12, 24, 24));
        assert!((e.combined.summary().hota - 1.0).abs() < 1e-9);
    }

    #[test]
    fn an_id_swap_costs_association_but_not_detection() {
        let d = Dirs::new("swap");
        d.gt("7", 12);
        d.pred("7", 12, Some(6));
        let s = d.eval(&["7"]).unwrap().videos[0].hota.summary();
        assert!((s.det_a - 1.0).abs() < 1e-9, "{s:?}");
        assert!(s.ass_a < 0.9 && s.hota < 0.95, "{s:?}");
    }

    #[test]
    fn no_predictions_score_zero() {
        let d = Dirs::new("empty");
        d.gt("7", 5);
        d.write_pred("7", "{\"frames\": []}");
        let v = &d.eval(&["7"]).unwrap().videos[0];
        assert_eq!(v.pred_boxes, 0);
        assert_eq!(v.hota.summary().hota, 0.0);
    }

    #[test]
    fn two_videos_combine_and_the_table_and_json_render() {
        let d = Dirs::new("two");
        d.gt("7", 6);
        d.gt("8", 6);
        d.pred("7", 6, None);
        d.pred("8", 6, Some(3));
        let e = d.eval(&["7", "8"]).unwrap();
        let (a, b, all) = (e.videos[0].hota.summary().hota, e.videos[1].hota.summary().hota, e.combined.summary().hota);
        assert!(all < a && all > b - 1e-9, "{a} {b} {all}");
        let t = table(&e);
        assert!(t.contains("all") && t.lines().count() == 4, "{t}");
        let j = parse_json(&results_json(&e).unwrap()).unwrap();
        assert_eq!(j.get_array("videos").unwrap().len(), 2);
        assert!((j.get_object("combined").unwrap().get_f64("hota").unwrap() - all).abs() < 1e-12);
    }

    #[test]
    fn prediction_frames_outside_the_ground_truth_are_counted_not_scored() {
        let d = Dirs::new("outside");
        d.gt("7", 4);
        d.pred("7", 9, None); // frames 4..8 have no ground truth
        let v = &d.eval(&["7"]).unwrap().videos[0];
        assert_eq!(v.pred_frames_outside_gt, 5);
        assert!((v.hota.summary().hota - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_missing_prediction_file_or_ground_truth_is_an_error() {
        let d = Dirs::new("missing");
        d.gt("7", 3);
        assert!(d.eval(&["7"]).unwrap_err().contains("7.json"));
        d.pred("9", 3, None);
        assert!(d.eval(&["9"]).unwrap_err().contains("ground truth"));
    }

    #[test]
    fn malformed_predictions_are_errors_with_a_location() {
        let ok = |s: &str| parse_predictions(s, "7");
        let obj = |extra: &str| format!("{{\"frames\": [{{\"frame\": 0, \"objects\": [{extra}]}}]}}");
        assert!(ok(&obj("{\"id\": 1, \"box_xyxy_norm\": [0.1, 0.1, 0.2, 0.2]}")).is_ok());
        for (bad, needle) in [
            ("not json", "not valid JSON"),
            ("{\"video\": \"8\", \"frames\": []}", "video"),
            ("{\"frames\": [], \"extra\": 1}", "extra"),
            ("{}", "frames"),
            ("{\"frames\": [{\"frame\": 0, \"objects\": []}, {\"frame\": 0, \"objects\": []}]}", "listed twice"),
            (&obj("{\"id\": 1, \"box_xyxy_norm\": [0.1, 0.1, 0.2, 0.2]}, {\"id\": \"1\", \"box_xyxy_norm\": [0.3, 0.3, 0.4, 0.4]}"), "appears twice"),
            (&obj("{\"id\": 1, \"box_xyxy_norm\": [0.2, 0.1, 0.1, 0.2]}"), "positive width"),
            (&obj("{\"id\": 1, \"box_xyxy_norm\": [0.1, 0.1, 0.2]}"), "expected 4"),
            (&obj("{\"id\": 1.5, \"box_xyxy_norm\": [0.1, 0.1, 0.2, 0.2]}"), "track id"),
            (&obj("{\"id\": \"\", \"box_xyxy_norm\": [0.1, 0.1, 0.2, 0.2]}"), "empty track id"),
            (&obj("{\"id\": 1}"), "box_xyxy_norm"),
            (&obj("{\"id\": 1, \"box_xyxy_norm\": [0.1, 0.1, 0.2, 0.2], \"mask\": 1}"), "mask"),
            ("{\"frames\": [{\"frame\": -1, \"objects\": []}]}", "frame"),
        ] {
            let e = ok(bad).unwrap_err();
            assert!(e.contains(needle), "{bad}: {e}");
        }
    }

    #[test]
    fn splits_are_parsed_strictly() {
        assert_eq!(parse_split("11\n# comment\n\n22 # trailing\n").unwrap(), ["11", "22"]);
        assert!(parse_split("").is_err());
        assert!(parse_split("# only a comment\n").is_err());
        assert!(parse_split("11\n11\n").unwrap_err().contains("twice"));
        assert!(parse_split("11 22\n").is_err());
        assert!(parse_split("../etc\n").is_err());
        assert!(parse_split("a/b\n").is_err());
    }

    /// The real ground truth of video 11, written in the prediction schema and
    /// scored against itself: round-trips the loader, the schema and HOTA.
    #[test]
    #[ignore = "needs data/visem (VISEM-Tracking)"]
    fn real_ground_truth_as_predictions_scores_one() {
        let visem = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(DEFAULT_VISEM);
        let gt = load_video(&visem.join("11")).expect("video 11");
        let mut frames = Vec::new();
        for (n, boxes) in &gt.frames {
            let objs: Vec<String> = boxes
                .iter()
                .filter(|b| b.class == VisemClass::Sperm)
                .map(|b| format!("{{\"id\": {}, \"box_xyxy_norm\": [{}, {}, {}, {}]}}", json::string(&b.track), b.bbox.x, b.bbox.y, b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h))
                .collect();
            frames.push(format!("{{\"frame\": {n}, \"objects\": [{}]}}", objs.join(", ")));
        }
        let dir = std::env::temp_dir().join(format!("spiketrack-eval-real-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("11.json"), format!("{{\"video\": \"11\", \"frames\": [{}]}}", frames.join(", "))).unwrap();
        let e = evaluate(&dir, &["11".to_string()], &visem).expect("evaluate");
        let _ = std::fs::remove_dir_all(&dir);
        let s = e.videos[0].hota.summary();
        println!("video 11: {} frames, {} gt boxes, HOTA {:.6}", e.videos[0].frames, e.videos[0].gt_boxes, s.hota);
        assert!((s.hota - 1.0).abs() < 1e-9, "{s:?}");
    }
}
