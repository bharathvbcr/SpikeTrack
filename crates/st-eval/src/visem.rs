//! VISEM-Tracking loader (Thambawita et al., Scientific Data 2023; CC BY 4.0).
//!
//! Layout (v4 train split): `Train/<video>/labels_ftid/<video>_frame_<n>_with_ftid.txt`,
//! one line per box: `<track id> <class> <cx> <cy> <w> <h>`, coordinates
//! normalized to the frame. Classes: 0 sperm, 1 cluster, 2 small or pinhead.
//! Frames are 640x480 JPEGs in `images/`.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::hota::{Frame, Xywh};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VisemClass {
    Sperm = 0,
    Cluster = 1,
    SmallOrPinhead = 2,
}

impl VisemClass {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "0" => Some(Self::Sperm),
            "1" => Some(Self::Cluster),
            "2" => Some(Self::SmallOrPinhead),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisemBox {
    pub track: String,
    pub class: VisemClass,
    /// Normalized to the frame: top-left corner and size.
    pub bbox: Xywh,
}

#[derive(Debug)]
pub enum VisemError {
    Io { path: PathBuf, err: std::io::Error },
    Parse { path: PathBuf, line: usize, detail: String },
    Layout { path: PathBuf, detail: String },
}

impl fmt::Display for VisemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, err } => write!(f, "{}: {err}", path.display()),
            Self::Parse { path, line, detail } => write!(f, "{}:{line}: {detail}", path.display()),
            Self::Layout { path, detail } => write!(f, "{}: {detail}", path.display()),
        }
    }
}

impl std::error::Error for VisemError {}

/// One annotated video: boxes by frame index (every labelled frame is present,
/// including frames with no boxes).
#[derive(Clone, Debug)]
pub struct VisemVideo {
    pub id: String,
    pub frames: BTreeMap<u32, Vec<VisemBox>>,
}

impl VisemVideo {
    /// Ground truth for HOTA, restricted to `classes`, in [`Frame`] order with
    /// empty tracker lists. Frames between the first and last labelled frame
    /// that have no label file are an error, not silently skipped.
    pub fn hota_frames(&self, classes: &[VisemClass]) -> Result<Vec<Frame<String>>, VisemError> {
        let (Some(&first), Some(&last)) = (self.frames.keys().next(), self.frames.keys().next_back()) else {
            return Ok(Vec::new());
        };
        (first..=last)
            .map(|t| {
                let boxes = self.frames.get(&t).ok_or_else(|| VisemError::Layout {
                    path: PathBuf::from(&self.id),
                    detail: format!("frame {t} has no label file"),
                })?;
                let gt = boxes.iter().filter(|b| classes.contains(&b.class)).map(|b| (b.track.clone(), b.bbox)).collect();
                Ok(Frame { gt, pred: Vec::new() })
            })
            .collect()
    }
}

fn frame_index(name: &str, video: &str) -> Option<u32> {
    name.strip_prefix(&format!("{video}_frame_"))?.strip_suffix("_with_ftid.txt")?.parse().ok()
}

pub fn parse_labels(path: &Path, text: &str) -> Result<Vec<VisemBox>, VisemError> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let err = |detail: String| VisemError::Parse { path: path.to_path_buf(), line: n + 1, detail };
        let f: Vec<&str> = line.split_whitespace().collect();
        let [track, class, cx, cy, w, h] = f.as_slice() else {
            return Err(err(format!("{} fields, expected 6", f.len())));
        };
        let class = VisemClass::parse(class).ok_or_else(|| err(format!("class {class:?}")))?;
        let num = |s: &str| -> Result<f64, VisemError> {
            let v: f64 = s.parse().map_err(|_| err(format!("not a number: {s:?}")))?;
            if v.is_finite() && (0.0..=1.0 + 1e-9).contains(&v) {
                Ok(v)
            } else {
                Err(err(format!("{v} is outside [0, 1]")))
            }
        };
        let (cx, cy, w, h) = (num(cx)?, num(cy)?, num(w)?, num(h)?);
        out.push(VisemBox {
            track: (*track).to_string(),
            class,
            bbox: Xywh { x: cx - w / 2.0, y: cy - h / 2.0, w, h },
        });
    }
    Ok(out)
}

/// Load one video directory (`Train/<id>`).
pub fn load_video(dir: &Path) -> Result<VisemVideo, VisemError> {
    let id = dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| VisemError::Layout { path: dir.to_path_buf(), detail: "no directory name".into() })?
        .to_string();
    let labels = dir.join("labels_ftid");
    let io = |path: &Path, err| VisemError::Io { path: path.to_path_buf(), err };
    let mut frames = BTreeMap::new();
    for entry in std::fs::read_dir(&labels).map_err(|e| io(&labels, e))? {
        let path = entry.map_err(|e| io(&labels, e))?.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else { continue };
        let t = frame_index(name, &id).ok_or_else(|| VisemError::Layout {
            path: path.clone(),
            detail: format!("not a {id}_frame_<n>_with_ftid.txt file"),
        })?;
        let text = std::fs::read_to_string(&path).map_err(|e| io(&path, e))?;
        frames.insert(t, parse_labels(&path, &text)?);
    }
    if frames.is_empty() {
        return Err(VisemError::Layout { path: labels, detail: "no label files".into() });
    }
    Ok(VisemVideo { id, frames })
}

/// Every video under `Train/`, sorted by id.
pub fn load_split(train: &Path) -> Result<Vec<VisemVideo>, VisemError> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(train)
        .map_err(|e| VisemError::Io { path: train.to_path_buf(), err: e })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("labels_ftid").is_dir())
        .collect();
    dirs.sort();
    dirs.iter().map(|d| load_video(d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_label_line() {
        let p = Path::new("x.txt");
        let b = parse_labels(p, "ckyw7 0 0.5 0.25 0.1 0.2\n\n").unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].class, VisemClass::Sperm);
        let r = b[0].bbox;
        assert!((r.x - 0.45).abs() < 1e-12 && (r.y - 0.15).abs() < 1e-12 && (r.w - 0.1).abs() < 1e-12);
    }

    #[test]
    fn refuses_bad_lines() {
        let p = Path::new("x.txt");
        assert!(parse_labels(p, "id 0 0.5 0.5 0.1").is_err());
        assert!(parse_labels(p, "id 3 0.5 0.5 0.1 0.1").is_err());
        assert!(parse_labels(p, "id 0 1.5 0.5 0.1 0.1").is_err());
        assert!(parse_labels(p, "id 0 nan 0.5 0.1 0.1").is_err());
    }

    #[test]
    fn frame_names() {
        assert_eq!(frame_index("22_frame_640_with_ftid.txt", "22"), Some(640));
        assert_eq!(frame_index("23_frame_640_with_ftid.txt", "22"), None);
    }
}
