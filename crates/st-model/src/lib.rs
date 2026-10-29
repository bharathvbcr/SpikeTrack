//! SAM 3.1 image model on the Ojas Backend.

use std::fmt;

use ojas_core::OjasError;

pub mod backbone;
pub mod detector;
pub mod host;
pub mod layers;
pub mod profile;
pub mod text;
pub mod tokenizer;
pub mod tracker;

pub use backbone::{Backbone, ExternalTrunk, Neck, Taps, IMAGE_SIZE};
pub use detector::{Detector, DetectorOut, DetectorTaps, DetectorWeights};
pub use text::{Text, TextFeatures};
pub use tokenizer::Tokenizer;

#[derive(Debug)]
pub enum ModelError {
    Ojas(OjasError),
    MissingWeight(String),
    WeightShape { name: String, expected: Vec<usize>, found: Vec<usize> },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ojas(e) => write!(f, "{e}"),
            Self::MissingWeight(n) => write!(f, "weight {n:?} is not in the checkpoint group"),
            Self::WeightShape { name, expected, found } => write!(f, "{name}: shape {found:?}, expected {expected:?}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<OjasError> for ModelError {
    fn from(e: OjasError) -> Self {
        Self::Ojas(e)
    }
}
