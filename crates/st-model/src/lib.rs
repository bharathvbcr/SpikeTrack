//! SAM 3.1 image model on the Ojas Backend.

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::cloned_ref_to_slice_refs,
    clippy::assertions_on_constants,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::type_complexity,
    clippy::doc_overindented_list_items
)]

use std::fmt;

use ojas_core::OjasError;

pub mod backbone;
pub mod detector;
pub mod host;
pub mod layers;
pub mod profile;
pub mod text;
pub mod tile;
pub mod tokenizer;
pub mod tracker;

pub use backbone::{Backbone, ExternalTrunk, Neck, Taps, IMAGE_SIZE};
pub use detector::{Detector, DetectorOut, DetectorTaps, DetectorWeights};
/// The model's input size: the one runtime value every shape in the trunk, necks, detector and
/// tracker derives from ([`Geometry::from_image_side`] validates it).
pub use st_gate::Geometry;
pub use text::{Text, TextFeatures};
pub use tile::TileSpec;
pub use tokenizer::Tokenizer;

#[derive(Debug)]
pub enum ModelError {
    Ojas(OjasError),
    MissingWeight(String),
    WeightShape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ojas(e) => write!(f, "{e}"),
            Self::MissingWeight(n) => write!(f, "weight {n:?} is not in the checkpoint group"),
            Self::WeightShape {
                name,
                expected,
                found,
            } => write!(f, "{name}: shape {found:?}, expected {expected:?}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<OjasError> for ModelError {
    fn from(e: OjasError) -> Self {
        Self::Ojas(e)
    }
}
