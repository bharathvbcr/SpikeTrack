//! Frame preprocessing for SpikeTrack.

pub mod resize;

pub use resize::{resize_bilinear_pil, sam_input, Rgb8};
