//! Evaluation for SpikeTrack: HOTA (TrackEval definition) and the VISEM-Tracking loader.

#![allow(clippy::needless_range_loop)]

pub mod assign;
pub mod hota;
pub mod jf;
pub mod visem;

pub use hota::{combine, hota_sequence, Frame, Hota, HotaSummary, Xywh};
