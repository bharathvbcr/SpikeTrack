//! Frame decode and preprocessing for SpikeTrack.
//!
//! [`decode`] reads still images (ImageIO) and video (AVFoundation) with no
//! third-party crate; [`resize`] is Pillow's bilinear resample and SAM 3's input
//! normalisation.

#[cfg(target_os = "macos")]
pub mod decode;
pub mod resize;

#[cfg(target_os = "macos")]
pub use decode::{decode_image, DecodeError, VideoInfo, VideoReader};
pub use resize::{resize_bilinear_aa_f32, resize_bilinear_pil, sam_input, Rgb8};
