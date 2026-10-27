//! Frame decode and preprocessing for SpikeTrack.
//!
//! [`decode`] reads still images and video (AVFoundation) with no third-party
//! crate; its JPEG path is [`jpeg`], a Rust port of libjpeg-turbo 3.2.0's
//! default decode that matches Pillow byte for byte, and ImageIO covers the rest.
//! [`resize`] is Pillow's bilinear resample and SAM 3's input normalisation.

#[cfg(target_os = "macos")]
pub mod decode;
pub mod jpeg;
pub mod resize;

#[cfg(target_os = "macos")]
pub use decode::{decode_image, decode_image_imageio, decode_image_with_source, DecodeError, Decoder, VideoInfo, VideoReader};
pub use jpeg::{decode_jpeg, JpegError, Unsupported};
pub use resize::{resize_bilinear_aa_f32, resize_bilinear_pil, sam_input, Rgb8};
