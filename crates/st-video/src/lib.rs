//! Frame decode and preprocessing for SpikeTrack.
//!
//! [`decode`] reads still images and video (AVFoundation) with no third-party
//! crate; its JPEG path is [`jpeg`], a Rust port of libjpeg-turbo 3.2.0's
//! default decode that matches Pillow byte for byte, and ImageIO covers the rest.
//! [`encode`] is its counterpart: an H.264 `.mp4` writer on AVFoundation's `AVAssetWriter`.
//! [`resize`] is Pillow's bilinear resample and SAM 3's input normalisation.

#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::byte_char_slices,
    clippy::approx_constant
)]

#[cfg(target_os = "macos")]
pub mod decode;
#[cfg(target_os = "macos")]
pub mod encode;
pub mod jpeg;
pub mod resize;

#[cfg(target_os = "macos")]
pub use decode::{
    decode_image, decode_image_imageio, decode_image_with_source, DecodeError, Decoder, VideoInfo,
    VideoReader,
};
#[cfg(target_os = "macos")]
pub use encode::{EncodeError, VideoFile, VideoWriter, WriterOptions};
pub use jpeg::{decode_jpeg, JpegError, Unsupported};
pub use resize::{resize_bilinear_aa_f32, resize_bilinear_pil, sam_input, Rgb8};
