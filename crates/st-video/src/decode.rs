//! Native image and video decode for macOS (Apple silicon): the Pillow-exact
//! JPEG decoder in [`crate::jpeg`] and ImageIO for still images, AVFoundation
//! for video. No Python, no ffmpeg, no third-party crate:
//! the frameworks are linked with `#[link(.., kind = "framework")]` and the
//! Objective-C calls go through libobjc's `objc_msgSend`, cast to the exact
//! function type of each call.
//!
//! Every declaration below was checked against the macOS SDK headers
//! (`xcrun --show-sdk-path`): `ImageIO/CGImageSource.h`, `CGImageProperties.h`,
//! `CoreGraphics/CGColorSpace.h`, `CGBitmapContext.h`, `CGContext.h`, `CGImage.h`,
//! `CoreMedia/CMSampleBuffer.h`, `CMTime.h`, `CoreVideo/CVPixelBuffer.h`,
//! `CVReturn.h`, `CoreFoundation/CFURL.h`, `CFData.h`, and
//! `AVFoundation/AVAsset.h`, `AVAssetTrack.h`, `AVAssetReader.h`,
//! `AVAssetReaderOutput.h`, `AVMediaFormat.h`.
//!
//! # Ownership
//!
//! Every Core Foundation object is held by a [`CfRef`] and every Objective-C
//! object this module retains by an [`Obj`]; both release on drop, so a `?` on
//! any path releases what was acquired. Calls that touch Objective-C run inside
//! an `NSAutoreleasePool` ([`Pool`]) that is drained before the call returns.
//! Nothing here panics on a framework failure: each one is a [`DecodeError`].
//!
//! # Images: [`decode_image`]
//!
//! A JPEG is decoded by [`crate::jpeg::decode_jpeg`], whose output is
//! byte-identical to Pillow 13.0.0 on libjpeg-turbo 3.2.0
//! (`Image.open(..).convert("RGB")`; `tests/decode.rs` checks six VISEM frames
//! at zero tolerance), for baseline sequential 8-bit grayscale and YCbCr/RGB
//! files. ICC profiles are ignored, as Pillow ignores them, and a grayscale
//! JPEG is replicated to three channels with no colour management; the EXIF
//! orientation is not applied (Pillow's `Image.open` does not either). A JPEG
//! that is damaged or truncated is an error. A JPEG in a variant the native
//! decoder refuses by name (progressive, arithmetic, 12-bit, CMYK, ...) and
//! every other format go to ImageIO, and [`decode_image_with_source`] says
//! which decoder ran ([`Decoder`]); [`decode_image_imageio`] forces ImageIO.
//!
//! What follows describes the ImageIO path. The file is decoded by ImageIO (`CGImageSourceCreateImageAtIndex`, first
//! image only) and drawn into an 8-bit RGBX bitmap context in the **sRGB**
//! colour space (`kCGColorSpaceSRGB`), then the X byte is dropped. What that
//! means, chosen deliberately:
//!
//! * An image already in sRGB (or untagged, which ImageIO treats as device RGB)
//!   comes through unconverted. An image with another ICC profile, or a gray,
//!   CMYK, 16-bit or floating-point image, is converted to sRGB by Quartz.
//!   PIL and libjpeg ignore profiles instead; the two agree on sRGB pictures
//!   (every VISEM frame) and differ on tagged ones.
//! * Alpha is not composited against anything: the bitmap starts black, so
//!   transparent pixels come out black (premultiplied-over-black).
//! * The EXIF orientation tag is **not** applied (PIL's `Image.open` does not
//!   either); the stored pixels are returned.
//! * ImageIO's JPEG decoder is not libjpeg-turbo: the IDCT and chroma
//!   upsampling differ in the last bits. `crates/st-video/tests/decode.rs`
//!   measures the difference against Pillow on VISEM frames, through
//!   [`decode_image_imageio`] (the fallback path for refused JPEG variants).
//! * A truncated or damaged file is an error: ImageIO's status for the image
//!   is read after the pixels were drawn, and anything but "complete" fails.
//!   ImageIO accepts cut-short PNG chunks lists as complete, so PNG chunk
//!   boundaries are checked first ([`check_png`]). A cut-short BMP or TIFF is
//!   not detected (ImageIO reports it complete and zero-fills; measured by
//!   tessl, not by us).
//!
//! # Video: [`VideoReader`]
//!
//! `AVAssetReader` + `AVAssetReaderTrackOutput` on the first video track, asked
//! for `kCVPixelFormatType_32BGRA`, so AVFoundation's own YCbCr to BGRA
//! conversion produces the pixels. (tessl's media notes measure that conversion
//! as altering chroma relative to libswscale/ffmpeg; the effect on the VISEM
//! frames is measured in `tests/decode.rs`.) [`VideoReader::open`] makes one
//! extra pass over the compressed samples, without decoding, to learn the exact
//! frame count and the duration, so [`VideoInfo::frame_count`] is always known
//! and the decode pass is checked against it: fewer or more frames than the
//! sample table lists is an error, not a short result.
//!
//! **Rotation is applied.** `AVAssetReaderTrackOutput` returns frames as stored;
//! the track's `preferredTransform` is a pure right-angle rotation and/or flip
//! for every camera video, and [`VideoReader`] applies it, so frames come out
//! upright. [`VideoInfo::transform`] reports the integer `[a, b, c, d]` that was
//! applied (identity is `[1, 0, 0, 1]`), and `width`/`height` are the delivered
//! (post-transform) size. A transform that is not a right-angle rotation or a
//! flip is refused at open. Presentation times are the decoded frames' own
//! (track time, edit list applied).
//!
//! # Preprocessing is not decided here
//!
//! This module only decodes to 8-bit RGB. What upstream SAM 3 does after
//! decoding depends on the input kind, and ours matches only some of them:
//!
//! | input | upstream decode and resize | ours |
//! |---|---|---|
//! | one image file | PIL or torchvision decode; `v2.Resize` (antialiased bilinear) on a uint8 tensor | JPEG: Pillow-exact decode; other formats: ImageIO; then [`crate::sam_input`] (Pillow bilinear): within one grey level |
//! | folder of JPEG frames | PIL decode; `TF.resize` on the PIL image (Pillow bilinear) | Pillow-exact decode for baseline JPEG, [`crate::sam_input`]: same resize |
//! | mp4 file | OpenCV decode; `cv2.resize(.., INTER_CUBIC)` after BGR to RGB (`io_utils.py:313-333`) | AVFoundation decode, bilinear resize: **does not match** |
//!
//! Bicubic for mp4 input is not implemented; the mismatch is a recorded gap.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("st-video's decode module is Apple-silicon macOS only: objc_msgSend is cast to typed function pointers, which needs the arm64 ABI (no _stret/_fpret variants)");

use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use crate::resize::Rgb8;

/// The largest image file [`decode_image`] reads.
pub const MAX_IMAGE_FILE_BYTES: u64 = 512 << 20;
/// The most pixels [`decode_image`] and [`VideoReader`] accept in one frame
/// (an RGBX bitmap of this size is 512 MiB). The JPEG decoder owns the number.
pub const MAX_FRAME_PIXELS: usize = crate::jpeg::MAX_PIXELS;
/// The most frames [`VideoReader::open`] counts before refusing the video.
pub const MAX_VIDEO_FRAMES: usize = 1 << 24;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum DecodeError {
    /// The file could not be read.
    Io { path: PathBuf, err: std::io::Error },
    /// The file has no bytes.
    Empty { path: PathBuf },
    /// The system does not recognise the file as an image or a video.
    Unrecognized { path: PathBuf, detail: String },
    /// Recognised, but cut short or damaged.
    Damaged { path: PathBuf, detail: String },
    /// Recognised and intact, but something this module refuses (a transform
    /// that is not a right-angle rotation or flip, an unexpected pixel format).
    Unsupported { path: PathBuf, detail: String },
    /// Larger than the bounds in this module.
    TooLarge { path: PathBuf, detail: String },
    /// A framework call failed in a way no other variant describes.
    Framework { api: &'static str, detail: String },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, err } => write!(f, "{}: {err}", path.display()),
            Self::Empty { path } => write!(f, "{}: the file is empty", path.display()),
            Self::Unrecognized { path, detail } => write!(f, "{}: not a recognised image or video: {detail}", path.display()),
            Self::Damaged { path, detail } => write!(f, "{}: damaged or truncated: {detail}", path.display()),
            Self::Unsupported { path, detail } => write!(f, "{}: unsupported: {detail}", path.display()),
            Self::TooLarge { path, detail } => write!(f, "{}: too large: {detail}", path.display()),
            Self::Framework { api, detail } => write!(f, "{api}: {detail}"),
        }
    }
}

impl std::error::Error for DecodeError {}

fn io_err(path: &Path, err: std::io::Error) -> DecodeError {
    DecodeError::Io { path: path.to_path_buf(), err }
}

// ---------------------------------------------------------------------------
// FFI: libobjc, CoreFoundation, ImageIO, CoreGraphics, CoreMedia, CoreVideo
// ---------------------------------------------------------------------------

pub(crate) type Id = *mut c_void;
pub(crate) type Sel = *const c_void;
pub(crate) type CfTypeRef = *const c_void;
pub(crate) type CfIndex = isize;

/// `CGRect` (`CGGeometry.h`); `CGFloat` is `double` on 64-bit.
#[repr(C)]
#[derive(Clone, Copy)]
struct CgRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// `CGSize` (`CGGeometry.h`).
#[repr(C)]
#[derive(Clone, Copy)]
struct CgSize {
    width: f64,
    height: f64,
}

/// `CGAffineTransform` (`CGAffineTransform.h`): `{a, b, c, d, tx, ty}`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CgAffineTransform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

/// `CMTime` (`CMTime.h`): `value / timescale` seconds when `flags & 1`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CmTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

/// `kCMTimeFlags_Valid` (`CMTime.h`).
const CMTIME_VALID: u32 = 1;
/// `kCGImageAlphaNoneSkipLast` (`CGImage.h`): RGBX, the X byte ignored.
const ALPHA_NONE_SKIP_LAST: u32 = 5;
/// `kCGImageStatusComplete` (`CGImageSource.h`).
const STATUS_COMPLETE: i32 = 0;
/// `kCVPixelFormatType_32BGRA` (`CVPixelBuffer.h`), the FourCC `'BGRA'`.
pub(crate) const PIXEL_FORMAT_32BGRA: u32 = u32::from_be_bytes(*b"BGRA");
/// `kCVPixelBufferLock_ReadOnly` (`CVPixelBuffer.h`).
const PIXEL_BUFFER_LOCK_READ_ONLY: u64 = 1;
/// `kCVReturnSuccess` (`CVReturn.h`).
pub(crate) const CV_RETURN_SUCCESS: i32 = 0;
/// `AVAssetReaderStatus` (`AVAssetReader.h`, `NSInteger`).
const READER_STATUS_READING: isize = 1;
const READER_STATUS_COMPLETED: isize = 2;
const READER_STATUS_FAILED: isize = 3;

#[link(name = "objc")]
extern "C" {
    pub(crate) fn objc_getClass(name: *const c_char) -> Id;
    pub(crate) fn sel_registerName(name: *const c_char) -> Sel;
    /// Cast to the exact signature of each call (see [`msg!`]); never called as declared.
    pub(crate) fn objc_msgSend();
}

#[link(name = "Foundation", kind = "framework")]
extern "C" {}

#[link(name = "AVFoundation", kind = "framework")]
extern "C" {
    static AVMediaTypeVideo: Id;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: CfTypeRef);
    fn CFDataCreate(allocator: CfTypeRef, bytes: *const u8, length: CfIndex) -> CfTypeRef;
    fn CFURLCreateFromFileSystemRepresentation(allocator: CfTypeRef, buffer: *const u8, buf_len: CfIndex, is_directory: u8) -> CfTypeRef;
    fn CFDictionaryGetValue(dict: CfTypeRef, key: *const c_void) -> *const c_void;
    fn CFGetTypeID(cf: CfTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFNumberGetValue(number: CfTypeRef, the_type: CfIndex, value_ptr: *mut c_void) -> u8;
}

/// `kCFNumberSInt64Type` (`CFNumber.h`).
const CF_NUMBER_SINT64_TYPE: CfIndex = 4;

#[link(name = "ImageIO", kind = "framework")]
extern "C" {
    static kCGImagePropertyPixelWidth: CfTypeRef;
    static kCGImagePropertyPixelHeight: CfTypeRef;

    fn CGImageSourceCreateWithData(data: CfTypeRef, options: CfTypeRef) -> CfTypeRef;
    fn CGImageSourceGetCount(isrc: CfTypeRef) -> usize;
    fn CGImageSourceGetStatusAtIndex(isrc: CfTypeRef, index: usize) -> i32;
    fn CGImageSourceCopyPropertiesAtIndex(isrc: CfTypeRef, index: usize, options: CfTypeRef) -> CfTypeRef;
    fn CGImageSourceCreateImageAtIndex(isrc: CfTypeRef, index: usize, options: CfTypeRef) -> CfTypeRef;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    static kCGColorSpaceSRGB: CfTypeRef;

    fn CGImageGetWidth(image: CfTypeRef) -> usize;
    fn CGImageGetHeight(image: CfTypeRef) -> usize;
    fn CGColorSpaceCreateWithName(name: CfTypeRef) -> CfTypeRef;
    fn CGBitmapContextCreate(
        data: *mut c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        space: CfTypeRef,
        bitmap_info: u32,
    ) -> CfTypeRef;
    fn CGContextDrawImage(c: CfTypeRef, rect: CgRect, image: CfTypeRef);
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMSampleBufferGetNumSamples(sbuf: CfTypeRef) -> CfIndex;
    fn CMSampleBufferGetPresentationTimeStamp(sbuf: CfTypeRef) -> CmTime;
    fn CMSampleBufferGetDuration(sbuf: CfTypeRef) -> CmTime;
    fn CMSampleBufferGetImageBuffer(sbuf: CfTypeRef) -> CfTypeRef;
    fn CMTimeGetSeconds(time: CmTime) -> f64;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CfTypeRef;

    fn CVPixelBufferLockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CfTypeRef) -> *const u8;
    fn CVPixelBufferGetBytesPerRow(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetPixelFormatType(pb: CfTypeRef) -> u32;
}

/// `objc_msgSend` cast to the signature of one call. Arguments are written as
/// `value => type`; the return type follows `->`. On arm64 `objc_msgSend` is
/// the only entry point (struct and floating-point returns need no `_stret` or
/// `_fpret`), and the function type must be non-variadic.
///
/// `msg!(receiver, "selector:with:", a => Id, b => usize; -> bool)`
///
/// The paths are `$crate`-qualified so that [`crate::encode`] shares the macro.
macro_rules! msg {
    ($recv:expr, $sel:literal $(, $arg:expr => $ty:ty)* ; -> $ret:ty) => {{
        let f: unsafe extern "C" fn($crate::decode::Id, $crate::decode::Sel $(, $ty)*) -> $ret = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn($crate::decode::Id, $crate::decode::Sel $(, $ty)*) -> $ret,
        >($crate::decode::objc_msgSend);
        f($recv, $crate::decode::sel_registerName(concat!($sel, "\0").as_ptr().cast()) $(, $arg)*)
    }};
}
pub(crate) use msg;

// ---------------------------------------------------------------------------
// RAII guards
// ---------------------------------------------------------------------------

/// An owned Core Foundation reference (a `Create`/`Copy`/`CF_RETURNS_RETAINED`
/// result), released on drop. `CFRelease` releases every CF-based type
/// (`CGImageSource`, `CGImage`, `CGColorSpace`, `CGContext`, `CMSampleBuffer`,
/// `CFData`, `CFURL`, `CFDictionary`).
pub(crate) struct CfRef(NonNull<c_void>);

impl CfRef {
    /// # Safety
    /// `ptr` is NULL or a live CF object the caller owns one reference to.
    pub(crate) unsafe fn take(ptr: CfTypeRef) -> Option<CfRef> {
        NonNull::new(ptr as *mut c_void).map(CfRef)
    }

    pub(crate) fn as_ptr(&self) -> CfTypeRef {
        self.0.as_ptr() as CfTypeRef
    }
}

impl Drop for CfRef {
    fn drop(&mut self) {
        // SAFETY: `self` owns one reference and drop runs once.
        unsafe { CFRelease(self.as_ptr()) }
    }
}

/// An Objective-C object this module owns one reference to, `release`d on drop.
pub(crate) struct Obj(NonNull<c_void>);

impl Obj {
    /// Take our own reference (`retain`) to an object a factory method returned
    /// autoreleased.
    ///
    /// # Safety
    /// `ptr` is NULL or a live Objective-C object.
    pub(crate) unsafe fn retain(ptr: Id) -> Option<Obj> {
        let n = NonNull::new(ptr)?;
        let _: Id = msg!(n.as_ptr(), "retain"; -> Id);
        Some(Obj(n))
    }

    pub(crate) fn id(&self) -> Id {
        self.0.as_ptr()
    }
}

impl Drop for Obj {
    fn drop(&mut self) {
        // SAFETY: `self` owns one reference and drop runs once.
        unsafe {
            let _: () = msg!(self.id(), "release"; -> ());
        }
    }
}

/// An `NSAutoreleasePool`, drained on drop. Declared first in a function so it
/// drops after every guard that borrows autoreleased memory.
pub(crate) struct Pool(Id);

impl Pool {
    pub(crate) fn new() -> Result<Pool, DecodeError> {
        // SAFETY: `+alloc` and `-init` on NSAutoreleasePool; Foundation is linked.
        unsafe {
            let class = objc_getClass(c"NSAutoreleasePool".as_ptr());
            if class.is_null() {
                return Err(framework("objc_getClass", "NSAutoreleasePool is not loaded"));
            }
            let raw: Id = msg!(class, "alloc"; -> Id);
            let pool: Id = msg!(raw, "init"; -> Id);
            if pool.is_null() {
                return Err(framework("NSAutoreleasePool", "init returned nil"));
            }
            Ok(Pool(pool))
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: the pool was created by `new` on this thread and is drained once.
        unsafe {
            let _: () = msg!(self.0, "drain"; -> ());
        }
    }
}

/// `CVPixelBufferLockBaseAddress(.., ReadOnly)`, unlocked on drop.
struct PixelLock(CfTypeRef);

impl PixelLock {
    /// # Safety
    /// `pb` is a live `CVPixelBuffer` that outlives the lock.
    unsafe fn new(pb: CfTypeRef) -> Result<PixelLock, DecodeError> {
        let r = CVPixelBufferLockBaseAddress(pb, PIXEL_BUFFER_LOCK_READ_ONLY);
        if r != CV_RETURN_SUCCESS {
            return Err(framework("CVPixelBufferLockBaseAddress", format!("CVReturn {r}")));
        }
        Ok(PixelLock(pb))
    }
}

impl Drop for PixelLock {
    fn drop(&mut self) {
        // SAFETY: locked in `new`; the buffer is still alive (see `new`).
        unsafe {
            CVPixelBufferUnlockBaseAddress(self.0, PIXEL_BUFFER_LOCK_READ_ONLY);
        }
    }
}

pub(crate) fn framework(api: &'static str, detail: impl Into<String>) -> DecodeError {
    DecodeError::Framework { api, detail: detail.into() }
}

pub(crate) fn class(name: &'static CStr) -> Result<Id, DecodeError> {
    // SAFETY: `name` is NUL-terminated.
    let c = unsafe { objc_getClass(name.as_ptr()) };
    if c.is_null() {
        Err(framework("objc_getClass", format!("{} is not loaded", name.to_string_lossy())))
    } else {
        Ok(c)
    }
}

/// `-[NSError localizedDescription]` as a Rust string; copied out before the
/// autorelease pool drains.
///
/// # Safety
/// `err` is NULL or a live `NSError`, and a pool is active.
pub(crate) unsafe fn ns_error_text(err: Id) -> String {
    if err.is_null() {
        return "no error object".to_string();
    }
    let desc: Id = msg!(err, "localizedDescription"; -> Id);
    if desc.is_null() {
        return "no description".to_string();
    }
    let utf8: *const c_char = msg!(desc, "UTF8String"; -> *const c_char);
    if utf8.is_null() {
        return "undescribable error".to_string();
    }
    CStr::from_ptr(utf8).to_string_lossy().into_owned()
}

pub(crate) fn file_url(path: &Path) -> Result<CfRef, DecodeError> {
    use std::os::unix::ffi::OsStrExt;
    let bytes = path.as_os_str().as_bytes();
    let len = CfIndex::try_from(bytes.len()).map_err(|_| framework("CFURLCreateFromFileSystemRepresentation", "path too long"))?;
    // SAFETY: pointer and length describe `bytes`; the result is +1 or NULL.
    unsafe { CfRef::take(CFURLCreateFromFileSystemRepresentation(std::ptr::null(), bytes.as_ptr(), len, 0)) }
        .ok_or_else(|| framework("CFURLCreateFromFileSystemRepresentation", format!("no URL for {}", path.display())))
}

/// Size and kind checks shared by images and videos; returns the file size.
fn stat_file(path: &Path) -> Result<u64, DecodeError> {
    let meta = std::fs::metadata(path).map_err(|e| io_err(path, e))?;
    if !meta.is_file() {
        return Err(DecodeError::Unrecognized { path: path.to_path_buf(), detail: "not a regular file".into() });
    }
    if meta.len() == 0 {
        return Err(DecodeError::Empty { path: path.to_path_buf() });
    }
    Ok(meta.len())
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

fn status_name(status: i32) -> &'static str {
    match status {
        -5 => "unexpected end of file",
        -4 => "invalid data",
        -3 => "unknown type",
        -2 => "reading header",
        -1 => "incomplete",
        0 => "complete",
        _ => "unknown",
    }
}

/// A PNG's chunks must be whole (length, type, data and CRC) up to `IEND`, or
/// to the end of the data at a chunk boundary after image data started. ImageIO
/// accepts a PNG cut inside its last chunk's CRC as a complete picture.
/// Lengths and boundaries only: no CRC is verified here (ImageIO checks them).
pub fn check_png(b: &[u8]) -> Result<(), String> {
    let mut at = 8usize;
    let mut saw_idat = false;
    while at < b.len() {
        if b.len() - at < 12 {
            return Err(format!("PNG cut inside a chunk header at byte {at}"));
        }
        let len = u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
        let tag = &b[at + 4..at + 8];
        let end = at.checked_add(12).and_then(|v| v.checked_add(len)).ok_or("PNG chunk length overflows")?;
        if end > b.len() {
            return Err(format!("PNG chunk {:?} at byte {at} declares {len} bytes but the file ends {} bytes short", String::from_utf8_lossy(tag), end - b.len()));
        }
        saw_idat |= tag == b"IDAT";
        if tag == b"IEND" {
            return Ok(());
        }
        at = end;
    }
    if saw_idat {
        Ok(())
    } else {
        Err("PNG has no image data chunk".to_string())
    }
}

/// A JPEG's marker segments must run, through at least one scan, to its
/// end-of-image marker. ImageIO reports "complete" for a JPEG cut a few bytes
/// into its entropy-coded data (measured on a VISEM frame: every cut from 14 to
/// 78 bytes after the scan header decodes as an all-black picture with status
/// 0), so the status alone does not catch truncation. The walk skips segment
/// payloads by their declared length (so an EXIF thumbnail's own scan and EOI
/// are not mistaken for the picture's), and inside a scan skips stuffed bytes
/// (`FF 00`), restarts (`FF D0..D7`) and fill (`FF FF`). A file that lost only
/// its EOI is refused too.
pub fn check_jpeg(b: &[u8]) -> Result<(), String> {
    let mut i = 2usize; // past SOI
    let mut saw_scan = false;
    while i + 1 < b.len() {
        if b[i] != 0xff {
            return Err(format!("JPEG: expected a marker at byte {i}"));
        }
        let marker = b[i + 1];
        match marker {
            0xff => i += 1, // fill byte
            0x01 | 0xd0..=0xd8 => i += 2,
            0xd9 => return if saw_scan { Ok(()) } else { Err("JPEG ends before any scan".to_string()) },
            _ => {
                let Some(len) = b.get(i + 2..i + 4).map(|l| usize::from(u16::from_be_bytes([l[0], l[1]]))) else {
                    return Err(format!("JPEG cut inside the segment header at byte {i}"));
                };
                if len < 2 || i + 2 + len > b.len() {
                    return Err(format!("JPEG segment ff{marker:02x} at byte {i} declares {len} bytes but the file ends first"));
                }
                i += 2 + len;
                if marker == 0xda {
                    saw_scan = true;
                    // Entropy-coded data up to the next real marker.
                    while i + 1 < b.len() && !(b[i] == 0xff && b[i + 1] != 0x00 && b[i + 1] != 0xff && !(0xd0..=0xd7).contains(&b[i + 1])) {
                        i += 1;
                    }
                }
            }
        }
    }
    Err("JPEG has no end-of-image marker".to_string())
}

/// Which decoder produced an image ([`decode_image_with_source`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    /// [`crate::jpeg::decode_jpeg`]: byte-identical to Pillow 13.0.0 on
    /// libjpeg-turbo 3.2.0 (`Image.open(..).convert("RGB")`).
    PillowExactJpeg,
    /// ImageIO, drawn into sRGB: PNG and every other format, and the JPEG
    /// variants [`crate::jpeg`] refuses by name (progressive, arithmetic,
    /// 12-bit, CMYK, ...). Its JPEG decode differs from Pillow's in the last
    /// bits and its colour management differs on tagged or gray images.
    ImageIo,
}

impl Decoder {
    /// A stable name for logs and result files.
    pub fn name(self) -> &'static str {
        match self {
            Self::PillowExactJpeg => "pillow-exact-jpeg",
            Self::ImageIo => "imageio",
        }
    }
}

impl fmt::Display for Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Decode an image file to 8-bit RGB: a JPEG through [`crate::jpeg`]
/// (Pillow-exact), anything else, and the JPEG variants that decoder refuses,
/// through ImageIO. See the module documentation for the colour, alpha,
/// orientation and truncation behaviour, and [`decode_image_with_source`] to
/// learn which decoder ran.
pub fn decode_image(path: impl AsRef<Path>) -> Result<Rgb8, DecodeError> {
    decode_image_with_source(path).map(|(img, _)| img)
}

/// [`decode_image`], also reporting the decoder that produced the pixels. A
/// damaged or truncated JPEG is an error, never a fall-back to ImageIO: only a
/// JPEG the native decoder refuses by name ([`crate::jpeg::Unsupported`])
/// is handed to ImageIO.
pub fn decode_image_with_source(path: impl AsRef<Path>) -> Result<(Rgb8, Decoder), DecodeError> {
    let path = path.as_ref();
    let bytes = read_image_file(path)?;
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        match crate::jpeg::decode_jpeg(&bytes) {
            Ok(img) => return Ok((img, Decoder::PillowExactJpeg)),
            Err(crate::jpeg::JpegError::Unsupported(_)) => {}
            Err(crate::jpeg::JpegError::TooLarge { width, height }) => {
                return Err(DecodeError::TooLarge { path: path.to_path_buf(), detail: format!("{width}x{height} exceeds {MAX_FRAME_PIXELS} pixels") });
            }
            Err(e) => return Err(DecodeError::Damaged { path: path.to_path_buf(), detail: e.to_string() }),
        }
    }
    decode_image_bytes(path, &bytes).map(|img| (img, Decoder::ImageIo))
}

/// Decode an image file with ImageIO only, JPEG included. This is the path
/// [`decode_image`] used for JPEG before [`crate::jpeg`] existed; it is kept so
/// that its difference from Pillow stays measurable (`tests/decode.rs`).
pub fn decode_image_imageio(path: impl AsRef<Path>) -> Result<Rgb8, DecodeError> {
    let path = path.as_ref();
    let bytes = read_image_file(path)?;
    decode_image_bytes(path, &bytes)
}

fn read_image_file(path: &Path) -> Result<Vec<u8>, DecodeError> {
    let size = stat_file(path)?;
    if size > MAX_IMAGE_FILE_BYTES {
        return Err(DecodeError::TooLarge { path: path.to_path_buf(), detail: format!("{size} bytes exceeds {MAX_IMAGE_FILE_BYTES}") });
    }
    std::fs::read(path).map_err(|e| io_err(path, e))
}

fn decode_image_bytes(path: &Path, bytes: &[u8]) -> Result<Rgb8, DecodeError> {
    let damaged = |detail: String| DecodeError::Damaged { path: path.to_path_buf(), detail };
    let unrecognized = |detail: &str| DecodeError::Unrecognized { path: path.to_path_buf(), detail: detail.to_string() };
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        check_png(bytes).map_err(damaged)?;
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        check_jpeg(bytes).map_err(damaged)?;
    }
    let len = CfIndex::try_from(bytes.len()).map_err(|_| framework("CFDataCreate", "file too large"))?;
    // SAFETY: pointer and length describe `bytes`; CFDataCreate copies; +1 or NULL.
    let data = unsafe { CfRef::take(CFDataCreate(std::ptr::null(), bytes.as_ptr(), len)) }.ok_or_else(|| framework("CFDataCreate", "no memory"))?;
    // SAFETY: `data` is a live CFData; NULL options are allowed; +1 or NULL.
    let source = unsafe { CfRef::take(CGImageSourceCreateWithData(data.as_ptr(), std::ptr::null())) }
        .ok_or_else(|| unrecognized("ImageIO makes no image source from these bytes"))?;
    // SAFETY: `source` is a live CGImageSource.
    if unsafe { CGImageSourceGetCount(source.as_ptr()) } == 0 {
        return Err(unrecognized("the source holds no images"));
    }
    // The header's size is read before any pixels are decoded, so the bound is
    // enforced before ImageIO allocates.
    // SAFETY: `source` is live; the copy is +1 or NULL.
    let props = unsafe { CfRef::take(CGImageSourceCopyPropertiesAtIndex(source.as_ptr(), 0, std::ptr::null())) }
        .ok_or_else(|| damaged(status_text(&source, "no properties for the first image")))?;
    // SAFETY: `props` is a live CFDictionary; the keys are SDK constants.
    let (pw, ph) = unsafe { (dict_i64(props.as_ptr(), kCGImagePropertyPixelWidth), dict_i64(props.as_ptr(), kCGImagePropertyPixelHeight)) };
    let (Some(pw), Some(ph)) = (pw, ph) else {
        return Err(damaged(status_text(&source, "the header has no pixel size")));
    };
    let (pw, ph) = (usize::try_from(pw).unwrap_or(0), usize::try_from(ph).unwrap_or(0));
    if pw == 0 || ph == 0 {
        return Err(damaged(status_text(&source, "the image has no pixels")));
    }
    match pw.checked_mul(ph) {
        Some(p) if p <= MAX_FRAME_PIXELS => {}
        _ => return Err(DecodeError::TooLarge { path: path.to_path_buf(), detail: format!("{pw}x{ph} exceeds {MAX_FRAME_PIXELS} pixels") }),
    }
    // SAFETY: `source` is live; the image is +1 or NULL.
    let image = unsafe { CfRef::take(CGImageSourceCreateImageAtIndex(source.as_ptr(), 0, std::ptr::null())) }
        .ok_or_else(|| damaged(status_text(&source, "ImageIO returned no image")))?;
    // SAFETY: `image` is a live CGImage.
    let (w, h) = unsafe { (CGImageGetWidth(image.as_ptr()), CGImageGetHeight(image.as_ptr())) };
    if (w, h) != (pw, ph) {
        return Err(damaged(status_text(&source, &format!("the header says {pw}x{ph} but the image is {w}x{h}"))));
    }
    let rgbx = draw_srgb_rgbx(&image, w, h)?;
    // A truncated or damaged file decodes as far as it goes; ImageIO says so only
    // through the status, which is final once the pixels have been drawn.
    // SAFETY: `source` is live.
    let status = unsafe { CGImageSourceGetStatusAtIndex(source.as_ptr(), 0) };
    if status != STATUS_COMPLETE {
        return Err(damaged(format!("ImageIO status {status} ({})", status_name(status))));
    }
    let mut rgb = Vec::with_capacity(w * h * 3);
    for px in rgbx.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    Ok(Rgb8::new(w, h, rgb))
}

fn status_text(source: &CfRef, why: &str) -> String {
    // SAFETY: `source` is a live CGImageSource.
    let status = unsafe { CGImageSourceGetStatusAtIndex(source.as_ptr(), 0) };
    format!("{why} (ImageIO status {status}, {})", status_name(status))
}

/// An `i64` from a `CFNumber` under `key` in `dict`; `None` when absent or not a number.
///
/// # Safety
/// `dict` is a live `CFDictionary` and `key` a live CF string constant.
unsafe fn dict_i64(dict: CfTypeRef, key: CfTypeRef) -> Option<i64> {
    let value = CFDictionaryGetValue(dict, key);
    if value.is_null() || CFGetTypeID(value) != CFNumberGetTypeID() {
        return None;
    }
    let mut out: i64 = 0;
    // SAFETY: `value` is a CFNumber (checked); `out` is the i64 kCFNumberSInt64Type names.
    if CFNumberGetValue(value, CF_NUMBER_SINT64_TYPE, (&mut out as *mut i64).cast()) != 0 {
        Some(out)
    } else {
        None
    }
}

/// Draw `image` into a zero-filled 8-bit RGBX bitmap in sRGB, 1:1 (no scaling).
fn draw_srgb_rgbx(image: &CfRef, w: usize, h: usize) -> Result<Vec<u8>, DecodeError> {
    let mut buf = vec![0u8; w * h * 4];
    // SAFETY: the name is an SDK constant; +1 or NULL.
    let space = unsafe { CfRef::take(CGColorSpaceCreateWithName(kCGColorSpaceSRGB)) }.ok_or_else(|| framework("CGColorSpaceCreateWithName", "no sRGB colour space"))?;
    {
        // SAFETY: `buf` holds `h` rows of `w * 4` bytes and outlives `ctx`, which is
        // dropped (released) before `buf` is read; 8 bits per component, 4 bytes
        // per pixel and kCGImageAlphaNoneSkipLast is a supported RGB format; +1 or NULL.
        let ctx = unsafe { CfRef::take(CGBitmapContextCreate(buf.as_mut_ptr().cast(), w, h, 8, w * 4, space.as_ptr(), ALPHA_NONE_SKIP_LAST)) }
            .ok_or_else(|| framework("CGBitmapContextCreate", format!("no {w}x{h} RGBX context")))?;
        let rect = CgRect { x: 0.0, y: 0.0, width: w as f64, height: h as f64 };
        // SAFETY: `ctx` and `image` are live; the rect is the whole bitmap.
        unsafe { CGContextDrawImage(ctx.as_ptr(), rect, image.as_ptr()) };
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

/// What [`VideoReader::open`] learned about the first video track.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    /// Delivered frame size, after [`VideoInfo::transform`] was applied.
    pub width: usize,
    pub height: usize,
    /// The size the track stores, before the transform.
    pub stored_width: usize,
    pub stored_height: usize,
    /// Frames in the track, counted from the sample table (not decoded).
    pub frame_count: usize,
    /// `frame_count / duration`.
    pub fps: f64,
    /// The track's `nominalFrameRate` (what the header says).
    pub nominal_fps: f64,
    /// Seconds from the earliest presentation time to the latest frame's end.
    pub duration: f64,
    /// `[a, b, c, d]` of the track's `preferredTransform`, applied to every
    /// frame; identity is `[1, 0, 0, 1]`.
    pub transform: [i32; 4],
}

/// `AVAssetReaderTrackOutput`'s frames, one at a time, as 8-bit RGB.
pub struct VideoReader {
    path: PathBuf,
    // Field order is drop order: output and reader before the track and asset
    // they were made from.
    output: Obj,
    reader: Obj,
    _track: Obj,
    _asset: Obj,
    info: VideoInfo,
    next_index: usize,
    done: bool,
}

/// The integer `[a, b, c, d]` of a pure right-angle rotation or flip, or `None`.
fn right_angle_transform(t: [f64; 4]) -> Option<[i32; 4]> {
    let snap = |v: f64| -> Option<i32> {
        [-1, 0, 1].into_iter().find(|&n| (v - f64::from(n)).abs() < 1e-3)
    };
    let [a, b, c, d] = [snap(t[0])?, snap(t[1])?, snap(t[2])?, snap(t[3])?];
    let axis_aligned = a != 0 && d != 0 && b == 0 && c == 0;
    let swapped = b != 0 && c != 0 && a == 0 && d == 0;
    (axis_aligned || swapped).then_some([a, b, c, d])
}

/// Apply `[a, b, c, d]` (`dst = [a c; b d] * src`, top-left origin, pixel
/// indices) to an image. `[1, 0, 0, 1]` returns the image unchanged.
fn apply_transform(src: Rgb8, m: [i32; 4]) -> Rgb8 {
    if m == [1, 0, 0, 1] {
        return src;
    }
    let [a, b, c, d] = m;
    let (w, h) = (src.width as i64, src.height as i64);
    let swapped = a == 0;
    let (ow, oh) = if swapped { (src.height, src.width) } else { (src.width, src.height) };
    let ox = if a < 0 { w - 1 } else { 0 } + if c < 0 { h - 1 } else { 0 };
    let oy = if b < 0 { w - 1 } else { 0 } + if d < 0 { h - 1 } else { 0 };
    let mut out = vec![0u8; ow * oh * 3];
    for y in 0..h {
        for x in 0..w {
            let dx = i64::from(a) * x + i64::from(c) * y + ox;
            let dy = i64::from(b) * x + i64::from(d) * y + oy;
            let s = ((y * w + x) * 3) as usize;
            let t = ((dy * ow as i64 + dx) * 3) as usize;
            out[t..t + 3].copy_from_slice(&src.data[s..s + 3]);
        }
    }
    Rgb8::new(ow, oh, out)
}

impl VideoReader {
    /// Open the first video track of `path` and prepare to decode it from frame 0.
    pub fn open(path: impl AsRef<Path>) -> Result<VideoReader, DecodeError> {
        let path = path.as_ref();
        stat_file(path)?;
        let path = std::fs::canonicalize(path).map_err(|e| io_err(path, e))?;
        let _pool = Pool::new()?;
        let url = file_url(&path)?;
        let err = |detail: String| DecodeError::Unrecognized { path: path.clone(), detail };
        // SAFETY: `url` is a CFURL, toll-free bridged to NSURL; the class method
        // takes (NSURL *, NSDictionary * _Nullable) and returns an autoreleased
        // AVURLAsset, retained here.
        let asset = unsafe {
            let c = class(c"AVURLAsset")?;
            Obj::retain(msg!(c, "URLAssetWithURL:options:", url.as_ptr() as Id => Id, std::ptr::null_mut::<c_void>() => Id; -> Id))
        }
        .ok_or_else(|| framework("AVURLAsset", "URLAssetWithURL:options: returned nil"))?;
        // `-tracksWithMediaType:` is deprecated since macOS 15 in favour of
        // `loadTracksWithMediaType:completionHandler:`, which needs an Objective-C
        // block. AVAsset.h documents that synchronous property access is allowed
        // and blocks the calling thread until the value is loaded, which is what
        // we want from a local file; the method is still in the SDK.
        // SAFETY: `asset` is a live AVAsset; `AVMediaTypeVideo` is an SDK constant;
        // the NSArray is autoreleased, and `track` is retained before the pool drains.
        let track = unsafe {
            let tracks: Id = msg!(asset.id(), "tracksWithMediaType:", AVMediaTypeVideo => Id; -> Id);
            if tracks.is_null() {
                return Err(err("AVFoundation returned no track list".into()));
            }
            let count: usize = msg!(tracks, "count"; -> usize);
            if count == 0 {
                return Err(err("AVFoundation found no video track (not a video file, or its container is damaged)".into()));
            }
            let first: Id = msg!(tracks, "objectAtIndex:", 0usize => usize; -> Id);
            Obj::retain(first).ok_or_else(|| framework("NSArray", "objectAtIndex:0 returned nil"))?
        };
        // SAFETY: `track` is a live AVAssetTrack; these are plain property reads
        // (deprecated for Swift only, per the header) returning float, CGSize and
        // CGAffineTransform by value.
        let (nominal, natural, transform) = unsafe {
            (
                msg!(track.id(), "nominalFrameRate"; -> f32),
                msg!(track.id(), "naturalSize"; -> CgSize),
                msg!(track.id(), "preferredTransform"; -> CgAffineTransform),
            )
        };
        let unsupported = |detail: String| DecodeError::Unsupported { path: path.clone(), detail };
        let m = right_angle_transform([transform.a, transform.b, transform.c, transform.d]).ok_or_else(|| {
            unsupported(format!("the track's preferredTransform [{} {} {} {}] is not a right-angle rotation or a flip", transform.a, transform.b, transform.c, transform.d))
        })?;
        let (sw, sh) = (natural.width.round(), natural.height.round());
        if !(sw >= 1.0 && sh >= 1.0 && sw.is_finite() && sh.is_finite()) || (sw * sh) > MAX_FRAME_PIXELS as f64 {
            return Err(unsupported(format!("track size {}x{} is outside 1..={MAX_FRAME_PIXELS} pixels", natural.width, natural.height)));
        }
        let (stored_width, stored_height) = (sw as usize, sh as usize);
        let (width, height) = if m[0] == 0 { (stored_height, stored_width) } else { (stored_width, stored_height) };

        let (frame_count, duration) = count_samples(&path, &asset, &track)?;
        let fps = frame_count as f64 / duration;

        // The decode reader: BGRA frames.
        // SAFETY: Foundation constructors; the key is a CFString constant (an
        // NSString) and the value an NSNumber, both live for the call.
        let settings = unsafe {
            let number: Id = msg!(class(c"NSNumber")?, "numberWithUnsignedInt:", PIXEL_FORMAT_32BGRA => u32; -> Id);
            msg!(class(c"NSDictionary")?, "dictionaryWithObject:forKey:", number => Id, kCVPixelBufferPixelFormatTypeKey as Id => Id; -> Id)
        };
        if settings.is_null() {
            return Err(framework("NSDictionary", "dictionaryWithObject:forKey: returned nil"));
        }
        let (reader, output) = new_reader(&path, &asset, &track, settings)?;
        start_reading(&path, &reader)?;
        Ok(VideoReader {
            path: path.clone(),
            output,
            reader,
            _track: track,
            _asset: asset,
            info: VideoInfo { width, height, stored_width, stored_height, frame_count, fps, nominal_fps: f64::from(nominal), duration, transform: m },
            next_index: 0,
            done: false,
        })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    /// The next frame as `(index, presentation time in seconds, RGB)`, or
    /// `Ok(None)` after the last one. Reaching the end with a different number of
    /// frames than [`VideoInfo::frame_count`] is an error; after any error the
    /// reader is finished and returns `Ok(None)`.
    pub fn next_frame(&mut self) -> Result<Option<(usize, f64, Rgb8)>, DecodeError> {
        if self.done {
            return Ok(None);
        }
        let r = self.read_next();
        if !matches!(r, Ok(Some(_))) {
            self.done = true;
        }
        r
    }

    fn read_next(&mut self) -> Result<Option<(usize, f64, Rgb8)>, DecodeError> {
        let _pool = Pool::new()?;
        loop {
            // SAFETY: `output` is a live AVAssetReaderTrackOutput on a started reader;
            // copyNextSampleBuffer is CF_RETURNS_RETAINED, so a non-NULL result is ours.
            let sample = unsafe { CfRef::take(msg!(self.output.id(), "copyNextSampleBuffer"; -> CfTypeRef)) };
            let Some(sample) = sample else {
                return self.finish();
            };
            // SAFETY: `sample` is a live CMSampleBuffer.
            if unsafe { CMSampleBufferGetNumSamples(sample.as_ptr()) } == 0 {
                continue; // a marker-only buffer
            }
            let damaged = |detail: String| DecodeError::Damaged { path: self.path.clone(), detail };
            if self.next_index >= self.info.frame_count {
                return Err(damaged(format!("more frames than the {} the sample table lists", self.info.frame_count)));
            }
            // SAFETY: as above; the time is returned by value.
            let pts = unsafe { CMSampleBufferGetPresentationTimeStamp(sample.as_ptr()) };
            if pts.flags & CMTIME_VALID == 0 {
                return Err(damaged(format!("frame {} has no valid presentation time", self.next_index)));
            }
            // SAFETY: as above.
            let seconds = unsafe { CMTimeGetSeconds(pts) };
            // SAFETY: `sample` is live; the image buffer is borrowed from it (a Get).
            let pb = unsafe { CMSampleBufferGetImageBuffer(sample.as_ptr()) };
            if pb.is_null() {
                return Err(damaged(format!("frame {} has no image buffer", self.next_index)));
            }
            let rgb = bgra_to_rgb(&self.path, pb, self.info.stored_width, self.info.stored_height)?;
            let rgb = apply_transform(rgb, self.info.transform);
            let index = self.next_index;
            self.next_index += 1;
            return Ok(Some((index, seconds, rgb)));
        }
    }

    /// `copyNextSampleBuffer` returned NULL: the reader's status says whether
    /// that was the end or a failure.
    fn finish(&self) -> Result<Option<(usize, f64, Rgb8)>, DecodeError> {
        // SAFETY: `reader` is a live AVAssetReader; status and error are thread-safe reads.
        let (status, text) = unsafe {
            let status: isize = msg!(self.reader.id(), "status"; -> isize);
            let err: Id = msg!(self.reader.id(), "error"; -> Id);
            (status, ns_error_text(err))
        };
        let damaged = |detail: String| DecodeError::Damaged { path: self.path.clone(), detail };
        match status {
            READER_STATUS_COMPLETED if self.next_index == self.info.frame_count => Ok(None),
            READER_STATUS_COMPLETED => Err(damaged(format!("decoded {} frames but the sample table lists {}", self.next_index, self.info.frame_count))),
            READER_STATUS_FAILED => Err(damaged(format!("AVAssetReader failed after {} frames: {text}", self.next_index))),
            s => Err(damaged(format!("AVAssetReader stopped with status {s} after {} frames", self.next_index))),
        }
    }
}

impl Iterator for VideoReader {
    type Item = Result<(usize, f64, Rgb8), DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_frame().transpose()
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        if let Ok(_pool) = Pool::new() {
            // SAFETY: `reader` is live; cancelling a reader that is still reading is
            // what AVAssetReader requires before it is released (and must not run
            // concurrently with copyNextSampleBuffer, which `&mut self` rules out).
            unsafe {
                let status: isize = msg!(self.reader.id(), "status"; -> isize);
                if status == READER_STATUS_READING {
                    let _: () = msg!(self.reader.id(), "cancelReading"; -> ());
                }
            }
        }
    }
}

impl fmt::Debug for VideoReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoReader").field("path", &self.path).field("info", &self.info).field("next_index", &self.next_index).finish()
    }
}

/// An `AVAssetReader` over `track` with one `AVAssetReaderTrackOutput`
/// (`settings` NULL: compressed samples), not yet started.
fn new_reader(path: &Path, asset: &Obj, track: &Obj, settings: Id) -> Result<(Obj, Obj), DecodeError> {
    let failed = |detail: String| DecodeError::Damaged { path: path.to_path_buf(), detail };
    // SAFETY: `+assetReaderWithAsset:error:` takes an AVAsset and an NSError**, and
    // returns an autoreleased reader or nil with the error set.
    let (reader, error_text) = unsafe {
        let mut error: Id = std::ptr::null_mut();
        let r: Id = msg!(class(c"AVAssetReader")?, "assetReaderWithAsset:error:", asset.id() => Id, &mut error as *mut Id => *mut Id; -> Id);
        (Obj::retain(r), ns_error_text(error))
    };
    let reader = reader.ok_or_else(|| failed(format!("AVAssetReader: {error_text}")))?;
    // SAFETY: `+assetReaderTrackOutputWithTrack:outputSettings:` takes a track and a
    // nullable settings dictionary and returns an autoreleased output, retained here.
    let output = unsafe {
        Obj::retain(msg!(class(c"AVAssetReaderTrackOutput")?, "assetReaderTrackOutputWithTrack:outputSettings:", track.id() => Id, settings => Id; -> Id))
    }
    .ok_or_else(|| framework("AVAssetReaderTrackOutput", "assetReaderTrackOutputWithTrack:outputSettings: returned nil"))?;
    // SAFETY: both are live; `canAddOutput:` is checked first because `addOutput:`
    // raises an Objective-C exception for an output that cannot be added.
    unsafe {
        let can: bool = msg!(reader.id(), "canAddOutput:", output.id() => Id; -> bool);
        if !can {
            return Err(failed("AVAssetReader cannot read this track with these settings".into()));
        }
        let _: () = msg!(reader.id(), "addOutput:", output.id() => Id; -> ());
    }
    Ok((reader, output))
}

fn start_reading(path: &Path, reader: &Obj) -> Result<(), DecodeError> {
    // SAFETY: `reader` is a live AVAssetReader with one output, not yet started.
    unsafe {
        let started: bool = msg!(reader.id(), "startReading"; -> bool);
        if started {
            return Ok(());
        }
        let err: Id = msg!(reader.id(), "error"; -> Id);
        Err(DecodeError::Damaged { path: path.to_path_buf(), detail: format!("AVAssetReader would not start: {}", ns_error_text(err)) })
    }
}

/// Pass one: read the track's compressed samples (no decoding) to count frames
/// and measure the duration (earliest presentation time to the latest frame's
/// end, in seconds). Errors when the reader fails, the video has no frames or no
/// positive duration, or it exceeds [`MAX_VIDEO_FRAMES`].
fn count_samples(path: &Path, asset: &Obj, track: &Obj) -> Result<(usize, f64), DecodeError> {
    let damaged = |detail: String| DecodeError::Damaged { path: path.to_path_buf(), detail };
    let (reader, output) = new_reader(path, asset, track, std::ptr::null_mut())?;
    start_reading(path, &reader)?;
    let (mut frames, mut begin, mut end) = (0usize, f64::INFINITY, f64::NEG_INFINITY);
    loop {
        let _pool = Pool::new()?;
        // SAFETY: as in `read_next`.
        let Some(sample) = (unsafe { CfRef::take(msg!(output.id(), "copyNextSampleBuffer"; -> CfTypeRef)) }) else {
            break;
        };
        // SAFETY: `sample` is a live CMSampleBuffer; times are returned by value.
        let (n, pts, dur) = unsafe { (CMSampleBufferGetNumSamples(sample.as_ptr()), CMSampleBufferGetPresentationTimeStamp(sample.as_ptr()), CMSampleBufferGetDuration(sample.as_ptr())) };
        if n == 0 {
            continue; // a marker-only buffer
        }
        if n != 1 {
            return Err(DecodeError::Unsupported { path: path.to_path_buf(), detail: format!("a sample buffer holds {n} samples; one frame per sample is expected") });
        }
        if pts.flags & CMTIME_VALID == 0 || dur.flags & CMTIME_VALID == 0 {
            return Err(damaged(format!("frame {frames} has no valid presentation time or duration")));
        }
        // SAFETY: plain value conversions.
        let (start, length) = unsafe { (CMTimeGetSeconds(pts), CMTimeGetSeconds(dur)) };
        if !(start.is_finite() && length.is_finite() && length >= 0.0) {
            return Err(damaged(format!("frame {frames} has a non-finite time")));
        }
        begin = begin.min(start);
        end = end.max(start + length);
        frames += 1;
        if frames > MAX_VIDEO_FRAMES {
            return Err(DecodeError::TooLarge { path: path.to_path_buf(), detail: format!("more than {MAX_VIDEO_FRAMES} frames") });
        }
    }
    // SAFETY: `reader` is live; status and error are thread-safe reads.
    let (status, text) = unsafe {
        let status: isize = msg!(reader.id(), "status"; -> isize);
        (status, ns_error_text(msg!(reader.id(), "error"; -> Id)))
    };
    if status != READER_STATUS_COMPLETED {
        return Err(damaged(format!("reading the sample table stopped with status {status} after {frames} frames: {text}")));
    }
    let duration = end - begin;
    if frames == 0 || !(duration > 0.0) {
        return Err(damaged(format!("{frames} frames spanning {duration} s: the video has no frame rate")));
    }
    Ok((frames, duration))
}

/// A locked 32BGRA `CVPixelBuffer` as tightly packed RGB. The buffer must be
/// the stored size and format; its row stride is honoured.
fn bgra_to_rgb(path: &Path, pb: CfTypeRef, width: usize, height: usize) -> Result<Rgb8, DecodeError> {
    // SAFETY: `pb` is a live CVPixelBuffer borrowed from a live sample buffer; the
    // getters read plain properties.
    let (format, w, h, stride) = unsafe { (CVPixelBufferGetPixelFormatType(pb), CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), CVPixelBufferGetBytesPerRow(pb)) };
    if format != PIXEL_FORMAT_32BGRA {
        return Err(DecodeError::Unsupported { path: path.to_path_buf(), detail: format!("pixel format {format:#x}, expected 32BGRA") });
    }
    if (w, h) != (width, height) {
        return Err(DecodeError::Unsupported { path: path.to_path_buf(), detail: format!("a decoded frame is {w}x{h} but the track is {width}x{height}") });
    }
    if stride < w * 4 {
        return Err(DecodeError::Damaged { path: path.to_path_buf(), detail: format!("row stride {stride} is under {} bytes", w * 4) });
    }
    // SAFETY: `pb` outlives the lock (see PixelLock::new).
    let _lock = unsafe { PixelLock::new(pb)? };
    // SAFETY: the buffer is locked.
    let base = unsafe { CVPixelBufferGetBaseAddress(pb) };
    if base.is_null() {
        return Err(framework("CVPixelBufferGetBaseAddress", "NULL for a locked buffer"));
    }
    // SAFETY: while locked, `base` addresses `h` rows of `stride` bytes (CVPixelBuffer
    // contract: `stride = CVPixelBufferGetBytesPerRow`, rows = height for a chunky format).
    let src = unsafe { std::slice::from_raw_parts(base, stride * h) };
    let mut rgb = Vec::with_capacity(w * h * 3);
    for row in src.chunks_exact(stride) {
        for px in row[..w * 4].chunks_exact(4) {
            rgb.extend_from_slice(&[px[2], px[1], px[0]]);
        }
    }
    Ok(Rgb8::new(w, h, rgb))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(i: u8) -> [u8; 3] {
        [i, i + 100, i + 200]
    }

    /// 3 wide, 2 high: pixel values 0..6 row-major.
    fn sample() -> Rgb8 {
        Rgb8::new(3, 2, (0..6u8).flat_map(px).collect())
    }

    fn at(img: &Rgb8, x: usize, y: usize) -> u8 {
        img.data[(y * img.width + x) * 3]
    }

    fn grid(img: &Rgb8) -> Vec<Vec<u8>> {
        (0..img.height).map(|y| (0..img.width).map(|x| at(img, x, y)).collect()).collect()
    }

    #[test]
    fn right_angle_transforms_are_recognised() {
        assert_eq!(right_angle_transform([1.0, 0.0, 0.0, 1.0]), Some([1, 0, 0, 1]));
        assert_eq!(right_angle_transform([0.0, 1.0, -1.0, 0.0]), Some([0, 1, -1, 0]));
        assert_eq!(right_angle_transform([6.1e-17, 1.0, -1.0, 6.1e-17]), Some([0, 1, -1, 0]));
        assert_eq!(right_angle_transform([-1.0, 0.0, 0.0, 1.0]), Some([-1, 0, 0, 1]));
        assert_eq!(right_angle_transform([0.7071, 0.7071, -0.7071, 0.7071]), None);
        assert_eq!(right_angle_transform([2.0, 0.0, 0.0, 2.0]), None);
        assert_eq!(right_angle_transform([1.0, 1.0, 0.0, 1.0]), None);
    }

    #[test]
    fn identity_is_a_no_op() {
        assert_eq!(apply_transform(sample(), [1, 0, 0, 1]), sample());
    }

    #[test]
    fn rotate_90_clockwise() {
        // [[0 1 2], [3 4 5]] turned clockwise: first column read bottom-up becomes the first row.
        let r = apply_transform(sample(), [0, 1, -1, 0]);
        assert_eq!((r.width, r.height), (2, 3));
        assert_eq!(grid(&r), vec![vec![3, 0], vec![4, 1], vec![5, 2]]);
    }

    #[test]
    fn rotate_90_counter_clockwise() {
        let r = apply_transform(sample(), [0, -1, 1, 0]);
        assert_eq!((r.width, r.height), (2, 3));
        assert_eq!(grid(&r), vec![vec![2, 5], vec![1, 4], vec![0, 3]]);
    }

    #[test]
    fn rotate_180_and_flips() {
        assert_eq!(grid(&apply_transform(sample(), [-1, 0, 0, -1])), vec![vec![5, 4, 3], vec![2, 1, 0]]);
        assert_eq!(grid(&apply_transform(sample(), [-1, 0, 0, 1])), vec![vec![2, 1, 0], vec![5, 4, 3]]);
        assert_eq!(grid(&apply_transform(sample(), [1, 0, 0, -1])), vec![vec![3, 4, 5], vec![0, 1, 2]]);
    }

    #[test]
    fn four_clockwise_turns_return_the_image() {
        let mut img = sample();
        for _ in 0..4 {
            img = apply_transform(img, [0, 1, -1, 0]);
        }
        assert_eq!(img, sample());
    }

    #[test]
    fn rgb_channels_stay_together_under_rotation() {
        let r = apply_transform(sample(), [0, 1, -1, 0]);
        for y in 0..r.height {
            for x in 0..r.width {
                let v = at(&r, x, y);
                assert_eq!(&r.data[(y * r.width + x) * 3..][..3], &px(v));
            }
        }
    }

    #[test]
    fn png_chunk_check_finds_cuts() {
        let chunk = |tag: &[u8; 4], data: &[u8]| {
            let mut v = (data.len() as u32).to_be_bytes().to_vec();
            v.extend_from_slice(tag);
            v.extend_from_slice(data);
            v.extend_from_slice(&[0, 0, 0, 0]);
            v
        };
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend(chunk(b"IHDR", &[0; 13]));
        png.extend(chunk(b"IDAT", &[1, 2, 3]));
        png.extend(chunk(b"IEND", &[]));
        // IHDR ends at byte 33, IDAT at 48, IEND (the last chunk) at 60.
        assert_eq!(png.len(), 60);
        assert!(check_png(&png).is_ok());
        assert!(check_png(&png[..59]).is_err(), "cut inside IEND");
        assert!(check_png(&png[..48]).is_ok(), "whole chunks and image data, no IEND (PIL reads these)");
        assert!(check_png(&png[..47]).is_err(), "cut inside IDAT");
        assert!(check_png(&png[..33]).is_err(), "no image data");
    }
}
