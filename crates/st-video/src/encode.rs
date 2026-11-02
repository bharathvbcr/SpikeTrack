//! H.264 `.mp4` writer for macOS (Apple silicon): [`VideoWriter`] encodes 8-bit RGB frames
//! with AVFoundation's `AVAssetWriter`, with no third-party crate. It is the counterpart of
//! [`crate::decode::VideoReader`] and follows its conventions: frameworks are reached through the
//! Objective-C runtime (the shared `msg!`, [`Obj`], [`CfRef`] and [`Pool`] of [`crate::decode`]),
//! every Objective-C object this module retains is released on drop, every call that touches
//! Objective-C runs inside an `NSAutoreleasePool`, and a framework failure is an [`EncodeError`],
//! never a panic.
//!
//! Every selector, constant and function below was checked against the macOS SDK headers
//! (`xcrun --show-sdk-path`, SDK 27.0): `AVFoundation/AVAssetWriter.h`, `AVAssetWriterInput.h`
//! (`AVAssetWriterInputPixelBufferAdaptor`), `AVVideoSettings.h`, `AVMediaFormat.h`,
//! `CoreVideo/CVPixelBuffer.h`, `CVPixelBufferPool.h`, `CoreMedia/CMTime.h` and `usr/include/Block.h`.
//!
//! # What is written
//!
//! `AVFileTypeMPEG4` with one `AVVideoCodecTypeH264` track, 8-bit 4:2:0, at the size of the frames
//! pushed. The file name must end in `.mp4`. Frames are handed to the encoder as 32BGRA pixel
//! buffers from the adaptor's `CVPixelBufferPool`, and the track is tagged BT.709
//! (`AVVideoColorPropertiesKey`): the source buffers are untagged, so AVFoundation tags the output
//! and converts RGB to Y'CbCr with the BT.709 matrix and does no colour matching, which is what a
//! player undoes. The average bit rate is [`BITS_PER_PIXEL`] bits per pixel per frame, clamped to
//! [`MIN_BIT_RATE`]..=[`MAX_BIT_RATE`], because a segmentation overlay is thin coloured outlines on
//! a noisy picture and the unconfigured rate is chosen for camera footage. H.264 is lossy, 4:2:0
//! halves the chroma resolution, and the encoder reorders frames: read-back pixels differ from the
//! input (`tests/encode.rs` measures how much).
//!
//! # Timestamps
//!
//! Frame `i` is presented at `i * 1000 / round(fps * 1000)` seconds, an exact rational: the track
//! time scale is `round(fps * 1000)` and a frame lasts 1000 ticks, so an fps given to three decimals
//! (29.97, 47.952) is stored exactly and one with more digits is rounded to the nearest 0.001
//! ([`VideoWriter::fps`] reports what was stored). The session ends at `frames / fps`, so the last
//! frame lasts one frame period and `duration = frames / fps` exactly.
//!
//! # Boundary checks
//!
//! [`VideoWriter::create`] refuses, with a typed error and before AVFoundation is touched: a
//! width or height that is odd (4:2:0 chroma is half resolution; `AVVideoSettings.h` asks for even
//! sizes), outside [`MIN_DIMENSION`]..=[`MAX_DIMENSION`] or over [`MAX_PIXELS`]; an fps outside
//! [`MIN_FPS`]..=[`MAX_FPS`]; a path that does not end in `.mp4`, has no existing parent directory
//! or is a directory; and a file that already exists, unless [`WriterOptions::overwrite`] is set.
//! The system encoder has the last word on what it accepts (`canApplyOutputSettings:forMediaType:`
//! is asked, and a refusal is [`EncodeError::Unsupported`]). [`VideoWriter::push`] refuses a frame
//! whose size is not the writer's, or whose pixel vector is not `width * height * 3` bytes
//! (the fields of [`Rgb8`] are public), before any buffer is locked.
//!
//! # Back pressure and the exceptions AVFoundation throws
//!
//! `appendPixelBuffer:withPresentationTime:` throws an Objective-C exception when the input is not
//! `readyForMoreMediaData`, and an exception that unwinds into Rust aborts the process. So every
//! push first waits for the flag in a loop bounded by [`READY_TIMEOUT`] (1 ms sleeps; a writer that
//! has failed in the meantime is reported at once, not after the timeout), and every other call that
//! documents an exception is guarded by its `can...` test or ordered after the call it needs
//! (`startWriting` before the pool, the session before the first append).
//!
//! # Finishing
//!
//! [`VideoWriter::finish`] calls `markAsFinished`, `endSessionAtSourceTime:` and then
//! `finishWritingWithCompletionHandler:`, and polls `AVAssetWriter.status` (documented thread safe)
//! until it leaves `Writing`, for at most [`FINISH_TIMEOUT`]. `Completed` is success; `Failed` is an
//! error carrying the `NSError`'s text. The deprecated synchronous `finishWriting` is not used:
//! `AVAssetWriter.h` says that calling it on the main thread "can cause the finishWriting operation
//! to fail", `spiketrack track` runs on the main thread, and it cannot be given a timeout. The
//! asynchronous call takes an Objective-C block, and the block is the smallest one there is: a
//! *global* block (`_NSConcreteGlobalBlock`, `BLOCK_IS_GLOBAL`) that captures nothing and does
//! nothing, so it needs no copy or dispose helper, `Block_copy` returns it unchanged, and it is a
//! `static`. Completion is read from the status, not signalled by the block. Its layout is the
//! one in Clang's Block ABI and in the `block2` crate's `BlockHeader`/`GlobalBlock` (read as a
//! reference, not used as a dependency); `tests/encode.rs` finishes real files, which would crash
//! if it were wrong.
//!
//! A writer dropped without [`VideoWriter::finish`] calls `cancelWriting` (which blocks until the
//! session is cancelled and deletes the output file) and removes whatever is left at the path, so
//! no process is left waiting on a half-written file.

use std::ffi::c_void;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::decode::{
    class, file_url, msg, ns_error_text, CfRef, CfTypeRef, CmTime, DecodeError, Id, Obj, Pool,
    CV_RETURN_SUCCESS, MAX_VIDEO_FRAMES, PIXEL_FORMAT_32BGRA,
};
use crate::resize::Rgb8;

/// Smallest width or height accepted, in pixels.
pub const MIN_DIMENSION: usize = 16;
/// Largest width or height accepted, in pixels.
pub const MAX_DIMENSION: usize = 8192;
/// Most pixels in one frame (8K UHD).
pub const MAX_PIXELS: usize = 8192 * 4320;
/// Lowest frame rate accepted.
pub const MIN_FPS: f64 = 1.0;
/// Highest frame rate accepted.
pub const MAX_FPS: f64 = 240.0;
/// The H.264 average bit rate, per pixel per frame.
pub const BITS_PER_PIXEL: f64 = 0.25;
/// Lower clamp of the average bit rate, bits per second.
pub const MIN_BIT_RATE: f64 = 1.0e6;
/// Upper clamp of the average bit rate, bits per second.
pub const MAX_BIT_RATE: f64 = 100.0e6;
/// The longest [`VideoWriter::push`] waits for the encoder to accept another frame.
pub const READY_TIMEOUT: Duration = Duration::from_secs(60);
/// The longest [`VideoWriter::finish`] waits for the file to be completed.
pub const FINISH_TIMEOUT: Duration = Duration::from_secs(180);
/// Poll interval of both waits.
const POLL: Duration = Duration::from_millis(1);
/// Ticks one frame lasts; the time scale is `round(fps * TICKS_PER_FRAME)`.
const TICKS_PER_FRAME: i64 = 1000;

/// `AVAssetWriterStatus` (`AVAssetWriter.h`, `NSInteger`).
const STATUS_WRITING: isize = 1;
const STATUS_COMPLETED: isize = 2;
const STATUS_FAILED: isize = 3;
const STATUS_CANCELLED: isize = 4;
/// `kCVPixelBufferLock_*` flags are 0 for a read-write lock (`CVPixelBuffer.h`).
const PIXEL_BUFFER_LOCK_READ_WRITE: u64 = 0;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum EncodeError {
    /// Width or height the writer refuses.
    InvalidSize {
        width: usize,
        height: usize,
        detail: String,
    },
    /// A frame rate outside [`MIN_FPS`]..=[`MAX_FPS`], or not a number.
    InvalidFps { fps: f64 },
    /// An output path the writer refuses (extension, parent directory, not a file).
    InvalidPath { path: PathBuf, detail: String },
    /// The output file exists and [`WriterOptions::overwrite`] is not set.
    Exists { path: PathBuf },
    /// A file system operation failed.
    Io { path: PathBuf, err: std::io::Error },
    /// A frame pushed with the wrong size, or whose pixel vector has the wrong length.
    BadFrame { index: usize, detail: String },
    /// The system encoder will not write this video (`canApplyOutputSettings:` said no).
    Unsupported { detail: String },
    /// `finish` was called on a writer that has no frames.
    Empty { path: PathBuf },
    /// More frames than [`MAX_VIDEO_FRAMES`], the most [`crate::decode::VideoReader`] reads.
    TooManyFrames { limit: usize },
    /// A wait was abandoned: the encoder did not accept a frame, or finish the file, in time.
    Timeout {
        what: &'static str,
        waited: Duration,
        frames: usize,
    },
    /// `AVAssetWriter` failed or was cancelled; `detail` is its `NSError`'s text. Every later call on
    /// the writer returns the same error.
    Failed { path: PathBuf, detail: String },
    /// A framework call failed in a way no other variant describes.
    Framework { api: &'static str, detail: String },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize {
                width,
                height,
                detail,
            } => write!(f, "video size {width}x{height}: {detail}"),
            Self::InvalidFps { fps } => {
                write!(f, "frame rate {fps} is outside {MIN_FPS}..={MAX_FPS} fps")
            }
            Self::InvalidPath { path, detail } => write!(f, "{}: {detail}", path.display()),
            Self::Exists { path } => write!(
                f,
                "{}: the file already exists (remove it, or choose another name)",
                path.display()
            ),
            Self::Io { path, err } => write!(f, "{}: {err}", path.display()),
            Self::BadFrame { index, detail } => write!(f, "frame {index}: {detail}"),
            Self::Unsupported { detail } => write!(
                f,
                "the system H.264 encoder will not write this video: {detail}"
            ),
            Self::Empty { path } => write!(
                f,
                "{}: no frame was written, so there is no video",
                path.display()
            ),
            Self::TooManyFrames { limit } => write!(f, "more than {limit} frames"),
            Self::Timeout {
                what,
                waited,
                frames,
            } => write!(
                f,
                "gave up waiting {:.0} s for {what} after {frames} frame(s)",
                waited.as_secs_f64()
            ),
            Self::Failed { path, detail } => {
                write!(f, "{}: AVAssetWriter failed: {detail}", path.display())
            }
            Self::Framework { api, detail } => write!(f, "{api}: {detail}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// The shared Objective-C helpers report a framework failure as a [`DecodeError`]; nothing else
/// can come from them here.
impl From<DecodeError> for EncodeError {
    fn from(e: DecodeError) -> Self {
        match e {
            DecodeError::Framework { api, detail } => Self::Framework { api, detail },
            other => Self::Framework {
                api: "objc",
                detail: other.to_string(),
            },
        }
    }
}

fn fw(api: &'static str, detail: impl Into<String>) -> EncodeError {
    EncodeError::Framework {
        api,
        detail: detail.into(),
    }
}

// ---------------------------------------------------------------------------
// FFI: constants and functions this module adds to decode's
// ---------------------------------------------------------------------------

#[link(name = "AVFoundation", kind = "framework")]
extern "C" {
    static AVMediaTypeVideo: Id;
    static AVFileTypeMPEG4: Id;
    static AVVideoCodecKey: Id;
    static AVVideoCodecTypeH264: Id;
    static AVVideoWidthKey: Id;
    static AVVideoHeightKey: Id;
    static AVVideoCompressionPropertiesKey: Id;
    static AVVideoAverageBitRateKey: Id;
    static AVVideoColorPropertiesKey: Id;
    static AVVideoColorPrimariesKey: Id;
    static AVVideoColorPrimaries_ITU_R_709_2: Id;
    static AVVideoTransferFunctionKey: Id;
    static AVVideoTransferFunction_ITU_R_709_2: Id;
    static AVVideoYCbCrMatrixKey: Id;
    static AVVideoYCbCrMatrix_ITU_R_709_2: Id;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMTimeMake(value: i64, timescale: i32) -> CmTime;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CfTypeRef;
    static kCVPixelBufferWidthKey: CfTypeRef;
    static kCVPixelBufferHeightKey: CfTypeRef;

    fn CVPixelBufferPoolCreatePixelBuffer(
        allocator: CfTypeRef,
        pool: CfTypeRef,
        out: *mut CfTypeRef,
    ) -> i32;
    fn CVPixelBufferLockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CfTypeRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CfTypeRef) -> *const u8;
    fn CVPixelBufferGetBytesPerRow(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CfTypeRef) -> usize;
    fn CVPixelBufferGetPixelFormatType(pb: CfTypeRef) -> u32;
}

// The completion block of `finishWritingWithCompletionHandler:`: a global block that does nothing
// (see the module documentation). Layout per Clang's Block ABI: `{isa, flags, reserved, invoke,
// descriptor}` with a descriptor of `{reserved, size}`, and no signature, copy or dispose.

extern "C" {
    /// `void *_NSConcreteGlobalBlock[32]` (`usr/include/Block.h`), in libSystem.
    static _NSConcreteGlobalBlock: [*const c_void; 32];
}

/// `BLOCK_IS_GLOBAL` (Clang Block ABI; `block2`'s `BlockFlags`).
const BLOCK_IS_GLOBAL: i32 = 1 << 28;

#[repr(C)]
struct BlockDescriptor {
    reserved: std::ffi::c_ulong,
    size: std::ffi::c_ulong,
}

#[repr(C)]
struct GlobalBlock {
    isa: *const c_void,
    flags: i32,
    reserved: i32,
    invoke: unsafe extern "C" fn(*const GlobalBlock),
    descriptor: *const BlockDescriptor,
}

// SAFETY: the block is immutable, holds no state and only points at other immutable statics.
unsafe impl Sync for GlobalBlock {}

unsafe extern "C" fn do_nothing(_: *const GlobalBlock) {}

static NOOP_DESCRIPTOR: BlockDescriptor = BlockDescriptor {
    reserved: 0,
    size: std::mem::size_of::<GlobalBlock>() as std::ffi::c_ulong,
};

static NOOP_BLOCK: GlobalBlock = GlobalBlock {
    isa: &raw const _NSConcreteGlobalBlock as *const c_void,
    flags: BLOCK_IS_GLOBAL,
    reserved: 0,
    invoke: do_nothing,
    descriptor: &NOOP_DESCRIPTOR,
};

// ---------------------------------------------------------------------------
// Boundary checks
// ---------------------------------------------------------------------------

/// What [`VideoWriter::create`] does beyond the defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriterOptions {
    /// Replace an existing regular file at the output path (the default refuses). The old file is
    /// removed before the new one is started, so a failure in between leaves neither.
    pub overwrite: bool,
}

/// The size and frame-rate checks of [`VideoWriter::create`], without touching the file system or
/// AVFoundation. Returns the track time scale, `round(fps * 1000)`.
pub fn check_format(width: usize, height: usize, fps: f64) -> Result<i32, EncodeError> {
    let size = |detail: String| EncodeError::InvalidSize {
        width,
        height,
        detail,
    };
    if width % 2 != 0 || height % 2 != 0 {
        return Err(size(
            "H.264 4:2:0 needs an even width and height".to_string(),
        ));
    }
    if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&width)
        || !(MIN_DIMENSION..=MAX_DIMENSION).contains(&height)
    {
        return Err(size(format!(
            "each side must be {MIN_DIMENSION} to {MAX_DIMENSION} pixels"
        )));
    }
    if width * height > MAX_PIXELS {
        return Err(size(format!("more than {MAX_PIXELS} pixels")));
    }
    if !(fps.is_finite() && (MIN_FPS..=MAX_FPS).contains(&fps)) {
        return Err(EncodeError::InvalidFps { fps });
    }
    Ok((fps * TICKS_PER_FRAME as f64).round() as i32)
}

/// The output path checks of [`VideoWriter::create`], without changing anything: an `.mp4` name in
/// an existing directory, no file there yet (or a regular file to replace, when `overwrite`).
/// Returns the absolute path, with the parent directory canonicalised. A caller that wants to fail early (before loading a model, say) calls this with the
/// options it will create the writer with.
pub fn check_output(path: impl AsRef<Path>, overwrite: bool) -> Result<PathBuf, EncodeError> {
    inspect_output(path.as_ref(), overwrite).map(|(full, _)| full)
}

/// [`check_output`], also saying whether a file is there to be replaced.
fn inspect_output(path: &Path, overwrite: bool) -> Result<(PathBuf, bool), EncodeError> {
    let bad = |detail: &str| EncodeError::InvalidPath {
        path: path.to_path_buf(),
        detail: detail.to_string(),
    };
    let io = |p: &Path, err| EncodeError::Io {
        path: p.to_path_buf(),
        err,
    };
    if !path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("mp4"))
    {
        return Err(bad("the output file name must end in .mp4"));
    }
    let name = path.file_name().ok_or_else(|| bad("no file name"))?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let parent = std::fs::canonicalize(parent).map_err(|e| io(parent, e))?;
    if !parent.is_dir() {
        return Err(bad("the parent is not a directory"));
    }
    let full = parent.join(name);
    match std::fs::symlink_metadata(&full) {
        Ok(meta) if meta.is_dir() => Err(bad("a directory is in the way")),
        Ok(_) if !overwrite => Err(EncodeError::Exists { path: full }),
        Ok(_) => Ok((full, true)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok((full, false)),
        Err(e) => Err(io(&full, e)),
    }
}

/// [`check_output`], then remove the file being replaced. With `overwrite`, the old file is gone
/// before AVFoundation is asked to write the new one: if [`VideoWriter::create_with`] then fails,
/// nothing replaces it.
fn prepare_output(path: &Path, overwrite: bool) -> Result<PathBuf, EncodeError> {
    let (full, exists) = inspect_output(path, overwrite)?;
    if exists {
        std::fs::remove_file(&full).map_err(|err| EncodeError::Io {
            path: full.clone(),
            err,
        })?;
    }
    Ok(full)
}

/// `NSNumber numberWithInteger:`; autoreleased, so a pool must be active.
///
/// # Safety
/// An autorelease pool is active.
unsafe fn ns_int(v: isize) -> Result<Id, EncodeError> {
    let n: Id = msg!(class(c"NSNumber")?, "numberWithInteger:", v => isize; -> Id);
    if n.is_null() {
        return Err(fw("NSNumber", "numberWithInteger: returned nil"));
    }
    Ok(n)
}

/// `NSDictionary dictionaryWithObjects:forKeys:count:`; autoreleased.
///
/// # Safety
/// An autorelease pool is active and every key and value is a live Objective-C object (keys
/// conform to `NSCopying`).
unsafe fn ns_dict(pairs: &[(Id, Id)]) -> Result<Id, EncodeError> {
    let keys: Vec<Id> = pairs.iter().map(|p| p.0).collect();
    let values: Vec<Id> = pairs.iter().map(|p| p.1).collect();
    let d: Id = msg!(class(c"NSDictionary")?, "dictionaryWithObjects:forKeys:count:", values.as_ptr() => *const Id, keys.as_ptr() => *const Id, pairs.len() => usize; -> Id);
    if d.is_null() {
        return Err(fw(
            "NSDictionary",
            "dictionaryWithObjects:forKeys:count: returned nil",
        ));
    }
    Ok(d)
}

/// The H.264 output settings for a `width x height` video at `fps`.
///
/// # Safety
/// An autorelease pool is active.
unsafe fn output_settings(width: usize, height: usize, fps: f64) -> Result<Id, EncodeError> {
    let bit_rate = (width as f64 * height as f64 * fps * BITS_PER_PIXEL)
        .clamp(MIN_BIT_RATE, MAX_BIT_RATE)
        .round() as isize;
    let compression = ns_dict(&[(AVVideoAverageBitRateKey, ns_int(bit_rate)?)])?;
    let colour = ns_dict(&[
        (AVVideoColorPrimariesKey, AVVideoColorPrimaries_ITU_R_709_2),
        (
            AVVideoTransferFunctionKey,
            AVVideoTransferFunction_ITU_R_709_2,
        ),
        (AVVideoYCbCrMatrixKey, AVVideoYCbCrMatrix_ITU_R_709_2),
    ])?;
    ns_dict(&[
        (AVVideoCodecKey, AVVideoCodecTypeH264),
        (AVVideoWidthKey, ns_int(width as isize)?),
        (AVVideoHeightKey, ns_int(height as isize)?),
        (AVVideoCompressionPropertiesKey, compression),
        (AVVideoColorPropertiesKey, colour),
    ])
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

/// A `CVPixelBufferLockBaseAddress(.., read-write)`, unlocked on drop.
struct WriteLock(CfTypeRef);

impl WriteLock {
    /// # Safety
    /// `pb` is a live `CVPixelBuffer` that outlives the lock.
    unsafe fn new(pb: CfTypeRef) -> Result<WriteLock, EncodeError> {
        let r = CVPixelBufferLockBaseAddress(pb, PIXEL_BUFFER_LOCK_READ_WRITE);
        if r != CV_RETURN_SUCCESS {
            return Err(fw("CVPixelBufferLockBaseAddress", format!("CVReturn {r}")));
        }
        Ok(WriteLock(pb))
    }
}

impl Drop for WriteLock {
    fn drop(&mut self) {
        // SAFETY: locked in `new`; the buffer is still alive (see `new`).
        unsafe {
            CVPixelBufferUnlockBaseAddress(self.0, PIXEL_BUFFER_LOCK_READ_WRITE);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum State {
    Writing,
    /// The writer failed (or a wait timed out); the text is returned by every later call.
    Failed(String),
    Finished,
}

/// A finished video file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoFile {
    /// The absolute path written.
    pub path: PathBuf,
    pub frames: usize,
    /// The file's size in bytes.
    pub bytes: u64,
}

/// Encodes RGB frames into an H.264 `.mp4`; see the module documentation. Not `Send`: the
/// Objective-C objects it holds are used from the thread that made them.
pub struct VideoWriter {
    // Field order is drop order: the adaptor and input before the writer they feed.
    adaptor: Obj,
    input: Obj,
    writer: Obj,
    path: PathBuf,
    width: usize,
    height: usize,
    fps: f64,
    timescale: i32,
    frames: usize,
    state: State,
}

impl VideoWriter {
    /// A writer for a `width x height` video at `fps` frames per second at `path`, refusing an
    /// existing file. See the module documentation for what is checked.
    pub fn create(
        path: impl AsRef<Path>,
        width: usize,
        height: usize,
        fps: f64,
    ) -> Result<VideoWriter, EncodeError> {
        Self::create_with(path, width, height, fps, WriterOptions::default())
    }

    /// [`VideoWriter::create`] with [`WriterOptions`].
    pub fn create_with(
        path: impl AsRef<Path>,
        width: usize,
        height: usize,
        fps: f64,
        options: WriterOptions,
    ) -> Result<VideoWriter, EncodeError> {
        let timescale = check_format(width, height, fps)?;
        let path = prepare_output(path.as_ref(), options.overwrite)?;
        let _pool = Pool::new()?;
        let url = file_url(&path)?;
        // SAFETY: `url` is a CFURL, toll-free bridged to NSURL; the file type is an SDK constant;
        // `+assetWriterWithURL:fileType:error:` returns an autoreleased writer or nil with the
        // NSError set, and the writer is retained before the pool drains. It throws only for a file
        // type not declared in AVMediaFormat.h.
        let writer = unsafe {
            let mut error: Id = std::ptr::null_mut();
            let w: Id = msg!(class(c"AVAssetWriter")?, "assetWriterWithURL:fileType:error:", url.as_ptr() as Id => Id, AVFileTypeMPEG4 => Id, &mut error as *mut Id => *mut Id; -> Id);
            Obj::retain(w).ok_or_else(|| EncodeError::Failed {
                path: path.clone(),
                detail: ns_error_text(error),
            })?
        };
        // SAFETY: `writer` is live; the pool is active; every object in the settings is live. The
        // `can...` tests precede `addInput:` and the adaptor, which throw for what they refuse.
        let (input, adaptor) = unsafe {
            let settings = output_settings(width, height, fps)?;
            let ok: bool = msg!(writer.id(), "canApplyOutputSettings:forMediaType:", settings => Id, AVMediaTypeVideo => Id; -> bool);
            if !ok {
                return Err(EncodeError::Unsupported {
                    detail: format!("{width}x{height} at {fps} fps as H.264 in an mp4"),
                });
            }
            let input = Obj::retain(msg!(class(c"AVAssetWriterInput")?, "assetWriterInputWithMediaType:outputSettings:", AVMediaTypeVideo => Id, settings => Id; -> Id))
                .ok_or_else(|| fw("AVAssetWriterInput", "assetWriterInputWithMediaType:outputSettings: returned nil"))?;
            let can: bool = msg!(writer.id(), "canAddInput:", input.id() => Id; -> bool);
            if !can {
                return Err(EncodeError::Unsupported {
                    detail: "AVAssetWriter cannot take this video input".to_string(),
                });
            }
            let _: () = msg!(writer.id(), "addInput:", input.id() => Id; -> ());
            // The pool needs the width, height and pixel format (the header: it throws without them).
            let attrs = ns_dict(&[
                (
                    kCVPixelBufferPixelFormatTypeKey as Id,
                    ns_int(PIXEL_FORMAT_32BGRA as isize)?,
                ),
                (kCVPixelBufferWidthKey as Id, ns_int(width as isize)?),
                (kCVPixelBufferHeightKey as Id, ns_int(height as isize)?),
            ])?;
            let adaptor = Obj::retain(msg!(class(c"AVAssetWriterInputPixelBufferAdaptor")?, "assetWriterInputPixelBufferAdaptorWithAssetWriterInput:sourcePixelBufferAttributes:", input.id() => Id, attrs => Id; -> Id))
                .ok_or_else(|| fw("AVAssetWriterInputPixelBufferAdaptor", "assetWriterInputPixelBufferAdaptorWithAssetWriterInput:sourcePixelBufferAttributes: returned nil"))?;
            (input, adaptor)
        };
        // From here a `VideoWriter` exists, so an early return cancels the session and removes the file.
        let mut w = VideoWriter {
            adaptor,
            input,
            writer,
            path,
            width,
            height,
            fps: f64::from(timescale) / TICKS_PER_FRAME as f64,
            timescale,
            frames: 0,
            state: State::Writing,
        };
        // SAFETY: the input is attached; writing starts after all inputs are added.
        let started: bool = unsafe { msg!(w.writer.id(), "startWriting"; -> bool) };
        if !started {
            let detail = w.writer_error();
            return Err(w.fail(detail));
        }
        // SAFETY: writing has started; the session starts before any sample is appended.
        unsafe {
            let _: () = msg!(w.writer.id(), "startSessionAtSourceTime:", CMTimeMake(0, timescale) => CmTime; -> ());
        }
        Ok(w)
    }

    /// The frame size in pixels, `(width, height)`.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// The frame rate as stored: `fps` rounded to the nearest 0.001.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Frames pushed so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The absolute path being written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `AVAssetWriter.error` as text.
    fn writer_error(&self) -> String {
        // SAFETY: `writer` is live; `error` is a thread-safe read, and the text is copied out
        // before the pool drains (the callers hold a pool or none is needed for a nil error).
        unsafe {
            let _pool = Pool::new();
            let status: isize = msg!(self.writer.id(), "status"; -> isize);
            let err: Id = msg!(self.writer.id(), "error"; -> Id);
            let name = match status {
                STATUS_CANCELLED => "cancelled",
                STATUS_FAILED => "failed",
                STATUS_COMPLETED => "completed",
                STATUS_WRITING => "writing",
                _ => "not started",
            };
            format!("status {name}: {}", ns_error_text(err))
        }
    }

    /// Record a failure: every later call returns it, and drop cleans the file up.
    fn fail(&mut self, detail: String) -> EncodeError {
        self.state = State::Failed(detail.clone());
        EncodeError::Failed {
            path: self.path.clone(),
            detail,
        }
    }

    fn status(&self) -> isize {
        // SAFETY: `writer` is live; `status` is a thread-safe read.
        unsafe { msg!(self.writer.id(), "status"; -> isize) }
    }

    /// Wait, bounded, until the input accepts another frame.
    fn wait_ready(&mut self) -> Result<(), EncodeError> {
        let start = Instant::now();
        loop {
            // SAFETY: `input` is live; `isReadyForMoreMediaData` is a plain property read.
            let ready: bool = unsafe { msg!(self.input.id(), "isReadyForMoreMediaData"; -> bool) };
            if ready {
                return Ok(());
            }
            // A failed writer never becomes ready again: report it now, not after the timeout.
            if self.status() != STATUS_WRITING {
                let detail = self.writer_error();
                return Err(self.fail(detail));
            }
            let waited = start.elapsed();
            if waited >= READY_TIMEOUT {
                let frames = self.frames;
                self.state = State::Failed("the encoder stopped accepting frames".to_string());
                return Err(EncodeError::Timeout {
                    what: "the encoder to accept a frame",
                    waited,
                    frames,
                });
            }
            std::thread::sleep(POLL);
        }
    }

    /// Append the next frame, in presentation order: frame `i` is presented at `i / fps`. The frame
    /// must be [`VideoWriter::size`] pixels. Blocks while the encoder is behind, for at most
    /// [`READY_TIMEOUT`].
    pub fn push(&mut self, frame: &Rgb8) -> Result<(), EncodeError> {
        let index = self.frames;
        match &self.state {
            State::Writing => {}
            State::Failed(detail) => {
                return Err(EncodeError::Failed {
                    path: self.path.clone(),
                    detail: detail.clone(),
                })
            }
            State::Finished => return Err(fw("VideoWriter", "the writer is finished")),
        }
        let bad = |detail: String| EncodeError::BadFrame { index, detail };
        if (frame.width, frame.height) != (self.width, self.height) {
            return Err(bad(format!(
                "{}x{}, but the video is {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }
        if frame.data.len() != self.width * self.height * 3 {
            return Err(bad(format!(
                "{} bytes of pixels, expected {}",
                frame.data.len(),
                self.width * self.height * 3
            )));
        }
        if index >= MAX_VIDEO_FRAMES {
            return Err(EncodeError::TooManyFrames {
                limit: MAX_VIDEO_FRAMES,
            });
        }
        let _pool = Pool::new()?;
        self.wait_ready()?;
        let buffer = self.pooled_buffer(frame)?;
        // SAFETY: `CMTimeMake` is a plain function; `index * 1000` fits an i64.
        let pts = unsafe { CMTimeMake(index as i64 * TICKS_PER_FRAME, self.timescale) };
        // SAFETY: `adaptor` is live; the input is attached to a writer that is writing, inside a
        // session, and was just seen ready (the call throws otherwise); the buffer is a live
        // CVPixelBuffer the adaptor retains and is no longer modified.
        let appended: bool = unsafe {
            msg!(self.adaptor.id(), "appendPixelBuffer:withPresentationTime:", buffer.as_ptr() as Id => Id, pts => CmTime; -> bool)
        };
        if !appended {
            let detail = self.writer_error();
            return Err(self.fail(detail));
        }
        self.frames += 1;
        Ok(())
    }

    /// A buffer from the adaptor's pool holding `frame` as 32BGRA.
    fn pooled_buffer(&self, frame: &Rgb8) -> Result<CfRef, EncodeError> {
        let (w, h) = (self.width, self.height);
        // SAFETY: `adaptor` is live and its writer has started (the pool is NULL before that); the
        // pool is owned by the adaptor and is used within this call.
        let pool: CfTypeRef = unsafe { msg!(self.adaptor.id(), "pixelBufferPool"; -> CfTypeRef) };
        if pool.is_null() {
            return Err(fw(
                "AVAssetWriterInputPixelBufferAdaptor",
                "pixelBufferPool is NULL after startWriting",
            ));
        }
        let mut raw: CfTypeRef = std::ptr::null();
        // SAFETY: `pool` is live; the out parameter receives a +1 buffer or stays NULL.
        let r = unsafe { CVPixelBufferPoolCreatePixelBuffer(std::ptr::null(), pool, &mut raw) };
        // SAFETY: on success `raw` is a buffer we own one reference to.
        let buffer = unsafe { CfRef::take(raw) };
        let Some(buffer) = buffer.filter(|_| r == CV_RETURN_SUCCESS) else {
            return Err(fw(
                "CVPixelBufferPoolCreatePixelBuffer",
                format!("CVReturn {r}"),
            ));
        };
        let pb = buffer.as_ptr();
        // SAFETY: `pb` is a live CVPixelBuffer; the getters read plain properties.
        let (format, bw, bh, stride) = unsafe {
            (
                CVPixelBufferGetPixelFormatType(pb),
                CVPixelBufferGetWidth(pb),
                CVPixelBufferGetHeight(pb),
                CVPixelBufferGetBytesPerRow(pb),
            )
        };
        if format != PIXEL_FORMAT_32BGRA || (bw, bh) != (w, h) || stride < w * 4 {
            return Err(fw("CVPixelBufferPool", format!("the pool vended a {bw}x{bh} buffer, format {format:#x}, stride {stride}; expected {w}x{h} 32BGRA")));
        }
        // SAFETY: `pb` outlives the lock.
        let _lock = unsafe { WriteLock::new(pb)? };
        // SAFETY: the buffer is locked.
        let base = unsafe { CVPixelBufferGetBaseAddress(pb) }.cast_mut();
        if base.is_null() {
            return Err(fw(
                "CVPixelBufferGetBaseAddress",
                "NULL for a locked buffer",
            ));
        }
        // SAFETY: while locked, `base` addresses `h` rows of `stride` bytes (a chunky format), and
        // nothing else holds the fresh buffer.
        let dst = unsafe { std::slice::from_raw_parts_mut(base, stride * h) };
        for (row, src) in dst
            .chunks_exact_mut(stride)
            .zip(frame.data.chunks_exact(w * 3))
        {
            for (px, rgb) in row[..w * 4].chunks_exact_mut(4).zip(src.chunks_exact(3)) {
                px.copy_from_slice(&[rgb[2], rgb[1], rgb[0], 255]);
            }
        }
        Ok(buffer)
    }

    /// Complete the file: blocks until it is written, for at most [`FINISH_TIMEOUT`], and returns
    /// an error if `AVAssetWriter` failed, at any point of the writing. A writer with no frame is
    /// [`EncodeError::Empty`] and leaves no file.
    pub fn finish(mut self) -> Result<VideoFile, EncodeError> {
        match &self.state {
            State::Writing => {}
            State::Failed(detail) => {
                return Err(EncodeError::Failed {
                    path: self.path.clone(),
                    detail: detail.clone(),
                })
            }
            State::Finished => return Err(fw("VideoWriter", "the writer is finished")),
        }
        if self.frames == 0 {
            return Err(EncodeError::Empty {
                path: self.path.clone(),
            });
        }
        let _pool = Pool::new()?;
        // SAFETY: the writer is writing; the input is attached; the end time follows the last
        // frame. `finishWritingWithCompletionHandler:` takes a `void (^)(void)` block: ours is a
        // static global block that captures nothing (see the module documentation).
        unsafe {
            let _: () = msg!(self.input.id(), "markAsFinished"; -> ());
            let end = CMTimeMake(self.frames as i64 * TICKS_PER_FRAME, self.timescale);
            let _: () = msg!(self.writer.id(), "endSessionAtSourceTime:", end => CmTime; -> ());
            let _: () = msg!(self.writer.id(), "finishWritingWithCompletionHandler:", &NOOP_BLOCK as *const GlobalBlock as *const c_void => *const c_void; -> ());
        }
        let start = Instant::now();
        loop {
            match self.status() {
                STATUS_WRITING => {}
                STATUS_COMPLETED => break,
                _ => {
                    let detail = self.writer_error();
                    return Err(self.fail(detail));
                }
            }
            let waited = start.elapsed();
            if waited >= FINISH_TIMEOUT {
                let frames = self.frames;
                self.state = State::Failed("the file was not completed in time".to_string());
                return Err(EncodeError::Timeout {
                    what: "the file to be completed",
                    waited,
                    frames,
                });
            }
            std::thread::sleep(POLL);
        }
        let bytes = std::fs::metadata(&self.path)
            .map_err(|err| EncodeError::Io {
                path: self.path.clone(),
                err,
            })?
            .len();
        self.state = State::Finished;
        Ok(VideoFile {
            path: self.path.clone(),
            frames: self.frames,
            bytes,
        })
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        if self.state == State::Finished {
            return;
        }
        if let Ok(_pool) = Pool::new() {
            // SAFETY: `writer` is live; `cancelWriting` is a no-op for a failed or completed writer
            // and otherwise blocks until the session is cancelled and deletes the file; it must not
            // run concurrently with an append, which `&mut self` rules out.
            unsafe {
                let _: () = msg!(self.writer.id(), "cancelWriting"; -> ());
            }
        }
        // A failed writer leaves what it wrote; nothing else can own a path we refused to overwrite.
        let _ = std::fs::remove_file(&self.path);
    }
}

impl fmt::Debug for VideoWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoWriter")
            .field("path", &self.path)
            .field("size", &(self.width, self.height))
            .field("fps", &self.fps)
            .field("frames", &self.frames)
            .field("state", &self.state)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_format_checks_name_what_they_refuse() {
        assert_eq!(check_format(320, 240, 24.0).unwrap(), 24_000);
        assert_eq!(check_format(640, 480, 29.97).unwrap(), 29_970);
        assert_eq!(check_format(640, 480, 47.9521).unwrap(), 47_952);
        assert!(matches!(
            check_format(321, 240, 24.0),
            Err(EncodeError::InvalidSize { .. })
        ));
        assert!(matches!(
            check_format(320, 241, 24.0),
            Err(EncodeError::InvalidSize { .. })
        ));
        assert!(matches!(
            check_format(0, 0, 24.0),
            Err(EncodeError::InvalidSize { .. })
        ));
        assert!(matches!(
            check_format(8, 240, 24.0),
            Err(EncodeError::InvalidSize { .. })
        ));
        assert!(matches!(
            check_format(MAX_DIMENSION + 2, 16, 24.0),
            Err(EncodeError::InvalidSize { .. })
        ));
        assert!(
            matches!(
                check_format(8192, 8192, 24.0),
                Err(EncodeError::InvalidSize { .. })
            ),
            "over the pixel bound"
        );
        for fps in [
            0.0,
            0.5,
            240.5,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(
                matches!(
                    check_format(320, 240, fps),
                    Err(EncodeError::InvalidFps { .. })
                ),
                "{fps}"
            );
        }
        assert!(check_format(320, 240, MIN_FPS).is_ok() && check_format(320, 240, MAX_FPS).is_ok());
    }

    #[test]
    fn the_noop_block_has_the_layout_the_runtime_expects() {
        // Clang's Block ABI: isa, int flags, int reserved, invoke, descriptor: 8 + 4 + 4 + 8 + 8.
        assert_eq!(std::mem::size_of::<GlobalBlock>(), 32);
        assert_eq!(NOOP_DESCRIPTOR.size, 32);
        assert_eq!(NOOP_BLOCK.flags, 1 << 28);
        assert!(std::ptr::eq(
            NOOP_BLOCK.isa,
            (&raw const _NSConcreteGlobalBlock).cast()
        ));
    }
}
