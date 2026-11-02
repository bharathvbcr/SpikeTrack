//! SAM 3.1's ViT trunk as a CoreML ML Program, handed to [`st_model::Backbone`] as its
//! [`ExternalTrunk`]. `tools/baselines/coreml_trunk.py` exports the package from the converted
//! checkpoint; it is derived from gated weights and lives next to them, never in the repo.
//!
//! The package's contract (checked on every call, not assumed): input `image` `[1, 3, 1008, 1008]`
//! f32, normalized; output `trunk` `[1, 72, 72, 1024]` f32, NHWC.
//!
//! Loading names the compiled model by the `.mlpackage`'s content hash (SHA-256 over its files'
//! relative paths and bytes, in path order) and loads `<package>.<first 16 hex>.mlmodelc` beside it,
//! compiling it there once if it is missing. A changed package gets a new compiled model, never a
//! stale one.
//!
//! Hashing the 1.7 GB fp32 package took 3.7-7.6 s of every start (bench/results/e18-coreml-load), so
//! the content hash is recorded once per package version: `<package>.stat-<first 16 hex>.sha256`
//! holds it, named by the package's stat key. The stat key covers every file's relative path, length,
//! inode and modification time, plus the full bytes of files up to [`STAT_KEY_FULL_BYTES`] (the
//! spec files). A re-export, a copy or an edit changes the key, and the next load hashes in full
//! again. An in-place rewrite that keeps a large file's length, inode and modification time is the
//! one change the key does not see.

use std::cell::RefCell;
use std::ffi::c_void;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use block2::{RcBlock, StackBlock};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AllocAnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSInteger, NSNumber, NSString, NSURL};
use ojas_core::OjasError;
use sha2::{Digest, Sha256};
use st_model::ExternalTrunk;

const INPUT: &str = "image";
const OUTPUT: &str = "trunk";
const IMAGE_SHAPE: [usize; 4] = [1, 3, 1008, 1008];
const TRUNK_SHAPE: [usize; 4] = [1, 72, 72, 1024];
/// Compiling the fp32 trunk takes seconds; a compile still running after this is reported as a failure.
const COMPILE_TIMEOUT: Duration = Duration::from_secs(600);
/// Files up to this length enter the stat key with their bytes (the package's `Manifest.json` and
/// `model.mlmodel`); longer ones (the weights) with their length, inode and modification time.
pub const STAT_KEY_FULL_BYTES: u64 = 4 << 20;

/// Where Core ML may run the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Units {
    /// `MLComputeUnitsCPUAndGPU`: what the trunk is measured on.
    CpuAndGpu,
    /// `MLComputeUnitsAll`: Core ML may also pick the Neural Engine.
    All,
    /// `MLComputeUnitsCPUOnly`.
    CpuOnly,
}

impl Units {
    fn ml(self) -> MLComputeUnits {
        match self {
            Units::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            Units::All => MLComputeUnits::All,
            Units::CpuOnly => MLComputeUnits::CPUOnly,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Units::CpuAndGpu => "cpu_and_gpu",
            Units::All => "all",
            Units::CpuOnly => "cpu_only",
        }
    }
}

/// A load or prediction failure, with Core ML's own message where it gave one.
#[derive(Debug)]
pub struct CoremlError(String);

impl fmt::Display for CoremlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "st-coreml: {}", self.0)
    }
}

impl std::error::Error for CoremlError {}

fn fail(what: impl Into<String>) -> CoremlError {
    CoremlError(what.into())
}

fn ns_fail(what: &str, e: &NSError) -> CoremlError {
    CoremlError(format!("{what}: {}", e.localizedDescription()))
}

/// The trunk package, compiled and loaded.
pub struct CoremlTrunk {
    model: Retained<MLModel>,
    package: PathBuf,
    sha256: String,
    units: Units,
    /// Whether this load compiled the package (false: the cached compiled model was reused).
    pub compiled: bool,
    /// Whether this load hashed the package in full (false: the hash recorded under its stat key).
    pub hashed: bool,
    /// Hashing, compiling (when needed) and loading, in seconds.
    pub load_seconds: f64,
    /// The parts of `load_seconds`: the stat key plus the hash (full or recorded), compiling (0 when
    /// reused) and recording the hash, and Core ML's model load.
    pub hash_seconds: f64,
    pub compile_seconds: f64,
    pub model_seconds: f64,
}

impl CoremlTrunk {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        let t0 = Instant::now();
        if !package.is_dir() {
            return Err(fail(format!(
                "{} is not an .mlpackage directory",
                package.display()
            )));
        }
        let resolved = resolve_sha256(package, STAT_KEY_FULL_BYTES)?;
        let t_hash = t0.elapsed().as_secs_f64();
        let sha256 = resolved.sha256.clone();
        let compiled_dir = compiled_path(package, &sha256);
        let compiled = !compiled_dir.is_dir();
        if compiled {
            compile(package, &compiled_dir)?;
        }
        if resolved.hashed {
            record_sha256(package, &resolved, STAT_KEY_FULL_BYTES)?;
        }
        let t_compile = t0.elapsed().as_secs_f64();
        let url = NSURL::from_file_path(&compiled_dir)
            .ok_or_else(|| fail(format!("{} is not a file URL", compiled_dir.display())))?;
        let model = autoreleasepool(|_| -> Result<Retained<MLModel>, CoremlError> {
            let cfg = unsafe { MLModelConfiguration::new() };
            unsafe { cfg.setComputeUnits(units.ml()) };
            unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg) }
                .map_err(|e| ns_fail(&format!("loading {}", compiled_dir.display()), &e))
        })?;
        let load_seconds = t0.elapsed().as_secs_f64();
        Ok(Self {
            model,
            package: package.to_path_buf(),
            sha256,
            units,
            compiled,
            hashed: resolved.hashed,
            load_seconds,
            hash_seconds: t_hash,
            compile_seconds: t_compile - t_hash,
            model_seconds: load_seconds - t_compile,
        })
    }

    pub fn package(&self) -> &Path {
        &self.package
    }

    /// SHA-256 of the package, as [`CoremlTrunk::load`] computed it.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn units(&self) -> Units {
        self.units
    }

    /// One prediction: `image` `[1, 3, 1008, 1008]` row-major to the trunk `[1, 72, 72, 1024]`.
    pub fn predict(&self, image: &[f32]) -> Result<Vec<f32>, CoremlError> {
        let n_in: usize = IMAGE_SHAPE.iter().product();
        if image.len() != n_in {
            return Err(fail(format!(
                "image has {} values, want {n_in} ({IMAGE_SHAPE:?})",
                image.len()
            )));
        }
        autoreleasepool(|_| {
            // Core ML reads the input through this pointer; the buffer outlives the prediction.
            let mut input = image.to_vec();
            let ptr = NonNull::new(input.as_mut_ptr().cast::<c_void>())
                .ok_or_else(|| fail("input buffer is null"))?;
            let arr = unsafe {
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(),
                    ptr,
                    &ns_ints(&IMAGE_SHAPE),
                    MLMultiArrayDataType::Float32,
                    &ns_ints(&contiguous_strides(&IMAGE_SHAPE)),
                    None,
                )
            }
            .map_err(|e| ns_fail("wrapping the input", &e))?;
            let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) };
            let key = NSString::from_str(INPUT);
            let obj: &AnyObject = value.as_ref();
            let dict: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_slices(&[&*key], &[obj]);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dict,
                )
            }
            .map_err(|e| ns_fail("building the input features", &e))?;
            let out = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|e| ns_fail("prediction", &e))?;
            let value = unsafe { out.featureValueForName(&NSString::from_str(OUTPUT)) }
                .ok_or_else(|| fail(format!("the model has no output {OUTPUT:?}")))?;
            let arr = unsafe { value.multiArrayValue() }
                .ok_or_else(|| fail(format!("output {OUTPUT:?} is not a multi-array")))?;
            let result = read_f32(&arr, &TRUNK_SHAPE);
            drop(input);
            result
        })
    }
}

impl ExternalTrunk for CoremlTrunk {
    fn describe(&self) -> String {
        let name = self
            .package
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!(
            "coreml {name} ({}) sha256 {}",
            self.units.name(),
            self.sha256
        )
    }

    fn run(&self, image: &[f32]) -> Result<Vec<f32>, OjasError> {
        self.predict(image).map_err(|e| OjasError::Unsupported {
            op: "st_coreml::CoremlTrunk::run",
            detail: e.to_string(),
        })
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn ns_ints(v: &[usize]) -> Retained<NSArray<NSNumber>> {
    let nums: Vec<Retained<NSNumber>> = v
        .iter()
        .map(|&x| NSNumber::numberWithInteger(x as NSInteger))
        .collect();
    let refs: Vec<&NSNumber> = nums.iter().map(|n| &**n).collect();
    NSArray::from_slice(&refs)
}

fn ints_of(a: &NSArray<NSNumber>) -> Result<Vec<usize>, CoremlError> {
    a.iter()
        .map(|n| usize::try_from(n.integerValue()).map_err(|_| fail("negative shape or stride")))
        .collect()
}

/// The f32 elements of `arr` in row-major order of `want`, honouring Core ML's strides (an output
/// may be padded); anything but an f32 array of exactly that shape is an error.
fn read_f32(arr: &MLMultiArray, want: &[usize]) -> Result<Vec<f32>, CoremlError> {
    if want.len() != 4 {
        return Err(fail("read_f32 reads 4-d arrays"));
    }
    let dtype = unsafe { arr.dataType() };
    if dtype != MLMultiArrayDataType::Float32 {
        return Err(fail(format!(
            "output dtype is {:#x}, want Float32",
            dtype.0
        )));
    }
    let shape = ints_of(&*unsafe { arr.shape() })?;
    if shape != want {
        return Err(fail(format!("output shape is {shape:?}, want {want:?}")));
    }
    let strides = ints_of(&*unsafe { arr.strides() })?;
    if strides.len() != want.len() {
        return Err(fail(format!(
            "output has {} strides for {} dims",
            strides.len(),
            want.len()
        )));
    }
    let n: usize = want.iter().product();
    let span = 1 + want
        .iter()
        .zip(&strides)
        .map(|(d, s)| (d - 1) * s)
        .sum::<usize>();
    let contiguous = strides == contiguous_strides(want);
    let got: RefCell<Result<Vec<f32>, CoremlError>> =
        RefCell::new(Err(fail("Core ML did not call the read handler")));
    let handler = StackBlock::new(|bytes: NonNull<c_void>, size: NSInteger| {
        let size = usize::try_from(size).unwrap_or(0);
        let r = if size < span * 4 {
            Err(fail(format!(
                "output buffer is {size} bytes, its strides span {} bytes",
                span * 4
            )))
        } else {
            // SAFETY: Core ML hands a buffer of `size` bytes, valid for the handler's duration; it
            // holds f32 elements (dtype checked above) and `span * 4 <= size`.
            let src = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), span) };
            if contiguous {
                Ok(src[..n].to_vec())
            } else {
                let mut out = Vec::with_capacity(n);
                for a in 0..want[0] {
                    for b in 0..want[1] {
                        for c in 0..want[2] {
                            let row = a * strides[0] + b * strides[1] + c * strides[2];
                            out.extend((0..want[3]).map(|d| src[row + d * strides[3]]));
                        }
                    }
                }
                Ok(out)
            }
        };
        *got.borrow_mut() = r;
    });
    unsafe { arr.getBytesWithHandler(&handler) };
    got.into_inner()
}

fn compiled_path(package: &Path, sha256: &str) -> PathBuf {
    let mut name = package.as_os_str().to_os_string();
    name.push(format!(".{}.mlmodelc", &sha256[..16]));
    PathBuf::from(name)
}

/// Every file under `package`, in sorted path order.
fn package_files(package: &Path) -> Result<Vec<PathBuf>, CoremlError> {
    let mut files = Vec::new();
    let mut stack = vec![package.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries =
            fs::read_dir(&dir).map_err(|e| fail(format!("reading {}: {e}", dir.display())))?;
        for entry in entries {
            let path = entry
                .map_err(|e| fail(format!("reading {}: {e}", dir.display())))?
                .path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Feeds `path`'s whole content to `h`.
fn hash_file(
    h: &mut Sha256,
    f: &mut fs::File,
    path: &Path,
    buf: &mut [u8],
) -> Result<(), CoremlError> {
    loop {
        let k = f
            .read(buf)
            .map_err(|e| fail(format!("reading {}: {e}", path.display())))?;
        if k == 0 {
            return Ok(());
        }
        h.update(&buf[..k]);
    }
}

/// SHA-256 over every file under `package`: its relative path, a NUL, its length, then its bytes,
/// in sorted path order.
fn package_sha256(package: &Path) -> Result<String, CoremlError> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    for path in &package_files(package)? {
        let rel = path
            .strip_prefix(package)
            .map_err(|_| fail("package walk left the package"))?;
        h.update(rel.to_string_lossy().as_bytes());
        h.update([0]);
        let mut f =
            fs::File::open(path).map_err(|e| fail(format!("opening {}: {e}", path.display())))?;
        let len = f
            .metadata()
            .map_err(|e| fail(format!("reading {}: {e}", path.display())))?
            .len();
        h.update(len.to_le_bytes());
        hash_file(&mut h, &mut f, path, &mut buf)?;
    }
    Ok(hex(&h.finalize()))
}

/// SHA-256 over every file under `package`, in sorted path order: its relative path, a NUL, its
/// length, then either its bytes (length <= `full_bytes`) or its inode and modification time.
fn stat_key(package: &Path, full_bytes: u64) -> Result<String, CoremlError> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    for path in &package_files(package)? {
        let rel = path
            .strip_prefix(package)
            .map_err(|_| fail("package walk left the package"))?;
        h.update(rel.to_string_lossy().as_bytes());
        h.update([0]);
        let mut f =
            fs::File::open(path).map_err(|e| fail(format!("opening {}: {e}", path.display())))?;
        let meta = f
            .metadata()
            .map_err(|e| fail(format!("reading {}: {e}", path.display())))?;
        h.update(meta.len().to_le_bytes());
        if meta.len() <= full_bytes {
            h.update([b'c']);
            hash_file(&mut h, &mut f, path, &mut buf)?;
        } else {
            let mtime = meta
                .modified()
                .map_err(|e| {
                    fail(format!(
                        "reading the modification time of {}: {e}",
                        path.display()
                    ))
                })?
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| fail(format!("{} is modified before 1970", path.display())))?;
            h.update([b's']);
            h.update(std::os::unix::fs::MetadataExt::ino(&meta).to_le_bytes());
            h.update(mtime.as_secs().to_le_bytes());
            h.update(mtime.subsec_nanos().to_le_bytes());
        }
    }
    Ok(hex(&h.finalize()))
}

/// Where the content hash of the package version with stat key `key` is recorded.
fn recorded_path(package: &Path, key: &str) -> PathBuf {
    let mut name = package.as_os_str().to_os_string();
    name.push(format!(".stat-{}.sha256", &key[..16]));
    PathBuf::from(name)
}

/// The hash recorded at `path`, if it is one: exactly 64 lowercase hex digits (a trailing newline
/// allowed). Anything else reads as no record, so the load hashes in full and rewrites it.
fn read_recorded(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let s = text.strip_suffix('\n').unwrap_or(&text);
    (s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then(|| s.to_string())
}

/// A package's content hash, and how it was found.
#[derive(Debug)]
struct Resolved {
    sha256: String,
    /// The stat key the hash belongs to.
    key: String,
    /// True when hashed in full this time (no usable record).
    hashed: bool,
}

/// The package's content hash: the one recorded under its stat key when that record is well-formed
/// and its compiled model exists, otherwise a full hash.
fn resolve_sha256(package: &Path, full_bytes: u64) -> Result<Resolved, CoremlError> {
    let key = stat_key(package, full_bytes)?;
    if let Some(sha256) = read_recorded(&recorded_path(package, &key)) {
        if compiled_path(package, &sha256).is_dir() {
            return Ok(Resolved {
                sha256,
                key,
                hashed: false,
            });
        }
    }
    Ok(Resolved {
        sha256: package_sha256(package)?,
        key,
        hashed: true,
    })
}

/// Records `r`'s full hash under its stat key, if the package's stat key is still the one taken
/// before the full hash (a package changed mid-hash is not recorded). Written via a staging name, so
/// a crash never leaves a partial record.
fn record_sha256(package: &Path, r: &Resolved, full_bytes: u64) -> Result<(), CoremlError> {
    if stat_key(package, full_bytes)? != r.key {
        return Ok(());
    }
    let dst = recorded_path(package, &r.key);
    let mut staging = dst.as_os_str().to_os_string();
    staging.push(format!(".staging-{}", std::process::id()));
    let staging = PathBuf::from(staging);
    fs::write(&staging, format!("{}\n", r.sha256))
        .map_err(|e| fail(format!("writing {}: {e}", staging.display())))?;
    fs::rename(&staging, &dst).map_err(|e| {
        fail(format!(
            "moving {} to {}: {e}",
            staging.display(),
            dst.display()
        ))
    })
}

/// Compile `package` with Core ML and move the result to `dst` (via a staging name beside it, so a
/// crash mid-move never leaves a partial `dst`).
fn compile(package: &Path, dst: &Path) -> Result<(), CoremlError> {
    let url = NSURL::from_file_path(package)
        .ok_or_else(|| fail(format!("{} is not a file URL", package.display())))?;
    let mut staging = dst.as_os_str().to_os_string();
    staging.push(format!(".staging-{}", std::process::id()));
    let staging = PathBuf::from(staging);
    let (tx, rx) = mpsc::channel::<Result<(), CoremlError>>();
    let move_to = staging.clone();
    // Core ML compiles into a temporary directory it may remove after the handler returns, so the
    // handler moves it out.
    let handler = RcBlock::new(move |compiled: *mut NSURL, err: *mut NSError| {
        // SAFETY: Core ML passes either a valid compiled-model URL or a valid error, for the call.
        let r = match (unsafe { compiled.as_ref() }, unsafe { err.as_ref() }) {
            (Some(u), _) => match u.to_file_path() {
                Some(tmp) => fs::rename(&tmp, &move_to).map_err(|e| {
                    fail(format!(
                        "moving {} to {}: {e}",
                        tmp.display(),
                        move_to.display()
                    ))
                }),
                None => Err(fail("the compiled model's URL is not a file path")),
            },
            (None, Some(e)) => Err(ns_fail("compiling", e)),
            (None, None) => Err(fail(
                "Core ML's compile reported neither a model nor an error",
            )),
        };
        let _ = tx.send(r);
    });
    autoreleasepool(|_| unsafe { MLModel::compileModelAtURL_completionHandler(&url, &handler) });
    let r = rx.recv_timeout(COMPILE_TIMEOUT).map_err(|_| {
        fail(format!(
            "compiling {} did not finish in {COMPILE_TIMEOUT:?}",
            package.display()
        ))
    })?;
    r?;
    match fs::rename(&staging, dst) {
        Ok(()) => Ok(()),
        // Another process compiled the same package first: keep its copy.
        Err(_) if dst.is_dir() => fs::remove_dir_all(&staging)
            .map_err(|e| fail(format!("removing {}: {e}", staging.display()))),
        Err(e) => Err(fail(format!(
            "moving {} to {}: {e}",
            staging.display(),
            dst.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_are_row_major() {
        assert_eq!(
            contiguous_strides(&TRUNK_SHAPE),
            vec![72 * 72 * 1024, 72 * 1024, 1024, 1]
        );
        assert_eq!(contiguous_strides(&[5]), vec![1]);
    }

    #[test]
    fn compiled_path_carries_the_hash_prefix() {
        let p = compiled_path(
            Path::new("/m/coreml/t_fp32.mlpackage"),
            "0123456789abcdef0123",
        );
        assert_eq!(
            p,
            PathBuf::from("/m/coreml/t_fp32.mlpackage.0123456789abcdef.mlmodelc")
        );
    }

    #[test]
    fn package_hash_changes_with_content_and_path() {
        let dir = std::env::temp_dir().join(format!("st-coreml-hash-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("Data")).unwrap();
        fs::write(dir.join("Manifest.json"), b"{}").unwrap();
        fs::write(dir.join("Data/w.bin"), b"abc").unwrap();
        let a = package_sha256(&dir).unwrap();
        assert_eq!(a, package_sha256(&dir).unwrap());
        fs::write(dir.join("Data/w.bin"), b"abd").unwrap();
        let b = package_sha256(&dir).unwrap();
        assert_ne!(a, b);
        fs::rename(dir.join("Data/w.bin"), dir.join("Data/v.bin")).unwrap();
        assert_ne!(b, package_sha256(&dir).unwrap());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A package `<tmp>/<tag>-<pid>/t.mlpackage` with a 2-byte `Manifest.json` and an 8-byte
    /// `Data/w.bin`; with `full_bytes` 4 the first enters the stat key by content, the second by
    /// metadata. Returns (the parent to remove, the package).
    fn fake_package(tag: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("st-coreml-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let pkg = root.join("t.mlpackage");
        fs::create_dir_all(pkg.join("Data")).unwrap();
        fs::write(pkg.join("Manifest.json"), b"{}").unwrap();
        fs::write(pkg.join("Data/w.bin"), b"abcdefgh").unwrap();
        (root, pkg)
    }

    fn set_mtime(path: &Path, t: std::time::SystemTime) {
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn stat_key_follows_small_files_bytes_and_large_files_metadata() {
        let (root, pkg) = fake_package("statkey");
        let w = pkg.join("Data/w.bin");
        let k0 = stat_key(&pkg, 4).unwrap();
        assert_eq!(k0, stat_key(&pkg, 4).unwrap(), "a re-read keeps the key");
        fs::write(pkg.join("Manifest.json"), b"[]").unwrap();
        let k1 = stat_key(&pkg, 4).unwrap();
        assert_ne!(k0, k1, "a small file's bytes are in the key");
        // The documented blind spot: same length, same inode, modification time put back.
        let mtime = fs::metadata(&w).unwrap().modified().unwrap();
        fs::write(&w, b"abcdefgX").unwrap();
        set_mtime(&w, mtime);
        assert_eq!(
            k1,
            stat_key(&pkg, 4).unwrap(),
            "a large file's bytes are not in the key"
        );
        set_mtime(&w, mtime + Duration::from_secs(1));
        let k2 = stat_key(&pkg, 4).unwrap();
        assert_ne!(k1, k2, "a large file's modification time is in the key");
        fs::write(&w, b"abcdefghi").unwrap();
        set_mtime(&w, mtime + Duration::from_secs(1));
        let k3 = stat_key(&pkg, 4).unwrap();
        assert_ne!(k2, k3, "a large file's length is in the key");
        fs::rename(&w, pkg.join("Data/v.bin")).unwrap();
        assert_ne!(k3, stat_key(&pkg, 4).unwrap(), "paths are in the key");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_recorded_hash_is_used_only_when_well_formed_and_its_compiled_model_exists() {
        let (root, pkg) = fake_package("record");
        let full = package_sha256(&pkg).unwrap();
        let r = resolve_sha256(&pkg, 4).unwrap();
        assert!(r.hashed, "no record yet");
        assert_eq!(r.sha256, full);
        record_sha256(&pkg, &r, 4).unwrap();
        let rec = recorded_path(&pkg, &r.key);
        assert_eq!(fs::read_to_string(&rec).unwrap(), format!("{full}\n"));
        assert!(
            resolve_sha256(&pkg, 4).unwrap().hashed,
            "a record whose compiled model is missing is not used"
        );
        fs::create_dir_all(compiled_path(&pkg, &full)).unwrap();
        let hit = resolve_sha256(&pkg, 4).unwrap();
        assert!(!hit.hashed, "the record is used");
        assert_eq!(hit.sha256, full);
        for bad in [
            "garbage\n".to_string(),
            full.to_uppercase(),
            full[..63].to_string(),
            format!("{full}0"),
        ] {
            fs::write(&rec, &bad).unwrap();
            assert!(
                resolve_sha256(&pkg, 4).unwrap().hashed,
                "a malformed record {bad:?} is not used"
            );
        }
        record_sha256(&pkg, &r, 4).unwrap();
        assert!(!resolve_sha256(&pkg, 4).unwrap().hashed);
        fs::write(pkg.join("Manifest.json"), b"[]").unwrap();
        let changed = resolve_sha256(&pkg, 4).unwrap();
        assert!(
            changed.hashed,
            "a changed package has a new stat key and no record"
        );
        assert_ne!(changed.sha256, full);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_package_changed_during_the_full_hash_is_not_recorded() {
        let (root, pkg) = fake_package("midhash");
        let r = resolve_sha256(&pkg, 4).unwrap();
        fs::write(pkg.join("Manifest.json"), b"[]").unwrap();
        record_sha256(&pkg, &r, 4).unwrap();
        assert!(!recorded_path(&pkg, &r.key).exists());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn load_refuses_a_missing_package() {
        let err = CoremlTrunk::load(Path::new("/nonexistent/x.mlpackage"), Units::CpuAndGpu)
            .err()
            .expect("refused");
        assert!(
            err.to_string().contains("not an .mlpackage directory"),
            "{err}"
        );
    }
}
