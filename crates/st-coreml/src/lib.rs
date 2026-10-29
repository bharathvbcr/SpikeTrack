//! SAM 3.1's ViT trunk as a CoreML ML Program, handed to [`st_model::Backbone`] as its
//! [`ExternalTrunk`]. `tools/baselines/coreml_trunk.py` exports the package from the converted
//! checkpoint; it is derived from gated weights and lives next to them, never in the repo.
//!
//! The package's contract (checked on every call, not assumed): input `image` `[1, 3, 1008, 1008]`
//! f32, normalized; output `trunk` `[1, 72, 72, 1024]` f32, NHWC.
//!
//! Loading hashes the `.mlpackage` (SHA-256 over its files' relative paths and bytes, in path
//! order) and loads `<package>.<first 16 hex>.mlmodelc` beside it, compiling it there once if it is
//! missing. A changed package gets a new compiled model, never a stale one.

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
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration,
    MLMultiArray, MLMultiArrayDataType,
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
    /// Hashing, compiling (when needed) and loading, in seconds.
    pub load_seconds: f64,
}

impl CoremlTrunk {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        let t0 = Instant::now();
        if !package.is_dir() {
            return Err(fail(format!("{} is not an .mlpackage directory", package.display())));
        }
        let sha256 = package_sha256(package)?;
        let compiled_dir = compiled_path(package, &sha256);
        let compiled = !compiled_dir.is_dir();
        if compiled {
            compile(package, &compiled_dir)?;
        }
        let url = NSURL::from_file_path(&compiled_dir).ok_or_else(|| fail(format!("{} is not a file URL", compiled_dir.display())))?;
        let model = autoreleasepool(|_| -> Result<Retained<MLModel>, CoremlError> {
            let cfg = unsafe { MLModelConfiguration::new() };
            unsafe { cfg.setComputeUnits(units.ml()) };
            unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg) }
                .map_err(|e| ns_fail(&format!("loading {}", compiled_dir.display()), &e))
        })?;
        Ok(Self { model, package: package.to_path_buf(), sha256, units, compiled, load_seconds: t0.elapsed().as_secs_f64() })
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
            return Err(fail(format!("image has {} values, want {n_in} ({IMAGE_SHAPE:?})", image.len())));
        }
        autoreleasepool(|_| {
            // Core ML reads the input through this pointer; the buffer outlives the prediction.
            let mut input = image.to_vec();
            let ptr = NonNull::new(input.as_mut_ptr().cast::<c_void>()).ok_or_else(|| fail("input buffer is null"))?;
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
            let dict: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(&[&*key], &[obj]);
            let provider = unsafe { MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict) }
                .map_err(|e| ns_fail("building the input features", &e))?;
            let out = unsafe { self.model.predictionFromFeatures_error(ProtocolObject::from_ref(&*provider)) }
                .map_err(|e| ns_fail("prediction", &e))?;
            let value = unsafe { out.featureValueForName(&NSString::from_str(OUTPUT)) }
                .ok_or_else(|| fail(format!("the model has no output {OUTPUT:?}")))?;
            let arr = unsafe { value.multiArrayValue() }.ok_or_else(|| fail(format!("output {OUTPUT:?} is not a multi-array")))?;
            let result = read_f32(&arr, &TRUNK_SHAPE);
            drop(input);
            result
        })
    }
}

impl ExternalTrunk for CoremlTrunk {
    fn describe(&self) -> String {
        let name = self.package.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        format!("coreml {name} ({}) sha256 {}", self.units.name(), self.sha256)
    }

    fn run(&self, image: &[f32]) -> Result<Vec<f32>, OjasError> {
        self.predict(image).map_err(|e| OjasError::Unsupported { op: "st_coreml::CoremlTrunk::run", detail: e.to_string() })
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
    let nums: Vec<Retained<NSNumber>> = v.iter().map(|&x| NSNumber::numberWithInteger(x as NSInteger)).collect();
    let refs: Vec<&NSNumber> = nums.iter().map(|n| &**n).collect();
    NSArray::from_slice(&refs)
}

fn ints_of(a: &NSArray<NSNumber>) -> Result<Vec<usize>, CoremlError> {
    a.iter().map(|n| usize::try_from(n.integerValue()).map_err(|_| fail("negative shape or stride"))).collect()
}

/// The f32 elements of `arr` in row-major order of `want`, honouring Core ML's strides (an output
/// may be padded); anything but an f32 array of exactly that shape is an error.
fn read_f32(arr: &MLMultiArray, want: &[usize]) -> Result<Vec<f32>, CoremlError> {
    if want.len() != 4 {
        return Err(fail("read_f32 reads 4-d arrays"));
    }
    let dtype = unsafe { arr.dataType() };
    if dtype != MLMultiArrayDataType::Float32 {
        return Err(fail(format!("output dtype is {:#x}, want Float32", dtype.0)));
    }
    let shape = ints_of(&*unsafe { arr.shape() })?;
    if shape != want {
        return Err(fail(format!("output shape is {shape:?}, want {want:?}")));
    }
    let strides = ints_of(&*unsafe { arr.strides() })?;
    if strides.len() != want.len() {
        return Err(fail(format!("output has {} strides for {} dims", strides.len(), want.len())));
    }
    let n: usize = want.iter().product();
    let span = 1 + want.iter().zip(&strides).map(|(d, s)| (d - 1) * s).sum::<usize>();
    let contiguous = strides == contiguous_strides(want);
    let got: RefCell<Result<Vec<f32>, CoremlError>> = RefCell::new(Err(fail("Core ML did not call the read handler")));
    let handler = StackBlock::new(|bytes: NonNull<c_void>, size: NSInteger| {
        let size = usize::try_from(size).unwrap_or(0);
        let r = if size < span * 4 {
            Err(fail(format!("output buffer is {size} bytes, its strides span {} bytes", span * 4)))
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

/// SHA-256 over every file under `package`: its relative path, a NUL, its length, then its bytes,
/// in sorted path order.
fn package_sha256(package: &Path) -> Result<String, CoremlError> {
    let mut files = Vec::new();
    let mut stack = vec![package.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| fail(format!("reading {}: {e}", dir.display())))?;
        for entry in entries {
            let path = entry.map_err(|e| fail(format!("reading {}: {e}", dir.display())))?.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    for path in &files {
        let rel = path.strip_prefix(package).map_err(|_| fail("package walk left the package"))?;
        h.update(rel.to_string_lossy().as_bytes());
        h.update([0]);
        let mut f = fs::File::open(path).map_err(|e| fail(format!("opening {}: {e}", path.display())))?;
        let len = f.metadata().map_err(|e| fail(format!("reading {}: {e}", path.display())))?.len();
        h.update(len.to_le_bytes());
        loop {
            let k = f.read(&mut buf).map_err(|e| fail(format!("reading {}: {e}", path.display())))?;
            if k == 0 {
                break;
            }
            h.update(&buf[..k]);
        }
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Compile `package` with Core ML and move the result to `dst` (via a staging name beside it, so a
/// crash mid-move never leaves a partial `dst`).
fn compile(package: &Path, dst: &Path) -> Result<(), CoremlError> {
    let url = NSURL::from_file_path(package).ok_or_else(|| fail(format!("{} is not a file URL", package.display())))?;
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
                Some(tmp) => fs::rename(&tmp, &move_to).map_err(|e| fail(format!("moving {} to {}: {e}", tmp.display(), move_to.display()))),
                None => Err(fail("the compiled model's URL is not a file path")),
            },
            (None, Some(e)) => Err(ns_fail("compiling", e)),
            (None, None) => Err(fail("Core ML's compile reported neither a model nor an error")),
        };
        let _ = tx.send(r);
    });
    autoreleasepool(|_| unsafe { MLModel::compileModelAtURL_completionHandler(&url, &handler) });
    let r = rx.recv_timeout(COMPILE_TIMEOUT).map_err(|_| fail(format!("compiling {} did not finish in {COMPILE_TIMEOUT:?}", package.display())))?;
    r?;
    match fs::rename(&staging, dst) {
        Ok(()) => Ok(()),
        // Another process compiled the same package first: keep its copy.
        Err(_) if dst.is_dir() => fs::remove_dir_all(&staging).map_err(|e| fail(format!("removing {}: {e}", staging.display()))),
        Err(e) => Err(fail(format!("moving {} to {}: {e}", staging.display(), dst.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_are_row_major() {
        assert_eq!(contiguous_strides(&TRUNK_SHAPE), vec![72 * 72 * 1024, 72 * 1024, 1024, 1]);
        assert_eq!(contiguous_strides(&[5]), vec![1]);
    }

    #[test]
    fn compiled_path_carries_the_hash_prefix() {
        let p = compiled_path(Path::new("/m/coreml/t_fp32.mlpackage"), "0123456789abcdef0123");
        assert_eq!(p, PathBuf::from("/m/coreml/t_fp32.mlpackage.0123456789abcdef.mlmodelc"));
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

    #[test]
    fn load_refuses_a_missing_package() {
        let err = CoremlTrunk::load(Path::new("/nonexistent/x.mlpackage"), Units::CpuAndGpu).err().expect("refused");
        assert!(err.to_string().contains("not an .mlpackage directory"), "{err}");
    }
}
