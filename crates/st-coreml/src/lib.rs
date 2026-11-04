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
use st_model::tracker::{ExternalMemAttn, ExternalMemEncoder, HostMemInputs};
use st_model::{ExternalEncoder, ExternalNeck, ExternalSegHead, ExternalTrunk, HostSegInputs};

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

/// A package hashed, compiled when its compiled model is missing, and loaded: what
/// [`CoremlTrunk::load`] and [`CoremlMemAttn::load`] share.
struct Loaded {
    model: Retained<MLModel>,
    sha256: String,
    compiled: bool,
    hashed: bool,
    load_seconds: f64,
    hash_seconds: f64,
    compile_seconds: f64,
    model_seconds: f64,
}

fn load_package(package: &Path, units: Units) -> Result<Loaded, CoremlError> {
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
    Ok(Loaded {
        model,
        sha256,
        compiled,
        hashed: resolved.hashed,
        load_seconds,
        hash_seconds: t_hash,
        compile_seconds: t_compile - t_hash,
        model_seconds: load_seconds - t_compile,
    })
}

/// One prediction of `model`: each input `(name, data, shape)` is wrapped in place (Core ML reads
/// it through the pointer while the prediction runs), and the f32 output `output` of shape `want`
/// is read back row-major.
fn predict_f32(
    model: &MLModel,
    inputs: &mut [(&str, &mut [f32], &[usize])],
    output: &str,
    want: &[usize],
) -> Result<Vec<f32>, CoremlError> {
    let mut outs = predict_f32_multi(model, inputs, &[(output, want)])?;
    outs.pop().ok_or_else(|| fail("no output read"))
}

/// [`predict_f32`] with several f32 outputs, each `(name, shape)`, read back in that order.
fn predict_f32_multi(
    model: &MLModel,
    inputs: &mut [(&str, &mut [f32], &[usize])],
    outputs: &[(&str, &[usize])],
) -> Result<Vec<Vec<f32>>, CoremlError> {
    autoreleasepool(|_| {
        let mut keys: Vec<Retained<NSString>> = Vec::with_capacity(inputs.len());
        let mut values: Vec<Retained<MLFeatureValue>> = Vec::with_capacity(inputs.len());
        for (name, data, shape) in inputs.iter_mut() {
            let n: usize = shape.iter().product();
            if data.len() != n {
                return Err(fail(format!(
                    "input {name:?} has {} values, want {n} ({shape:?})",
                    data.len()
                )));
            }
            let ptr = NonNull::new(data.as_mut_ptr().cast::<c_void>())
                .ok_or_else(|| fail(format!("input {name:?} is null")))?;
            // SAFETY: `data` is borrowed mutably for this whole call, which ends after the
            // prediction; Core ML only reads it, and no deallocator is passed.
            let arr = unsafe {
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(),
                    ptr,
                    &ns_ints(shape),
                    MLMultiArrayDataType::Float32,
                    &ns_ints(&contiguous_strides(shape)),
                    None,
                )
            }
            .map_err(|e| ns_fail(&format!("wrapping input {name:?}"), &e))?;
            keys.push(NSString::from_str(name));
            values.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) });
        }
        let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
        let value_refs: Vec<&AnyObject> = values.iter().map(|v| v.as_ref()).collect();
        let dict: Retained<NSDictionary<NSString, AnyObject>> =
            NSDictionary::from_slices(&key_refs, &value_refs);
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| ns_fail("building the input features", &e))?;
        let out =
            unsafe { model.predictionFromFeatures_error(ProtocolObject::from_ref(&*provider)) }
                .map_err(|e| ns_fail("prediction", &e))?;
        outputs
            .iter()
            .map(|&(output, want)| {
                let value = unsafe { out.featureValueForName(&NSString::from_str(output)) }
                    .ok_or_else(|| fail(format!("the model has no output {output:?}")))?;
                let arr = unsafe { value.multiArrayValue() }
                    .ok_or_else(|| fail(format!("output {output:?} is not a multi-array")))?;
                read_f32(&arr, want)
            })
            .collect()
    })
}

impl CoremlTrunk {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        let l = load_package(package, units)?;
        Ok(Self {
            model: l.model,
            package: package.to_path_buf(),
            sha256: l.sha256,
            units,
            compiled: l.compiled,
            hashed: l.hashed,
            load_seconds: l.load_seconds,
            hash_seconds: l.hash_seconds,
            compile_seconds: l.compile_seconds,
            model_seconds: l.model_seconds,
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
        let mut input = image.to_vec();
        predict_f32(
            &self.model,
            &mut [(INPUT, &mut input, &IMAGE_SHAPE)],
            OUTPUT,
            &TRUNK_SHAPE,
        )
    }
}

impl ExternalTrunk for CoremlTrunk {
    fn describe(&self) -> String {
        describe(&self.package, self.units, &self.sha256)
    }

    fn run(&self, image: &[f32]) -> Result<Vec<f32>, OjasError> {
        self.predict(image).map_err(|e| OjasError::Unsupported {
            op: "st_coreml::CoremlTrunk::run",
            detail: e.to_string(),
        })
    }
}

/// What a [`PipelinedTrunk`]'s worker reports after loading its programs.
pub struct TrunkLoadReport {
    pub describe: String,
    /// One per neck program the worker runs after the trunk, in the order they were given.
    pub necks: Vec<String>,
    pub compiled: bool,
    pub hashed: bool,
    pub load_seconds: f64,
    pub hash_seconds: f64,
    pub compile_seconds: f64,
    pub model_seconds: f64,
}

/// One frame from the worker: the trunk and, per neck program, its three levels.
struct FrameOut {
    trunk: Vec<f32>,
    necks: Vec<[Vec<f32>; 3]>,
}

enum Job {
    Frame {
        image: Vec<f32>,
        reply: mpsc::Sender<Result<FrameOut, String>>,
    },
    /// One neck program on a trunk the worker did not produce (or whose levels were already taken).
    Neck {
        which: usize,
        trunk: Vec<f32>,
        reply: mpsc::Sender<Result<[Vec<f32>; 3], String>>,
    },
}

/// An image sent to the worker, and where its frame will arrive.
struct Queued {
    image: Vec<f32>,
    reply: mpsc::Receiver<Result<FrameOut, String>>,
}

/// Neck levels computed with a trunk already handed out, until [`PipelinedNeck::run`] takes them.
struct Delivered {
    trunk: Vec<f32>,
    necks: Vec<Option<[Vec<f32>; 3]>>,
}

/// Images [`PipelinedTrunk::prefetch`] keeps queued at most: the frame being tracked and the next.
const MAX_QUEUED: usize = 2;
/// Frames whose neck levels are kept for [`PipelinedNeck::run`].
const MAX_DELIVERED: usize = 2;

struct PipeInner {
    describe: String,
    neck_describe: Vec<String>,
    tx: std::sync::Mutex<Option<mpsc::Sender<Job>>>,
    queue: std::sync::Mutex<std::collections::VecDeque<Queued>>,
    delivered: std::sync::Mutex<std::collections::VecDeque<Delivered>>,
    worker: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The trunk package, and optionally neck packages fed by it, on their own thread. The caller starts
/// the next frame ([`PipelinedTrunk::prefetch`]) while the current frame's detector and tracker run.
/// `run` returns a queued frame's trunk when its image is bit-identical to the one asked for, and
/// otherwise queues the image and waits for it. Either way the result is the same program's
/// prediction of that image. The neck levels the worker computed from that trunk wait for
/// [`PipelinedNeck`]. Clones share one worker: keep one to prefetch with and hand the others to the
/// backbone.
#[derive(Clone)]
pub struct PipelinedTrunk(std::rc::Rc<PipeInner>);

/// One of a [`PipelinedTrunk`]'s neck programs, for [`st_model::Backbone::with_external_neck`].
pub struct PipelinedNeck {
    pipe: PipelinedTrunk,
    which: usize,
}

impl PipelinedTrunk {
    /// Starts the worker and waits for it to load `package` and `necks` itself (a Core ML model
    /// stays on the thread that loaded it).
    pub fn load(
        package: &Path,
        necks: &[PathBuf],
        units: Units,
    ) -> Result<(Self, TrunkLoadReport), CoremlError> {
        let (tx_job, rx_job) = mpsc::channel::<Job>();
        let (tx_load, rx_load) = mpsc::channel::<Result<TrunkLoadReport, String>>();
        let package = package.to_path_buf();
        let neck_packages = necks.to_vec();
        let worker = std::thread::Builder::new()
            .name("coreml-trunk".into())
            .spawn(move || {
                let loaded = CoremlTrunk::load(&package, units).and_then(|t| {
                    let necks = neck_packages
                        .iter()
                        .map(|p| CoremlNeck::load(p, units))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((t, necks))
                });
                let (trunk, necks) = match loaded {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx_load.send(Err(e.to_string()));
                        return;
                    }
                };
                let report = TrunkLoadReport {
                    describe: format!("{} pipelined", describe(&package, units, &trunk.sha256)),
                    necks: necks
                        .iter()
                        .map(|n| format!("{} pipelined", ExternalNeck::describe(n)))
                        .collect(),
                    compiled: trunk.compiled,
                    hashed: trunk.hashed,
                    load_seconds: trunk.load_seconds
                        + necks.iter().map(|n| n.load_seconds).sum::<f64>(),
                    hash_seconds: trunk.hash_seconds,
                    compile_seconds: trunk.compile_seconds,
                    model_seconds: trunk.model_seconds,
                };
                if tx_load.send(Ok(report)).is_err() {
                    return;
                }
                while let Ok(job) = rx_job.recv() {
                    match job {
                        Job::Frame { image, reply } => {
                            let out = trunk.predict(&image).and_then(|t| {
                                let levels = necks.iter().map(|n| n.predict(&t)).collect::<Result<
                                    Vec<_>,
                                    _,
                                >>(
                                )?;
                                Ok(FrameOut {
                                    trunk: t,
                                    necks: levels,
                                })
                            });
                            let _ = reply.send(out.map_err(|e| e.to_string()));
                        }
                        Job::Neck {
                            which,
                            trunk,
                            reply,
                        } => {
                            let out = match necks.get(which) {
                                Some(n) => n.predict(&trunk).map_err(|e| e.to_string()),
                                None => Err(format!("no neck program {which}")),
                            };
                            let _ = reply.send(out);
                        }
                    }
                }
            })
            .map_err(|e| fail(format!("starting the trunk worker: {e}")))?;
        let report = match rx_load.recv() {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                let _ = worker.join();
                return Err(fail(e));
            }
            Err(_) => {
                let _ = worker.join();
                return Err(fail("the trunk worker exited while loading"));
            }
        };
        Ok((
            Self(std::rc::Rc::new(PipeInner {
                describe: report.describe.clone(),
                neck_describe: report.necks.clone(),
                tx: std::sync::Mutex::new(Some(tx_job)),
                queue: std::sync::Mutex::new(std::collections::VecDeque::new()),
                delivered: std::sync::Mutex::new(std::collections::VecDeque::new()),
                worker: std::sync::Mutex::new(Some(worker)),
            })),
            report,
        ))
    }

    /// The external neck for the worker's neck program `which` (the order `load` was given).
    pub fn neck(&self, which: usize) -> Result<PipelinedNeck, CoremlError> {
        if which >= self.0.neck_describe.len() {
            return Err(fail(format!(
                "neck {which} of a pipeline with {} neck program(s)",
                self.0.neck_describe.len()
            )));
        }
        Ok(PipelinedNeck {
            pipe: self.clone(),
            which,
        })
    }

    fn send(&self, job: Job) -> Result<(), CoremlError> {
        let tx = self
            .0
            .tx
            .lock()
            .map_err(|_| fail("trunk sender poisoned"))?;
        tx.as_ref()
            .ok_or_else(|| fail("the trunk worker was shut down"))?
            .send(job)
            .map_err(|_| fail("the trunk worker exited"))
    }

    fn submit(&self, image: Vec<f32>) -> Result<Queued, CoremlError> {
        let (reply, rx) = mpsc::channel();
        self.send(Job::Frame {
            image: image.clone(),
            reply,
        })?;
        Ok(Queued { image, reply: rx })
    }

    /// Queue `image` on the worker. When [`MAX_QUEUED`] images are queued already, the oldest is
    /// dropped (the worker still finishes it; nothing waits for it).
    pub fn prefetch(&self, image: Vec<f32>) -> Result<(), CoremlError> {
        let mut q = self
            .0
            .queue
            .lock()
            .map_err(|_| fail("trunk queue poisoned"))?;
        while q.len() >= MAX_QUEUED {
            q.pop_front();
        }
        let queued = self.submit(image)?;
        q.push_back(queued);
        Ok(())
    }

    /// The trunk of `image`: a queued frame when `image` is one queued (bit for bit), otherwise one
    /// predicted after whatever is queued ahead of it. Its neck levels are kept for [`PipelinedNeck`].
    pub fn predict(&self, image: &[f32]) -> Result<Vec<f32>, CoremlError> {
        let entry = {
            let mut q = self
                .0
                .queue
                .lock()
                .map_err(|_| fail("trunk queue poisoned"))?;
            match q.iter().position(|e| same_bits(&e.image, image)) {
                Some(i) => q
                    .remove(i)
                    .ok_or_else(|| fail("trunk queue lost an entry"))?,
                None => self.submit(image.to_vec())?,
            }
        };
        let out = entry
            .reply
            .recv()
            .map_err(|_| fail("the trunk worker exited"))?
            .map_err(fail)?;
        if !out.necks.is_empty() {
            let mut d = self
                .0
                .delivered
                .lock()
                .map_err(|_| fail("neck store poisoned"))?;
            d.push_back(Delivered {
                trunk: out.trunk.clone(),
                necks: out.necks.into_iter().map(Some).collect(),
            });
            while d.len() > MAX_DELIVERED {
                d.pop_front();
            }
        }
        Ok(out.trunk)
    }

    /// Neck program `which` on `trunk`: the levels the worker computed with it when this trunk came
    /// from it (bit for bit) and they were not taken yet, otherwise a prediction made now.
    fn neck_levels(&self, which: usize, trunk: &[f32]) -> Result<[Vec<f32>; 3], CoremlError> {
        {
            let mut d = self
                .0
                .delivered
                .lock()
                .map_err(|_| fail("neck store poisoned"))?;
            if let Some(levels) = d
                .iter_mut()
                .rev()
                .find(|e| same_bits(&e.trunk, trunk))
                .and_then(|e| e.necks.get_mut(which).and_then(Option::take))
            {
                return Ok(levels);
            }
        }
        let (reply, rx) = mpsc::channel();
        self.send(Job::Neck {
            which,
            trunk: trunk.to_vec(),
            reply,
        })?;
        rx.recv()
            .map_err(|_| fail("the trunk worker exited"))?
            .map_err(fail)
    }
}

impl Drop for PipeInner {
    fn drop(&mut self) {
        if let Ok(mut tx) = self.tx.lock() {
            tx.take();
        }
        if let Some(w) = self.worker.lock().ok().and_then(|mut w| w.take()) {
            let _ = w.join();
        }
    }
}

impl ExternalTrunk for PipelinedTrunk {
    fn describe(&self) -> String {
        self.0.describe.clone()
    }

    fn run(&self, image: &[f32]) -> Result<Vec<f32>, OjasError> {
        self.predict(image).map_err(|e| OjasError::Unsupported {
            op: "st_coreml::PipelinedTrunk::run",
            detail: e.to_string(),
        })
    }
}

impl ExternalNeck for PipelinedNeck {
    fn describe(&self) -> String {
        self.pipe.0.neck_describe[self.which].clone()
    }

    fn run(&self, trunk: &[f32]) -> Result<[Vec<f32>; 3], OjasError> {
        self.pipe
            .neck_levels(self.which, trunk)
            .map_err(|e| OjasError::Unsupported {
                op: "st_coreml::PipelinedNeck::run",
                detail: e.to_string(),
            })
    }
}

/// One neck as a Core ML program (`<model-dir>/coreml/sam3.1_neck_<det|prop>_<precision>.mlpackage`,
/// written by tools/baselines/coreml_necks.py --neck), for [`st_model::Backbone::with_external_neck`].
/// Input `trunk` `[1, 72, 72, 1024]` NHWC; outputs `l0`, `l1`, `l2` `[1, 256, h, w]` at 288, 144, 72.
pub struct CoremlNeck {
    model: Retained<MLModel>,
    package: PathBuf,
    sha256: String,
    units: Units,
    input: std::sync::Mutex<Vec<f32>>,
    pub compiled: bool,
    pub load_seconds: f64,
}

impl CoremlNeck {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        let l = load_package(package, units)?;
        Ok(Self {
            model: l.model,
            package: package.to_path_buf(),
            sha256: l.sha256,
            units,
            input: std::sync::Mutex::new(vec![0f32; TRUNK_SHAPE.iter().product()]),
            compiled: l.compiled,
            load_seconds: l.load_seconds,
        })
    }

    /// One prediction: the three levels, row-major NCHW.
    pub fn predict(&self, trunk: &[f32]) -> Result<[Vec<f32>; 3], CoremlError> {
        let mut input = self
            .input
            .lock()
            .map_err(|_| fail("neck input buffer poisoned by an earlier panic"))?;
        if trunk.len() != input.len() {
            return Err(fail(format!(
                "trunk has {} values, want {} ({TRUNK_SHAPE:?})",
                trunk.len(),
                input.len()
            )));
        }
        input.copy_from_slice(trunk);
        let g = TRUNK_SHAPE[1];
        let shapes = [
            [1, 256, 4 * g, 4 * g],
            [1, 256, 2 * g, 2 * g],
            [1, 256, g, g],
        ];
        let outs = predict_f32_multi(
            &self.model,
            &mut [("trunk", &mut input, &TRUNK_SHAPE)],
            &[("l0", &shapes[0]), ("l1", &shapes[1]), ("l2", &shapes[2])],
        )?;
        let [a, b, c]: [Vec<f32>; 3] = outs
            .try_into()
            .map_err(|_| fail("the neck package returned other than three levels"))?;
        Ok([a, b, c])
    }
}

impl ExternalNeck for CoremlNeck {
    fn describe(&self) -> String {
        describe(&self.package, self.units, &self.sha256)
    }

    fn run(&self, trunk: &[f32]) -> Result<[Vec<f32>; 3], OjasError> {
        self.predict(trunk).map_err(|e| OjasError::Unsupported {
            op: "st_coreml::CoremlNeck::run",
            detail: e.to_string(),
        })
    }
}

/// The memory-attention package's fixed contract (tools/baselines/coreml_mattn.py): frame and
/// pointer slots past the ones in use hold zeros and are masked out by `key_bias`.
pub mod mattn {
    pub const T: usize = 72 * 72;
    pub const D: usize = 256;
    pub const MAX_FRAMES: usize = 7;
    pub const MAX_PTRS: usize = 256;
    pub const NKEYS: usize = MAX_FRAMES * T + MAX_PTRS;
    /// Added to a padded key's logit: large and finite, so an fp16 build cannot produce NaN.
    pub const MASK: f32 = -1e4;
}

/// The padded host inputs, kept between calls so only what changed is rewritten.
struct MattnBufs {
    image: Vec<f32>,
    src_pos: Vec<f32>,
    mem_image: Vec<f32>,
    mem: Vec<f32>,
    mem_pos: Vec<f32>,
    ptr: Vec<f32>,
    ptr_pos: Vec<f32>,
    key_bias: Vec<f32>,
    /// Frame and pointer slots the last call filled (the rest are already zero).
    frames: usize,
    ptrs: usize,
}

/// SAM 3.1's tracker memory attention as a Core ML program
/// (`<model-dir>/coreml/sam3.1_mattn_<precision>.mlpackage`, written by
/// tools/baselines/coreml_mattn.py), for [`st_model::tracker::Vos::with_external_mem_attn`].
pub struct CoremlMemAttn {
    model: Retained<MLModel>,
    package: PathBuf,
    sha256: String,
    units: Units,
    bufs: std::sync::Mutex<MattnBufs>,
    pub compiled: bool,
    pub load_seconds: f64,
}

impl CoremlMemAttn {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        use mattn::*;
        let l = load_package(package, units)?;
        let zeros = |n: usize| vec![0f32; n];
        let key_bias = vec![MASK; NKEYS];
        Ok(Self {
            model: l.model,
            package: package.to_path_buf(),
            sha256: l.sha256,
            units,
            bufs: std::sync::Mutex::new(MattnBufs {
                image: zeros(T * D),
                src_pos: zeros(T * D),
                mem_image: zeros(MAX_FRAMES * T * D),
                mem: zeros(MAX_FRAMES * T * D),
                mem_pos: zeros(MAX_FRAMES * T * D),
                ptr: zeros(MAX_PTRS * D),
                ptr_pos: zeros(MAX_PTRS * D),
                key_bias,
                frames: 0,
                ptrs: 0,
            }),
            compiled: l.compiled,
            load_seconds: l.load_seconds,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// One prediction from [`HostMemInputs`]: `[T, D]` after the final norm.
    pub fn predict(&self, x: &HostMemInputs<'_>) -> Result<Vec<f32>, CoremlError> {
        use mattn::*;
        let (f, p) = (x.frames.len(), x.ptrs.len() / D);
        if f > MAX_FRAMES
            || p > MAX_PTRS
            || x.ptrs.len() != p * D
            || x.ptr_pos.len() != x.ptrs.len()
        {
            return Err(fail(format!(
                "{f} memory frames and {} pointer values ({} positions): the package holds at most {MAX_FRAMES} frames and {MAX_PTRS} pointer tokens of {D}",
                x.ptrs.len(),
                x.ptr_pos.len()
            )));
        }
        let one = T * D;
        if x.image.len() != one || x.src_pos.len() != one {
            return Err(fail(format!(
                "image {} and src_pos {} values, want {one}",
                x.image.len(),
                x.src_pos.len()
            )));
        }
        for (i, fr) in x.frames.iter().enumerate() {
            if fr.image.len() != one || fr.memory.len() != one || fr.pos.len() != one {
                return Err(fail(format!(
                    "memory frame {i}: inputs are not [{T}, {D}] each"
                )));
            }
        }
        let mut guard = self
            .bufs
            .lock()
            .map_err(|_| fail("memory-attention buffers poisoned by an earlier panic"))?;
        let b = &mut *guard;
        b.image.copy_from_slice(x.image);
        b.src_pos.copy_from_slice(x.src_pos);
        for (i, fr) in x.frames.iter().enumerate() {
            let r = i * one..(i + 1) * one;
            b.mem_image[r.clone()].copy_from_slice(fr.image);
            b.mem[r.clone()].copy_from_slice(fr.memory);
            b.mem_pos[r].copy_from_slice(fr.pos);
        }
        for i in f..b.frames {
            let r = i * one..(i + 1) * one;
            b.mem_image[r.clone()].fill(0.0);
            b.mem[r.clone()].fill(0.0);
            b.mem_pos[r].fill(0.0);
        }
        b.ptr[..p * D].copy_from_slice(x.ptrs);
        b.ptr_pos[..p * D].copy_from_slice(x.ptr_pos);
        if b.ptrs > p {
            b.ptr[p * D..b.ptrs * D].fill(0.0);
            b.ptr_pos[p * D..b.ptrs * D].fill(0.0);
        }
        b.key_bias[..MAX_FRAMES * T].fill(MASK);
        b.key_bias[..f * T].fill(0.0);
        b.key_bias[MAX_FRAMES * T..].fill(MASK);
        b.key_bias[MAX_FRAMES * T..MAX_FRAMES * T + p].fill(0.0);
        b.frames = f;
        b.ptrs = p;
        let (t3, frames3, ptrs3, bias4) = (
            [1, T, D],
            [1, MAX_FRAMES * T, D],
            [1, MAX_PTRS, D],
            [1, 1, 1, NKEYS],
        );
        let out = predict_f32(
            &self.model,
            &mut [
                ("image", &mut b.image, &t3),
                ("src_pos", &mut b.src_pos, &t3),
                ("mem_image", &mut b.mem_image, &frames3),
                ("mem", &mut b.mem, &frames3),
                ("mem_pos", &mut b.mem_pos, &frames3),
                ("ptr", &mut b.ptr, &ptrs3),
                ("ptr_pos", &mut b.ptr_pos, &ptrs3),
                ("key_bias", &mut b.key_bias, &bias4),
            ],
            "out",
            &t3,
        )?;
        Ok(out)
    }
}

impl ExternalMemAttn for CoremlMemAttn {
    fn describe(&self) -> String {
        describe(&self.package, self.units, &self.sha256)
    }

    fn run(&self, x: &HostMemInputs<'_>) -> Result<Vec<f32>, OjasError> {
        self.predict(x).map_err(|e| OjasError::Unsupported {
            op: "st_coreml::CoremlMemAttn::run",
            detail: e.to_string(),
        })
    }
}

/// The segmentation head's fixed contract (tools/baselines/coreml_seghead.py): prompt slots past
/// the ones in use hold zeros and are masked out by `prompt_bias`.
pub mod seghead {
    pub const T: usize = 72 * 72;
    pub const D: usize = 256;
    pub const Q: usize = 200;
    pub const M: usize = 288;
    pub const MAX_PROMPT: usize = 33;
    /// Added to a padded prompt key's logit: large and finite, so an fp16 build cannot produce NaN.
    pub const MASK: f32 = -1e4;
}

struct SegBufs {
    memory: Vec<f32>,
    prompt: Vec<f32>,
    prompt_bias: Vec<f32>,
    fpn0: Vec<f32>,
    fpn1: Vec<f32>,
    hs: Vec<f32>,
}

/// SAM 3.1's detector segmentation head as a Core ML program
/// (`<model-dir>/coreml/sam3.1_seghead_<precision>.mlpackage`, written by
/// tools/baselines/coreml_seghead.py), for [`st_model::Detector::set_external_seg_head`].
pub struct CoremlSegHead {
    model: Retained<MLModel>,
    package: PathBuf,
    sha256: String,
    units: Units,
    bufs: std::sync::Mutex<SegBufs>,
    pub compiled: bool,
    pub load_seconds: f64,
}

impl CoremlSegHead {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        use seghead::*;
        let l = load_package(package, units)?;
        Ok(Self {
            model: l.model,
            package: package.to_path_buf(),
            sha256: l.sha256,
            units,
            bufs: std::sync::Mutex::new(SegBufs {
                memory: vec![0.0; T * D],
                prompt: vec![0.0; MAX_PROMPT * D],
                prompt_bias: vec![MASK; MAX_PROMPT],
                fpn0: vec![0.0; D * M * M],
                fpn1: vec![0.0; D * (M / 2) * (M / 2)],
                hs: vec![0.0; Q * D],
            }),
            compiled: l.compiled,
            load_seconds: l.load_seconds,
        })
    }

    /// One prediction: `pred_masks` `[Q, M, M]` and `semantic_seg` `[1, 1, M, M]`.
    pub fn predict(&self, x: &HostSegInputs<'_>) -> Result<(Vec<f32>, Vec<f32>), CoremlError> {
        use seghead::*;
        let plen = x.prompt.len() / D;
        if x.prompt.len() != plen * D || plen == 0 || plen > MAX_PROMPT {
            return Err(fail(format!(
                "prompt has {} values: want 1 to {MAX_PROMPT} tokens of {D}",
                x.prompt.len()
            )));
        }
        let mut guard = self
            .bufs
            .lock()
            .map_err(|_| fail("segmentation-head buffers poisoned by an earlier panic"))?;
        let b = &mut *guard;
        for (name, src, dst) in [
            ("memory", x.memory, &mut b.memory),
            ("fpn0", x.fpn0, &mut b.fpn0),
            ("fpn1", x.fpn1, &mut b.fpn1),
            ("hs", x.hs, &mut b.hs),
        ] {
            if src.len() != dst.len() {
                return Err(fail(format!(
                    "{name} has {} values, want {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
        }
        b.prompt[..plen * D].copy_from_slice(x.prompt);
        b.prompt[plen * D..].fill(0.0);
        b.prompt_bias.fill(MASK);
        b.prompt_bias[..plen].fill(0.0);
        let mut outs = predict_f32_multi(
            &self.model,
            &mut [
                ("mem", &mut b.memory, &[1, T, D]),
                ("prompt", &mut b.prompt, &[1, MAX_PROMPT, D]),
                ("prompt_bias", &mut b.prompt_bias, &[1, 1, 1, MAX_PROMPT]),
                ("fpn0", &mut b.fpn0, &[1, D, M, M]),
                ("fpn1", &mut b.fpn1, &[1, D, M / 2, M / 2]),
                ("hs", &mut b.hs, &[1, Q, D]),
            ],
            &[
                ("pred_masks", &[1, Q, M, M]),
                ("semantic_seg", &[1, 1, M, M]),
            ],
        )?;
        let semantic = outs.pop().ok_or_else(|| fail("no semantic_seg read"))?;
        let masks = outs.pop().ok_or_else(|| fail("no pred_masks read"))?;
        Ok((masks, semantic))
    }
}

impl ExternalSegHead for CoremlSegHead {
    fn describe(&self) -> String {
        describe(&self.package, self.units, &self.sha256)
    }

    fn run(&self, x: &HostSegInputs<'_>) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
        self.predict(x).map_err(|e| OjasError::Unsupported {
            op: "st_coreml::CoremlSegHead::run",
            detail: e.to_string(),
        })
    }
}

pub mod detenc {
    pub const T: usize = 72 * 72;
    pub const D: usize = 256;
    pub const GRID: usize = 72;
    /// Text slots the program takes; the geometry token is one more key.
    pub const TEXT: usize = 32;
    /// Added to a padded text key's logit: large and finite, so an fp16 build cannot produce NaN.
    pub const MASK: f32 = -1e4;
}

/// The tag a detector-encoder package carries for the adapter baked into it: the first 12 hex of
/// the adapter file's sha256, or `base` without one (tools/baselines/coreml_detenc.py `adapter_tag`).
pub fn adapter_tag(adapter: Option<&Path>) -> Result<String, CoremlError> {
    let Some(path) = adapter else {
        return Ok("base".into());
    };
    let bytes =
        std::fs::read(path).map_err(|e| fail(format!("reading {}: {e}", path.display())))?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().take(6).map(|b| format!("{b:02x}")).collect())
}

struct EncBufs {
    fpn2: Vec<f32>,
    pos: Vec<f32>,
    text: Vec<f32>,
    text_bias: Vec<f32>,
}

/// SAM 3.1's detector geometry + fusion encoders as one Core ML program
/// (`<model-dir>/coreml/sam3.1_detenc_<precision>_<adapter tag>.mlpackage`, written by
/// tools/baselines/coreml_detenc.py), for [`st_model::Detector::set_external_encoder`]. The adapter
/// it was exported with is baked in; [`adapter_tag`] names which.
pub struct CoremlDetEncoder {
    model: Retained<MLModel>,
    package: PathBuf,
    sha256: String,
    units: Units,
    bufs: std::sync::Mutex<EncBufs>,
    pub compiled: bool,
    pub load_seconds: f64,
}

impl CoremlDetEncoder {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        use detenc::*;
        let l = load_package(package, units)?;
        Ok(Self {
            model: l.model,
            package: package.to_path_buf(),
            sha256: l.sha256,
            units,
            bufs: std::sync::Mutex::new(EncBufs {
                fpn2: vec![0.0; D * T],
                pos: vec![0.0; T * D],
                text: vec![0.0; TEXT * D],
                text_bias: vec![MASK; TEXT + 1],
            }),
            compiled: l.compiled,
            load_seconds: l.load_seconds,
        })
    }

    /// One prediction: the encoder memory `[T, D]` and the geometry token `[D]`.
    pub fn predict(
        &self,
        fpn2: &[f32],
        pos: &[f32],
        text: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), CoremlError> {
        use detenc::*;
        let n = text.len() / D;
        if text.len() != n * D || n == 0 || n > TEXT {
            return Err(fail(format!(
                "text has {} values: want 1 to {TEXT} tokens of {D}",
                text.len()
            )));
        }
        let mut guard = self
            .bufs
            .lock()
            .map_err(|_| fail("encoder buffers poisoned by an earlier panic"))?;
        let b = &mut *guard;
        for (name, src, dst) in [("fpn2", fpn2, &mut b.fpn2), ("pos", pos, &mut b.pos)] {
            if src.len() != dst.len() {
                return Err(fail(format!(
                    "{name} has {} values, want {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
        }
        b.text[..n * D].copy_from_slice(text);
        b.text[n * D..].fill(0.0);
        b.text_bias.fill(MASK);
        b.text_bias[..n].fill(0.0);
        b.text_bias[TEXT] = 0.0;
        let mut outs = predict_f32_multi(
            &self.model,
            &mut [
                ("fpn2", &mut b.fpn2, &[1, D, GRID, GRID]),
                ("pos", &mut b.pos, &[1, T, D]),
                ("text", &mut b.text, &[1, TEXT, D]),
                ("text_bias", &mut b.text_bias, &[1, 1, 1, TEXT + 1]),
            ],
            &[("mem", &[1, T, D]), ("geo", &[1, 1, D])],
        )?;
        let geo = outs.pop().ok_or_else(|| fail("no geo read"))?;
        let mem = outs.pop().ok_or_else(|| fail("no mem read"))?;
        Ok((mem, geo))
    }
}

impl ExternalEncoder for CoremlDetEncoder {
    fn describe(&self) -> String {
        describe(&self.package, self.units, &self.sha256)
    }

    fn run(
        &self,
        fpn2: &[f32],
        pos: &[f32],
        text: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
        self.predict(fpn2, pos, text)
            .map_err(|e| OjasError::Unsupported {
                op: "st_coreml::CoremlDetEncoder::run",
                detail: e.to_string(),
            })
    }
}

pub mod memenc {
    pub const GRID: usize = 72;
    pub const D: usize = 256;
    /// Multiplex slots: the program takes one mask plane and one condition flag per slot.
    pub const MUX: usize = 16;
    /// The downsampler's input side, 16 x the grid.
    pub const S: usize = 16 * GRID;
}

struct MemEncBufs {
    pix: Vec<f32>,
    masks: Vec<f32>,
    cond: Vec<f32>,
}

/// SAM 3.1's multiplex memory encoder as one Core ML program
/// (`<model-dir>/coreml/sam3.1_memenc_<precision>.mlpackage`, written by
/// tools/baselines/coreml_memenc.py), for [`st_model::tracker::vos::Vos::with_external_mem_encoder`].
/// Its mask buffer (16 planes of 1152 x 1152) is allocated once, and the caller writes the planes
/// into it directly.
pub struct CoremlMemEncoder {
    model: Retained<MLModel>,
    package: PathBuf,
    sha256: String,
    units: Units,
    bufs: std::sync::Mutex<MemEncBufs>,
    pub compiled: bool,
    pub load_seconds: f64,
}

impl CoremlMemEncoder {
    pub fn load(package: &Path, units: Units) -> Result<Self, CoremlError> {
        use memenc::*;
        let l = load_package(package, units)?;
        Ok(Self {
            model: l.model,
            package: package.to_path_buf(),
            sha256: l.sha256,
            units,
            bufs: std::sync::Mutex::new(MemEncBufs {
                pix: vec![0.0; D * GRID * GRID],
                masks: vec![0.0; MUX * S * S],
                cond: vec![0.0; MUX],
            }),
            compiled: l.compiled,
            load_seconds: l.load_seconds,
        })
    }

    /// One prediction: the features `[256, 72, 72]`.
    pub fn predict(
        &self,
        pix_feat: &[f32],
        fill: &mut dyn FnMut(&mut [f32]),
        cond: &[f32],
    ) -> Result<Vec<f32>, CoremlError> {
        use memenc::*;
        let mut guard = self
            .bufs
            .lock()
            .map_err(|_| fail("memory encoder buffers poisoned by an earlier panic"))?;
        let b = &mut *guard;
        for (name, src, dst) in [
            ("pix_feat", pix_feat, &mut b.pix),
            ("cond", cond, &mut b.cond),
        ] {
            if src.len() != dst.len() {
                return Err(fail(format!(
                    "{name} has {} values, want {}",
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
        }
        fill(&mut b.masks);
        let mut outs = predict_f32_multi(
            &self.model,
            &mut [
                ("pix_feat", &mut b.pix, &[1, D, GRID, GRID]),
                ("masks", &mut b.masks, &[1, MUX, S, S]),
                ("cond", &mut b.cond, &[1, MUX, 1, 1]),
            ],
            &[("features", &[1, D, GRID, GRID])],
        )?;
        outs.pop().ok_or_else(|| fail("no features read"))
    }
}

impl ExternalMemEncoder for CoremlMemEncoder {
    fn describe(&self) -> String {
        describe(&self.package, self.units, &self.sha256)
    }

    fn run(
        &self,
        pix_feat: &[f32],
        fill: &mut dyn FnMut(&mut [f32]),
        cond: &[f32],
    ) -> Result<Vec<f32>, OjasError> {
        self.predict(pix_feat, fill, cond)
            .map_err(|e| OjasError::Unsupported {
                op: "st_coreml::CoremlMemEncoder::run",
                detail: e.to_string(),
            })
    }
}

/// What `run.json` records about a loaded package.
fn describe(package: &Path, units: Units, sha256: &str) -> String {
    let name = package
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("coreml {name} ({}) sha256 {sha256}", units.name())
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
    if want.is_empty() || want.contains(&0) {
        return Err(fail(format!(
            "read_f32 needs a non-empty shape, got {want:?}"
        )));
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
                // Row-major walk over `want`, the last axis innermost, through Core ML's strides.
                let last = want.len() - 1;
                let mut out = Vec::with_capacity(n);
                let mut idx = vec![0usize; want.len()];
                'rows: loop {
                    let row: usize = idx[..last].iter().zip(&strides).map(|(i, s)| i * s).sum();
                    out.extend((0..want[last]).map(|d| src[row + d * strides[last]]));
                    for ax in (0..last).rev() {
                        idx[ax] += 1;
                        if idx[ax] < want[ax] {
                            continue 'rows;
                        }
                        idx[ax] = 0;
                    }
                    break;
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
