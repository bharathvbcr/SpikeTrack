//! SAM 3.1 multiplex checkpoint loader.
//!
//! Reads the safetensors file that `tools/reference/convert_checkpoint.py`
//! writes from Meta's `sam3.1_multiplex.pt` (the conversion runs offline; no
//! Python runs here). Every tensor is claimed by exactly one field of the typed
//! structs in [`names`]; a missing tensor, a wrong shape or a tensor nothing
//! claims is a [`WeightsError`] naming it.
//!
//! The structs are written against one [`Source`], so the same code that loads
//! the weights also lists the names and shapes it expects ([`schema`]), which a
//! test checks against `tools/reference/weights_manifest.json` without needing
//! the gated weights.

pub mod lora;
pub mod names;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use ojas_io::{IoError, SafeTensors, StDtype};

pub use names::{
    Conv, FpnLevel, FpnNeck, LayerNorm, Linear, Sam31Weights, TensorGroup, TextBlock, TextEncoder,
    VisionBackbone, VitBlock, VitTrunk,
};

/// Metadata key the converter uses to list the complex64 RoPE tables it stored
/// as f32 `[..., 2]` (re, im) pairs.
pub const COMPLEX_PAIRS_KEY: &str = "complex64_as_f32_pairs";

/// An f32 tensor, row-major, in the checkpoint's own (PyTorch) layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Weight {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Weight {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug)]
pub enum WeightsError {
    Io(IoError),
    Missing { name: String },
    Shape { name: String, expected: Vec<usize>, found: Vec<usize> },
    Dtype { name: String, found: StDtype },
    /// A RoPE table the converter did not mark as a complex pair, or the reverse.
    ComplexPair { name: String, detail: String },
    ClaimedTwice { name: String },
    Unclaimed { names: Vec<String> },
}

impl fmt::Display for WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "checkpoint I/O: {}", e.detail()),
            Self::Missing { name } => write!(f, "checkpoint has no tensor {name:?}"),
            Self::Shape { name, expected, found } => {
                write!(f, "{name}: shape {found:?}, expected {expected:?}")
            }
            Self::Dtype { name, found } => write!(f, "{name}: dtype {found:?}, expected F32"),
            Self::ComplexPair { name, detail } => write!(f, "{name}: {detail}"),
            Self::ClaimedTwice { name } => write!(f, "{name} is claimed by two fields"),
            Self::Unclaimed { names } => {
                let shown: Vec<&str> = names.iter().take(8).map(String::as_str).collect();
                write!(f, "{} checkpoint tensors claimed by no field: {shown:?}", names.len())
            }
        }
    }
}

impl std::error::Error for WeightsError {}

impl From<IoError> for WeightsError {
    fn from(e: IoError) -> Self {
        Self::Io(e)
    }
}

/// What the typed structs read from: a checkpoint, or a recorder of names.
pub trait Source {
    /// The tensor `name`, which must have exactly `shape`.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError>;
    /// A RoPE table stored complex64 upstream, here f32 `shape ++ [2]`.
    fn complex_pairs(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError>;
    /// Every tensor under `prefix` that no other field claimed, for module
    /// families whose typed structs land with their own port.
    fn group(&self, prefix: &str) -> Result<TensorGroup, WeightsError>;
}

/// An opened checkpoint that records which tensors have been claimed.
pub struct Checkpoint {
    st: SafeTensors<'static>,
    complex: BTreeSet<String>,
    claimed: RefCell<BTreeSet<String>>,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self, WeightsError> {
        let st = SafeTensors::open(path)?;
        let complex = match st.metadata().get(COMPLEX_PAIRS_KEY) {
            Some(list) => parse_name_list(list).map_err(|detail| WeightsError::ComplexPair {
                name: COMPLEX_PAIRS_KEY.to_string(),
                detail,
            })?,
            None => BTreeSet::new(),
        };
        Ok(Self { st, complex, claimed: RefCell::new(BTreeSet::new()) })
    }

    /// Load the whole model and refuse any tensor left unclaimed.
    pub fn load_all(path: &Path) -> Result<Sam31Weights, WeightsError> {
        let ck = Self::open(path)?;
        let w = Sam31Weights::load(&ck)?;
        ck.finish()?;
        Ok(w)
    }

    /// Names in the file that no field has claimed yet.
    pub fn unclaimed(&self) -> Vec<String> {
        let claimed = self.claimed.borrow();
        self.st.names().filter(|n| !claimed.contains(*n)).map(str::to_string).collect()
    }

    /// Succeeds only when every tensor in the file was claimed.
    pub fn finish(self) -> Result<(), WeightsError> {
        let names = self.unclaimed();
        if names.is_empty() {
            Ok(())
        } else {
            Err(WeightsError::Unclaimed { names })
        }
    }

    fn claim(&self, name: &str) -> Result<(), WeightsError> {
        if !self.claimed.borrow_mut().insert(name.to_string()) {
            return Err(WeightsError::ClaimedTwice { name: name.to_string() });
        }
        Ok(())
    }

    fn read(&self, name: &str) -> Result<Weight, WeightsError> {
        let info = self.st.info(name).map_err(|_| WeightsError::Missing { name: name.to_string() })?;
        if info.dtype != StDtype::F32 {
            return Err(WeightsError::Dtype { name: name.to_string(), found: info.dtype });
        }
        let (shape, data) = self.st.read_f32(name)?;
        let shape = shape.into_iter().map(|d| d as usize).collect();
        Ok(Weight { name: name.to_string(), shape, data })
    }

    fn read_shaped(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError> {
        // Check the header before reading the bytes or claiming the name.
        let info = self.st.info(name).map_err(|_| WeightsError::Missing { name: name.to_string() })?;
        let found: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        if found != shape {
            return Err(WeightsError::Shape { name: name.to_string(), expected: shape.to_vec(), found });
        }
        self.claim(name)?;
        self.read(name)
    }
}

impl Source for Checkpoint {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError> {
        if self.complex.contains(name) {
            return Err(WeightsError::ComplexPair {
                name: name.to_string(),
                detail: "stored as complex pairs but read as a plain tensor".to_string(),
            });
        }
        self.read_shaped(name, shape)
    }

    fn complex_pairs(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError> {
        if !self.complex.contains(name) {
            return Err(WeightsError::ComplexPair {
                name: name.to_string(),
                detail: format!("not listed under {COMPLEX_PAIRS_KEY} in the file metadata"),
            });
        }
        let mut full = shape.to_vec();
        full.push(2);
        self.read_shaped(name, &full)
    }

    fn group(&self, prefix: &str) -> Result<TensorGroup, WeightsError> {
        let names: Vec<String> = {
            let claimed = self.claimed.borrow();
            self.st
                .names()
                .filter(|n| n.starts_with(prefix) && !claimed.contains(*n))
                .map(str::to_string)
                .collect()
        };
        if names.is_empty() {
            return Err(WeightsError::Missing { name: format!("{prefix}*") });
        }
        let mut tensors = BTreeMap::new();
        for name in names {
            if self.complex.contains(&name) {
                return Err(WeightsError::ComplexPair {
                    name,
                    detail: "complex RoPE table inside an untyped group".to_string(),
                });
            }
            self.claim(&name)?;
            let w = self.read(&name)?;
            tensors.insert(name, w);
        }
        Ok(TensorGroup { prefix: prefix.to_string(), tensors })
    }
}

/// One expected tensor, as the typed structs request it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expected {
    Tensor { name: String, shape: Vec<usize> },
    /// Complex upstream; the file holds f32 `shape ++ [2]`.
    ComplexPairs { name: String, shape: Vec<usize> },
    Group { prefix: String },
}

/// A [`Source`] that loads nothing and records what was asked for.
#[derive(Default)]
pub struct SchemaRecorder {
    pub expected: RefCell<Vec<Expected>>,
}

impl Source for SchemaRecorder {
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError> {
        self.expected.borrow_mut().push(Expected::Tensor { name: name.to_string(), shape: shape.to_vec() });
        Ok(Weight { name: name.to_string(), shape: shape.to_vec(), data: Vec::new() })
    }

    fn complex_pairs(&self, name: &str, shape: &[usize]) -> Result<Weight, WeightsError> {
        self.expected
            .borrow_mut()
            .push(Expected::ComplexPairs { name: name.to_string(), shape: shape.to_vec() });
        let mut full = shape.to_vec();
        full.push(2);
        Ok(Weight { name: name.to_string(), shape: full, data: Vec::new() })
    }

    fn group(&self, prefix: &str) -> Result<TensorGroup, WeightsError> {
        self.expected.borrow_mut().push(Expected::Group { prefix: prefix.to_string() });
        Ok(TensorGroup { prefix: prefix.to_string(), tensors: BTreeMap::new() })
    }
}

/// Every tensor (and untyped group) the loader asks for, in load order.
pub fn schema() -> Vec<Expected> {
    let rec = SchemaRecorder::default();
    Sam31Weights::load(&rec).expect("the schema recorder never fails");
    rec.expected.into_inner()
}

/// Parse the converter's metadata value: a JSON array of strings.
fn parse_name_list(s: &str) -> Result<BTreeSet<String>, String> {
    let v = ojas_io::parse_json(s).map_err(|e| format!("metadata is not JSON: {}", e.detail()))?;
    let ojas_io::JsonValue::Array(items) = v else {
        return Err("metadata is not a JSON array".to_string());
    };
    items
        .into_iter()
        .map(|it| match it {
            ojas_io::JsonValue::String(s) => Ok(s),
            other => Err(format!("metadata entry is not a string: {other:?}")),
        })
        .collect()
}
