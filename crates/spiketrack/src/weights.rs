//! The checkpoint a command runs, optionally with a LoRA adapter folded into it at load time
//! (`--lora FILE`, written offline by `tools/reference/train_lora.py`; see `st_weights::lora`).
//!
//! The adapter is opened and validated before the checkpoint is read, so a malformed adapter
//! fails in a second. Its merge checks every target against the checkpoint before changing any
//! tensor. The adapter's recorded `base_sha256` is reported, not checked: the binary does not
//! hash the multi-gigabyte checkpoint (GAP-LORA-BASE-HASH-UNCHECKED). A wrong base of another
//! shape is refused by the merge; one of the same shapes is not.

use std::path::{Path, PathBuf};

use st_weights::lora::LoraAdapter;
use st_weights::{Checkpoint, Sam31Weights};

use crate::cli::{existing_path, CliError};
use crate::json;

/// What `--lora` folded in, for the command's output files.
#[derive(Clone, Debug, PartialEq)]
pub struct Applied {
    pub path: PathBuf,
    /// Adapted (tensor, row range) targets.
    pub targets: usize,
    /// Distinct checkpoint tensors changed.
    pub tensors: usize,
    pub base_sha256: Option<String>,
}

/// `null` without an adapter, else `{"adapter": ..., "targets": ..., "tensors": ..., "base_sha256": ...}`.
pub fn lora_json(lora: Option<&Applied>) -> String {
    match lora {
        None => "null".to_string(),
        Some(a) => format!(
            "{{\"adapter\": {}, \"targets\": {}, \"tensors\": {}, \"base_sha256\": {}}}",
            json::string(&a.path.to_string_lossy()),
            a.targets,
            a.tensors,
            a.base_sha256.as_deref().map(json::string).unwrap_or_else(|| "null".to_string())
        ),
    }
}

/// The `--lora` path, which must exist when given. There is deliberately no environment
/// default: an adapter is applied only when the command line names it.
pub fn lora_path(arg: Option<&str>) -> Result<Option<PathBuf>, CliError> {
    arg.map(|a| existing_path(Some(a), "", "lora adapter", "lora")).transpose()
}

/// Load the checkpoint at `weights` and, given `lora`, fold that adapter into it.
pub fn load(weights: &Path, lora: Option<&Path>) -> Result<(Sam31Weights, Option<Applied>), CliError> {
    let adapter = lora
        .map(|p| LoraAdapter::open(p).map_err(|e| CliError::failed(format!("lora adapter {}: {e}", p.display()))))
        .transpose()?;
    let ck = Checkpoint::open(weights).map_err(|e| CliError::failed(format!("checkpoint {}: {e}", weights.display())))?;
    let mut w = Sam31Weights::load(&ck).map_err(|e| CliError::failed(format!("checkpoint {}: {e}", weights.display())))?;
    let applied = match (lora, adapter) {
        (Some(path), Some(adapter)) => {
            let report = adapter.apply(&mut w).map_err(|e| CliError::failed(format!("lora adapter {} on {}: {e}", path.display(), weights.display())))?;
            Some(Applied { path: path.to_path_buf(), targets: report.targets, tensors: report.tensors.len(), base_sha256: adapter.base_sha256 })
        }
        _ => None,
    };
    Ok((w, applied))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_io::parse_json;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("spiketrack-weights-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_malformed_adapter_fails_before_the_checkpoint_is_read() {
        let dir = scratch("malformed");
        let adapter = dir.join("adapter.safetensors");
        std::fs::write(&adapter, b"not a safetensors file").unwrap();
        // The checkpoint path does not exist: the error must still be the adapter's.
        let err = load(&dir.join("no-such-checkpoint.safetensors"), Some(&adapter)).unwrap_err().to_string();
        assert!(err.starts_with("lora adapter"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_adapter_path_is_refused() {
        let err = lora_path(Some("/definitely/not/here.safetensors")).unwrap_err().to_string();
        assert!(err.contains("lora adapter") && err.contains("does not exist"), "{err}");
        assert_eq!(lora_path(None).unwrap(), None);
    }

    #[test]
    fn lora_json_is_null_or_a_valid_object() {
        assert_eq!(lora_json(None), "null");
        let a = Applied { path: PathBuf::from("out/a \"b\".safetensors"), targets: 30, tensors: 12, base_sha256: Some("ab".into()) };
        let v = parse_json(&lora_json(Some(&a))).expect("valid JSON");
        assert_eq!(v.get_str("adapter").unwrap(), "out/a \"b\".safetensors");
        assert_eq!(v.get_u64("targets").unwrap(), 30);
        assert_eq!(v.get_u64("tensors").unwrap(), 12);
        assert_eq!(v.get_str("base_sha256").unwrap(), "ab");
        let none = Applied { base_sha256: None, ..a };
        assert!(parse_json(&lora_json(Some(&none))).is_ok());
    }
}
