//! The checkpoint a command runs, optionally with a LoRA adapter folded into it (`--lora FILE`,
//! see `st_weights::lora`) and/or a declarative surgical weight-space edit manifest applied on top
//! (`--weight-edit FILE`, see `st_weights::edit`).
//!
//! Both the adapter and the weight-edit manifest are opened and validated before the checkpoint is
//! read, so a malformed manifest or adapter fails in milliseconds. Both apply atomically (checking
//! every target and finiteness before mutating any tensor). When both `--lora` and `--weight-edit`
//! are given, `--lora` folds into the base weights first and `--weight-edit` applies on top.

use std::path::{Path, PathBuf};

use st_weights::edit::WeightEditPlan;
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

/// What `--weight-edit` changed, for the command's output files.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedEdit {
    pub path: PathBuf,
    pub description: Option<String>,
    pub edits: usize,
    pub tensors: usize,
    pub values_changed: usize,
    pub total_values: usize,
}

impl AppliedEdit {
    pub fn fraction_changed(&self) -> f64 {
        if self.total_values == 0 {
            0.0
        } else {
            self.values_changed as f64 / self.total_values as f64
        }
    }
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
            a.base_sha256
                .as_deref()
                .map(json::string)
                .unwrap_or_else(|| "null".to_string())
        ),
    }
}

/// `null` without a weight edit, else an object recording the manifest, counts, and fraction changed.
pub fn edit_json(edit: Option<&AppliedEdit>) -> Result<String, String> {
    match edit {
        None => Ok("null".to_string()),
        Some(e) => Ok(format!(
            "{{\"manifest\": {}, \"description\": {}, \"edits\": {}, \"tensors\": {}, \"values_changed\": {}, \"total_values\": {}, \"fraction_changed\": {}}}",
            json::string(&e.path.to_string_lossy()),
            e.description
                .as_deref()
                .map(json::string)
                .unwrap_or_else(|| "null".to_string()),
            e.edits,
            e.tensors,
            e.values_changed,
            e.total_values,
            json::number(e.fraction_changed(), "fraction_changed")?,
        )),
    }
}

/// The `--lora` path, which must exist when given. There is deliberately no environment
/// default: an adapter is applied only when the command line names it.
pub fn lora_path(arg: Option<&str>) -> Result<Option<PathBuf>, CliError> {
    arg.map(|a| existing_path(Some(a), "", "lora adapter", "lora"))
        .transpose()
}

/// The `--weight-edit` path, which must exist when given (no environment default).
pub fn edit_path(arg: Option<&str>) -> Result<Option<PathBuf>, CliError> {
    arg.map(|a| existing_path(Some(a), "", "weight edit manifest", "weight-edit"))
        .transpose()
}

/// Open and validate optional `--lora` and `--weight-edit` inputs before touching the checkpoint.
pub fn open_modifiers(
    lora: Option<&Path>,
    weight_edit: Option<&Path>,
) -> Result<(Option<LoraAdapter>, Option<WeightEditPlan>), CliError> {
    let adapter = lora
        .map(|p| {
            LoraAdapter::open(p)
                .map_err(|e| CliError::failed(format!("lora adapter {}: {e}", p.display())))
        })
        .transpose()?;
    let plan = weight_edit
        .map(|p| {
            WeightEditPlan::open(p)
                .map_err(|e| CliError::failed(format!("weight edit {}: {e}", p.display())))
        })
        .transpose()?;
    Ok((adapter, plan))
}

/// Apply validated modifiers (`--lora` then `--weight-edit`) to an already-loaded checkpoint.
pub fn apply_modifiers(
    w: &mut Sam31Weights,
    weights_path: &Path,
    lora: Option<(&Path, LoraAdapter)>,
    weight_edit: Option<(&Path, WeightEditPlan)>,
) -> Result<(Option<Applied>, Option<AppliedEdit>), CliError> {
    let applied_lora = match lora {
        Some((path, adapter)) => {
            let report = adapter.apply(w).map_err(|e| {
                CliError::failed(format!(
                    "lora adapter {} on {}: {e}",
                    path.display(),
                    weights_path.display()
                ))
            })?;
            Some(Applied {
                path: path.to_path_buf(),
                targets: report.targets,
                tensors: report.tensors.len(),
                base_sha256: adapter.base_sha256,
            })
        }
        None => None,
    };
    let applied_edit = match weight_edit {
        Some((path, plan)) => {
            let report = plan.apply(w).map_err(|e| {
                CliError::failed(format!(
                    "weight edit {} on {}: {e}",
                    path.display(),
                    weights_path.display()
                ))
            })?;
            Some(AppliedEdit {
                path: path.to_path_buf(),
                description: report.description,
                edits: report.edits,
                tensors: report.tensors.len(),
                values_changed: report.values_changed,
                total_values: report.total_values,
            })
        }
        None => None,
    };
    Ok((applied_lora, applied_edit))
}

/// Read and schema-verify the raw checkpoint at `weights` without applying modifiers.
pub fn load_raw(weights: &Path) -> Result<Sam31Weights, CliError> {
    let ck = Checkpoint::open(weights)
        .map_err(|e| CliError::failed(format!("checkpoint {}: {e}", weights.display())))?;
    let w = Sam31Weights::load(&ck)
        .map_err(|e| CliError::failed(format!("checkpoint {}: {e}", weights.display())))?;
    ck.finish()
        .map_err(|e| CliError::failed(format!("checkpoint {}: {e}", weights.display())))?;
    Ok(w)
}

/// Load `weights` and apply already-opened `lora` and `weight_edit` modifiers once.
pub fn load_opened(
    weights: &Path,
    lora: Option<(&Path, LoraAdapter)>,
    weight_edit: Option<(&Path, WeightEditPlan)>,
) -> Result<(Sam31Weights, Option<Applied>, Option<AppliedEdit>), CliError> {
    let mut w = load_raw(weights)?;
    let (applied_lora, applied_edit) = apply_modifiers(&mut w, weights, lora, weight_edit)?;
    Ok((w, applied_lora, applied_edit))
}

/// Load the checkpoint at `weights`, fold `lora` (if any), and apply `weight_edit` (if any).
pub fn load(
    weights: &Path,
    lora: Option<&Path>,
    weight_edit: Option<&Path>,
) -> Result<(Sam31Weights, Option<Applied>, Option<AppliedEdit>), CliError> {
    let (adapter, plan) = open_modifiers(lora, weight_edit)?;
    load_opened(weights, lora.zip(adapter), weight_edit.zip(plan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_io::parse_json;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("spiketrack-weights-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_malformed_adapter_fails_before_the_checkpoint_is_read() {
        let dir = scratch("malformed");
        let adapter = dir.join("adapter.safetensors");
        std::fs::write(&adapter, b"not a safetensors file").unwrap();
        // The checkpoint path does not exist: the error must still be the adapter's.
        let err = load(
            &dir.join("no-such-checkpoint.safetensors"),
            Some(&adapter),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(err.starts_with("lora adapter"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_malformed_weight_edit_fails_before_the_checkpoint_is_read() {
        let dir = scratch("malformed-edit");
        let manifest = dir.join("edit.json");
        std::fs::write(&manifest, br#"{"format": "wrong-v1", "edits": []}"#).unwrap();
        let err = load(
            &dir.join("no-such-checkpoint.safetensors"),
            None,
            Some(&manifest),
        )
        .unwrap_err()
        .to_string();
        assert!(err.starts_with("weight edit"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_adapter_or_edit_path_is_refused() {
        let err = lora_path(Some("/definitely/not/here.safetensors"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("lora adapter") && err.contains("does not exist"),
            "{err}"
        );
        assert_eq!(lora_path(None).unwrap(), None);

        let edit_err = edit_path(Some("/definitely/not/here.json"))
            .unwrap_err()
            .to_string();
        assert!(
            edit_err.contains("weight edit manifest") && edit_err.contains("does not exist"),
            "{edit_err}"
        );
        assert_eq!(edit_path(None).unwrap(), None);
    }

    #[test]
    fn lora_and_edit_json_are_null_or_valid_objects() {
        assert_eq!(lora_json(None), "null");
        let a = Applied {
            path: PathBuf::from("out/a \"b\".safetensors"),
            targets: 30,
            tensors: 12,
            base_sha256: Some("ab".into()),
        };
        let v = parse_json(&lora_json(Some(&a))).expect("valid JSON");
        assert_eq!(v.get_str("adapter").unwrap(), "out/a \"b\".safetensors");
        assert_eq!(v.get_u64("targets").unwrap(), 30);
        assert_eq!(v.get_u64("tensors").unwrap(), 12);
        assert_eq!(v.get_str("base_sha256").unwrap(), "ab");
        let none = Applied {
            base_sha256: None,
            ..a
        };
        assert!(parse_json(&lora_json(Some(&none))).is_ok());

        assert_eq!(edit_json(None).unwrap(), "null");
        let e = AppliedEdit {
            path: PathBuf::from("bench/edit \"1\".json"),
            description: Some("graft sperm<-cell".into()),
            edits: 2,
            tensors: 2,
            values_changed: 1026,
            total_values: 1_000_000,
        };
        let ev = parse_json(&edit_json(Some(&e)).unwrap()).expect("valid edit JSON");
        assert_eq!(ev.get_str("manifest").unwrap(), "bench/edit \"1\".json");
        assert_eq!(ev.get_str("description").unwrap(), "graft sperm<-cell");
        assert_eq!(ev.get_u64("edits").unwrap(), 2);
        assert_eq!(ev.get_u64("values_changed").unwrap(), 1026);
        assert!((ev.get_f64("fraction_changed").unwrap() - 0.001026).abs() < 1e-12);
    }
}
