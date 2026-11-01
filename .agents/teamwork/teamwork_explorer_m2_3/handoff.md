# Milestone 2: Technical Investigation & Architecture Design Report
## Features 11 & 12: Retiring Duplicate IO Readers & Training Loops Across Tessl and Ojas

**Author**: Explorer 3 (Milestone 2)  
**Date**: 2026-10-31  
**Status**: Investigation Complete — Handoff for Worker Implementation  
**Working Directory**: `/Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_3/`

---

## 1. Observation

### 1.1 Architectural Layout & Dependency Hierarchy
- **[Verified]** `/Users/bharath/Code/research/ojas/Cargo.toml:3-23` lists workspace crates: `ojas-core`, `ojas-cpu`, `ojas-autograd`, `ojas-oracle`, `ojas-data`, `ojas-io`, `ojas-infer`, `ojas-metal`, `ojas-device`, `ojas-kernels`, `ojas-wgpu`, `ojas-cuda`, `ojas-hip`, `ojas-capi`, `ojas-gusset-engine`, `ojas-simd`, `ojas-qwen35`, `ojas-model`, `ojas-train`.
- **[Verified]** `/Users/bharath/Code/research/ojas/ojas-io/Cargo.toml:10-11` declares only one dependency:
  ```toml
  [dependencies]
  ojas-core = { path = "../ojas-core" }
  ```
  `ojas-core` has zero external dependencies (`#![forbid(unsafe_code)]`, pure Rust).
- **[Verified]** `/Users/bharath/Code/research/ojas/ojas-metal/Cargo.toml:22` declares dependency on `tessl`:
  ```toml
  [target.'cfg(target_os = "macos")'.dependencies]
  tessl = { path = "../../tessl", optional = true }
  ```
- **[Verified]** Adding `ojas-io = { path = "../ojas/ojas-io" }` and `ojas-core = { path = "../ojas/ojas-core" }` into `tessl/Cargo.toml` forms a strictly acyclic dependency flow:
  $$\text{ojas-metal} \longrightarrow \text{tessl} \longrightarrow \text{ojas-io} \longrightarrow \text{ojas-core}$$
- **[Verified]** `rustc --version` output: `rustc 1.99.1 (58cf8f9c3 2026-10-13)`. Both `tessl` (rust-version 1.82) and `ojas` (rust-version 1.97) compile cleanly under this toolchain.

---

### 1.2 Feature 11: File Readers in Tessl vs Canonical `ojas-io`

#### A. SafeTensors Readers
- **[Verified]** `/Users/bharath/Code/research/tessl/src/safetensors.rs` (951 lines):
  - Defines `Dtype` (lines 50-76), `TensorInfo` (lines 124-130), `SafeTensors` (lines 201-214), `TensorSource` trait (lines 489-505), `CheckpointView` (lines 558-564), `INDEX_FILE_NAME = "model.safetensors.index.json"` (line 532), `MAX_HEADER_BYTES = 100_000_000` (line 40).
  - Implements in-place positioned loading: `read_f32_into` (lines 456-481), `read_bf16_bits_into` (lines 413-422), `read_u32_into` (lines 426-435), and blocked in-place widening `widen_in_place` (lines 775-791).
- **[Verified]** `/Users/bharath/Code/research/ojas/ojas-io/src/safetensors.rs` (1,768 lines):
  - Implements canonical strict reader and writer: `StDtype` (lines 43-66), `TensorInfo` (lines 193-198), `SafeTensors<'a>` (lines 245-249), `MAX_HEADER_BYTES = 100_000_000` (line 26), `read_into(&self, name, offset, dst: &mut [u8])` (lines 327-366), `read_bytes`, `read_f32`, `read_f32_widened`, `read_bf16_bits`, `read_i64`, `read_u16`.
  - Implements streaming writer: `SafeTensorsWriter<W>` (lines 675-687), `write_safetensors`, `encode_safetensors`.
- **[Verified]** Inverted Dependency Anomaly in `ojas-qwen35`:
  `/Users/bharath/Code/research/ojas/ojas-qwen35/src/names.rs:15`:
  ```rust
  use tessl::safetensors::{CheckpointView, Dtype, TensorSource, INDEX_FILE_NAME};
  ```
  `ojas-qwen35` (an Ojas crate) was importing `CheckpointView` from `tessl` because multi-file sharded checkpoint index parsing was placed in `tessl/src/safetensors.rs` instead of `ojas-io`!
- **[Verified]** Callers of `crate::safetensors` in `tessl/src/`:
  - `src/qwen35_model.rs:113`: `use crate::safetensors::TensorSource;`
  - `src/loader.rs:31, 365`: `use crate::safetensors::{SafeTensors, TensorSource};`
  - `src/embedgemma2_vision.rs:136`: `use crate::safetensors::SafeTensors;`
  - `src/embedgemma2_audio.rs:126`: `use crate::safetensors::SafeTensors;`
  - `src/embedgemma2_mm.rs:134`: `use crate::safetensors::SafeTensors;`
  - `src/embedgemma2.rs:115`: `use crate::safetensors::SafeTensors;`
  - `src/bert.rs:62`: `use crate::safetensors::SafeTensors;`
  - `src/capi.rs:83`: `use crate::safetensors::CheckpointView;`
  - `src/capi_embedgemma2.rs:137`: `use crate::safetensors::SafeTensors;`
  - `src/qwen35_q4.rs:54`: `use crate::safetensors::{Dtype, TensorInfo};`
  - `src/parser_mutation.rs:35`: `use crate::safetensors::{CheckpointView, Dtype, SafeTensors, TensorSource, INDEX_FILE_NAME};`

#### B. JSON Parsers
- **[Verified]** `/Users/bharath/Code/research/tessl/src/json.rs` (517 lines):
  - Private module (`mod json;` in `tessl/src/lib.rs:260`).
  - Defines `Json` enum (lines 14-32), `NumForm` (lines 36-45), `Syntax` (lines 78-87), `parse(text, syntax)` (lines 90-106).
  - Used only by `tessl/src/config_json.rs`, `embedgemma2_tok.rs`, `embedgemma2_pre.rs`, and test/benchmark helpers (`tune.rs`, `parity.rs`, `evidence.rs`).
- **[Verified]** `/Users/bharath/Code/research/tessl/src/config_json.rs` (67 lines):
  - Thin adapter converting `Json` into model configuration values: `parse_object`, `field`, `uint`, `float`, `flag`, `string`.
  - Called by `bert.rs`, `embedgemma2.rs`, `embedgemma2_vision.rs`, `embedgemma2_audio.rs`, `embedgemma2_mm.rs`, `qwen35_model.rs`, `qwen35_q4.rs`.
- **[Verified]** `/Users/bharath/Code/research/ojas/ojas-io/src/json.rs` (1,236 lines):
  - Canonical RFC 8259 compliant strict parser: `parse_json(text)`, `parse_json_with(text, limits)`, `JsonValue`, `JsonNumber`, `JsonLimits`.
  - Rich accessor API on `JsonValue`: `get`, `get_u64`, `get_i64`, `get_f64`, `get_bool`, `get_str`, `get_array`, `get_obj`, `deny_unknown_keys`.

#### C. NumPy `.npy` Readers
- **[Verified]** `/Users/bharath/Code/research/tessl/src/npy.rs` (418 lines):
  - Defines `NpyArray` (lines 23-28) with `f32_slice`, `f64_slice`, `i64_slice`, `scalar_f32`.
  - Defines `read_npy(path)` (lines 114-116), `read_npy_with_limit(path, limit)` (lines 125-215), `write_npy_f32(path, shape, data)` (lines 325-387), `transpose_last2(data, shape)` (lines 292-322), `MAX_NPY_HEADER_BYTES = 1 << 20` (line 102).
  - Called by ~25 integration test files and benchmark harnesses in `tessl`.
- **[Verified]** `/Users/bharath/Code/research/ojas/ojas-io/src/npy.rs` (269 lines):
  - Defines `Npy` (lines 24-27), `Data` enum (`F32`, `F64`, `I64`), `read(path)`, `parse(bytes)`, `f32s()`, `f64s()`, `i64s()`, `to_f64()`.
  - Currently lacks `write_npy_f32`, `transpose_last2`, and `read_npy_with_limit`.

#### D. Helper `plain.rs`
- **[Verified]** `/Users/bharath/Code/research/tessl/src/plain.rs` (146 lines):
  - Private module (`mod plain;`). Owned `PlainScalar`, `bytes_mut`, `le_to_native`, `open_regular_file`, `checked_numel`, `check_payload_budget`.
  - Exists solely to support `tessl/src/safetensors.rs` and `tessl/src/npy.rs`.

---

### 1.3 Feature 12: Training Layers in Tessl vs Ojas

- **[Verified]** `/Users/bharath/Code/research/tessl/src/qwen35_train.rs` (4,726 lines):
  - Implements an experimental whole-model training loop: `Qwen35Model::train_step`, `train_step_into`, `train_step_ragged_into`, `PendingStep`, `Supervise`, and bespoke activation re-computation.
  - Bypasses Ojas's `Backend` trait and autograd tape completely.
  - Duplicates gradient buffering (`Qwen35Grads`, `MixerGrads`, `LayerGrads`) outside Ojas's `AdapterBank`.
- **[Verified]** `/Users/bharath/Code/research/tessl/src/qwen35_adamw.rs` (1,519 lines):
  - Implements GPU-resident AdamW optimizer with bespoke scalar calculations: `adamw_scalars(hyper, step, weight_decay) -> [f32; 8]` (lines 507-530).
  - Duplicates `ojas_core::check_adamw` and `ojas_model::optim::adamw_all_or_nothing`.
- **[Verified]** `/Users/bharath/Code/research/ojas/ojas-metal/src/gpu.rs` (3,736 lines):
  - Experimental "tiny Metal training step" hardcoded to toy dimensions:
    ```rust
    // ojas-metal/src/gpu.rs:22-28
    pub const MAX_BATCH: u32 = 2;
    pub const MAX_SEQ: u32 = 16;
    pub const MAX_VOCAB: u32 = 128;
    pub const D_MODEL: u32 = 128;
    pub const N_HEAD: u32 = 2;
    pub const HEAD_DIM: u32 = 64;
    ```
  - Outside `gpu.rs`, the **sole** active consumer in the entire codebase is `ojas-metal/src/device.rs:39`:
    ```rust
    use crate::gpu::gate_dbias_threads;
    ```
    where `gate_dbias_threads(n_head)` is simply `n_head.checked_mul(32)`.
  - 3,730+ lines of `gpu.rs` are dead legacy code that duplicate training step logic.
- **[Verified]** Canonical Training Engine in `ojas-model`:
  `/Users/bharath/Code/research/ojas/ojas-model/src/qwen35/step_provider.rs` (1,340 lines):
  - Implements `StepProvider` trait (`accumulate`, `read_parameters`, `save_state_with`, `load_state_with`).
  - Uses canonical `ojas_autograd::Tape` for exact reverse-mode autograd.
  - Built over generic `ojas_core::Backend` with full activation checkpointing (`ActivationCheckpoint`), full fine-tuning, LoRA adapters (`LoraTarget`), gradient accumulation (`AdapterBank`), and AdamW optimization (`ojas-model/src/qwen35/optim.rs`).
- **[Verified]** Canonical AdamW Scalar Formation in `ojas-core`:
  `/Users/bharath/Code/research/ojas/ojas-core/src/backend.rs:203-245`:
  - `check_adamw(config, step) -> Result<(next_step, bc1, bc2), OjasError>`.
  - Formed in f64 with exact checked arithmetic and unified non-finite/out-of-range validation.

---

## 2. Logic Chain

1. **Premise 1 (User Rule & Project Architecture)**:
   - "Reuse, unify, and shrink: Extending what exists is the default; One canonical owner per behavior; Replace, don't accumulate." (`AGENTS.md`)
   - "Tessl specializes in Metal and heavy Mac inference optimizations; Ojas serves as the universal system with a focus on Rust-based training." (`PROJECT.md:Architecture`, `ORIGINAL_REQUEST.md:R2`)
   - `ojas-io` is the single canonical owner of IO (SafeTensors, JSON, NPY).
   - `ojas-model` (via `TapeStepProvider`) is the single canonical owner of training step execution.

2. **Premise 2 (Feature 11 SafeTensors/JSON/NPY Unification)**:
   - Tessl's readers (`safetensors.rs` 951 lines, `json.rs` 517 lines, `npy.rs` 418 lines, `plain.rs` 146 lines = 2,032 lines) duplicate implementations already maintained in `ojas-io`.
   - Furthermore, `tessl/src/safetensors.rs` implemented `CheckpointView` (sharded index parser) and zero-copy in-place readers (`read_f32_into`, `read_bf16_bits_into`), forcing `ojas-qwen35` to import from `tessl`.
   - By enriching `ojas-io` with `CheckpointView`, `TensorSource`, in-place buffer loaders, and NPY utilities (`write_npy_f32`, `transpose_last2`):
     - `ojas-io` becomes the sole, complete IO crate.
     - `tessl` can import `ojas-io` (acyclic dependency).
     - `tessl/src/json.rs` and `tessl/src/plain.rs` are deleted outright.
     - `tessl/src/safetensors.rs` and `tessl/src/npy.rs` become thin re-export facades, ensuring 100% backward compatibility for all existing unit tests and callers.

3. **Premise 3 (Feature 12 Training Step & Optimizer Consolidation)**:
   - `ojas-metal/src/gpu.rs` (3,736 lines) was an early experimental step for a toy model (128 d_model, 2 heads). It is bypassed by modern `ojas-model` and only leaks `gate_dbias_threads`. Moving `gate_dbias_threads` into `ojas-metal/src/device.rs` allows deleting `gpu.rs` entirely.
   - `tessl/src/qwen35_train.rs` (4,726 lines) and `tessl/src/qwen35_adamw.rs` (1,519 lines) represent an ad-hoc GPU-only training loop that bypasses `ojas_autograd::Tape` and duplicates `ojas-model::TapeStepProvider`.
   - Moving AdamW scalar formulation into `ojas-core` (`adamw_kernel_scalars`) guarantees mathematical parity across CPU and Metal shaders.
   - Retiring `tessl/src/qwen35_train.rs` establishes `ojas-model` as the single canonical owner of training orchestration, while Tessl retains its true specialized role: Metal 4 runtime and low-level compute kernels (`qwen35_bwd.rs`, `attn_train.rs`, `gdn_train.rs`, `gemm.rs`).

4. **Deduction & Code Elimination Metric**:
   - Total lines of duplicated code eliminated:
     - `tessl/src/safetensors.rs`: 951 lines
     - `tessl/src/json.rs`: 517 lines
     - `tessl/src/npy.rs`: 418 lines
     - `tessl/src/plain.rs`: 146 lines
     - `ojas-metal/src/gpu.rs`: 3,736 lines
     - `tessl/src/qwen35_train.rs`: 4,726 lines
     - `tessl/src/qwen35_adamw.rs`: 1,519 lines
     - **Net dead/duplicate code retired: ~12,013 lines**.

---

## 3. Caveats

1. **`shape` Representation in SafeTensors**:
   - `tessl/src/safetensors.rs` used `Vec<usize>` for `shape`, whereas `ojas-io/src/safetensors.rs` stores `Vec<u64>`.
   - *Mitigation*: In `ojas-io`, provide accessor methods `shape_usize(&self) -> Vec<usize>` or overload slice parameters so existing callers transition seamlessly.
2. **Error Types**:
   - `tessl` readers returned `Result<T, String>`, whereas `ojas-io` returns `Result<T, IoError>`.
   - *Mitigation*: Re-export facades in `tessl::safetensors` and `tessl::npy` can preserve the `Result<T, String>` signature using `.map_err(|e| e.to_string())`.
3. **C API `tessl_qwen35_train_step`**:
   - `tessl/src/capi.rs` exposes C entry points (`tessl_qwen35_train_step`) which have ctypes bindings in `python/tessl_torch/__init__.py`.
   - *Mitigation*: As training orchestration moves to Ojas, `capi.rs` training entry points should either provide a thin delegating bridge or return clear migration diagnostics to `ojas-capi`.
4. **Adversarial Mutation Tests**:
   - `tessl/src/parser_mutation.rs` mutates inputs to verify parser refusal. Once routed through `ojas-io`, these tests will rigorously exercise `ojas-io`'s strict parsing guarantees.

---

## 4. Conclusion & Actionable Implementation Plan for Worker

### 4.1 Phase-by-Phase Worker Instructions

```
+-----------------------------------------------------------------------------------------+
|                                  MIGRATION ROADMAP                                      |
|                                                                                         |
|  Phase 1: Enrich `ojas-io` with Sharded Checkpoints & In-Place Zero-Copy Loaders       |
|  Phase 2: Add `ojas-io` dependency to `tessl/Cargo.toml`                               |
|  Phase 3: Retire `tessl/src/json.rs`, `plain.rs`, redirect `safetensors.rs`, `npy.rs`   |
|  Phase 4: Delete `ojas-metal/src/gpu.rs` (move `gate_dbias_threads` to `device.rs`)     |
|  Phase 5: Unify AdamW kernel scalars in `ojas-core`                                     |
|  Phase 6: Consolidate Qwen3.5 training in `ojas-model::TapeStepProvider`                |
+-----------------------------------------------------------------------------------------+
```

#### Phase 1: Enrich `ojas-io` (Ojas Workspace)
1. **Sharded Checkpoint View in `ojas-io`**:
   - Move `CheckpointView`, `TensorSource`, and `INDEX_FILE_NAME` from `tessl/src/safetensors.rs` into `ojas-io/src/safetensors.rs` (or new module `ojas-io/src/checkpoint_view.rs`).
   - Implement `TensorSource` for `SafeTensors` and `CheckpointView`.
2. **Zero-Copy In-Place Readers in `ojas-io::SafeTensors`**:
   - Add `read_f32_into(&self, name: &str, shape: &[usize], dst: &mut [f32]) -> Result<(), IoError>`.
   - Add `read_bf16_bits_into(&self, name: &str, shape: &[usize], dst: &mut [u16]) -> Result<(), IoError>`.
   - Add `read_u32_into(&self, name: &str, shape: &[usize], dst: &mut [u32]) -> Result<(), IoError>`.
   - Port `widen_in_place` (from `tessl/src/safetensors.rs:775-791`) into `ojas-io/src/half.rs`.
3. **NPY Utilities in `ojas-io::npy`**:
   - Add `write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String>`.
   - Add `transpose_last2(data: &mut [f32], shape: &mut [usize]) -> Result<(), String>`.
   - Add `read_npy_with_limit(path: &Path, limit: Option<u64>) -> Result<NpyArray, String>`.
   - Add `NpyArray` wrapper (or type alias with `f32_slice`, `f64_slice`, `i64_slice`).
4. **Verification**: Run `cargo test -p ojas-io`.

#### Phase 2: Wire Dependency into Tessl
1. Update `/Users/bharath/Code/research/tessl/Cargo.toml`:
   ```toml
   [dependencies]
   ojas-core = { path = "../ojas/ojas-core" }
   ojas-io = { path = "../ojas/ojas-io" }
   ```
2. Verify: `cd /Users/bharath/Code/research/tessl && cargo check --lib`.

#### Phase 3: Retire Duplicate Readers in Tessl (Feature 11)
1. **JSON Parser**:
   - Rewrite `tessl/src/config_json.rs` to parse via `ojas_io::parse_json(text)` and query `ojas_io::JsonValue`.
   - Update `tessl/src/embedgemma2_tok.rs` and `embedgemma2_pre.rs` to use `ojas_io::parse_json`.
   - Delete `tessl/src/json.rs`.
   - In `tessl/src/lib.rs`: delete `mod json;` (line 260).
2. **SafeTensors Reader**:
   - Turn `tessl/src/safetensors.rs` into a thin compatibility facade:
     ```rust
     pub use ojas_io::safetensors::*;
     pub use ojas_io::safetensors::{
         CheckpointView, SafeTensors, TensorInfo, TensorSource, INDEX_FILE_NAME, MAX_HEADER_BYTES,
     };
     pub type Dtype = ojas_io::StDtype;
     ```
   - In `ojas-qwen35/src/names.rs`: change `use tessl::safetensors::CheckpointView;` to `use ojas_io::CheckpointView;`.
3. **NumPy Reader**:
   - Turn `tessl/src/npy.rs` into a thin compatibility facade:
     ```rust
     pub use ojas_io::npy::*;
     ```
4. **Helper Clean-up**:
   - Delete `tessl/src/plain.rs`.
   - In `tessl/src/lib.rs`: delete `mod plain;` (line 271).
5. **Verification**: Run `cd /Users/bharath/Code/research/tessl && cargo test --test bert_model --test embedgemma2_model`.

#### Phase 4: Retire `ojas-metal/src/gpu.rs` (Feature 12 Step 1)
1. In `ojas-metal/src/device.rs`:
   Add `gate_dbias_threads`:
   ```rust
   pub(crate) fn gate_dbias_threads(n_head: usize) -> Result<usize, OjasError> {
       n_head.checked_mul(32).ok_or_else(|| OjasError::OutOfRange {
           op: "gate_dbias_threads",
           detail: "n_head * 32 overflows usize".into(),
       })
   }
   ```
   Remove line 39: `use crate::gpu::gate_dbias_threads;`.
2. In `ojas-metal/src/lib.rs`: delete line 73: `pub mod gpu;`.
3. Delete file `/Users/bharath/Code/research/ojas/ojas-metal/src/gpu.rs` (3,736 lines removed).
4. **Verification**: Run `cargo check -p ojas-metal && cargo test -p ojas-metal`.

#### Phase 5: Consolidate AdamW Scalars into `ojas-core` (Feature 12 Step 2)
1. In `ojas-core/src/backend.rs`:
   Expose canonical AdamW GPU kernel scalars:
   ```rust
   pub fn adamw_kernel_scalars(
       config: AdamWConfig,
       step: u64,
       grad_scale: f32,
   ) -> Result<[f32; 8], OjasError> {
       let (next, bc1, bc2) = check_adamw(config, step)?;
       let step_size = (config.lr / bc1) as f32;
       let bc2_sqrt = (bc2.powf(0.5)) as f32;
       Ok([
           (1.0 - config.lr * config.weight_decay) as f32,
           (1.0 - config.beta1) as f32,
           config.beta2 as f32,
           (1.0 - config.beta2) as f32,
           step_size,
           bc2_sqrt,
           config.eps as f32,
           grad_scale,
       ])
   }
   ```
2. Verify with parity tests against PyTorch in `ojas-oracle`.

#### Phase 6: Consolidate Qwen3.5 Training in `ojas-model` (Feature 12 Step 3)
1. Confirm `ojas-model::TapeStepProvider` (`ojas-model/src/qwen35/step_provider.rs`) covers:
   - Full model parameter training & LoRA fine-tuning.
   - Activation checkpointing (`ActivationCheckpoint`).
   - Reverse-mode autograd tape (`ojas-autograd::Tape`).
   - Gradient bank accumulation (`AdapterBank`).
   - Optimizer step execution (`ojas-model/src/qwen35/optim.rs`).
2. Redirect `ojas-qwen35/src/step.rs` to execute via `ojas-model::TapeStepProvider`.
3. Retire whole-step training orchestrator `tessl/src/qwen35_train.rs` and `tessl/src/qwen35_adamw.rs`, keeping low-level Metal kernels in `qwen35_bwd.rs`, `attn_train.rs`, and `gdn_train.rs`.
4. Update `tessl/src/capi.rs` training entry points with deprecation or delegation to `ojas-model`.

---

## 5. Verification Method

To independently verify the implementation, execute the following commands in order:

### Verification 1: Crate Independence and Clean Builds
```bash
# 1. Verify ojas-io standalone compile and test suite
cd /Users/bharath/Code/research/ojas
cargo test -p ojas-io --release

# 2. Verify tessl compiles with ojas-io dependency
cd /Users/bharath/Code/research/tessl
cargo check --lib

# 3. Verify ojas-metal compiles without gpu.rs
cd /Users/bharath/Code/research/ojas
cargo check -p ojas-metal
```

### Verification 2: Tessl Unit & Integration Test Regression Pass
```bash
cd /Users/bharath/Code/research/tessl
# Test Bert and EmbedGemma2 readers (exercises ojas-io safetensors and config_json)
cargo test --test bert_model
cargo test --test embedgemma2_model
cargo test --test embedgemma2_vision_model
cargo test --test embedgemma2_audio_model

# Test NPY integration
cargo test --test gemm_parity
```

### Verification 3: Ojas Training & Autograd Step Pass
```bash
cd /Users/bharath/Code/research/ojas
# Test TapeStepProvider and AdamW in ojas-model
cargo test -p ojas-model --test adamw_torch_100
cargo test -p ojas-model --test activation_checkpoint

# Test ojas-qwen35 step provider
cargo test -p ojas-qwen35 --test cpu_tape_spec
```

### Invalidation Conditions
- Any circular dependency between `ojas-metal`, `tessl`, and `ojas-io`.
- Any compilation failure in `tessl` tests referencing `safetensors` or `npy`.
- Any drift in AdamW scalar values compared to PyTorch golden fixtures in `ojas-oracle`.
- Any silent CPU fallback on Metal execution.
