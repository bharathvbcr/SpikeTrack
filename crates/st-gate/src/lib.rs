//! Training-free event gating, pixel side: which attention windows of the SAM
//! 3.1 backbone saw a changed input since they were last computed.
//!
//! - [`bits`]: the bitset types. The only module that knows the storage; the
//!   owner's sparsl replaces it later.
//! - [`change_map`]: per-patch and per-window change bits from two frames, in
//!   integer math (threshold rules in its module docs).
//! - [`policy`]: the per-frame [`Decision`] (full, reuse, or a window subset),
//!   refresh rules and the reference-frame bookkeeping.
//!
//! This crate has no dependencies and no notion of tensors. `st-model` depends
//! on it (for [`WindowSet`] and [`GatePolicy`]) and its gated trunk executes the
//! decisions, so there is no cycle: the pixel crate is the leaf.

pub mod bits;
pub mod change_map;
pub mod policy;

pub use bits::{BitSet, WindowSet};
pub use change_map::{change_map, ChangeMap, ChangeMapper, Frame, GateError, Geometry};
pub use policy::{Decision, FullReason, Gate, GateConfig, GatePolicy};
