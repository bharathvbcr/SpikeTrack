//! Per-frame gating decisions: which windows of the backbone to recompute.
//!
//! [`Gate::decide`] takes the model-input frame, compares it with a
//! *reference* (below), and answers one of three things:
//!
//! - [`Decision::Full`]: run the whole backbone and refresh every cache.
//! - [`Decision::Reuse`]: reuse the previous trunk output untouched.
//! - [`Decision::Windows`]: recompute only these attention windows (a proper,
//!   non-empty subset), carry the rest from the cache.
//!
//! # Policies, simplest first
//!
//! 1. [`GatePolicy::Frame`]: skip the whole backbone when nothing changed
//!    (`Reuse`), otherwise recompute everything. With `tau = 0` and
//!    `reuse_patches = 0` this is **exact**: the model input is bit-identical,
//!    so the previous trunk output is the output.
//! 2. [`GatePolicy::Windows`]: recompute only windows whose input changed.
//!    What is exact and what is not depends on `approximate_after_global`
//!    (the backbone, `st_model::Backbone::trunk_gated`, enforces it):
//!    - `false`: blocks of the first windowed run (before the first global
//!      block, blocks 0-6 for SAM 3.1) run only on changed windows; an
//!      unchanged window's input is bit-identical there, so its cached output
//!      is exactly what a recompute would give. The first global block mixes
//!      all windows, so any change reaches every token; every block from there
//!      on runs on all windows. At `tau = 0` the whole trunk is bit-identical
//!      to the ungated one. The saving is capped at 7 of 32 blocks times the
//!      unchanged fraction: at most about 22% of the blocks, less for the
//!      global blocks' larger cost share. **This variant cannot reach a 1.5x
//!      speedup on its own.**
//!    - `true`: also in the later windowed runs, only changed windows are
//!      recomputed and unchanged windows carry the output cached from the last
//!      time they were recomputed. Their true input changed (it passed through
//!      a global block that saw the changed windows), so this is an
//!      approximation; its error is bounded only by how stale the carried
//!      value can be, which `refresh_every` limits.
//!
//!    In both variants `tau > 0` makes the "unchanged" verdict itself
//!    approximate (a sub-threshold change is treated as no change); only
//!    `tau = 0` is exact. See `change_map` for the exact threshold rule.
//! 3. Skipping multiplex-slot re-decoding is not part of this crate.
//!
//! # The reference
//!
//! Changes are measured against the frame as it was at the last recompute of
//! each window, not against the previous frame. A window that is skipped keeps
//! its old pixels in the reference, so slow drift below `tau` accumulates until
//! it crosses it instead of being skipped forever. A `Full` decision replaces
//! the whole reference; `Windows` replaces only the recomputed windows;
//! `Reuse` leaves it alone.
//!
//! # Forced refreshes
//!
//! - `refresh_every = N` (0 = never): every `N`-th frame since the last `Full`
//!   is `Full`, so no cached value is more than `N - 1` frames old. This is the
//!   staleness bound for the approximate variant, and the guard against
//!   `reuse_patches` drift.
//! - `full_above_percent = P` (100 = never): when more than `P` percent of the
//!   windows changed, recompute everything: the gather/scatter overhead is not
//!   worth a handful of skipped windows, and for the approximate variant it
//!   resets all staleness.
//! - A frame with every window changed is `Full` (`AllChanged`), never a full
//!   `Windows` set.
//!
//! Everything here is integer math over the frame bytes and the configuration:
//! the same video and configuration give the same decisions, bit for bit.

use crate::bits::WindowSet;
use crate::change_map::{ChangeMap, ChangeMapper, Frame, GateError, Geometry};

/// What the backbone may skip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatePolicy {
    /// No gating: every frame is `Full`, and no frame is ever compared.
    Off,
    /// Policy 1: reuse the trunk output when nothing changed, else recompute all.
    Frame,
    /// Policy 2: recompute only changed windows. See the module docs for what
    /// `approximate_after_global` gives up.
    Windows { approximate_after_global: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateConfig {
    pub policy: GatePolicy,
    /// Change threshold, mean absolute difference per patch on the 0..=255
    /// sample scale; a patch is changed above it. `0` is exact.
    pub tau: u32,
    /// Reuse the previous trunk output when at most this many patches changed
    /// (`0`: only when none did). Anything above `0` is approximate.
    pub reuse_patches: u32,
    /// Every this many frames since the last `Full` is a `Full`; `0` never.
    pub refresh_every: u32,
    /// `Full` when more than this percent of the windows changed; `100` never.
    pub full_above_percent: u8,
}

impl GateConfig {
    /// `policy` at threshold `tau`, with no forced refresh and no changed-window cap.
    pub fn new(policy: GatePolicy, tau: u32) -> Self {
        Self { policy, tau, reuse_patches: 0, refresh_every: 0, full_above_percent: 100 }
    }

    /// Policy 1 or 2 with nothing approximate in it: `tau = 0`, window carrying
    /// only in the first windowed run. The result equals the ungated trunk
    /// bit for bit (on a backend whose arithmetic does not depend on batch size).
    pub fn exact(windows: bool) -> Self {
        Self::new(if windows { GatePolicy::Windows { approximate_after_global: false } } else { GatePolicy::Frame }, 0)
    }

    fn validate(&self) -> Result<(), GateError> {
        if self.full_above_percent > 100 {
            return Err(GateError::Config(format!("full_above_percent {} is over 100", self.full_above_percent)));
        }
        Ok(())
    }
}

/// Why a frame is recomputed in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullReason {
    /// [`GatePolicy::Off`].
    Disabled,
    /// No reference yet: the first frame, or after [`Gate::reset`].
    First,
    /// `refresh_every` frames since the last `Full`.
    Refresh,
    /// [`GatePolicy::Frame`] and something changed.
    Changed,
    /// More than `full_above_percent` of the windows changed.
    TooMuchChanged,
    /// Every window changed.
    AllChanged,
}

/// What to do with one frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Full(FullReason),
    Reuse,
    /// A non-empty, non-full set.
    Windows(WindowSet),
}

impl Decision {
    /// The windows to recompute, in the form the backbone takes: everything
    /// for `Full`, nothing for `Reuse`.
    pub fn recompute_set(&self, side: usize) -> WindowSet {
        match self {
            Self::Full(_) => WindowSet::full(side),
            Self::Reuse => WindowSet::new(side),
            Self::Windows(s) => s.clone(),
        }
    }
}

/// The per-stream gate: reference frame, change map and counters.
pub struct Gate {
    cfg: GateConfig,
    geo: Geometry,
    mapper: ChangeMapper,
    map: ChangeMap,
    reference: Vec<u8>,
    channels: usize,
    has_reference: bool,
    since_full: u32,
}

impl Gate {
    pub fn new(cfg: GateConfig, geo: Geometry) -> Result<Self, GateError> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            geo,
            mapper: ChangeMapper::new(geo)?,
            map: ChangeMap::new(geo),
            reference: Vec::new(),
            channels: 0,
            has_reference: false,
            since_full: 0,
        })
    }

    pub fn config(&self) -> &GateConfig {
        &self.cfg
    }

    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    /// The change map of the last [`Gate::decide`] that compared frames
    /// (empty after a decision that did not).
    pub fn last_map(&self) -> &ChangeMap {
        &self.map
    }

    /// Forget the reference: the next frame is `Full(First)`. Call it when the
    /// backbone call that executed a decision failed, or the stream restarts.
    pub fn reset(&mut self) {
        self.has_reference = false;
        self.since_full = 0;
        self.map.clear();
    }

    /// Decides the frame and updates the reference as if the decision were
    /// executed. If executing it fails, call [`Gate::reset`].
    pub fn decide(&mut self, frame: &Frame<'_>) -> Result<Decision, GateError> {
        frame.check(self.geo)?;
        self.map.clear();
        if self.cfg.policy == GatePolicy::Off {
            return Ok(Decision::Full(FullReason::Disabled));
        }
        if !self.has_reference {
            return Ok(self.full(frame, FullReason::First));
        }
        let reference = Frame::new(&self.reference, frame.width(), frame.height(), self.channels)?;
        self.mapper.compute(&reference, frame, self.cfg.tau, &mut self.map)?;

        if self.cfg.refresh_every > 0 && self.since_full + 1 >= self.cfg.refresh_every {
            return Ok(self.full(frame, FullReason::Refresh));
        }
        if self.map.patches_changed() <= self.cfg.reuse_patches as usize {
            self.since_full += 1;
            return Ok(Decision::Reuse);
        }
        let GatePolicy::Windows { .. } = self.cfg.policy else {
            return Ok(self.full(frame, FullReason::Changed));
        };
        let (changed, total) = (self.map.windows_changed(), self.map.windows().windows());
        if changed == total {
            return Ok(self.full(frame, FullReason::AllChanged));
        }
        if changed * 100 > usize::from(self.cfg.full_above_percent) * total {
            return Ok(self.full(frame, FullReason::TooMuchChanged));
        }
        let set = self.map.windows().clone();
        self.copy_windows(frame, &set);
        self.since_full += 1;
        Ok(Decision::Windows(set))
    }

    fn full(&mut self, frame: &Frame<'_>, why: FullReason) -> Decision {
        self.reference.clear();
        self.reference.extend_from_slice(frame.data());
        self.channels = frame.channels();
        self.has_reference = true;
        self.since_full = 0;
        Decision::Full(why)
    }

    /// Replaces the reference's pixels under `set` with the frame's.
    fn copy_windows(&mut self, frame: &Frame<'_>, set: &WindowSet) {
        let geo = self.geo;
        let ch = self.channels;
        let side_px = geo.window * geo.patch;
        let row_bytes = geo.pixels() * ch;
        let run = side_px * ch;
        for w in set.iter() {
            let (wy, wx) = (w / geo.side(), w % geo.side());
            for y in wy * side_px..(wy + 1) * side_px {
                let at = y * row_bytes + wx * run;
                self.reference[at..at + run].copy_from_slice(&frame.data()[at..at + run]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: Geometry = Geometry { grid: 6, patch: 2, window: 2 };
    const PX: usize = 12;

    fn img(v: u8) -> Vec<u8> {
        vec![v; PX * PX * 3]
    }

    fn frame(d: &[u8]) -> Frame<'_> {
        Frame::new(d, PX, PX, 3).unwrap()
    }

    /// Adds `delta` to every sample of window `w` (windows are 4x4 pixels here).
    fn bump(d: &mut [u8], w: usize, delta: u8) {
        let (wy, wx) = (w / 3, w % 3);
        for y in wy * 4..wy * 4 + 4 {
            for x in wx * 4..wx * 4 + 4 {
                for c in 0..3 {
                    let i = (y * PX + x) * 3 + c;
                    d[i] = d[i].wrapping_add(delta);
                }
            }
        }
    }

    fn gate(policy: GatePolicy, tau: u32) -> Gate {
        Gate::new(GateConfig::new(policy, tau), TINY).unwrap()
    }

    const WIN: GatePolicy = GatePolicy::Windows { approximate_after_global: false };

    #[test]
    fn first_frame_is_full_then_an_identical_frame_is_reused() {
        for policy in [GatePolicy::Frame, WIN] {
            let mut g = gate(policy, 0);
            let a = img(50);
            assert_eq!(g.decide(&frame(&a)).unwrap(), Decision::Full(FullReason::First));
            assert_eq!(g.decide(&frame(&a)).unwrap(), Decision::Reuse);
            assert_eq!(g.decide(&frame(&a)).unwrap(), Decision::Reuse);
        }
    }

    #[test]
    fn off_never_compares_and_is_always_full() {
        let mut g = gate(GatePolicy::Off, 0);
        for v in [1, 1, 200] {
            assert_eq!(g.decide(&frame(&img(v))).unwrap(), Decision::Full(FullReason::Disabled));
        }
        assert_eq!(g.last_map().patches_changed(), 0);
    }

    #[test]
    fn frame_policy_is_all_or_nothing() {
        let mut g = gate(GatePolicy::Frame, 0);
        let a = img(10);
        g.decide(&frame(&a)).unwrap();
        let mut b = a.clone();
        bump(&mut b, 4, 1);
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Full(FullReason::Changed));
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Reuse);
    }

    #[test]
    fn windows_policy_names_exactly_the_changed_windows() {
        let mut g = gate(WIN, 0);
        let a = img(10);
        g.decide(&frame(&a)).unwrap();
        let mut b = a.clone();
        bump(&mut b, 1, 5);
        bump(&mut b, 8, 5);
        let d = g.decide(&frame(&b)).unwrap();
        assert_eq!(d, Decision::Windows(WindowSet::from_indices(3, &[1, 8])));
        assert_eq!(g.last_map().windows_changed(), 2);
        assert_eq!(d.recompute_set(3).count(), 2);
        assert_eq!(Decision::Reuse.recompute_set(3).count(), 0);
        assert!(Decision::Full(FullReason::First).recompute_set(3).is_full());
    }

    #[test]
    fn all_windows_changed_is_full_not_a_full_window_set() {
        let mut g = gate(WIN, 0);
        g.decide(&frame(&img(10))).unwrap();
        assert_eq!(g.decide(&frame(&img(11))).unwrap(), Decision::Full(FullReason::AllChanged));
    }

    #[test]
    fn the_reference_is_the_frame_at_the_last_recompute_so_slow_drift_is_caught() {
        // Each step brightens window 4 by 3 levels; tau 8 ignores one step, not three.
        let mut g = gate(WIN, 8);
        let mut a = img(100);
        g.decide(&frame(&a)).unwrap();
        let mut verdicts = Vec::new();
        for _ in 0..4 {
            bump(&mut a, 4, 3);
            verdicts.push(g.decide(&frame(&a)).unwrap());
        }
        assert_eq!(
            verdicts,
            vec![Decision::Reuse, Decision::Reuse, Decision::Windows(WindowSet::from_indices(3, &[4])), Decision::Reuse]
        );
    }

    #[test]
    fn a_windows_decision_updates_only_those_windows_of_the_reference() {
        let mut g = gate(WIN, 0);
        let a = img(10);
        g.decide(&frame(&a)).unwrap();
        let mut b = a.clone();
        bump(&mut b, 0, 1);
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Windows(WindowSet::from_indices(3, &[0])));
        // Window 0 is now current; window 5 changing alone is reported alone.
        bump(&mut b, 5, 1);
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Windows(WindowSet::from_indices(3, &[5])));
        // The reference equals the last frame exactly.
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Reuse);
    }

    #[test]
    fn refresh_every_forces_a_full_on_the_nth_frame_and_counts_from_it() {
        let mut cfg = GateConfig::new(WIN, 0);
        cfg.refresh_every = 3;
        let mut g = Gate::new(cfg, TINY).unwrap();
        let a = img(10);
        let kinds: Vec<_> = (0..7).map(|_| matches!(g.decide(&frame(&a)).unwrap(), Decision::Full(_))).collect();
        // First, then reuse, reuse, refresh, reuse, reuse, refresh.
        assert_eq!(kinds, vec![true, false, false, true, false, false, true]);
        let mut cfg = GateConfig::new(WIN, 0);
        cfg.refresh_every = 1;
        let mut g = Gate::new(cfg, TINY).unwrap();
        assert!((0..4).all(|_| matches!(g.decide(&frame(&a)).unwrap(), Decision::Full(_))));
    }

    #[test]
    fn full_above_percent_caps_the_changed_windows() {
        let mut cfg = GateConfig::new(WIN, 0);
        cfg.full_above_percent = 40; // 3 of 9 is 33%, 4 of 9 is 44%
        let mut g = Gate::new(cfg, TINY).unwrap();
        let a = img(10);
        g.decide(&frame(&a)).unwrap();
        let mut three = a.clone();
        [0, 4, 8].iter().for_each(|&w| bump(&mut three, w, 1));
        assert_eq!(g.decide(&frame(&three)).unwrap(), Decision::Windows(WindowSet::from_indices(3, &[0, 4, 8])));
        let mut four = three.clone();
        [1, 2, 3, 5].iter().for_each(|&w| bump(&mut four, w, 1));
        assert_eq!(g.decide(&frame(&four)).unwrap(), Decision::Full(FullReason::TooMuchChanged));
        assert!(Gate::new(GateConfig { full_above_percent: 101, ..GateConfig::new(WIN, 0) }, TINY).is_err());
    }

    #[test]
    fn reuse_patches_trades_exactness_for_skipping() {
        let mut cfg = GateConfig::new(GatePolicy::Frame, 0);
        cfg.reuse_patches = 4;
        let mut g = Gate::new(cfg, TINY).unwrap();
        let a = img(10);
        g.decide(&frame(&a)).unwrap();
        let mut b = a.clone();
        b[0] = 99; // one patch
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Reuse);
        bump(&mut b, 0, 9); // window 0 is 4 patches here, which still fits the budget
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Reuse);
        bump(&mut b, 1, 9); // 8 patches differ from the reference now
        assert_eq!(g.decide(&frame(&b)).unwrap(), Decision::Full(FullReason::Changed));
    }

    #[test]
    fn reset_forgets_the_reference() {
        let mut g = gate(WIN, 0);
        let a = img(10);
        g.decide(&frame(&a)).unwrap();
        g.reset();
        assert_eq!(g.decide(&frame(&a)).unwrap(), Decision::Full(FullReason::First));
    }

    #[test]
    fn same_stream_same_decisions() {
        let run = || {
            let mut g = Gate::new(GateConfig { refresh_every: 5, full_above_percent: 60, ..GateConfig::new(WIN, 2) }, TINY).unwrap();
            let mut d = img(30);
            let mut out = Vec::new();
            for step in 0..30usize {
                bump(&mut d, (step * 7) % 9, (step % 4) as u8 + 1);
                out.push(g.decide(&frame(&d)).unwrap());
            }
            out
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn wrong_sized_frames_are_errors_not_decisions() {
        let mut g = gate(WIN, 0);
        let small = vec![0u8; 6 * 6 * 3];
        assert!(g.decide(&Frame::new(&small, 6, 6, 3).unwrap()).is_err());
        g.decide(&frame(&img(1))).unwrap();
        let grey = vec![0u8; PX * PX];
        assert!(g.decide(&Frame::new(&grey, PX, PX, 1).unwrap()).is_err());
    }
}
