//! Per-patch and per-window change bits from two frames, in integer math.
//!
//! # What is compared
//!
//! The 8-bit RGB frame the model input is made from: SAM 3's preprocessing is
//! `resize_bilinear_pil(frame, 1008, 1008)` and then, per sample,
//! `(v / 255 - 0.5) / 0.5`. That map is injective on `u8`, so two frames have
//! bit-identical model inputs exactly when their resized `u8` samples are
//! identical. Comparing the `u8` frame instead of the `f32` tensor therefore
//! loses nothing at threshold 0, reads a quarter of the bytes, and needs no
//! float math (which is also why the result cannot differ between runs or
//! machines).
//!
//! It is computed on the model-input frame, not on the decoded frame mapped
//! to the token grid, for one reason: the ViT's patch embedding is a 14x14
//! stride-14 convolution, so a token depends on exactly the 14x14 pixels of
//! the *resized* frame under it. A decoded 480x640 frame resampled to 1008x1008
//! smears every source pixel over several tokens, so a decoded-frame map would
//! have to be dilated by the filter support to be safe, and a "no change"
//! verdict could not be made exact. The resize already runs once per frame for
//! the model; the map reuses its output.
//!
//! # Threshold semantics
//!
//! Patch `p` (the `patch x patch` pixels of one token, all channels, so
//! `patch * patch * channels` samples) is **changed** when
//!
//! ```text
//! sum(|current - reference|) over its samples  >  tau * (patch * patch * channels)
//! ```
//!
//! i.e. its mean absolute difference, on the 0..=255 sample scale, is strictly
//! greater than `tau`. So:
//!
//! - `tau = 0`: changed if and only if any sample differs. This is the exact
//!   setting: an unchanged patch has a bit-identical model input.
//! - `tau >= 255`: nothing is ever changed (the mean cannot exceed 255).
//! - The mean is over the whole patch, so one sample can differ by 255 and
//!   still stay under `tau = 1` (255 / 588 = 0.43 for 3 channels). A small
//!   bright moving object can hide under a coarse `tau`; that is the price of
//!   noise robustness, and why only `tau = 0` is exact.
//!
//! A window is changed when any of its patches is. The earlier no-model
//! measurement (`tools/analysis/window_change_rates.py`) used a grey, cv2
//! bilinear frame and the same mean-over-patch, strictly-greater rule, so its
//! rates are comparable but not identical to this map's.

use std::fmt;

use crate::bits::{BitSet, WindowSet};

/// The token grid and how frame pixels map onto it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Tokens per side of the token grid.
    pub grid: usize,
    /// Pixels per token side (the patch-embed kernel and stride).
    pub patch: usize,
    /// Tokens per side of one attention window.
    pub window: usize,
}

impl Geometry {
    /// SAM 3.1's PE ViT: a 72x72 token grid of 14x14 patches on a 1008x1008
    /// frame, attention windows of 24x24 tokens (a 3x3 grid of windows).
    pub const SAM3: Geometry = Geometry { grid: 72, patch: 14, window: 24 };

    /// Frame side in pixels.
    pub fn pixels(self) -> usize {
        self.grid * self.patch
    }

    /// Windows per grid row.
    pub fn side(self) -> usize {
        self.grid / self.window
    }

    /// Tokens in the grid.
    pub fn tokens(self) -> usize {
        self.grid * self.grid
    }

    fn validate(self) -> Result<(), GateError> {
        if self.grid == 0 || self.patch == 0 || self.window == 0 {
            return Err(GateError::Geometry(format!("{self:?} has a zero dimension")));
        }
        if self.grid % self.window != 0 {
            return Err(GateError::Geometry(format!("window {} does not divide grid {}", self.window, self.grid)));
        }
        Ok(())
    }
}

/// Why a frame, geometry or configuration was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateError {
    Geometry(String),
    /// `data.len()` is not `width * height * channels`.
    DataLength { expected: usize, found: usize },
    /// The frame is not `pixels x pixels` for the geometry.
    FrameSize { expected: usize, found: (usize, usize) },
    /// Zero channels, more than 8, or a different count than the reference.
    Channels { expected: usize, found: usize },
    Config(String),
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(d) => write!(f, "bad geometry: {d}"),
            Self::DataLength { expected, found } => write!(f, "frame data is {found} bytes, expected {expected}"),
            Self::FrameSize { expected, found } => write!(f, "frame is {} x {}, expected {expected} x {expected}", found.0, found.1),
            Self::Channels { expected, found } => write!(f, "frame has {found} channels, expected {expected}"),
            Self::Config(d) => write!(f, "bad gate configuration: {d}"),
        }
    }
}

impl std::error::Error for GateError {}

/// A borrowed interleaved 8-bit frame, row-major: sample `c` of pixel
/// `(x, y)` is `data[(y * width + x) * channels + c]`. The layout of
/// `st_video::Rgb8` with `channels = 3`; this crate does not depend on it.
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    channels: usize,
}

impl<'a> Frame<'a> {
    pub fn new(data: &'a [u8], width: usize, height: usize, channels: usize) -> Result<Self, GateError> {
        if channels == 0 || channels > 8 {
            return Err(GateError::Channels { expected: 3, found: channels });
        }
        let expected = width.checked_mul(height).and_then(|p| p.checked_mul(channels));
        if expected != Some(data.len()) {
            return Err(GateError::DataLength { expected: expected.unwrap_or(usize::MAX), found: data.len() });
        }
        Ok(Self { data, width, height, channels })
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub(crate) fn check(&self, geo: Geometry) -> Result<(), GateError> {
        let px = geo.pixels();
        if self.width != px || self.height != px {
            return Err(GateError::FrameSize { expected: px, found: (self.width, self.height) });
        }
        Ok(())
    }
}

/// Which patches and windows changed between two frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChangeMap {
    geo: Geometry,
    patches: BitSet,
    windows: WindowSet,
}

impl ChangeMap {
    /// Nothing changed.
    pub fn new(geo: Geometry) -> Self {
        Self { geo, patches: BitSet::new(geo.tokens()), windows: WindowSet::new(geo.side()) }
    }

    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    /// One bit per token, row-major (`y * grid + x`).
    pub fn patches(&self) -> &BitSet {
        &self.patches
    }

    /// The windows containing at least one changed patch.
    pub fn windows(&self) -> &WindowSet {
        &self.windows
    }

    pub fn patch_changed(&self, x: usize, y: usize) -> bool {
        self.patches.get(y * self.geo.grid + x)
    }

    pub fn patches_changed(&self) -> usize {
        self.patches.count()
    }

    pub fn windows_changed(&self) -> usize {
        self.windows.count()
    }

    pub fn patch_fraction(&self) -> f64 {
        self.patches.count() as f64 / self.geo.tokens() as f64
    }

    pub fn window_fraction(&self) -> f64 {
        self.windows.count() as f64 / self.windows.windows() as f64
    }

    pub(crate) fn clear(&mut self) {
        self.patches.clear();
        self.windows = WindowSet::new(self.geo.side());
    }
}

/// Computes change maps; owns the per-patch scratch so a frame allocates
/// nothing.
pub struct ChangeMapper {
    geo: Geometry,
    sad: Vec<u32>,
}

impl ChangeMapper {
    pub fn new(geo: Geometry) -> Result<Self, GateError> {
        geo.validate()?;
        Ok(Self { geo, sad: vec![0; geo.tokens()] })
    }

    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    /// Fills `out` with the patches of `current` that differ from `reference`
    /// by more than `tau` (see the module docs). Both frames must be
    /// `geometry().pixels()` square with the same channel count.
    pub fn compute(&mut self, reference: &Frame<'_>, current: &Frame<'_>, tau: u32, out: &mut ChangeMap) -> Result<(), GateError> {
        let geo = self.geo;
        reference.check(geo)?;
        current.check(geo)?;
        if reference.channels != current.channels {
            return Err(GateError::Channels { expected: reference.channels, found: current.channels });
        }
        if out.geo != geo {
            return Err(GateError::Geometry(format!("map is for {:?}, mapper for {geo:?}", out.geo)));
        }
        let ch = current.channels;
        let row_bytes = geo.pixels() * ch;
        let patch_bytes = geo.patch * ch;
        // One patch is at most 255 * patch * patch * 8 samples, which must fit u32.
        if geo.patch * geo.patch * ch * 255 > u32::MAX as usize {
            return Err(GateError::Geometry(format!("patch {} x {ch} channels overflows the u32 accumulator", geo.patch)));
        }

        self.sad.iter_mut().for_each(|s| *s = 0);
        for (y, (a, b)) in reference.data.chunks_exact(row_bytes).zip(current.data.chunks_exact(row_bytes)).enumerate() {
            let acc = &mut self.sad[(y / geo.patch) * geo.grid..][..geo.grid];
            for ((s, pa), pb) in acc.iter_mut().zip(a.chunks_exact(patch_bytes)).zip(b.chunks_exact(patch_bytes)) {
                *s += pa.iter().zip(pb).map(|(&p, &q)| u32::from(p.abs_diff(q))).sum::<u32>();
            }
        }

        out.clear();
        let limit = u64::from(tau) * (geo.patch * geo.patch * ch) as u64;
        let side = geo.side();
        for (i, &s) in self.sad.iter().enumerate() {
            if u64::from(s) > limit {
                out.patches.set(i, true);
                let (x, y) = (i % geo.grid, i / geo.grid);
                out.windows.insert((y / geo.window) * side + x / geo.window);
            }
        }
        Ok(())
    }
}

/// [`ChangeMapper::compute`] into a fresh map; for one-off use and tests.
pub fn change_map(reference: &Frame<'_>, current: &Frame<'_>, geo: Geometry, tau: u32) -> Result<ChangeMap, GateError> {
    let mut mapper = ChangeMapper::new(geo)?;
    let mut out = ChangeMap::new(geo);
    mapper.compute(reference, current, tau, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 6x6 tokens of 2x2 pixels, windows of 2x2 tokens: a 3x3 grid of windows on a 12x12 frame.
    const TINY: Geometry = Geometry { grid: 6, patch: 2, window: 2 };

    fn blank(geo: Geometry, ch: usize, v: u8) -> Vec<u8> {
        vec![v; geo.pixels() * geo.pixels() * ch]
    }

    fn frame(data: &[u8], geo: Geometry, ch: usize) -> Frame<'_> {
        Frame::new(data, geo.pixels(), geo.pixels(), ch).unwrap()
    }

    fn put(data: &mut [u8], geo: Geometry, ch: usize, x: usize, y: usize, c: usize, v: u8) {
        data[(y * geo.pixels() + x) * ch + c] = v;
    }

    #[test]
    fn identical_frames_change_nothing_at_every_threshold() {
        let a = blank(TINY, 3, 90);
        for tau in [0, 1, 255] {
            let m = change_map(&frame(&a, TINY, 3), &frame(&a, TINY, 3), TINY, tau).unwrap();
            assert_eq!((m.patches_changed(), m.windows_changed()), (0, 0), "tau {tau}");
        }
    }

    #[test]
    fn tau_zero_is_exact_one_sample_one_level_flags_exactly_its_patch_and_window() {
        let a = blank(TINY, 3, 100);
        let mut b = a.clone();
        // Pixel (5, 3) is in patch (2, 1), which is in window (row 0, col 1) = window 1.
        put(&mut b, TINY, 3, 5, 3, 2, 101);
        let m = change_map(&frame(&a, TINY, 3), &frame(&b, TINY, 3), TINY, 0).unwrap();
        assert_eq!(m.patches().iter_ones().collect::<Vec<_>>(), vec![TINY.grid + 2]);
        assert!(m.patch_changed(2, 1));
        assert_eq!(m.windows().iter().collect::<Vec<_>>(), vec![1]);
        // The same single-level change is invisible to any tau >= 1: the mean is 1/12.
        let m = change_map(&frame(&a, TINY, 3), &frame(&b, TINY, 3), TINY, 1).unwrap();
        assert_eq!(m.patches_changed(), 0);
    }

    #[test]
    fn the_threshold_is_a_strict_mean_over_the_whole_patch() {
        // 2x2 patch, 3 channels: 12 samples. tau 2 -> changed iff sum > 24.
        let a = blank(TINY, 3, 0);
        for (sum, changed) in [(24u32, false), (25, true)] {
            let mut b = a.clone();
            let mut left = sum;
            // Pixels of patch (0, 0), spread over its 12 samples, at most 255 each.
            for i in 0..12usize {
                let v = left.min(255);
                left -= v;
                put(&mut b, TINY, 3, i / 3 % 2, i / 6, i % 3, v as u8);
            }
            let m = change_map(&frame(&a, TINY, 3), &frame(&b, TINY, 3), TINY, 2).unwrap();
            assert_eq!(m.patch_changed(0, 0), changed, "sum {sum}");
        }
    }

    #[test]
    fn tau_255_never_fires_even_on_a_full_swing() {
        let (a, b) = (blank(TINY, 3, 0), blank(TINY, 3, 255));
        let m = change_map(&frame(&a, TINY, 3), &frame(&b, TINY, 3), TINY, 255).unwrap();
        assert_eq!(m.patches_changed(), 0);
        let m = change_map(&frame(&a, TINY, 3), &frame(&b, TINY, 3), TINY, 254).unwrap();
        assert_eq!((m.patches_changed(), m.windows_changed()), (36, 9));
    }

    #[test]
    fn the_difference_is_absolute_so_brighter_and_darker_agree() {
        let (mid, hi, lo) = (blank(TINY, 1, 100), blank(TINY, 1, 110), blank(TINY, 1, 90));
        let up = change_map(&frame(&mid, TINY, 1), &frame(&hi, TINY, 1), TINY, 9).unwrap();
        let down = change_map(&frame(&mid, TINY, 1), &frame(&lo, TINY, 1), TINY, 9).unwrap();
        assert_eq!(up, down);
        assert_eq!(up.patches_changed(), 36);
        let none = change_map(&frame(&mid, TINY, 1), &frame(&hi, TINY, 1), TINY, 10).unwrap();
        assert_eq!(none.patches_changed(), 0);
    }

    #[test]
    fn the_last_row_and_column_of_patches_and_windows_map_correctly() {
        let a = blank(TINY, 1, 0);
        let mut b = a.clone();
        put(&mut b, TINY, 1, 11, 11, 0, 9); // bottom-right pixel: patch (5, 5), window 8
        put(&mut b, TINY, 1, 0, 11, 0, 9); // bottom-left: patch (0, 5), window 6
        let m = change_map(&frame(&a, TINY, 1), &frame(&b, TINY, 1), TINY, 0).unwrap();
        assert!(m.patch_changed(5, 5) && m.patch_changed(0, 5));
        assert_eq!(m.windows().iter().collect::<Vec<_>>(), vec![6, 8]);
    }

    /// An independent, direct implementation of the module-doc rule.
    fn naive(a: &[u8], b: &[u8], geo: Geometry, ch: usize, tau: u32) -> (Vec<bool>, Vec<bool>) {
        let n = geo.pixels();
        let mut patches = vec![false; geo.tokens()];
        let mut windows = vec![false; geo.side() * geo.side()];
        for py in 0..geo.grid {
            for px in 0..geo.grid {
                let mut sum = 0u64;
                for dy in 0..geo.patch {
                    for dx in 0..geo.patch {
                        for c in 0..ch {
                            let i = (((py * geo.patch + dy) * n) + px * geo.patch + dx) * ch + c;
                            sum += (i64::from(a[i]) - i64::from(b[i])).unsigned_abs();
                        }
                    }
                }
                let samples = (geo.patch * geo.patch * ch) as u64;
                if sum > u64::from(tau) * samples {
                    patches[py * geo.grid + px] = true;
                    windows[(py / geo.window) * geo.side() + px / geo.window] = true;
                }
            }
        }
        (patches, windows)
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[test]
    fn matches_a_direct_implementation_on_random_sparse_changes() {
        let mut s = 7u64;
        for (geo, ch) in [(TINY, 3), (TINY, 1), (Geometry { grid: 8, patch: 3, window: 4 }, 3), (Geometry { grid: 4, patch: 5, window: 2 }, 2)] {
            let mut mapper = ChangeMapper::new(geo).unwrap();
            let mut out = ChangeMap::new(geo);
            for round in 0..40 {
                let a: Vec<u8> = (0..geo.pixels() * geo.pixels() * ch).map(|_| (lcg(&mut s) % 256) as u8).collect();
                let mut b = a.clone();
                for _ in 0..(lcg(&mut s) % 12) {
                    let i = (lcg(&mut s) as usize) % b.len();
                    b[i] = (lcg(&mut s) % 256) as u8;
                }
                let tau = [0, 0, 1, 3, 20][round % 5];
                mapper.compute(&frame(&a, geo, ch), &frame(&b, geo, ch), tau, &mut out).unwrap();
                let (p, w) = naive(&a, &b, geo, ch, tau);
                let got_p: Vec<bool> = (0..geo.tokens()).map(|i| out.patches().get(i)).collect();
                let got_w: Vec<bool> = (0..geo.side() * geo.side()).map(|i| out.windows().contains(i)).collect();
                assert_eq!((got_p, got_w), (p, w), "{geo:?} ch {ch} round {round} tau {tau}");
            }
        }
    }

    #[test]
    fn a_reused_map_is_fully_overwritten() {
        let a = blank(TINY, 3, 0);
        let mut b = a.clone();
        put(&mut b, TINY, 3, 0, 0, 0, 200);
        let mut mapper = ChangeMapper::new(TINY).unwrap();
        let mut out = ChangeMap::new(TINY);
        mapper.compute(&frame(&a, TINY, 3), &frame(&b, TINY, 3), 0, &mut out).unwrap();
        assert_eq!(out.patches_changed(), 1);
        mapper.compute(&frame(&a, TINY, 3), &frame(&a, TINY, 3), 0, &mut out).unwrap();
        assert_eq!((out.patches_changed(), out.windows_changed()), (0, 0));
    }

    #[test]
    fn sam3_geometry_on_a_real_size_frame() {
        let g = Geometry::SAM3;
        assert_eq!((g.pixels(), g.side(), g.tokens()), (1008, 3, 5184));
        let a = blank(g, 3, 10);
        let mut b = a.clone();
        // Pixel (336, 0) is the first pixel of window 1; (1007, 1007) the last of window 8.
        put(&mut b, g, 3, 336, 0, 0, 11);
        put(&mut b, g, 3, 1007, 1007, 1, 11);
        let m = change_map(&frame(&a, g, 3), &frame(&b, g, 3), g, 0).unwrap();
        assert_eq!(m.windows().iter().collect::<Vec<_>>(), vec![1, 8]);
        assert_eq!(m.patches().iter_ones().collect::<Vec<_>>(), vec![24, 71 * 72 + 71]);
    }

    #[test]
    fn refuses_frames_that_do_not_fit() {
        let a = blank(TINY, 3, 0);
        assert!(matches!(Frame::new(&a[1..], 12, 12, 3), Err(GateError::DataLength { .. })));
        assert!(matches!(Frame::new(&a, 12, 12, 0), Err(GateError::Channels { .. })));
        let small = Frame::new(&a[..6 * 6 * 3], 6, 6, 3).unwrap();
        let ok = frame(&a, TINY, 3);
        let mut mapper = ChangeMapper::new(TINY).unwrap();
        let mut out = ChangeMap::new(TINY);
        assert!(matches!(mapper.compute(&small, &ok, 0, &mut out), Err(GateError::FrameSize { .. })));
        let grey = blank(TINY, 1, 0);
        assert!(matches!(mapper.compute(&frame(&grey, TINY, 1), &ok, 0, &mut out), Err(GateError::Channels { .. })));
        assert!(ChangeMapper::new(Geometry { grid: 7, patch: 2, window: 2 }).is_err());
        assert!(ChangeMapper::new(Geometry { grid: 6, patch: 0, window: 2 }).is_err());
        let mut other = ChangeMap::new(Geometry { grid: 4, patch: 2, window: 2 });
        assert!(matches!(mapper.compute(&ok, &ok, 0, &mut other), Err(GateError::Geometry(_))));
    }
}
