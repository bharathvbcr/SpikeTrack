//! SAM 3.1's shared vision backbone: the PE ViT trunk and the three simple-FPN
//! necks, written once against the Ojas [`Backend`] so the same code runs on
//! the CPU reference backend and on Metal.
//!
//! Upstream: external/sam3 f1647e8 `model/vitdet.py` (ViT, Block, Attention)
//! and `model/necks.py` (`Sam3TriViTDetNeck`), verified op by op against the
//! fp32 fixtures (`tools/reference/check_fixtures.py`).

use ojas_core::{
    window_merge, window_partition, AttentionOpts, Backend, Conv2dParams, ConvTranspose2dParams,
    OjasError, Tensor,
};
use st_gate::{GatePolicy, WindowSet};
use st_weights::names::{VIT_DIM, VIT_GRID, VIT_PATCH, VIT_ROPE_PAIRS, VIT_WINDOW};
use st_weights::{Conv, FpnLevel, FpnNeck, LayerNorm, Linear, VisionBackbone, VitBlock, Weight};

pub const IMAGE_SIZE: usize = VIT_GRID * VIT_PATCH;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = VIT_DIM / HEADS;
/// Block LayerNorm epsilon (`vitdet.py` builds `partial(nn.LayerNorm, eps=1e-6)`).
pub const BLOCK_LN_EPS: f32 = 1e-6;
/// `ln_pre` is a plain `nn.LayerNorm` (torch's default eps).
pub const LN_PRE_EPS: f32 = 1e-5;

/// Which of SAM 3.1's three necks to run on the trunk features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Neck {
    /// `convs`: the detector's.
    Detector,
    /// `interactive_convs`: the interactive (point/box) tracker path.
    Interactive,
    /// `propagation_convs`: video propagation.
    Propagation,
}

/// Per-stage outputs a caller may want to keep (parity tests, the gate cache).
#[derive(Default)]
pub struct Taps {
    pub patch_embed: Option<Tensor>,
    pub ln_pre: Option<Tensor>,
    /// Indexed by block; filled only for the blocks in `keep_blocks`.
    pub blocks: Vec<Option<Tensor>>,
}

/// What the previous frames left behind for [`Backbone::trunk_gated`].
///
/// One tensor per windowed run (the blocks between two global blocks: blocks
/// 0-6, 8-14, 16-22 and 24-30 of SAM 3.1), each the run's last block output
/// `[1, 72, 72, 1024]`, plus the last trunk output. Five 21 MB tensors, shared
/// by reference count with whatever else holds them.
///
/// A window's slice of a run output is "what this window's blocks gave the
/// last time the window was computed": that is the value carried when the
/// window is skipped.
#[derive(Clone, Default)]
pub struct TrunkCache {
    run_out: Vec<Option<Tensor>>,
    trunk_out: Option<Tensor>,
    valid: bool,
}

impl TrunkCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// True when every tensor is from one completed gated or capturing run, so a
    /// partial recompute or a reuse may build on it.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Forget everything; the next gated run is a full one.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// What a [`Backbone::trunk_gated`] call did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedMode {
    /// The previous trunk output was returned; no block ran.
    Reused,
    /// Every block ran on every window (and the cache was refilled).
    Full,
    /// Only some windows ran in the windowed runs the policy allows.
    Windows,
}

/// Work done by one trunk run, in blocks and in model FLOPs.
///
/// A "window-block" is one windowed block on one 24x24 window; the ungated
/// trunk runs 28 x 9 = 252 of them plus 4 global blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedStats {
    pub mode: GatedMode,
    pub window_blocks_run: usize,
    pub window_blocks_total: usize,
    pub global_blocks_run: usize,
    pub global_blocks_total: usize,
}

impl GatedStats {
    /// Matmul FLOPs of the blocks that ran, plus the patch embedding if the
    /// image was embedded. Per block and token: qkv, proj and the two MLP
    /// matmuls are `2 * 12 * C^2` (`24 C^2`), attention is `4 * keys * C`, with
    /// 576 keys in a window and 5184 in a global block. A model, not a
    /// measurement: it ignores norms, RoPE, GELU and the host gather/scatter.
    pub fn flops(&self) -> f64 {
        let c = VIT_DIM as f64;
        let (tw, tg) = ((VIT_WINDOW * VIT_WINDOW) as f64, (VIT_GRID * VIT_GRID) as f64);
        let window_block = tw * (24.0 * c * c + 4.0 * tw * c);
        let global_block = tg * (24.0 * c * c + 4.0 * tg * c);
        let embed = if self.mode == GatedMode::Reused { 0.0 } else { 2.0 * tg * (3 * VIT_PATCH * VIT_PATCH) as f64 * c };
        self.window_blocks_run as f64 * window_block + self.global_blocks_run as f64 * global_block + embed
    }

    /// [`Self::flops`] over the ungated trunk's.
    pub fn flop_fraction(&self) -> f64 {
        let full = Self { mode: GatedMode::Full, window_blocks_run: self.window_blocks_total, global_blocks_run: self.global_blocks_total, ..*self };
        self.flops() / full.flops()
    }

    /// Window-blocks run over the ungated trunk's, counting a global block as
    /// nine window-blocks (the rows it processes, ignoring attention's quadratic share).
    pub fn block_fraction(&self) -> f64 {
        let weight = (VIT_GRID / VIT_WINDOW).pow(2);
        let done = self.window_blocks_run + weight * self.global_blocks_run;
        done as f64 / (self.window_blocks_total + weight * self.global_blocks_total) as f64
    }
}

/// How a block groups its tokens into attention sequences. Everything else in a
/// block (norms, projections, MLP, residuals) is per token and the same for all.
#[derive(Clone, Copy)]
enum Mix {
    /// `x` is `[1, 72, 72, C]`; one sequence of 5184 tokens.
    Global,
    /// `x` is `[1, 72, 72, C]`; the nine windows are partitioned into a batch of 9.
    Grid,
    /// `x` is `[k, 24, 24, C]`: `k` windows already gathered, a batch of `k`.
    Gathered(usize),
}

/// A maximal run of windowed blocks and the global block that ends it, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage {
    windowed: std::ops::Range<usize>,
    global: Option<usize>,
}

/// Splits a block list (`true` = global) into stages. SAM 3.1: `[0..7)+7`,
/// `[8..15)+15`, `[16..23)+23`, `[24..31)+31`.
fn stages_of(is_global: &[bool]) -> Vec<Stage> {
    let mut stages = Vec::new();
    let mut i = 0;
    while i < is_global.len() {
        let start = i;
        while i < is_global.len() && !is_global[i] {
            i += 1;
        }
        let global = (i < is_global.len()).then_some(i);
        stages.push(Stage { windowed: start..i, global });
        i += usize::from(global.is_some());
    }
    stages
}

/// How one trunk run treats the windows of each windowed run.
enum Plan<'a> {
    /// `trunk`: every block on every window, no cache.
    Ungated,
    /// Every block on every window, then fill the cache.
    Capture(&'a mut TrunkCache),
    /// Recompute `set` in the first windowed run, and in the later ones when
    /// `approximate`; carry the cache for the other windows.
    Partial { cache: &'a mut TrunkCache, set: &'a WindowSet, approximate: bool },
}

fn out_of_range(op: &'static str, detail: String) -> OjasError {
    OjasError::OutOfRange { op, detail }
}

fn host_f32<B: Backend + ?Sized>(be: &B, t: &Tensor) -> Result<Vec<f32>, OjasError> {
    be.download(t)?.to_f32_vec()
}

/// Element offset of token row `y` of window `w` in a `[1, 72, 72, C]` buffer;
/// the 24 tokens of that row are one contiguous run of `24 * C` elements.
fn window_row_offset(w: usize, y: usize) -> usize {
    let side = VIT_GRID / VIT_WINDOW;
    let (wy, wx) = (w / side, w % side);
    ((wy * VIT_WINDOW + y) * VIT_GRID + wx * VIT_WINDOW) * VIT_DIM
}

fn check_grid(op: &'static str, t: &Tensor) -> Result<(), OjasError> {
    if t.shape() != [1, VIT_GRID, VIT_GRID, VIT_DIM] {
        return Err(out_of_range(op, format!("expected [1, {VIT_GRID}, {VIT_GRID}, {VIT_DIM}], got {:?}", t.shape())));
    }
    Ok(())
}

/// The windows of `set` (ascending, the order of `window_partition`) of a
/// `[1, 72, 72, C]` tensor, as `[k, 24, 24, C]`. Host-side: the Backend has no
/// gather over a window subset, so this reads the tensor back and uploads the
/// copy (free on the CPU backend; a round trip on a device backend).
fn gather_windows<B: Backend + ?Sized>(be: &B, x: &Tensor, set: &WindowSet) -> Result<Tensor, OjasError> {
    check_grid("st_model::backbone::gather_windows", x)?;
    let data = host_f32(be, x)?;
    let run = VIT_WINDOW * VIT_DIM;
    let mut out = Vec::with_capacity(set.count() * VIT_WINDOW * run);
    for w in set.iter() {
        for y in 0..VIT_WINDOW {
            let at = window_row_offset(w, y);
            out.extend_from_slice(&data[at..at + run]);
        }
    }
    upload(be, &out, &[set.count(), VIT_WINDOW, VIT_WINDOW, VIT_DIM])
}

/// `base` `[1, 72, 72, C]` with the windows of `set` replaced by the rows of
/// `sub` `[k, 24, 24, C]` (the inverse of [`gather_windows`]); `base` is not modified.
fn scatter_windows<B: Backend + ?Sized>(be: &B, base: &Tensor, sub: &Tensor, set: &WindowSet) -> Result<Tensor, OjasError> {
    const OP: &str = "st_model::backbone::scatter_windows";
    check_grid(OP, base)?;
    if sub.shape() != [set.count(), VIT_WINDOW, VIT_WINDOW, VIT_DIM] {
        return Err(out_of_range(OP, format!("{} windows selected, tensor is {:?}", set.count(), sub.shape())));
    }
    let mut data = host_f32(be, base)?;
    let rows = host_f32(be, sub)?;
    let run = VIT_WINDOW * VIT_DIM;
    for (j, w) in set.iter().enumerate() {
        for y in 0..VIT_WINDOW {
            let at = window_row_offset(w, y);
            data[at..at + run].copy_from_slice(&rows[(j * VIT_WINDOW + y) * run..][..run]);
        }
    }
    upload(be, &data, &[1, VIT_GRID, VIT_GRID, VIT_DIM])
}

struct DevLinear {
    w: Tensor,
    b: Tensor,
}

struct DevNorm {
    w: Tensor,
    b: Tensor,
}

struct DevBlock {
    global: bool,
    norm1: DevNorm,
    q: DevLinear,
    k: DevLinear,
    v: DevLinear,
    proj: DevLinear,
    rope_cos: Tensor,
    rope_sin: Tensor,
    norm2: DevNorm,
    fc1: DevLinear,
    fc2: DevLinear,
}

struct DevConv {
    w: Tensor,
    b: Tensor,
}

enum DevLevel {
    Up4 { dconv0: DevConv, dconv1: DevConv, c1: DevConv, c3: DevConv },
    Up2 { dconv: DevConv, c1: DevConv, c3: DevConv },
    Same { c1: DevConv, c3: DevConv },
}

/// The backbone with its weights resident on one backend.
pub struct Backbone<'b, B: Backend + ?Sized> {
    be: &'b B,
    patch_embed: Tensor,
    /// `pos_embed` without cls, tiled 3x3 to `[1, 72, 72, 1024]` once at load.
    pos_tiled: Tensor,
    ln_pre: DevNorm,
    blocks: Vec<DevBlock>,
    necks: [Vec<DevLevel>; 3],
}

fn upload<B: Backend + ?Sized>(be: &B, data: &[f32], shape: &[usize]) -> Result<Tensor, OjasError> {
    let host = Tensor::from_f32(data, shape, be.budget())?;
    be.upload(&host)
}

fn weight<B: Backend + ?Sized>(be: &B, w: &Weight) -> Result<Tensor, OjasError> {
    upload(be, &w.data, &w.shape)
}

fn norm<B: Backend + ?Sized>(be: &B, n: &LayerNorm) -> Result<DevNorm, OjasError> {
    Ok(DevNorm { w: weight(be, &n.weight)?, b: weight(be, &n.bias)? })
}

fn linear<B: Backend + ?Sized>(be: &B, l: &Linear) -> Result<DevLinear, OjasError> {
    Ok(DevLinear { w: weight(be, &l.weight)?, b: weight(be, &l.bias)? })
}

fn conv<B: Backend + ?Sized>(be: &B, c: &Conv) -> Result<DevConv, OjasError> {
    Ok(DevConv { w: weight(be, &c.weight)?, b: weight(be, &c.bias)? })
}

/// Rows `[part * dim, (part + 1) * dim)` of the fused qkv projection.
fn qkv_part<B: Backend + ?Sized>(be: &B, qkv: &Linear, part: usize) -> Result<DevLinear, OjasError> {
    let rows = part * VIT_DIM..(part + 1) * VIT_DIM;
    let w = &qkv.weight.data[rows.start * VIT_DIM..rows.end * VIT_DIM];
    let b = &qkv.bias.data[rows];
    Ok(DevLinear { w: upload(be, w, &[VIT_DIM, VIT_DIM])?, b: upload(be, b, &[VIT_DIM])? })
}

/// The checkpoint's complex RoPE table `[T, 32, 2]` (re, im) as cos and sin `[T, 32]`.
fn rope_tables<B: Backend + ?Sized>(be: &B, f: &Weight) -> Result<(Tensor, Tensor), OjasError> {
    let t = f.shape[0];
    let (mut cos, mut sin) = (Vec::with_capacity(t * VIT_ROPE_PAIRS), Vec::with_capacity(t * VIT_ROPE_PAIRS));
    for pair in f.data.chunks_exact(2) {
        cos.push(pair[0]);
        sin.push(pair[1]);
    }
    Ok((upload(be, &cos, &[t, VIT_ROPE_PAIRS])?, upload(be, &sin, &[t, VIT_ROPE_PAIRS])?))
}

/// `get_abs_pos(..., tiling=True)`: drop cls, tile the 24x24 grid to 72x72, NHWC.
fn tiled_pos(pos: &Weight) -> Vec<f32> {
    let src = &pos.data[VIT_DIM..]; // skip cls
    let side = VIT_WINDOW;
    let mut out = Vec::with_capacity(VIT_GRID * VIT_GRID * VIT_DIM);
    for y in 0..VIT_GRID {
        for x in 0..VIT_GRID {
            let s = ((y % side) * side + (x % side)) * VIT_DIM;
            out.extend_from_slice(&src[s..s + VIT_DIM]);
        }
    }
    out
}

impl<'b, B: Backend + ?Sized> Backbone<'b, B> {
    /// Uploads the trunk and the three necks to `be`.
    pub fn new(be: &'b B, w: &VisionBackbone) -> Result<Self, OjasError> {
        let t = &w.trunk;
        let mut blocks = Vec::with_capacity(t.blocks.len());
        for blk in &t.blocks {
            blocks.push(Self::block(be, blk)?);
        }
        let neck = |n: &FpnNeck| -> Result<Vec<DevLevel>, OjasError> {
            n.levels
                .iter()
                .map(|lvl| {
                    Ok(match lvl {
                        FpnLevel::Up4 { dconv0, dconv1, conv_1x1, conv_3x3 } => DevLevel::Up4 {
                            dconv0: conv(be, dconv0)?,
                            dconv1: conv(be, dconv1)?,
                            c1: conv(be, conv_1x1)?,
                            c3: conv(be, conv_3x3)?,
                        },
                        FpnLevel::Up2 { dconv, conv_1x1, conv_3x3 } => DevLevel::Up2 {
                            dconv: conv(be, dconv)?,
                            c1: conv(be, conv_1x1)?,
                            c3: conv(be, conv_3x3)?,
                        },
                        FpnLevel::Same { conv_1x1, conv_3x3 } => {
                            DevLevel::Same { c1: conv(be, conv_1x1)?, c3: conv(be, conv_3x3)? }
                        }
                    })
                })
                .collect()
        };
        Ok(Self {
            be,
            patch_embed: weight(be, &t.patch_embed)?,
            pos_tiled: upload(be, &tiled_pos(&t.pos_embed), &[1, VIT_GRID, VIT_GRID, VIT_DIM])?,
            ln_pre: norm(be, &t.ln_pre)?,
            blocks,
            necks: [neck(&w.convs)?, neck(&w.interactive_convs)?, neck(&w.propagation_convs)?],
        })
    }

    fn block(be: &B, blk: &VitBlock) -> Result<DevBlock, OjasError> {
        let (rope_cos, rope_sin) = rope_tables(be, &blk.freqs_cis)?;
        Ok(DevBlock {
            global: blk.global,
            norm1: norm(be, &blk.norm1)?,
            q: qkv_part(be, &blk.qkv, 0)?,
            k: qkv_part(be, &blk.qkv, 1)?,
            v: qkv_part(be, &blk.qkv, 2)?,
            proj: linear(be, &blk.proj)?,
            rope_cos,
            rope_sin,
            norm2: norm(be, &blk.norm2)?,
            fc1: linear(be, &blk.fc1)?,
            fc2: linear(be, &blk.fc2)?,
        })
    }

    /// Trunk: `image` `[1, 3, 1008, 1008]` (normalized) to `[1, 72, 72, 1024]`.
    /// `keep_blocks` lists the block outputs to return in `taps`.
    pub fn trunk(&self, image: &Tensor, keep_blocks: &[usize], taps: &mut Taps) -> Result<Tensor, OjasError> {
        self.run_trunk(image, keep_blocks, taps, Plan::Ungated).map(|(x, _)| x)
    }

    /// The trunk with event gating: recompute only the windows in `recompute`
    /// where `policy` allows it, and take the rest from `cache`. Returns the
    /// trunk output and what ran. The caller (st-gate's `Gate`) decides
    /// `recompute` from the frame; this method executes it.
    ///
    /// What each policy does, for `recompute` = the windows whose input changed:
    ///
    /// - [`GatePolicy::Off`]: the ungated trunk; the cache is cleared.
    /// - [`GatePolicy::Frame`]: an empty `recompute` on a valid cache returns
    ///   the cached trunk output without touching `image`; anything else is a
    ///   full run that refills the cache.
    /// - [`GatePolicy::Windows`]: an empty `recompute` on a valid cache is the
    ///   same reuse. A full `recompute`, or an invalid cache, is a full run that
    ///   refills the cache (`image` is run through the one `block_apply` that
    ///   `trunk` uses, so it is bitwise `trunk`'s output). Otherwise, in the
    ///   first windowed run (blocks 0-6) the `k` windows of `recompute` are
    ///   gathered into a batch of `k`, run through the same blocks, and scattered
    ///   over the cached run output; with `approximate_after_global` the later
    ///   windowed runs (8-14, 16-22, 24-30) do the same with the windows of
    ///   `recompute` and carry the cached outputs of the others. Global blocks
    ///   always run on everything.
    ///
    /// Exactness, per case:
    ///
    /// - An unchanged window's input to block 0 is bit-identical to the cached
    ///   frame's *when `recompute` was built with threshold 0* (st-gate), and
    ///   windowed blocks never read another window, so its block 6 output is
    ///   exactly the cached one. Blocks 7 onward then see exactly the tensor the
    ///   ungated trunk would, so the whole output equals the ungated trunk's.
    ///   This needs the backend's per-token arithmetic to ignore batch size (a
    ///   `k`-window batch gives each row the bits the 9-window batch would): true
    ///   of the CPU backend under `Numerics::Exact`, not guaranteed under `Fast`
    ///   (whole-matrix Accelerate products pick their blocking by shape) or on
    ///   Metal. This is [I] from the sources; `tests/gating_parity.rs` is the check.
    /// - With a threshold above 0 the same path is exact only for the frame the
    ///   cache reference describes; sub-threshold changes are ignored.
    /// - `approximate_after_global` is an approximation by construction: an
    ///   unchanged window's true input to blocks 8+ changed (global attention
    ///   saw the changed windows), and its cached output is from an older frame.
    ///
    /// `taps`: `patch_embed` and `ln_pre` are set whenever the image is embedded;
    /// `blocks[i]` is set for `i` in `keep_blocks` only where the full tensor
    /// exists, which is every block of a full run and, in a partial run, the last
    /// block of each windowed run and each global block (blocks 6, 7, 14, 15, 22,
    /// 23, 30, 31). A reuse sets none.
    pub fn trunk_gated(
        &self,
        image: &Tensor,
        cache: &mut TrunkCache,
        recompute: &WindowSet,
        policy: GatePolicy,
        keep_blocks: &[usize],
        taps: &mut Taps,
    ) -> Result<(Tensor, GatedStats), OjasError> {
        let side = VIT_GRID / VIT_WINDOW;
        if recompute.side() != side {
            return Err(out_of_range("st_model::Backbone::trunk_gated", format!("window set is {0} x {0}, the trunk has {side} x {side}", recompute.side())));
        }
        let reuse = cache.valid && recompute.is_empty() && policy != GatePolicy::Off;
        if reuse {
            if let Some(out) = cache.trunk_out.clone() {
                taps.patch_embed = None;
                taps.ln_pre = None;
                taps.blocks = vec![None; self.blocks.len()];
                let (windowed, global) = self.block_counts();
                let stats = GatedStats { mode: GatedMode::Reused, window_blocks_run: 0, window_blocks_total: windowed * side * side, global_blocks_run: 0, global_blocks_total: global };
                return Ok((out, stats));
            }
        }
        match policy {
            GatePolicy::Off => {
                cache.clear();
                self.run_trunk(image, keep_blocks, taps, Plan::Ungated)
            }
            GatePolicy::Frame => self.run_trunk(image, keep_blocks, taps, Plan::Capture(cache)),
            GatePolicy::Windows { approximate_after_global } => {
                if cache.valid && !recompute.is_full() {
                    self.run_trunk(image, keep_blocks, taps, Plan::Partial { cache, set: recompute, approximate: approximate_after_global })
                } else {
                    self.run_trunk(image, keep_blocks, taps, Plan::Capture(cache))
                }
            }
        }
    }

    fn block_counts(&self) -> (usize, usize) {
        let global = self.blocks.iter().filter(|b| b.global).count();
        (self.blocks.len() - global, global)
    }

    fn stages(&self) -> Vec<Stage> {
        stages_of(&self.blocks.iter().map(|b| b.global).collect::<Vec<_>>())
    }

    /// The one trunk loop: `trunk` is `Plan::Ungated`; the gated paths differ
    /// only in which windows a windowed run is applied to and what is cached.
    fn run_trunk(&self, image: &Tensor, keep_blocks: &[usize], taps: &mut Taps, mut plan: Plan<'_>) -> Result<(Tensor, GatedStats), OjasError> {
        let be = self.be;
        let side = VIT_GRID / VIT_WINDOW;
        let mut p = Conv2dParams::new();
        p.stride = [VIT_PATCH, VIT_PATCH];
        let pe = be.conv2d_forward(image, &self.patch_embed, None, p)?; // [1, 1024, 72, 72]
        let pe = be.permute(&pe, &[0, 2, 3, 1])?; // NHWC
        let x = be.residual_add_forward(&pe, &self.pos_tiled)?;
        let mut x = be.layer_norm_forward(&x, &self.ln_pre.w, &self.ln_pre.b, LN_PRE_EPS)?;
        taps.patch_embed = Some(pe);
        taps.ln_pre = Some(x.clone());
        taps.blocks = vec![None; self.blocks.len()];

        let stages = self.stages();
        let (windowed_blocks, global_blocks) = self.block_counts();
        let mut stats = GatedStats {
            mode: match plan {
                Plan::Ungated | Plan::Capture(_) => GatedMode::Full,
                Plan::Partial { .. } => GatedMode::Windows,
            },
            window_blocks_run: 0,
            window_blocks_total: windowed_blocks * side * side,
            global_blocks_run: 0,
            global_blocks_total: global_blocks,
        };
        match &mut plan {
            Plan::Ungated => {}
            Plan::Capture(cache) | Plan::Partial { cache, .. } => {
                // Invalid until this run completes: an error mid-run leaves a mix of frames.
                cache.valid = false;
                cache.run_out.resize(stages.len(), None);
            }
        }

        for (s, stage) in stages.iter().enumerate() {
            if !stage.windowed.is_empty() {
                // Only the first run's input is a per-window function of the frame; later
                // runs follow a global block, so carrying them is the approximation.
                let partial = match &plan {
                    Plan::Partial { set, approximate, .. } => (s == 0 || *approximate) && !set.is_full(),
                    _ => false,
                };
                if let (true, Plan::Partial { cache, set, .. }) = (partial, &mut plan) {
                    let set: &WindowSet = set;
                    let base = cache.run_out[s].clone().ok_or_else(|| out_of_range("st_model::Backbone::trunk_gated", format!("no cached output for windowed run {s}")))?;
                    let k = set.count();
                    let mut sub = gather_windows(be, &x, set)?;
                    for i in stage.windowed.clone() {
                        sub = self.block_apply(&self.blocks[i], &sub, Mix::Gathered(k))?;
                        stats.window_blocks_run += k;
                    }
                    x = scatter_windows(be, &base, &sub, set)?;
                    let last = stage.windowed.end - 1;
                    if keep_blocks.contains(&last) {
                        taps.blocks[last] = Some(x.clone());
                    }
                } else {
                    for i in stage.windowed.clone() {
                        x = self.block_apply(&self.blocks[i], &x, Mix::Grid)?;
                        stats.window_blocks_run += side * side;
                        if keep_blocks.contains(&i) {
                            taps.blocks[i] = Some(x.clone());
                        }
                    }
                }
                match &mut plan {
                    Plan::Capture(cache) | Plan::Partial { cache, .. } => cache.run_out[s] = Some(x.clone()),
                    Plan::Ungated => {}
                }
            }
            if let Some(g) = stage.global {
                x = self.block_apply(&self.blocks[g], &x, Mix::Global)?;
                stats.global_blocks_run += 1;
                if keep_blocks.contains(&g) {
                    taps.blocks[g] = Some(x.clone());
                }
            }
        }
        match &mut plan {
            Plan::Capture(cache) | Plan::Partial { cache, .. } => {
                cache.trunk_out = Some(x.clone());
                cache.valid = true;
            }
            Plan::Ungated => {}
        }
        Ok((x, stats))
    }

    /// One block, the only implementation of it: windowed blocks (`Grid`,
    /// `Gathered`) and global blocks (`Global`) differ in how the tokens are
    /// grouped into attention sequences; every other op is per token. For
    /// `Global` and `Grid` the sequence of backend calls, and their shapes, is
    /// the ungated trunk's block exactly.
    fn block_apply(&self, blk: &DevBlock, x: &Tensor, mix: Mix) -> Result<Tensor, OjasError> {
        debug_assert_eq!(blk.global, matches!(mix, Mix::Global), "a block is global exactly when it mixes globally");
        let be = self.be;
        let h = be.layer_norm_forward(x, &blk.norm1.w, &blk.norm1.b, BLOCK_LN_EPS)?;
        // Windowed blocks attend within each 24x24 window: a batch of 9 x 576 (or k x 576).
        let (seqs, tokens) = match mix {
            Mix::Global => (1, VIT_GRID * VIT_GRID),
            Mix::Grid => ((VIT_GRID / VIT_WINDOW).pow(2), VIT_WINDOW * VIT_WINDOW),
            Mix::Gathered(k) => (k, VIT_WINDOW * VIT_WINDOW),
        };
        let h = if matches!(mix, Mix::Grid) { window_partition(be, &h, VIT_WINDOW)? } else { h };
        let h = h.reshape(&[seqs, tokens, VIT_DIM])?;
        let heads = |lin: &DevLinear| -> Result<Tensor, OjasError> {
            be.linear_bias_forward(&h, &lin.w, &lin.b)?.reshape(&[seqs, tokens, HEADS, HEAD_DIM])
        };
        // RoPE on [B, T, H, D]; attention wants [B, H, T, D].
        let q = be.rope_interleaved_forward(&heads(&blk.q)?, &blk.rope_cos, &blk.rope_sin)?;
        let k = be.rope_interleaved_forward(&heads(&blk.k)?, &blk.rope_cos, &blk.rope_sin)?;
        let v = heads(&blk.v)?;
        let (q, k, v) = (be.permute(&q, &[0, 2, 1, 3])?, be.permute(&k, &[0, 2, 1, 3])?, be.permute(&v, &[0, 2, 1, 3])?);
        let opts = AttentionOpts { causal: false, window: None, bias: None, key_lens: None };
        let a = be.attention_forward(&q, &k, &v, opts)?; // [B, H, T, D]
        let a = be.permute(&a, &[0, 2, 1, 3])?.reshape(&[seqs, tokens, VIT_DIM])?;
        let a = be.linear_bias_forward(&a, &blk.proj.w, &blk.proj.b)?;
        let a = match mix {
            Mix::Global => a.reshape(&[1, VIT_GRID, VIT_GRID, VIT_DIM])?,
            Mix::Grid => window_merge(be, &a.reshape(&[seqs, VIT_WINDOW, VIT_WINDOW, VIT_DIM])?, VIT_GRID, VIT_GRID)?,
            Mix::Gathered(_) => a.reshape(&[seqs, VIT_WINDOW, VIT_WINDOW, VIT_DIM])?,
        };
        let x = be.residual_add_forward(x, &a)?;
        let m = be.layer_norm_forward(&x, &blk.norm2.w, &blk.norm2.b, BLOCK_LN_EPS)?;
        let m = be.linear_bias_forward(&m, &blk.fc1.w, &blk.fc1.b)?;
        let m = be.gelu_erf_forward(&m)?;
        let m = be.linear_bias_forward(&m, &blk.fc2.w, &blk.fc2.b)?;
        be.residual_add_forward(&x, &m)
    }

    /// One neck over trunk features `[1, 72, 72, 1024]` (NHWC): levels at
    /// 288x288, 144x144 and 72x72, each `[1, 256, h, w]` (NCHW).
    pub fn neck(&self, which: Neck, trunk: &Tensor) -> Result<[Tensor; 3], OjasError> {
        let be = self.be;
        let x = be.permute(trunk, &[0, 3, 1, 2])?;
        let mut up = ConvTranspose2dParams::new();
        up.stride = [2, 2];
        let one = Conv2dParams::new();
        let mut three = Conv2dParams::new();
        three.padding = [1, 1];
        let tail = |y: &Tensor, c1: &DevConv, c3: &DevConv| -> Result<Tensor, OjasError> {
            let y = be.conv2d_forward(y, &c1.w, Some(&c1.b), one)?;
            be.conv2d_forward(&y, &c3.w, Some(&c3.b), three)
        };
        let levels = &self.necks[which as usize];
        let mut out = Vec::with_capacity(3);
        for lvl in levels {
            out.push(match lvl {
                DevLevel::Up4 { dconv0, dconv1, c1, c3 } => {
                    let y = be.conv_transpose2d_forward(&x, &dconv0.w, Some(&dconv0.b), up)?;
                    let y = be.gelu_erf_forward(&y)?;
                    let y = be.conv_transpose2d_forward(&y, &dconv1.w, Some(&dconv1.b), up)?;
                    tail(&y, c1, c3)?
                }
                DevLevel::Up2 { dconv, c1, c3 } => {
                    let y = be.conv_transpose2d_forward(&x, &dconv.w, Some(&dconv.b), up)?;
                    tail(&y, c1, c3)?
                }
                DevLevel::Same { c1, c3 } => tail(&x, c1, c3)?,
            });
        }
        let [a, b, c]: [Tensor; 3] = out.try_into().map_err(|_| OjasError::OutOfRange {
            op: "st_model::Backbone::neck",
            detail: "a neck has three levels".to_string(),
        })?;
        Ok([a, b, c])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::Budget;
    use ojas_cpu::CpuBackend;

    #[test]
    fn stages_split_at_global_blocks() {
        let sam: Vec<bool> = (0..32).map(|i| i % 8 == 7).collect();
        let st = stages_of(&sam);
        assert_eq!(st.len(), 4);
        for (s, stage) in st.iter().enumerate() {
            assert_eq!(*stage, Stage { windowed: s * 8..s * 8 + 7, global: Some(s * 8 + 7) });
        }
        // Odd shapes keep every block exactly once.
        let odd = [false, false, true, true, false, true, false];
        assert_eq!(
            stages_of(&odd),
            vec![
                Stage { windowed: 0..2, global: Some(2) },
                Stage { windowed: 3..3, global: Some(3) },
                Stage { windowed: 4..5, global: Some(5) },
                Stage { windowed: 6..7, global: None },
            ]
        );
        assert!(stages_of(&[]).is_empty());
    }

    /// `[1, 72, 72, C]` whose element is a function of its position, so a misplaced window shows.
    fn ramp() -> Vec<f32> {
        (0..VIT_GRID * VIT_GRID * VIT_DIM).map(|i| (i % 4093) as f32 + (i / 4093) as f32 * 0.25).collect()
    }

    #[test]
    fn gather_matches_window_partition_and_scatter_inverts_it() {
        let be = CpuBackend::new(Budget::new(2 << 30));
        let data = ramp();
        let x = upload(&be, &data, &[1, VIT_GRID, VIT_GRID, VIT_DIM]).unwrap();
        let all = window_partition(&be, &x, VIT_WINDOW).unwrap();
        let all = host_f32(&be, &all).unwrap();
        let per = VIT_WINDOW * VIT_WINDOW * VIT_DIM;

        let set = WindowSet::from_indices(3, &[7, 0, 4]);
        let sub = gather_windows(&be, &x, &set).unwrap();
        assert_eq!(sub.shape(), &[3, VIT_WINDOW, VIT_WINDOW, VIT_DIM]);
        let sub_host = host_f32(&be, &sub).unwrap();
        for (j, w) in [0usize, 4, 7].into_iter().enumerate() {
            assert_eq!(sub_host[j * per..(j + 1) * per], all[w * per..(w + 1) * per], "window {w}");
        }

        // Scatter the gathered windows, negated, over the original: only those windows change.
        let neg: Vec<f32> = sub_host.iter().map(|v| -v).collect();
        let sub_neg = upload(&be, &neg, sub.shape()).unwrap();
        let merged = scatter_windows(&be, &x, &sub_neg, &set).unwrap();
        let merged_parts = window_partition(&be, &merged, VIT_WINDOW).unwrap();
        let merged_parts = host_f32(&be, &merged_parts).unwrap();
        for w in 0..9 {
            let (got, want) = (&merged_parts[w * per..(w + 1) * per], &all[w * per..(w + 1) * per]);
            if set.contains(w) {
                assert!(got.iter().zip(want).all(|(g, v)| *g == -*v), "window {w} replaced");
            } else {
                assert_eq!(got, want, "window {w} untouched");
            }
        }
        assert_eq!(host_f32(&be, &x).unwrap(), data, "the base is not modified");

        // The identity: gather everything, scatter it back.
        let full = WindowSet::full(3);
        let g = gather_windows(&be, &x, &full).unwrap();
        assert_eq!(host_f32(&be, &g).unwrap(), all, "a full gather is window_partition");
        let back = scatter_windows(&be, &merged, &g, &full).unwrap();
        assert_eq!(host_f32(&be, &back).unwrap(), data);
    }

    #[test]
    fn gather_and_scatter_refuse_wrong_shapes() {
        let be = CpuBackend::new(Budget::new(1 << 30));
        let small = upload(&be, &[0.0; 8], &[1, 2, 2, 2]).unwrap();
        let set = WindowSet::from_indices(3, &[1]);
        assert!(gather_windows(&be, &small, &set).is_err());
        let x = upload(&be, &ramp(), &[1, VIT_GRID, VIT_GRID, VIT_DIM]).unwrap();
        let wrong_count = upload(&be, &vec![0.0; 2 * VIT_WINDOW * VIT_WINDOW * VIT_DIM], &[2, VIT_WINDOW, VIT_WINDOW, VIT_DIM]).unwrap();
        assert!(scatter_windows(&be, &x, &wrong_count, &set).is_err());
    }

    #[test]
    fn stats_count_work_and_a_full_run_is_one() {
        let full = GatedStats { mode: GatedMode::Full, window_blocks_run: 252, window_blocks_total: 252, global_blocks_run: 4, global_blocks_total: 4 };
        assert_eq!(full.flop_fraction(), 1.0);
        assert_eq!(full.block_fraction(), 1.0);
        let reuse = GatedStats { mode: GatedMode::Reused, window_blocks_run: 0, global_blocks_run: 0, ..full };
        assert_eq!((reuse.flops(), reuse.flop_fraction(), reuse.block_fraction()), (0.0, 0.0, 0.0));
        // Exact policy, 4 of 9 windows changed: 7 blocks on 4 windows, the other 21 on all 9, all globals.
        let exact = GatedStats { mode: GatedMode::Windows, window_blocks_run: 7 * 4 + 21 * 9, global_blocks_run: 4, ..full };
        assert!(exact.flop_fraction() < 0.9 && exact.flop_fraction() > 0.85, "{}", exact.flop_fraction());
        // The ViT's global blocks cost a large share, so the saving is small even for 7 of 32 blocks.
    }
}
