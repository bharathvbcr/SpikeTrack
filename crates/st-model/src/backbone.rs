//! SAM 3.1's shared vision backbone: the PE ViT trunk and the three simple-FPN
//! necks, written once against the Ojas [`Backend`] so the same code runs on
//! the CPU reference backend and on Metal.
//!
//! Upstream: external/sam3 f1647e8 `model/vitdet.py` (ViT, Block, Attention)
//! and `model/necks.py` (`Sam3TriViTDetNeck`), verified op by op against the
//! fp32 fixtures (`tools/reference/check_fixtures.py`).

use ojas_core::{
    axial_rope_2d_tables, window_merge, window_partition, AttentionOpts, AutocastMode, Backend, Budget, Conv2dParams,
    ConvTranspose2dParams, OjasError, Tensor,
};
use st_gate::{GatePolicy, Geometry, WindowSet};
use st_weights::names::{VIT_DIM, VIT_GRID, VIT_PATCH, VIT_ROPE_PAIRS, VIT_WINDOW};
use st_weights::{Conv, FpnLevel, FpnNeck, LayerNorm, Linear, VisionBackbone, VitBlock, Weight};

/// The checkpoint's native image side (1008): the default of every geometry-taking constructor.
pub const IMAGE_SIZE: usize = VIT_GRID * VIT_PATCH;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = VIT_DIM / HEADS;
/// LayerNorm epsilon of `ln_pre` and of every block's norm1/norm2: the builder passes
/// `norm_layer="LayerNorm"` (model_builder.py:88), which `vitdet.py:868-869` turns into
/// `partial(nn.LayerNorm, eps=1e-5)` and hands to ln_pre and to each Block.
pub const VIT_LN_EPS: f32 = 1e-5;

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
    /// [`Self::flops_at`] for the native 72 x 72 grid. The counts do not say which grid they
    /// were measured on, so a run at another resolution asks [`Self::flops_at`] with
    /// [`Backbone::geometry`].
    pub fn flops(&self) -> f64 {
        self.flops_at(Geometry::SAM3)
    }

    /// Matmul FLOPs of the blocks that ran, plus the patch embedding if the
    /// image was embedded. Per block and token: qkv, proj and the two MLP
    /// matmuls are `2 * 12 * C^2` (`24 C^2`), attention is `4 * keys * C`, with
    /// 576 keys in a window and `grid^2` (5184 at 1008) in a global block. A model, not a
    /// measurement: it ignores norms, RoPE, GELU and the host gather/scatter.
    pub fn flops_at(&self, geo: Geometry) -> f64 {
        let c = VIT_DIM as f64;
        let (tw, tg) = ((geo.window * geo.window) as f64, geo.tokens() as f64);
        let window_block = tw * (24.0 * c * c + 4.0 * tw * c);
        let global_block = tg * (24.0 * c * c + 4.0 * tg * c);
        let embed = if self.mode == GatedMode::Reused { 0.0 } else { 2.0 * tg * (3 * geo.patch * geo.patch) as f64 * c };
        self.window_blocks_run as f64 * window_block + self.global_blocks_run as f64 * global_block + embed
    }

    /// [`Self::flop_fraction_at`] for the native grid.
    pub fn flop_fraction(&self) -> f64 {
        self.flop_fraction_at(Geometry::SAM3)
    }

    /// [`Self::flops_at`] over the ungated trunk's.
    pub fn flop_fraction_at(&self, geo: Geometry) -> f64 {
        let full = Self { mode: GatedMode::Full, window_blocks_run: self.window_blocks_total, global_blocks_run: self.global_blocks_total, ..*self };
        self.flops_at(geo) / full.flops_at(geo)
    }

    /// [`Self::block_fraction_at`] for the native grid.
    pub fn block_fraction(&self) -> f64 {
        self.block_fraction_at(Geometry::SAM3)
    }

    /// Window-blocks run over the ungated trunk's, counting a global block as
    /// as many window-blocks as there are windows (the rows it processes, ignoring attention's
    /// quadratic share): nine at 1008.
    pub fn block_fraction_at(&self, geo: Geometry) -> f64 {
        let weight = geo.side().pow(2);
        let done = self.window_blocks_run + weight * self.global_blocks_run;
        done as f64 / (self.window_blocks_total + weight * self.global_blocks_total) as f64
    }
}

/// How a block groups its tokens into attention sequences. Everything else in a
/// block (norms, projections, MLP, residuals) is per token and the same for all.
#[derive(Clone, Copy)]
enum Mix {
    /// `x` is `[1, G, G, C]`; one sequence of `G^2` tokens (5184 at 1008).
    Global,
    /// `x` is `[1, G, G, C]`; the `(G / 24)^2` windows (nine at 1008) are partitioned into a batch.
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

/// Element offset of token row `y` of window `w` in a `[1, grid, grid, C]` buffer;
/// the 24 tokens of that row are one contiguous run of `24 * C` elements.
fn window_row_offset(grid: usize, w: usize, y: usize) -> usize {
    let side = grid / VIT_WINDOW;
    let (wy, wx) = (w / side, w % side);
    ((wy * VIT_WINDOW + y) * grid + wx * VIT_WINDOW) * VIT_DIM
}

/// The token grid of a `[1, grid, grid, C]` trunk tensor holding exactly the windows `set`
/// indexes: `grid` is read off the tensor, and refused unless it is a whole number of windows
/// and `set` is a grid of them.
fn checked_grid(op: &'static str, t: &Tensor, set: &WindowSet) -> Result<usize, OjasError> {
    let grid = t.shape().get(1).copied().unwrap_or(0);
    if t.shape() != [1, grid, grid, VIT_DIM] || grid == 0 || grid % VIT_WINDOW != 0 || grid / VIT_WINDOW != set.side() {
        return Err(out_of_range(
            op,
            format!("expected [1, {0}, {0}, {VIT_DIM}] for a {1} x {1} window set, got {2:?}", set.side() * VIT_WINDOW, set.side(), t.shape()),
        ));
    }
    Ok(grid)
}

/// The windows of `set` (ascending, the order of `window_partition`) of a
/// `[1, grid, grid, C]` tensor, as `[k, 24, 24, C]`. Host-side: the Backend has no
/// gather over a window subset, so this reads the tensor back and uploads the
/// copy (free on the CPU backend; a round trip on a device backend).
fn gather_windows<B: Backend + ?Sized>(be: &B, x: &Tensor, set: &WindowSet) -> Result<Tensor, OjasError> {
    let grid = checked_grid("st_model::backbone::gather_windows", x, set)?;
    let data = host_f32(be, x)?;
    let run = VIT_WINDOW * VIT_DIM;
    let mut out = Vec::with_capacity(set.count() * VIT_WINDOW * run);
    for w in set.iter() {
        for y in 0..VIT_WINDOW {
            let at = window_row_offset(grid, w, y);
            out.extend_from_slice(&data[at..at + run]);
        }
    }
    upload(be, &out, &[set.count(), VIT_WINDOW, VIT_WINDOW, VIT_DIM])
}

/// `base` `[1, grid, grid, C]` with the windows of `set` replaced by the rows of
/// `sub` `[k, 24, 24, C]` (the inverse of [`gather_windows`]); `base` is not modified.
fn scatter_windows<B: Backend + ?Sized>(be: &B, base: &Tensor, sub: &Tensor, set: &WindowSet) -> Result<Tensor, OjasError> {
    const OP: &str = "st_model::backbone::scatter_windows";
    let grid = checked_grid(OP, base, set)?;
    if sub.shape() != [set.count(), VIT_WINDOW, VIT_WINDOW, VIT_DIM] {
        return Err(out_of_range(OP, format!("{} windows selected, tensor is {:?}", set.count(), sub.shape())));
    }
    let mut data = host_f32(be, base)?;
    let rows = host_f32(be, sub)?;
    let run = VIT_WINDOW * VIT_DIM;
    for (j, w) in set.iter().enumerate() {
        for y in 0..VIT_WINDOW {
            let at = window_row_offset(grid, w, y);
            data[at..at + run].copy_from_slice(&rows[(j * VIT_WINDOW + y) * run..][..run]);
        }
    }
    upload(be, &data, &[1, grid, grid, VIT_DIM])
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
    /// The input size every shape in the trunk and necks follows.
    geo: Geometry,
    patch_embed: Tensor,
    /// `pos_embed` without cls, tiled to `[1, grid, grid, 1024]` once at load (3x3 at 1008).
    pos_tiled: Tensor,
    ln_pre: DevNorm,
    blocks: Vec<DevBlock>,
    necks: [Vec<DevLevel>; 3],
    /// [`Self::with_trunk_linears_bf16`]: the trunk's six linears per block run inside an Ojas
    /// Autocast bf16 region (bf16 operands on the matrix units, f32 sums), everything else f32.
    trunk_linears_bf16: bool,
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

/// The trunk's `freqs_cis` for a `grid x grid` token map, as cos and sin `[grid^2, 32]` on the
/// host: upstream's `compute_axial_cis(dim=64, end_x=grid, end_y=grid, scale_pos=24 / grid)`
/// (`vitdet.py:88`, reached through `Attention._setup_rope_freqs` with `rope_pt_size=(24, 24)`,
/// `rope_interp=True`). The scale is the 24-token pretrain window over the grid, so a
/// global block's positions span the same 0..24 range at any resolution (1/3 at 72, 1/2 at 48),
/// and a windowed block (grid 24) is the integer grid at scale 1. The 4 global blocks store
/// the 72-grid table in the checkpoint; any other grid needs this.
pub fn vit_rope_tables(grid: usize, budget: &Budget) -> Result<(Tensor, Tensor), OjasError> {
    if grid == 0 {
        return Err(out_of_range("st_model::backbone::vit_rope_tables", "an empty grid".to_string()));
    }
    // Python's `rope_pt_size[0] / input_size[0]` is an f64 quotient that torch rounds to f32.
    let scale_pos = (VIT_WINDOW as f64 / grid as f64) as f32;
    axial_rope_2d_tables(HEAD_DIM, grid, grid, 10000.0, scale_pos, budget)
}

/// `get_abs_pos(..., tiling=True)`: drop cls, tile the 24x24 grid to `grid x grid`, NHWC.
fn tiled_pos(pos: &Weight, grid: usize) -> Vec<f32> {
    let src = &pos.data[VIT_DIM..]; // skip cls
    let side = VIT_WINDOW;
    let mut out = Vec::with_capacity(grid * grid * VIT_DIM);
    for y in 0..grid {
        for x in 0..grid {
            let s = ((y % side) * side + (x % side)) * VIT_DIM;
            out.extend_from_slice(&src[s..s + VIT_DIM]);
        }
    }
    out
}

impl<'b, B: Backend + ?Sized> Backbone<'b, B> {
    /// Uploads the trunk and the three necks to `be` for the native 1008 x 1008 input.
    pub fn new(be: &'b B, w: &VisionBackbone) -> Result<Self, OjasError> {
        Self::new_at(be, w, Geometry::SAM3)
    }

    /// As [`Self::new`] for the input size `geo` (see [`Geometry::from_image_side`]). Nothing
    /// stored in the checkpoint changes shape except the 4 global blocks' RoPE tables: they are
    /// used as stored when their row count is `geo`'s token count, and regenerated by
    /// [`vit_rope_tables`] otherwise. The position embedding is the checkpoint's 24 x 24 grid
    /// tiled to `geo.grid`, and the windowed blocks' tables (24 x 24) are unchanged.
    pub fn new_at(be: &'b B, w: &VisionBackbone, geo: Geometry) -> Result<Self, OjasError> {
        if geo.patch != VIT_PATCH || geo.window != VIT_WINDOW || geo.grid == 0 || geo.grid % geo.window != 0 {
            return Err(out_of_range("st_model::Backbone::new_at", format!("{geo:?} is not a {VIT_PATCH}-pixel patch, {VIT_WINDOW}-token window ViT grid")));
        }
        let t = &w.trunk;
        let mut blocks = Vec::with_capacity(t.blocks.len());
        for blk in &t.blocks {
            blocks.push(Self::block(be, blk, geo)?);
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
            geo,
            patch_embed: weight(be, &t.patch_embed)?,
            pos_tiled: upload(be, &tiled_pos(&t.pos_embed, geo.grid), &[1, geo.grid, geo.grid, VIT_DIM])?,
            ln_pre: norm(be, &t.ln_pre)?,
            blocks,
            necks: [neck(&w.convs)?, neck(&w.interactive_convs)?, neck(&w.propagation_convs)?],
            trunk_linears_bf16: false,
        })
    }

    /// An experiment, not upstream's numerics: the trunk's q, k, v, proj, fc1 and fc2 linears run
    /// in bf16 (each block's weights rounded to bf16 once here, the input rounded per call by the
    /// Autocast region, the output rounded; f32 bias and f32 accumulation), while attention, RoPE,
    /// norms, GELU, the residual stream, the necks and everything downstream stay f32. Whole-model
    /// Autocast (every matmul-class op, attention and convolutions included) missed the D2 and D3
    /// gates (bench/results/module-gates-cbd5565); this is the narrower variant those runs left
    /// untested. `be` must be an [`ojas_core::Autocast`] backend: anything else is refused here,
    /// rather than silently running f32.
    pub fn with_trunk_linears_bf16(mut self) -> Result<Self, OjasError> {
        let be = self.be;
        drop(be.autocast_region(AutocastMode::Bf16)?);
        for blk in &mut self.blocks {
            for lin in [&mut blk.q, &mut blk.k, &mut blk.v, &mut blk.proj, &mut blk.fc1, &mut blk.fc2] {
                lin.w = be.cast_bf16(&lin.w)?;
            }
        }
        self.trunk_linears_bf16 = true;
        Ok(self)
    }

    /// `x @ lin.w^T + lin.b`, inside a bf16 Autocast region when the trunk linears are bf16.
    fn trunk_linear(&self, x: &Tensor, lin: &DevLinear) -> Result<Tensor, OjasError> {
        let _region = if self.trunk_linears_bf16 { Some(self.be.autocast_region(AutocastMode::Bf16)?) } else { None };
        self.be.linear_bias_forward(x, &lin.w, &lin.b)
    }

    /// The geometry this backbone was built for; the input and every output follow it.
    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    fn block(be: &B, blk: &VitBlock, geo: Geometry) -> Result<DevBlock, OjasError> {
        // A windowed block's table is the 24 x 24 grid at any input size; a global block's is the
        // stored one only on the grid it was stored for.
        let tokens = if blk.global { geo.tokens() } else { VIT_WINDOW * VIT_WINDOW };
        let (rope_cos, rope_sin) = if blk.freqs_cis.shape[0] == tokens {
            rope_tables(be, &blk.freqs_cis)?
        } else if blk.global {
            let (cos, sin) = vit_rope_tables(geo.grid, be.budget())?;
            (be.upload(&cos)?, be.upload(&sin)?)
        } else {
            return Err(out_of_range("st_model::Backbone::new_at", format!("a windowed block's RoPE table has {} rows, not {tokens}", blk.freqs_cis.shape[0])));
        };
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

    /// Trunk: `image` `[1, 3, S, S]` (normalized, `S` = [`Geometry::pixels`], 1008 by default) to
    /// `[1, S / 14, S / 14, 1024]`.
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
        let side = self.geo.side();
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
        let side = self.geo.side();
        let px = self.geo.pixels();
        if image.shape() != [1, 3, px, px] {
            return Err(out_of_range("st_model::Backbone::trunk", format!("expected a [1, 3, {px}, {px}] image for this backbone, got {:?}", image.shape())));
        }
        let mut p = Conv2dParams::new();
        p.stride = [VIT_PATCH, VIT_PATCH];
        let pe = be.conv2d_forward(image, &self.patch_embed, None, p)?; // [1, 1024, grid, grid]
        let pe = be.permute(&pe, &[0, 2, 3, 1])?; // NHWC
        let x = be.residual_add_forward(&pe, &self.pos_tiled)?;
        let mut x = be.layer_norm_forward(&x, &self.ln_pre.w, &self.ln_pre.b, VIT_LN_EPS)?;
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
        let h = be.layer_norm_forward(x, &blk.norm1.w, &blk.norm1.b, VIT_LN_EPS)?;
        // Windowed blocks attend within each 24x24 window: a batch of 9 x 576 (or k x 576).
        let (seqs, tokens) = match mix {
            Mix::Global => (1, self.geo.tokens()),
            Mix::Grid => (self.geo.side().pow(2), VIT_WINDOW * VIT_WINDOW),
            Mix::Gathered(k) => (k, VIT_WINDOW * VIT_WINDOW),
        };
        let h = if matches!(mix, Mix::Grid) { window_partition(be, &h, VIT_WINDOW)? } else { h };
        let h = h.reshape(&[seqs, tokens, VIT_DIM])?;
        let heads = |lin: &DevLinear| -> Result<Tensor, OjasError> {
            self.trunk_linear(&h, lin)?.reshape(&[seqs, tokens, HEADS, HEAD_DIM])
        };
        // RoPE on [B, T, H, D]; attention wants [B, H, T, D].
        let q = be.rope_interleaved_forward(&heads(&blk.q)?, &blk.rope_cos, &blk.rope_sin)?;
        let k = be.rope_interleaved_forward(&heads(&blk.k)?, &blk.rope_cos, &blk.rope_sin)?;
        let v = heads(&blk.v)?;
        let (q, k, v) = (be.permute(&q, &[0, 2, 1, 3])?, be.permute(&k, &[0, 2, 1, 3])?, be.permute(&v, &[0, 2, 1, 3])?);
        let opts = AttentionOpts { causal: false, window: None, bias: None, key_lens: None };
        let a = be.attention_forward(&q, &k, &v, opts)?; // [B, H, T, D]
        let a = be.permute(&a, &[0, 2, 1, 3])?.reshape(&[seqs, tokens, VIT_DIM])?;
        let a = self.trunk_linear(&a, &blk.proj)?;
        let a = match mix {
            Mix::Global => a.reshape(&[1, self.geo.grid, self.geo.grid, VIT_DIM])?,
            Mix::Grid => window_merge(be, &a.reshape(&[seqs, VIT_WINDOW, VIT_WINDOW, VIT_DIM])?, self.geo.grid, self.geo.grid)?,
            Mix::Gathered(_) => a.reshape(&[seqs, VIT_WINDOW, VIT_WINDOW, VIT_DIM])?,
        };
        let x = be.residual_add_forward(x, &a)?;
        let m = be.layer_norm_forward(&x, &blk.norm2.w, &blk.norm2.b, VIT_LN_EPS)?;
        let m = self.trunk_linear(&m, &blk.fc1)?;
        let m = be.gelu_erf_forward(&m)?;
        let m = self.trunk_linear(&m, &blk.fc2)?;
        be.residual_add_forward(&x, &m)
    }

    /// One neck over trunk features `[1, G, G, 1024]` (NHWC, `G` the token grid, 72 by default):
    /// levels at `4G`, `2G` and `G` squared (288, 144, 72), each `[1, 256, h, w]` (NCHW).
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

    /// `compute_axial_cis` written out in f64: pair `j` of the first 16 rotates by `col * scale * f_j`,
    /// pair `16 + j` by `row * scale * f_j`, `f_j = 10000^(-4j / 64)`.
    fn expected_angle(grid: usize, token: usize, pair: usize, scale: f64) -> f64 {
        let (col, row) = ((token % grid) as f64, (token / grid) as f64);
        let j = pair % 16;
        let f = 10000f64.powf(-(4.0 * j as f64) / HEAD_DIM as f64);
        (if pair < 16 { col } else { row }) * scale * f
    }

    #[test]
    fn generated_rope_tables_follow_compute_axial_cis_at_every_grid() {
        let budget = Budget::new(1 << 30);
        // Windowed (24, scale 1), the native global grid (72, scale 1/3) and the 672 grid (48, scale 1/2).
        for (grid, scale) in [(24usize, 1.0f64), (72, 1.0 / 3.0), (48, 0.5), (96, 0.25)] {
            let (cos, sin) = vit_rope_tables(grid, &budget).unwrap();
            assert_eq!(cos.shape(), &[grid * grid, VIT_ROPE_PAIRS], "grid {grid}");
            let (c, s) = (cos.to_host(&budget).unwrap().to_f32_vec().unwrap(), sin.to_host(&budget).unwrap().to_f32_vec().unwrap());
            let mut worst = 0f64;
            for t in (0..grid * grid).step_by(7).chain([grid * grid - 1]) {
                for p in 0..VIT_ROPE_PAIRS {
                    let a = expected_angle(grid, t, p, scale);
                    worst = worst.max((f64::from(c[t * VIT_ROPE_PAIRS + p]) - a.cos()).abs()).max((f64::from(s[t * VIT_ROPE_PAIRS + p]) - a.sin()).abs());
                }
            }
            // f32 angles (as torch forms them) against the f64 formula: a few 1e-6 at the largest angles.
            assert!(worst < 2e-5, "grid {grid}: max abs {worst:e}");
        }
        assert!(vit_rope_tables(0, &budget).is_err());
    }

    #[test]
    fn a_global_block_spans_the_same_positions_at_every_grid() {
        // Scale 24 / grid puts the last column at 24 * (grid - 1) / grid, so tokens one window
        // apart (24 tokens at grid 72, 16 at grid 48) share angles only through the scale: the
        // position of token `col` is `col * 24 / grid` at every grid.
        let budget = Budget::new(1 << 30);
        let at = |grid: usize| {
            let (c, s) = vit_rope_tables(grid, &budget).unwrap();
            (c.to_host(&budget).unwrap().to_f32_vec().unwrap(), s.to_host(&budget).unwrap().to_f32_vec().unwrap())
        };
        let (c72, s72) = at(72);
        let (c48, s48) = at(48);
        // Column 3 of row 0 at grid 72 sits at 1.0 (3 * 24 / 72); column 2 at grid 48 sits at 1.0 too.
        for p in 0..VIT_ROPE_PAIRS {
            assert!((c72[3 * VIT_ROPE_PAIRS + p] - c48[2 * VIT_ROPE_PAIRS + p]).abs() < 1e-6, "cos pair {p}");
            assert!((s72[3 * VIT_ROPE_PAIRS + p] - s48[2 * VIT_ROPE_PAIRS + p]).abs() < 1e-6, "sin pair {p}");
        }
    }

    #[test]
    fn the_position_embedding_tiles_at_any_whole_number_of_windows() {
        let pos = Weight { name: "pos_embed".into(), shape: vec![1, 577, VIT_DIM], data: (0..577 * VIT_DIM).map(|i| (i % 8191) as f32).collect() };
        let native = tiled_pos(&pos, 72);
        assert_eq!(native.len(), 72 * 72 * VIT_DIM);
        for grid in [24usize, 48] {
            let t = tiled_pos(&pos, grid);
            assert_eq!(t.len(), grid * grid * VIT_DIM);
            // Each row of the smaller grid is the same row of the native one: the tile is
            // anchored at the origin and the grid is cropped, never resampled.
            for y in 0..grid {
                assert_eq!(t[y * grid * VIT_DIM..(y + 1) * grid * VIT_DIM], native[y * 72 * VIT_DIM..y * 72 * VIT_DIM + grid * VIT_DIM], "grid {grid} row {y}");
            }
        }
        // 336 is the pretrain grid: the stored embedding without its cls row.
        assert_eq!(tiled_pos(&pos, 24), pos.data[VIT_DIM..]);
    }

    #[test]
    fn stats_follow_the_grid_they_are_asked_about() {
        let g48 = Geometry::from_image_side(672).unwrap();
        let full = GatedStats { mode: GatedMode::Full, window_blocks_run: 112, window_blocks_total: 112, global_blocks_run: 4, global_blocks_total: 4 };
        assert_eq!((full.flop_fraction_at(g48), full.block_fraction_at(g48)), (1.0, 1.0));
        // Half the windows of a 2 x 2 grid: 2 of 4 window-blocks per windowed block.
        let half = GatedStats { window_blocks_run: 56, global_blocks_run: 4, ..full };
        let want = (56 + 4 * 4) as f64 / (112 + 4 * 4) as f64;
        assert!((half.block_fraction_at(g48) - want).abs() < 1e-12);
        // The 1008 default is not the 672 answer.
        assert!((half.block_fraction() - half.block_fraction_at(g48)).abs() > 1e-3);
        // A global block is 2304 tokens at 672, 5184 at 1008: the attention term shrinks quadratically.
        let one_global = GatedStats { window_blocks_run: 0, global_blocks_run: 1, ..full };
        assert!(one_global.flops_at(g48) < one_global.flops() / 2.0);
    }

    #[test]
    fn gather_and_scatter_follow_the_grid_of_the_tensor() {
        let be = CpuBackend::new(Budget::new(2 << 30));
        let grid = 48;
        let data: Vec<f32> = (0..grid * grid * VIT_DIM).map(|i| (i % 4093) as f32).collect();
        let x = upload(&be, &data, &[1, grid, grid, VIT_DIM]).unwrap();
        let all = host_f32(&be, &window_partition(&be, &x, VIT_WINDOW).unwrap()).unwrap();
        let per = VIT_WINDOW * VIT_WINDOW * VIT_DIM;
        let set = WindowSet::from_indices(2, &[3, 1]);
        let sub = gather_windows(&be, &x, &set).unwrap();
        let sub_host = host_f32(&be, &sub).unwrap();
        for (j, w) in [1usize, 3].into_iter().enumerate() {
            assert_eq!(sub_host[j * per..(j + 1) * per], all[w * per..(w + 1) * per], "window {w}");
        }
        assert_eq!(host_f32(&be, &scatter_windows(&be, &x, &sub, &set).unwrap()).unwrap(), data);
        // A 3 x 3 window set does not describe a 2 x 2 window grid.
        assert!(gather_windows(&be, &x, &WindowSet::from_indices(3, &[0])).is_err());
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
