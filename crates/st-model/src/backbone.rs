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
        let be = self.be;
        let mut p = Conv2dParams::new();
        p.stride = [VIT_PATCH, VIT_PATCH];
        let pe = be.conv2d_forward(image, &self.patch_embed, None, p)?; // [1, 1024, 72, 72]
        let pe = be.permute(&pe, &[0, 2, 3, 1])?; // NHWC
        let x = be.residual_add_forward(&pe, &self.pos_tiled)?;
        let mut x = be.layer_norm_forward(&x, &self.ln_pre.w, &self.ln_pre.b, LN_PRE_EPS)?;
        taps.patch_embed = Some(pe);
        taps.ln_pre = Some(x.clone());
        taps.blocks = vec![None; self.blocks.len()];
        for (i, blk) in self.blocks.iter().enumerate() {
            x = self.block_forward(blk, &x)?;
            if keep_blocks.contains(&i) {
                taps.blocks[i] = Some(x.clone());
            }
        }
        Ok(x)
    }

    fn block_forward(&self, blk: &DevBlock, x: &Tensor) -> Result<Tensor, OjasError> {
        let be = self.be;
        let h = be.layer_norm_forward(x, &blk.norm1.w, &blk.norm1.b, BLOCK_LN_EPS)?;
        // Windowed blocks attend within each 24x24 window: a batch of 9 x 576.
        let (seqs, tokens) = if blk.global {
            (1, VIT_GRID * VIT_GRID)
        } else {
            let n = VIT_GRID / VIT_WINDOW;
            (n * n, VIT_WINDOW * VIT_WINDOW)
        };
        let h = if blk.global { h } else { window_partition(be, &h, VIT_WINDOW)? };
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
        let a = if blk.global {
            a.reshape(&[1, VIT_GRID, VIT_GRID, VIT_DIM])?
        } else {
            window_merge(be, &a.reshape(&[seqs, VIT_WINDOW, VIT_WINDOW, VIT_DIM])?, VIT_GRID, VIT_GRID)?
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
