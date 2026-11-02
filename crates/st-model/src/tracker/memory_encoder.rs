//! SAM 3.1's multiplex memory encoder (`memory.py` `SimpleMaskEncoder` with the
//! multiplex `SimpleMaskDownSampler`, as `_create_multiplex_maskmem_backbone`
//! builds it).
//!
//! Masks `[B, 32, S, S]` (16 muxed mask channels then 16 muxed "conditioned"
//! flags; `S` = 1008 at the native size) are resized to `16 * grid` (1152;
//! antialiased bilinear; for this upscale identical to plain bilinear), downsampled 16x by four stride-2 3x3
//! convs (32 -> 16 -> 64 -> 256 -> 1024, each LayerNorm2d + GELU) and a 1x1 to
//! 256, added to `pix_feat_proj(pix_feat)`, then two ConvNeXt blocks.

use ojas_core::{Backend, Conv2dParams, OjasError, Tensor};
use st_gate::Geometry;
use st_weights::TensorGroup;

use crate::host;
use crate::layers::{get, up, Conv, Lin};
use crate::ModelError;

pub const D: usize = 256;
const LN2D_EPS: f32 = 1e-6;

struct CxBlock {
    dwconv: Conv,
    norm: (Tensor, Tensor),
    pw1: Lin,
    pw2: Lin,
    gamma: Tensor,
}

pub struct MemoryEncoder {
    down: Vec<(Conv, (Tensor, Tensor))>,
    down_out: Conv,
    pix_feat_proj: Conv,
    fuser: Vec<CxBlock>,
    /// The side masks are resized to before the downsampler (1152 at 1008): 16 x the grid.
    interp: usize,
    /// Sine position encoding of the grid x grid map, `[1, 256, grid, grid]`.
    pos: Tensor,
}

/// What the encoder returns for one call.
pub struct MemoryOut {
    /// `[B, 256, grid, grid]` (72 at 1008).
    pub features: Tensor,
    /// `[1, 256, grid, grid]` (the same for every bucket).
    pub pos: Tensor,
}

impl MemoryEncoder {
    /// The encoder for the native 1008 x 1008 input.
    pub fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        Self::load_at(be, g, p, Geometry::SAM3)
    }

    /// The encoder for the input size `geo`. The four stride-2 convolutions of the downsampler
    /// reduce the resized mask 16x, so it must be resized to `16 * grid`
    /// ([`Geometry::input_mask`]: upstream's `interpol_size=[1152, 1152]` at 1008) to land on the
    /// feature map it is added to; no weight depends on it.
    pub fn load_at<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        geo: Geometry,
    ) -> Result<Self, ModelError> {
        let vec = |name: &str, n: usize| -> Result<Tensor, ModelError> {
            Ok(up(be, &get(g, &format!("{p}.{name}"), &[n])?.data, &[n])?)
        };
        let chans = [(32usize, 16usize), (16, 64), (64, 256), (256, 1024)];
        let down = chans
            .iter()
            .enumerate()
            .map(|(i, &(cin, cout))| -> Result<_, ModelError> {
                let conv = Conv::load(
                    be,
                    g,
                    &format!("{p}.mask_downsampler.encoder.{}", 3 * i),
                    [cout, cin, 3, 3],
                )?;
                let ln = (
                    vec(
                        &format!("mask_downsampler.encoder.{}.weight", 3 * i + 1),
                        cout,
                    )?,
                    vec(
                        &format!("mask_downsampler.encoder.{}.bias", 3 * i + 1),
                        cout,
                    )?,
                );
                Ok((conv, ln))
            })
            .collect::<Result<_, _>>()?;
        let fuser = (0..2)
            .map(|i| -> Result<CxBlock, ModelError> {
                let f = format!("{p}.fuser.layers.{i}");
                let gamma = get(g, &format!("{f}.gamma"), &[D])?;
                Ok(CxBlock {
                    dwconv: Conv::load(be, g, &format!("{f}.dwconv"), [D, 1, 7, 7])?,
                    norm: (
                        vec(&format!("fuser.layers.{i}.norm.weight"), D)?,
                        vec(&format!("fuser.layers.{i}.norm.bias"), D)?,
                    ),
                    pw1: Lin::load(be, g, &format!("{f}.pwconv1"), 4 * D, D)?,
                    pw2: Lin::load(be, g, &format!("{f}.pwconv2"), D, 4 * D)?,
                    gamma: up(be, &gamma.data, &[D])?,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            down,
            down_out: Conv::load(
                be,
                g,
                &format!("{p}.mask_downsampler.encoder.12"),
                [D, 1024, 1, 1],
            )?,
            pix_feat_proj: Conv::load(be, g, &format!("{p}.pix_feat_proj"), [D, D, 1, 1])?,
            fuser,
            interp: geo.input_mask(),
            pos: up(
                be,
                &host::sine_position_map(geo.grid, geo.grid),
                &[1, D, geo.grid, geo.grid],
            )?,
        })
    }

    /// `pix_feat` `[B, 256, grid, grid]`, `masks` `[B, 32, H, W]` (already sigmoid-
    /// scaled and muxed, with the conditioning channels; upstream passes
    /// `skip_mask_sigmoid=True`).
    pub fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        pix_feat: &Tensor,
        masks: &Tensor,
    ) -> Result<MemoryOut, OjasError> {
        let interp = self.interp;
        let mut x = if masks.shape()[2..] == [interp, interp] {
            masks.clone()
        } else {
            be.interpolate_bilinear_forward(masks, [interp, interp], false)?
        };
        let mut s2 = Conv2dParams::new();
        s2.stride = [2, 2];
        s2.padding = [1, 1];
        for (conv, ln) in &self.down {
            x = be.conv2d_forward(&x, &conv.w, Some(&conv.b), s2)?;
            x = be.gelu_erf_forward(&be.channel_layer_norm_forward(&x, &ln.0, &ln.1, LN2D_EPS)?)?;
        }
        let masks = self.down_out.fwd(be, &x, 0)?;
        let mut x = be.residual_add_forward(&self.pix_feat_proj.fwd(be, pix_feat, 0)?, &masks)?;
        for blk in &self.fuser {
            let mut dw = Conv2dParams::new();
            dw.padding = [3, 3];
            dw.groups = D;
            let y = be.conv2d_forward(&x, &blk.dwconv.w, Some(&blk.dwconv.b), dw)?;
            let y = be.channel_layer_norm_forward(&y, &blk.norm.0, &blk.norm.1, LN2D_EPS)?;
            let y = be.permute(&y, &[0, 2, 3, 1])?; // NHWC
            let y = blk
                .pw2
                .fwd(be, &be.gelu_erf_forward(&blk.pw1.fwd(be, &y)?)?)?;
            let y = be.broadcast_mul_forward(&y, &blk.gamma)?;
            let y = be.permute(&y, &[0, 3, 1, 2])?;
            x = be.residual_add_forward(&x, &y)?;
        }
        Ok(MemoryOut {
            features: x,
            pos: self.pos.clone(),
        })
    }
}
