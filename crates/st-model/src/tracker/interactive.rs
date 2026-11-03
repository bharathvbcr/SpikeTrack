//! The tracker's interactive (per-object, non-multiplexed) SAM mask decoder
//! (`sam/mask_decoder.py` `MaskDecoder` as `_build_sam_heads` builds it):
//! one object-score token, one IoU token, 4 mask tokens (1 + 3 multimask),
//! the prompt's sparse tokens, the image repeated per object with the dense
//! prompt added. In VOS it runs once, on the conditioning frame, to derive
//! each object's pointer from its given mask.

use ojas_core::{Backend, ConvTranspose2dParams, OjasError, Tensor};
use st_weights::TensorGroup;

use crate::host;
use crate::layers::{get, host_slice, up, Mlp};
use crate::tracker::mask_decoder::{checked_feat, TwoWay};
use crate::ModelError;

pub const D: usize = 256;
pub const MASK_TOKENS: usize = 4;
const LN2D_EPS: f32 = 1e-6;

pub struct InteractiveOut {
    /// `[N, 4, M, M]`, `M` = 4 x the feature grid (288 at 1008).
    pub masks: Tensor,
    /// `[N, 4]`, host.
    pub iou_pred: Vec<f32>,
    /// `[N, 4, D]` mask tokens, host.
    pub mask_tokens: Vec<f32>,
    /// `[N, 1]`, host.
    pub object_score_logits: Vec<f32>,
}

pub struct InteractiveDecoder {
    transformer: TwoWay,
    out_tokens: Vec<f32>,
    dc1: (Tensor, Tensor),
    ln1: (Tensor, Tensor),
    dc2: (Tensor, Tensor),
    hyper: Vec<Mlp>,
    iou_head: Mlp,
    obj_score_head: Mlp,
}

impl InteractiveDecoder {
    pub fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        let t = |name: &str, shape: &[usize]| -> Result<Tensor, ModelError> {
            Ok(up(be, &get(g, &format!("{p}.{name}"), shape)?.data, shape)?)
        };
        // [obj_score; iou; mask x4], the order predict_masks concatenates them in.
        let mut out_tokens = get(g, &format!("{p}.obj_score_token.weight"), &[1, D])?
            .data
            .clone();
        out_tokens.extend_from_slice(&get(g, &format!("{p}.iou_token.weight"), &[1, D])?.data);
        out_tokens.extend_from_slice(
            &get(g, &format!("{p}.mask_tokens.weight"), &[MASK_TOKENS, D])?.data,
        );
        Ok(Self {
            transformer: TwoWay::load(be, g, &format!("{p}.transformer"))?,
            out_tokens,
            dc1: (
                t("output_upscaling.0.weight", &[D, D / 4, 2, 2])?,
                t("output_upscaling.0.bias", &[D / 4])?,
            ),
            ln1: (
                t("output_upscaling.1.weight", &[D / 4])?,
                t("output_upscaling.1.bias", &[D / 4])?,
            ),
            dc2: (
                t("output_upscaling.3.weight", &[D / 4, D / 8, 2, 2])?,
                t("output_upscaling.3.bias", &[D / 8])?,
            ),
            hyper: (0..MASK_TOKENS)
                .map(|i| {
                    Mlp::load_host(
                        be,
                        g,
                        &format!("{p}.output_hypernetworks_mlps.{i}"),
                        &[D, D, D, D / 8],
                        false,
                        false,
                    )
                })
                .collect::<Result<_, _>>()?,
            iou_head: Mlp::load_host(
                be,
                g,
                &format!("{p}.iou_prediction_head"),
                &[D, D, D, MASK_TOKENS],
                false,
                false,
            )?,
            obj_score_head: Mlp::load_host(
                be,
                g,
                &format!("{p}.pred_obj_score_head"),
                &[D, D, D, 1],
                false,
                false,
            )?,
        })
    }

    /// `image` `[1, D, G, G]` (repeated per object), `image_pe` `[1, D, G, G]`,
    /// `sparse` `[N, S, D]` host, `dense` `[N, D, G, G]`, high-res skips
    /// `[1, 32, 4G, 4G]` and `[1, 64, 2G, 2G]` (repeated per object). `G` is the token grid
    /// (72 at 1008), read from `image`.
    pub fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        image: &Tensor,
        image_pe: &Tensor,
        sparse: &[f32],
        n_sparse: usize,
        dense: &Tensor,
        s0: &Tensor,
        s1: &Tensor,
    ) -> Result<InteractiveOut, OjasError> {
        let n = dense.shape()[0];
        let feat = checked_feat("st_model::InteractiveDecoder::fwd", image, image_pe, s0, s1)?;
        if dense.shape()[1..] != [D, feat, feat] {
            return Err(OjasError::OutOfRange {
                op: "st_model::InteractiveDecoder::fwd",
                detail: format!("dense {:?} is not [N, {D}, {feat}, {feat}]", dense.shape()),
            });
        }
        let hw = feat * feat;
        let ntok = 2 + MASK_TOKENS + n_sparse;
        let mut tok = Vec::with_capacity(n * ntok * D);
        for i in 0..n {
            tok.extend_from_slice(&self.out_tokens);
            tok.extend_from_slice(&sparse[i * n_sparse * D..(i + 1) * n_sparse * D]);
        }
        let tokens = up(be, &tok, &[n, ntok, D])?;
        let rep = |x: &Tensor| -> Result<Tensor, OjasError> {
            if n == 1 {
                return Ok(x.clone());
            }
            let mut s = vec![1; x.shape().len()];
            s[0] = n;
            let zeros = be.fill(&s, ojas_core::DType::F32, 0.0)?;
            be.broadcast_add_forward(x, &zeros)
        };
        let src = be.residual_add_forward(&rep(image)?, dense)?;
        let src_tok = be.permute(&src.reshape(&[n, D, hw])?, &[0, 2, 1])?;
        let pe = be.permute(&rep(image_pe)?.reshape(&[n, D, hw])?, &[0, 2, 1])?;
        let (hs_dev, keys) = self.transformer.fwd(be, &src_tok, &pe, &tokens)?;
        drop((src_tok, pe, tokens));

        let src = be
            .permute(&keys, &[0, 2, 1])?
            .reshape(&[n, D, feat, feat])?;
        drop(keys);
        let mut up2 = ConvTranspose2dParams::new();
        up2.stride = [2, 2];
        let x = be.residual_add_forward(
            &be.conv_transpose2d_forward(&src, &self.dc1.0, Some(&self.dc1.1), up2)?,
            &rep(s1)?,
        )?;
        drop(src);
        let x = be.gelu_erf_forward(&be.channel_layer_norm_forward(
            &x,
            &self.ln1.0,
            &self.ln1.1,
            LN2D_EPS,
        )?)?;
        let x = be.residual_add_forward(
            &be.conv_transpose2d_forward(&x, &self.dc2.0, Some(&self.dc2.1), up2)?,
            &rep(s0)?,
        )?;
        let upscaled_dev = be.gelu_erf_forward(&x)?; // [N, 32, 4G, 4G]
        drop(x);

        let hs_view = host_slice(&hs_dev, be)?;
        drop(hs_dev);
        let hs = hs_view.as_slice();
        let row = |i: usize, t: usize| &hs[(i * ntok + t) * D..(i * ntok + t + 1) * D];
        let ch = D / 8;
        let mut mask_tokens = Vec::with_capacity(n * MASK_TOKENS * D);
        for i in 0..n {
            for m in 0..MASK_TOKENS {
                mask_tokens.extend_from_slice(row(i, 2 + m));
            }
        }
        let mut hyper_all = vec![0f32; n * MASK_TOKENS * ch];
        let mut m_in = Vec::with_capacity(n * D);
        for m in 0..MASK_TOKENS {
            m_in.clear();
            for i in 0..n {
                m_in.extend_from_slice(row(i, 2 + m));
            }
            let h_m = self.hyper[m].host(&m_in, n);
            for i in 0..n {
                hyper_all[(i * MASK_TOKENS + m) * ch..(i * MASK_TOKENS + m + 1) * ch]
                    .copy_from_slice(&h_m[i * ch..(i + 1) * ch]);
            }
        }
        let mut iou_in = Vec::with_capacity(n * D);
        let mut obj_in = Vec::with_capacity(n * D);
        for i in 0..n {
            iou_in.extend_from_slice(row(i, 1));
            obj_in.extend_from_slice(row(i, 0));
        }
        drop(hs_view);
        let iou_pred = self.iou_head.host(&iou_in, n);
        let object_score_logits = self.obj_score_head.host(&obj_in, n);

        let upscaled_view = host_slice(&upscaled_dev, be)?;
        drop(upscaled_dev);
        let upscaled = upscaled_view.as_slice();
        let low_res = 4 * feat;
        let px = low_res * low_res;
        let mut masks = vec![0f32; n * MASK_TOKENS * px];
        host::for_each_row_chunk_mut(&mut masks, px, 1, |im0, piece| {
            for (k, out) in piece.chunks_exact_mut(px).enumerate() {
                let im = im0 + k;
                let i = im / MASK_TOKENS;
                let emb = &upscaled[i * ch * px..(i + 1) * ch * px];
                let hyper = &hyper_all[im * ch..(im + 1) * ch];
                for (c, &hv) in hyper.iter().enumerate() {
                    let plane = &emb[c * px..(c + 1) * px];
                    out.iter_mut().zip(plane).for_each(|(o, &e)| *o += hv * e);
                }
            }
        });
        drop(upscaled_view);
        Ok(InteractiveOut {
            masks: up(be, &masks, &[n, MASK_TOKENS, low_res, low_res])?,
            iou_pred,
            mask_tokens,
            object_score_logits,
        })
    }
}
