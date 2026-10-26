//! The tracker's interactive (per-object, non-multiplexed) SAM mask decoder
//! (`sam/mask_decoder.py` `MaskDecoder` as `_build_sam_heads` builds it):
//! one object-score token, one IoU token, 4 mask tokens (1 + 3 multimask),
//! the prompt's sparse tokens, the image repeated per object with the dense
//! prompt added. In VOS it runs once, on the conditioning frame, to derive
//! each object's pointer from its given mask.

use ojas_core::{Backend, ConvTranspose2dParams, OjasError, Tensor};
use st_weights::TensorGroup;

use crate::layers::{get, up, Mlp};
use crate::tracker::mask_decoder::TwoWay;
use crate::ModelError;

pub const D: usize = 256;
pub const MASK_TOKENS: usize = 4;
pub const FEAT: usize = 72;
const LN2D_EPS: f32 = 1e-6;

pub struct InteractiveOut {
    /// `[N, 4, 288, 288]`.
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
        let t = |name: &str, shape: &[usize]| -> Result<Tensor, ModelError> { Ok(up(be, &get(g, &format!("{p}.{name}"), shape)?.data, shape)?) };
        // [obj_score; iou; mask x4], the order predict_masks concatenates them in.
        let mut out_tokens = get(g, &format!("{p}.obj_score_token.weight"), &[1, D])?.data.clone();
        out_tokens.extend_from_slice(&get(g, &format!("{p}.iou_token.weight"), &[1, D])?.data);
        out_tokens.extend_from_slice(&get(g, &format!("{p}.mask_tokens.weight"), &[MASK_TOKENS, D])?.data);
        Ok(Self {
            transformer: TwoWay::load(be, g, &format!("{p}.transformer"))?,
            out_tokens,
            dc1: (t("output_upscaling.0.weight", &[D, D / 4, 2, 2])?, t("output_upscaling.0.bias", &[D / 4])?),
            ln1: (t("output_upscaling.1.weight", &[D / 4])?, t("output_upscaling.1.bias", &[D / 4])?),
            dc2: (t("output_upscaling.3.weight", &[D / 4, D / 8, 2, 2])?, t("output_upscaling.3.bias", &[D / 8])?),
            hyper: (0..MASK_TOKENS)
                .map(|i| Mlp::load(be, g, &format!("{p}.output_hypernetworks_mlps.{i}"), &[D, D, D, D / 8], false, false))
                .collect::<Result<_, _>>()?,
            iou_head: Mlp::load(be, g, &format!("{p}.iou_prediction_head"), &[D, D, D, MASK_TOKENS], false, false)?,
            obj_score_head: Mlp::load(be, g, &format!("{p}.pred_obj_score_head"), &[D, D, D, 1], false, false)?,
        })
    }

    /// `image` `[1, D, 72, 72]` (repeated per object), `image_pe` `[1, D, 72, 72]`,
    /// `sparse` `[N, S, D]` host, `dense` `[N, D, 72, 72]`, high-res skips
    /// `[1, 32, 288, 288]` and `[1, 64, 144, 144]` (repeated per object).
    pub fn fwd<B: Backend + ?Sized>(&self, be: &B, image: &Tensor, image_pe: &Tensor, sparse: &[f32], n_sparse: usize, dense: &Tensor, s0: &Tensor, s1: &Tensor) -> Result<InteractiveOut, OjasError> {
        let n = dense.shape()[0];
        let hw = FEAT * FEAT;
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
            let h = x.to_host(be.budget())?.to_f32_vec()?;
            let mut s = x.shape().to_vec();
            s[0] = n;
            up(be, &h.repeat(n), &s)
        };
        let src = be.residual_add_forward(&rep(image)?, dense)?;
        let src_tok = be.permute(&src.reshape(&[n, D, hw])?, &[0, 2, 1])?;
        let pe = be.permute(&rep(image_pe)?.reshape(&[n, D, hw])?, &[0, 2, 1])?;
        let (hs, keys) = self.transformer.fwd(be, &src_tok, &pe, &tokens)?;
        let hs = hs.to_host(be.budget())?.to_f32_vec()?;

        let src = be.permute(&keys, &[0, 2, 1])?.reshape(&[n, D, FEAT, FEAT])?;
        let mut up2 = ConvTranspose2dParams::new();
        up2.stride = [2, 2];
        let x = be.residual_add_forward(&be.conv_transpose2d_forward(&src, &self.dc1.0, Some(&self.dc1.1), up2)?, &rep(s1)?)?;
        let x = be.gelu_erf_forward(&be.channel_layer_norm_forward(&x, &self.ln1.0, &self.ln1.1, LN2D_EPS)?)?;
        let x = be.residual_add_forward(&be.conv_transpose2d_forward(&x, &self.dc2.0, Some(&self.dc2.1), up2)?, &rep(s0)?)?;
        let upscaled = be.gelu_erf_forward(&x)?.to_host(be.budget())?.to_f32_vec()?; // [N, 32, 288, 288]

        let row = |i: usize, t: usize| &hs[(i * ntok + t) * D..(i * ntok + t + 1) * D];
        let px = 288 * 288;
        let mut masks = Vec::with_capacity(n * MASK_TOKENS * px);
        let (mut iou_pred, mut mask_tokens, mut object_score_logits) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let emb = &upscaled[i * (D / 8) * px..(i + 1) * (D / 8) * px];
            for m in 0..MASK_TOKENS {
                let t = row(i, 2 + m);
                mask_tokens.extend_from_slice(t);
                let hyper = self.hyper[m].host(t, 1); // [32]
                let mut out = vec![0f32; px];
                for (c, &hv) in hyper.iter().enumerate() {
                    let plane = &emb[c * px..(c + 1) * px];
                    out.iter_mut().zip(plane).for_each(|(o, &e)| *o += hv * e);
                }
                masks.extend(out);
            }
            iou_pred.extend(self.iou_head.host(row(i, 1), 1));
            object_score_logits.extend(self.obj_score_head.host(row(i, 0), 1));
        }
        Ok(InteractiveOut { masks: up(be, &masks, &[n, MASK_TOKENS, 288, 288])?, iou_pred, mask_tokens, object_score_logits })
    }
}
