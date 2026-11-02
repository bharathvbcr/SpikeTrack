//! SAM 3.1's multiplex mask decoder (`multiplex_mask_decoder.py`,
//! `MultiplexMaskDecoder` as `_build_sam_heads` builds it) and the SAM two-way
//! transformer inside it (`sam/transformer.py`).
//!
//! Configuration, read from the checkpoint shapes: 16 multiplexed objects per
//! bucket, 3 masks per object (`multimask_outputs_only`; 48 mask tokens), one
//! IoU and one object-score token per object, high-res features from
//! `conv_s0`/`conv_s1`, depth-2 two-way transformer (ReLU MLP 2048,
//! token-to-image and image-to-token attention at width 128).

use ojas_core::{Backend, ConvTranspose2dParams, OjasError, Tensor};
use st_weights::TensorGroup;

use crate::layers::{get, up, Lin, Mha, Mlp, Norm};
use crate::ModelError;

pub const D: usize = 256;
pub const MUX: usize = 16;
pub const MASKS_PER_OBJ: usize = 3;
const LN2D_EPS: f32 = 1e-6;

struct TwoWayBlock {
    self_attn: Mha,
    norm1: Norm,
    t2i: Mha,
    norm2: Norm,
    mlp: (Lin, Lin),
    norm3: Norm,
    norm4: Norm,
    i2t: Mha,
    skip_first_pe: bool,
}

/// `TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)`.
pub struct TwoWay {
    blocks: Vec<TwoWayBlock>,
    final_t2i: Mha,
    norm_final: Norm,
}

impl TwoWay {
    pub(crate) fn load<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
    ) -> Result<Self, ModelError> {
        let blocks = (0..2)
            .map(|i| {
                let b = format!("{p}.layers.{i}");
                Ok(TwoWayBlock {
                    self_attn: Mha::load_split(be, g, &format!("{b}.self_attn"), D, D, 8)?,
                    norm1: Norm::load(be, g, &format!("{b}.norm1"))?,
                    t2i: Mha::load_split(
                        be,
                        g,
                        &format!("{b}.cross_attn_token_to_image"),
                        D,
                        D / 2,
                        8,
                    )?,
                    norm2: Norm::load(be, g, &format!("{b}.norm2"))?,
                    mlp: (
                        Lin::load(be, g, &format!("{b}.mlp.lin1"), 2048, D)?,
                        Lin::load(be, g, &format!("{b}.mlp.lin2"), D, 2048)?,
                    ),
                    norm3: Norm::load(be, g, &format!("{b}.norm3"))?,
                    norm4: Norm::load(be, g, &format!("{b}.norm4"))?,
                    i2t: Mha::load_split(
                        be,
                        g,
                        &format!("{b}.cross_attn_image_to_token"),
                        D,
                        D / 2,
                        8,
                    )?,
                    skip_first_pe: i == 0,
                })
            })
            .collect::<Result<_, ModelError>>()?;
        Ok(Self {
            blocks,
            final_t2i: Mha::load_split(
                be,
                g,
                &format!("{p}.final_attn_token_to_image"),
                D,
                D / 2,
                8,
            )?,
            norm_final: Norm::load(be, g, &format!("{p}.norm_final_attn"))?,
        })
    }

    /// `image` and `image_pe` `[B, HW, D]` (tokens), `tokens` `[B, N, D]`.
    /// Returns `(queries [B, N, D], keys [B, HW, D])`.
    pub(crate) fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        image: &Tensor,
        image_pe: &Tensor,
        tokens: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let add = |a: &Tensor, b: &Tensor| be.residual_add_forward(a, b);
        let (mut q, mut k) = (tokens.clone(), image.clone());
        for blk in &self.blocks {
            q = if blk.skip_first_pe {
                blk.self_attn.fwd(be, &q, &q, &q, None)?
            } else {
                let qp = add(&q, tokens)?;
                add(&q, &blk.self_attn.fwd(be, &qp, &qp, &q, None)?)?
            };
            q = blk.norm1.fwd(be, &q)?;
            let (qp, kp) = (add(&q, tokens)?, add(&k, image_pe)?);
            q = blk
                .norm2
                .fwd(be, &add(&q, &blk.t2i.fwd(be, &qp, &kp, &k, None)?)?)?;
            let m = blk
                .mlp
                .1
                .fwd(be, &be.relu_forward(&blk.mlp.0.fwd(be, &q)?)?)?;
            q = blk.norm3.fwd(be, &add(&q, &m)?)?;
            let (qp, kp) = (add(&q, tokens)?, add(&k, image_pe)?);
            k = blk
                .norm4
                .fwd(be, &add(&k, &blk.i2t.fwd(be, &kp, &qp, &q, None)?)?)?;
        }
        let (qp, kp) = (add(&q, tokens)?, add(&k, image_pe)?);
        let q = self
            .norm_final
            .fwd(be, &add(&q, &self.final_t2i.fwd(be, &qp, &kp, &k, None)?)?)?;
        Ok((q, k))
    }
}

/// Outputs of one multiplexed decode for `B` buckets of `MUX` objects.
pub struct MuxDecoderOut {
    /// `[B, MUX, 3, M, M]` mask logits, `M` = 4 x the feature grid (288 at 1008).
    pub masks: Tensor,
    /// `[B, MUX, 3]`, host.
    pub iou_pred: Vec<f32>,
    /// `[B, MUX, 1]`, host.
    pub object_score_logits: Vec<f32>,
    /// `[B, MUX, 3, D]` output mask tokens (the object pointers' source), host.
    pub sam_tokens: Vec<f32>,
}

pub struct MuxMaskDecoder {
    transformer: TwoWay,
    obj_score_token: Vec<f32>,
    iou_token: Vec<f32>,
    mask_tokens: Vec<f32>,
    dc1: (Tensor, Tensor),
    ln1: (Tensor, Tensor),
    dc2: (Tensor, Tensor),
    pub(crate) conv_s0: (Tensor, Tensor),
    pub(crate) conv_s1: (Tensor, Tensor),
    hyper: Vec<Mlp>,
    iou_head: Mlp,
    obj_score_head: Mlp,
}

impl MuxMaskDecoder {
    pub fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        let t = |name: &str, shape: &[usize]| -> Result<Tensor, ModelError> {
            Ok(up(be, &get(g, &format!("{p}.{name}"), shape)?.data, shape)?)
        };
        let host = |name: &str, shape: &[usize]| -> Result<Vec<f32>, ModelError> {
            Ok(get(g, &format!("{p}.{name}"), shape)?.data.clone())
        };
        Ok(Self {
            transformer: TwoWay::load(be, g, &format!("{p}.transformer"))?,
            obj_score_token: host("obj_score_token.weight", &[MUX, D])?,
            iou_token: host("iou_token.weight", &[MUX, D])?,
            mask_tokens: host("mask_tokens.weight", &[MUX * MASKS_PER_OBJ, D])?,
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
            conv_s0: (
                t("conv_s0.weight", &[D / 8, D, 1, 1])?,
                t("conv_s0.bias", &[D / 8])?,
            ),
            conv_s1: (
                t("conv_s1.weight", &[D / 4, D, 1, 1])?,
                t("conv_s1.bias", &[D / 4])?,
            ),
            hyper: (0..MASKS_PER_OBJ)
                .map(|i| {
                    Mlp::load(
                        be,
                        g,
                        &format!("{p}.output_hypernetworks_mlps.{i}"),
                        &[D, D, D, D / 8],
                        false,
                        false,
                    )
                })
                .collect::<Result<_, _>>()?,
            iou_head: Mlp::load(
                be,
                g,
                &format!("{p}.iou_prediction_head"),
                &[D, D, D, MASKS_PER_OBJ],
                false,
                false,
            )?,
            obj_score_head: Mlp::load(
                be,
                g,
                &format!("{p}.pred_obj_score_head"),
                &[D, D, D, 1],
                false,
                false,
            )?,
        })
    }

    /// `image` `[B, 256, G, G]`, `image_pe` `[1, 256, G, G]`, high-res
    /// features after `conv_s0`/`conv_s1` (`[B, 32, 4G, 4G]`, `[B, 64, 2G, 2G]`),
    /// per-slot embeddings `[B, MUX, D]` (valid/invalid output embeddings) on the host.
    /// `G` is the token grid (72 at 1008) and is read from `image`; the decoder has no
    /// resolution of its own.
    pub fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        image: &Tensor,
        image_pe: &Tensor,
        s0: &Tensor,
        s1: &Tensor,
        extra: &[f32],
    ) -> Result<MuxDecoderOut, OjasError> {
        let b = image.shape()[0];
        let feat = checked_feat("st_model::MuxMaskDecoder::fwd", image, image_pe, s0, s1)?;
        let (hw, low_res) = (feat * feat, 4 * feat);
        // Tokens per bucket: [obj_score x16; iou x16; mask x48 (+ the slot's extra embedding)].
        let ntok = 2 * MUX + MUX * MASKS_PER_OBJ;
        let mut tok = Vec::with_capacity(b * ntok * D);
        for bi in 0..b {
            tok.extend_from_slice(&self.obj_score_token);
            tok.extend_from_slice(&self.iou_token);
            for s in 0..MUX {
                let e = &extra[(bi * MUX + s) * D..(bi * MUX + s + 1) * D];
                for m in 0..MASKS_PER_OBJ {
                    let mt = &self.mask_tokens
                        [(s * MASKS_PER_OBJ + m) * D..(s * MASKS_PER_OBJ + m + 1) * D];
                    tok.extend(mt.iter().zip(e).map(|(a, c)| a + c));
                }
            }
        }
        let tokens = up(be, &tok, &[b, ntok, D])?;
        let img_tok = be.permute(&image.reshape(&[b, D, hw])?, &[0, 2, 1])?;
        let pe1 = be.permute(&image_pe.reshape(&[1, D, hw])?, &[0, 2, 1])?;
        let pe = if b == 1 {
            pe1
        } else {
            let zeros = be.fill(&[b, 1, 1], ojas_core::DType::F32, 0.0)?;
            be.broadcast_add_forward(&pe1, &zeros)?
        };
        let (hs, src) = self.transformer.fwd(be, &img_tok, &pe, &tokens)?;
        let hs_host = hs.to_host(be.budget())?.to_f32_vec()?;

        // Upscale the image tokens with the high-res skips.
        let src = be.permute(&src, &[0, 2, 1])?.reshape(&[b, D, feat, feat])?;
        let mut up2 = ConvTranspose2dParams::new();
        up2.stride = [2, 2];
        let x = be.conv_transpose2d_forward(&src, &self.dc1.0, Some(&self.dc1.1), up2)?;
        let x = be.residual_add_forward(&x, s1)?;
        let x = be.gelu_erf_forward(&be.channel_layer_norm_forward(
            &x,
            &self.ln1.0,
            &self.ln1.1,
            LN2D_EPS,
        )?)?;
        let x = be.conv_transpose2d_forward(&x, &self.dc2.0, Some(&self.dc2.1), up2)?;
        let upscaled = be.gelu_erf_forward(&be.residual_add_forward(&x, s0)?)?; // [B, 32, 4G, 4G]

        // Per-object hypernetwork vectors [B, MUX, 3, 32] from the mask tokens.
        let row =
            |bi: usize, tidx: usize| &hs_host[(bi * ntok + tidx) * D..(bi * ntok + tidx + 1) * D];
        let mut hyper_in = Vec::with_capacity(b * MUX * MASKS_PER_OBJ * (D / 8));
        let mut sam_tokens = Vec::with_capacity(b * MUX * MASKS_PER_OBJ * D);
        for bi in 0..b {
            for s in 0..MUX {
                for m in 0..MASKS_PER_OBJ {
                    let t = row(bi, 2 * MUX + s * MASKS_PER_OBJ + m);
                    hyper_in.extend(self.hyper[m].host(t, 1));
                    sam_tokens.extend_from_slice(t);
                }
            }
        }
        let n = low_res * low_res;
        let mut mask_tensors = Vec::with_capacity(b);
        for bi in 0..b {
            let hyper = Tensor::from_f32(
                &hyper_in
                    [bi * MUX * MASKS_PER_OBJ * (D / 8)..(bi + 1) * MUX * MASKS_PER_OBJ * (D / 8)],
                &[MUX * MASKS_PER_OBJ, D / 8],
                be.budget(),
            )?;
            let hyper = be.upload(&hyper)?;
            let emb = be.permute(&upscaled.narrow_batch(bi)?.reshape(&[D / 8, n])?, &[1, 0])?; // [n, 32]
            let m = be.linear_forward(&hyper, &emb)?; // [48, n]
            mask_tensors.push(m.reshape(&[1, MUX, MASKS_PER_OBJ, low_res, low_res])?);
        }
        let masks = if b == 1 {
            mask_tensors.pop().expect("mask tensor")
        } else {
            let refs: Vec<&Tensor> = mask_tensors.iter().collect();
            be.concat(&refs, 0)?
        };
        let mut iou_pred = Vec::with_capacity(b * MUX * MASKS_PER_OBJ);
        let mut object_score_logits = Vec::with_capacity(b * MUX);
        for bi in 0..b {
            for s in 0..MUX {
                iou_pred.extend(self.iou_head.host(row(bi, MUX + s), 1));
                object_score_logits.extend(self.obj_score_head.host(row(bi, s), 1));
            }
        }
        Ok(MuxDecoderOut {
            masks,
            iou_pred,
            object_score_logits,
            sam_tokens,
        })
    }
}

/// The token grid `G` of a decoder call, read from `image` `[B, 256, G, G]` and checked against
/// the positions `[1, 256, G, G]` and the two high-resolution skips (`4G` and `2G` squared):
/// the transformer's two upsampling steps are fixed 2x each, so any other pairing would only
/// fail later as an opaque shape error, or not at all.
pub(crate) fn checked_feat(
    op: &'static str,
    image: &Tensor,
    image_pe: &Tensor,
    s0: &Tensor,
    s1: &Tensor,
) -> Result<usize, OjasError> {
    let feat = image.shape().get(2).copied().unwrap_or(0);
    let ok = feat > 0
        && [image.shape().len(), s0.shape().len(), s1.shape().len()] == [4, 4, 4]
        && image.shape()[1..] == [D, feat, feat]
        && image_pe.shape() == [1, D, feat, feat]
        && s0.shape()[2..] == [4 * feat, 4 * feat]
        && s1.shape()[2..] == [2 * feat, 2 * feat];
    if !ok {
        return Err(OjasError::OutOfRange {
            op,
            detail: format!("image {:?}, image_pe {:?}, s0 {:?}, s1 {:?} do not share one token grid G (expected [B, {D}, G, G], [1, {D}, G, G], [B, _, 4G, 4G], [B, _, 2G, 2G])", image.shape(), image_pe.shape(), s0.shape(), s1.shape()),
        });
    }
    Ok(feat)
}

/// Per-bucket view of a `[B, ...]` tensor.
trait NarrowBatch {
    fn narrow_batch(&self, i: usize) -> Result<Tensor, OjasError>;
}

impl NarrowBatch for Tensor {
    fn narrow_batch(&self, i: usize) -> Result<Tensor, OjasError> {
        let shape = self.shape();
        if shape[0] == 1 && i == 0 {
            return Ok(self.clone());
        }
        let inner: usize = shape[1..].iter().product();
        let mut s = shape.to_vec();
        s[0] = 1;
        let strides: Vec<usize> = self.strides().to_vec();
        self.narrow(i * inner * 4, &s, &strides)
    }
}

/// `conv_s0`/`conv_s1` on the propagation neck's 288 and 144 levels (upstream
/// applies them in `forward_image` before tracking).
pub fn high_res<B: Backend + ?Sized>(
    be: &B,
    dec: &MuxMaskDecoder,
    fpn0: &Tensor,
    fpn1: &Tensor,
) -> Result<(Tensor, Tensor), OjasError> {
    let p = ojas_core::Conv2dParams::new();
    Ok((
        be.conv2d_forward(fpn0, &dec.conv_s0.0, Some(&dec.conv_s0.1), p)?,
        be.conv2d_forward(fpn1, &dec.conv_s1.0, Some(&dec.conv_s1.1), p)?,
    ))
}
