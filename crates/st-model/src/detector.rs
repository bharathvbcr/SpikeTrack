//! SAM 3.1's open-vocabulary detector for a text prompt (upstream
//! `Sam3Image.forward_grounding` as built by `build_sam3_multiplex_video_predictor`):
//! geometry encoder (an empty prompt: its CLS token only), fusion encoder,
//! DETR decoder with box-RPB and a presence token, dot-product scoring and the
//! segmentation head. Inference only (no DAC queries: upstream applies them in
//! training only).
//!
//! Prompt padding is dropped instead of masked: every use of the prompt
//! (attention keys, mean pooling) ignores padded tokens, so attending to the
//! valid tokens alone is the same computation.

use ojas_core::{Backend, DType, OjasError, Tensor};
use st_weights::TensorGroup;

use crate::host;
use crate::layers::{get, host_of, up, Conv, Lin, Mha, Mlp, Norm};
use crate::ModelError;

pub const D: usize = 256;
pub const HEADS: usize = 8;
pub const QUERIES: usize = 200;
pub const FEAT: usize = 72;
pub const TOKENS: usize = FEAT * FEAT;
pub const MASK: usize = 288;
/// Every detector LayerNorm is `nn.LayerNorm(256)` (torch default eps).
pub const LN_EPS: f32 = 1e-5;
pub const GN_GROUPS: usize = 8;
pub const GN_EPS: f32 = 1e-5;













/// Pre-norm `TransformerEncoderLayer` (encoder.py): self-attention, cross-attention, ReLU FFN.
struct EncLayer {
    sa: Mha,
    ca: Mha,
    l1: Lin,
    l2: Lin,
    n1: Norm,
    n2: Norm,
    n3: Norm,
}

impl EncLayer {
    fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        Ok(Self {
            sa: Mha::load(be, g, &format!("{p}.self_attn"))?,
            ca: Mha::load(be, g, &format!("{p}.cross_attn_image"))?,
            l1: Lin::load(be, g, &format!("{p}.linear1"), 2048, D)?,
            l2: Lin::load(be, g, &format!("{p}.linear2"), D, 2048)?,
            n1: Norm::load(be, g, &format!("{p}.norm1"))?,
            n2: Norm::load(be, g, &format!("{p}.norm2"))?,
            n3: Norm::load(be, g, &format!("{p}.norm3"))?,
        })
    }

    /// `self_pos`: added to self-attention q and k (`pos_enc_at_attn`).
    /// `mem_pos`: added to cross-attention keys (`pos_enc_at_cross_attn_keys`).
    fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor, self_pos: Option<&Tensor>, memory: &Tensor, mem_pos: Option<&Tensor>) -> Result<Tensor, OjasError> {
        let t = self.n1.fwd(be, x)?;
        let qk = match self_pos {
            Some(p) => be.residual_add_forward(&t, p)?,
            None => t.clone(),
        };
        let x = be.residual_add_forward(x, &self.sa.fwd(be, &qk, &qk, &t, None)?)?;
        let t = self.n2.fwd(be, &x)?;
        let keys = match mem_pos {
            Some(p) => be.residual_add_forward(memory, p)?,
            None => memory.clone(),
        };
        let x = be.residual_add_forward(&x, &self.ca.fwd(be, &t, &keys, memory, None)?)?;
        let t = self.n3.fwd(be, &x)?;
        let f = self.l2.fwd(be, &be.relu_forward(&self.l1.fwd(be, &t)?)?)?;
        be.residual_add_forward(&x, &f)
    }
}

/// Post-norm DETR decoder layer (decoder.py `TransformerDecoderLayer`).
struct DecLayer {
    sa: Mha,
    ca_text: Mha,
    ca: Mha,
    norm1: Norm,
    norm2: Norm,
    catext_norm: Norm,
    norm3: Norm,
    l1: Lin,
    l2: Lin,
}

impl DecLayer {
    fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        Ok(Self {
            sa: Mha::load(be, g, &format!("{p}.self_attn"))?,
            ca_text: Mha::load(be, g, &format!("{p}.ca_text"))?,
            ca: Mha::load(be, g, &format!("{p}.cross_attn"))?,
            norm1: Norm::load(be, g, &format!("{p}.norm1"))?,
            norm2: Norm::load(be, g, &format!("{p}.norm2"))?,
            catext_norm: Norm::load(be, g, &format!("{p}.catext_norm"))?,
            norm3: Norm::load(be, g, &format!("{p}.norm3"))?,
            l1: Lin::load(be, g, &format!("{p}.linear1"), 2048, D)?,
            l2: Lin::load(be, g, &format!("{p}.linear2"), D, 2048)?,
        })
    }

    /// `x` `[1, 1 + Q, D]` (presence token first), `pos` the same shape (zero row first).
    fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor, pos: &Tensor, prompt: &Tensor, memory: &Tensor, mem_keys: &Tensor, rpb: &Tensor) -> Result<Tensor, OjasError> {
        let qk = be.residual_add_forward(x, pos)?;
        let x = self.norm2.fwd(be, &be.residual_add_forward(x, &self.sa.fwd(be, &qk, &qk, x, None)?)?)?;
        let q = be.residual_add_forward(&x, pos)?;
        let x = self.catext_norm.fwd(be, &be.residual_add_forward(&x, &self.ca_text.fwd(be, &q, prompt, prompt, None)?)?)?;
        let q = be.residual_add_forward(&x, pos)?;
        let x = self.norm1.fwd(be, &be.residual_add_forward(&x, &self.ca.fwd(be, &q, mem_keys, memory, Some(rpb))?)?)?;
        let f = self.l2.fwd(be, &be.relu_forward(&self.l1.fwd(be, &x)?)?)?;
        self.norm3.fwd(be, &be.residual_add_forward(&x, &f)?)
    }
}



/// Per-stage values kept for parity tests.
#[derive(Default)]
pub struct DetectorTaps {
    pub geo: Option<Vec<f32>>,
    pub enc_layers: Vec<Tensor>,
    /// Normed decoder queries per layer, `[Q, D]` each, host.
    pub hs: Vec<Vec<f32>>,
    /// Reference boxes entering each layer (cxcywh, sigmoid space), `[Q, 4]` each.
    pub ref_boxes: Vec<Vec<f32>>,
    pub presence_logits: Vec<f32>,
    pub scores: Vec<Vec<f32>>,
    pub seg_cross_attn: Option<Tensor>,
    pub pixel_embed: Option<Tensor>,
}

/// The detector's final outputs for one image and one prompt.
pub struct DetectorOut {
    /// `[Q]`, after upstream's joint presence scoring (`supervise_joint_box_scores`).
    pub pred_logits: Vec<f32>,
    pub presence_logit: f32,
    /// `[Q, 4]` cxcywh in [0, 1].
    pub pred_boxes: Vec<f32>,
    /// `[Q, 288, 288]` logits.
    pub pred_masks: Tensor,
    /// `[1, 1, 288, 288]`.
    pub semantic_seg: Tensor,
}

impl DetectorOut {
    /// What upstream's `Sam3Processor` thresholds: `sigmoid(logit) * sigmoid(presence)`.
    pub fn probabilities(&self) -> Vec<f32> {
        let p = host::sigmoid(self.presence_logit);
        self.pred_logits.iter().map(|&l| host::sigmoid(l) * p).collect()
    }
}

pub struct Detector<'b, B: Backend + ?Sized> {
    be: &'b B,
    // geometry encoder (text-only path)
    geo_cls: Tensor,
    geo_final_proj: Lin,
    geo_norm: Norm,
    geo_layers: Vec<EncLayer>,
    geo_encode_norm: Norm,
    enc_layers: Vec<EncLayer>,
    // decoder
    query_embed: Tensor,
    ref_points: Vec<f32>,
    presence_token: Tensor,
    dec_layers: Vec<DecLayer>,
    dec_norm: Norm,
    bbox_embed: Mlp,
    ref_point_head: Mlp,
    rpb_x: Mlp,
    rpb_y: Mlp,
    presence_head: Mlp,
    presence_out_norm: Norm,
    // scoring
    prompt_mlp: Mlp,
    prompt_proj: Lin,
    hs_proj: Lin,
    // segmentation head
    seg_cross_attn: Mha,
    seg_cross_norm: Norm,
    pd_convs: Vec<Conv>,
    pd_gn: Vec<(Tensor, Tensor)>,
    instance_head: Conv,
    semantic_head: Conv,
    mask_embed: Mlp,
    /// Sine position encoding of the 72x72 level, `[1, 5184, 256]`.
    pos: Tensor,
}

/// Detector weight groups, as `st_weights::Sam31Weights` holds them.
pub struct DetectorWeights<'a> {
    pub transformer: &'a TensorGroup,
    pub geometry: &'a TensorGroup,
    pub segmentation: &'a TensorGroup,
    pub scoring: &'a TensorGroup,
}

impl<'b, B: Backend + ?Sized> Detector<'b, B> {
    pub fn new(be: &'b B, w: DetectorWeights<'_>) -> Result<Self, ModelError> {
        let (t, g, s, sc) = (w.transformer, w.geometry, w.segmentation, w.scoring);
        let gp = "detector.geometry_encoder";
        let tp = "detector.transformer";
        let sp = "detector.segmentation_head";
        let cp = "detector.dot_prod_scoring";
        let pos_chw = host::sine_position_map(FEAT, FEAT);
        let mut pos = vec![0f32; TOKENS * D];
        for c in 0..D {
            for i in 0..TOKENS {
                pos[i * D + c] = pos_chw[c * TOKENS + i];
            }
        }
        let gn = |i: usize| -> Result<(Tensor, Tensor), ModelError> {
            let wv = get(s, &format!("{sp}.pixel_decoder.norms.{i}.weight"), &[D])?;
            let bv = get(s, &format!("{sp}.pixel_decoder.norms.{i}.bias"), &[D])?;
            Ok((up(be, &wv.data, &[D])?, up(be, &bv.data, &[D])?))
        };
        Ok(Self {
            be,
            geo_cls: up(be, &get(g, &format!("{gp}.cls_embed.weight"), &[1, D])?.data, &[1, 1, D])?,
            geo_final_proj: Lin::load(be, g, &format!("{gp}.final_proj"), D, D)?,
            geo_norm: Norm::load(be, g, &format!("{gp}.norm"))?,
            geo_layers: (0..3).map(|i| EncLayer::load(be, g, &format!("{gp}.encode.{i}"))).collect::<Result<_, _>>()?,
            geo_encode_norm: Norm::load(be, g, &format!("{gp}.encode_norm"))?,
            enc_layers: (0..6).map(|i| EncLayer::load(be, t, &format!("{tp}.encoder.layers.{i}"))).collect::<Result<_, _>>()?,
            query_embed: up(be, &get(t, &format!("{tp}.decoder.query_embed.weight"), &[QUERIES, D])?.data, &[QUERIES, D])?,
            ref_points: get(t, &format!("{tp}.decoder.reference_points.weight"), &[QUERIES, 4])?.data.clone(),
            presence_token: up(be, &get(t, &format!("{tp}.decoder.presence_token.weight"), &[1, D])?.data, &[1, D])?,
            dec_layers: (0..6).map(|i| DecLayer::load(be, t, &format!("{tp}.decoder.layers.{i}"))).collect::<Result<_, _>>()?,
            dec_norm: Norm::load(be, t, &format!("{tp}.decoder.norm"))?,
            bbox_embed: Mlp::load(be, t, &format!("{tp}.decoder.bbox_embed"), &[D, D, D, 4], false, false)?,
            ref_point_head: Mlp::load(be, t, &format!("{tp}.decoder.ref_point_head"), &[2 * D, D, D], false, false)?,
            rpb_x: Mlp::load(be, t, &format!("{tp}.decoder.boxRPB_embed_x"), &[2, D, HEADS], false, false)?,
            rpb_y: Mlp::load(be, t, &format!("{tp}.decoder.boxRPB_embed_y"), &[2, D, HEADS], false, false)?,
            presence_head: Mlp::load(be, t, &format!("{tp}.decoder.presence_token_head"), &[D, D, D, 1], false, false)?,
            presence_out_norm: Norm::load(be, t, &format!("{tp}.decoder.presence_token_out_norm"))?,
            prompt_mlp: Mlp::load(be, sc, &format!("{cp}.prompt_mlp"), &[D, 2048, D], true, true)?,
            prompt_proj: Lin::load(be, sc, &format!("{cp}.prompt_proj"), D, D)?,
            hs_proj: Lin::load(be, sc, &format!("{cp}.hs_proj"), D, D)?,
            seg_cross_attn: Mha::load(be, s, &format!("{sp}.cross_attend_prompt"))?,
            seg_cross_norm: Norm::load(be, s, &format!("{sp}.cross_attn_norm"))?,
            pd_convs: (0..2).map(|i| Conv::load(be, s, &format!("{sp}.pixel_decoder.conv_layers.{i}"), [D, D, 3, 3])).collect::<Result<_, _>>()?,
            pd_gn: (0..2).map(gn).collect::<Result<_, _>>()?,
            instance_head: Conv::load(be, s, &format!("{sp}.instance_seg_head"), [D, D, 1, 1])?,
            semantic_head: Conv::load(be, s, &format!("{sp}.semantic_seg_head"), [1, D, 1, 1])?,
            mask_embed: Mlp::load(be, s, &format!("{sp}.mask_predictor.mask_embed"), &[D, D, D, D], false, false)?,
            pos: up(be, &pos, &[1, TOKENS, D])?,
        })
    }

    /// Weights the port reads, for the claim-everything check: every tensor
    /// in the four groups except the ones upstream's inference path never
    /// touches (geometry point/box/mask encoders: text-only prompts;
    /// `pixel_decoder.conv_layers.2`/`norms.2`: 3 stages built, 2 used).
    pub fn unused_by_text_prompts(name: &str) -> bool {
        let geo_unused = ["label_embed", "points_", "boxes_", "img_pre_norm", "mask_encoder"];
        name.starts_with("detector.geometry_encoder.") && geo_unused.iter().any(|u| name.contains(u))
            || name.contains("pixel_decoder.conv_layers.2.")
            || name.contains("pixel_decoder.norms.2.")
    }

    /// `fpn`: the detector neck's three levels `[1, 256, 288|144|72, ...]`
    /// (`Backbone::neck(Neck::Detector, ..)`). `text`: the prompt's valid
    /// (non-padding) resized text tokens, `[1, L, 256]`.
    pub fn forward(&self, fpn: &[Tensor; 3], text: &Tensor, taps: &mut DetectorTaps) -> Result<DetectorOut, ModelError> {
        let be = self.be;
        // Image tokens [1, 5184, 256] from the 72x72 level (NCHW -> tokens).
        let img = be.permute(&fpn[2].reshape(&[1, D, TOKENS])?, &[0, 2, 1])?;

        // Geometry encoder, empty prompt: CLS -> final_proj -> norm -> 3 layers -> encode_norm.
        let mut geo = self.geo_norm.fwd(be, &self.geo_final_proj.fwd(be, &self.geo_cls)?)?;
        for l in &self.geo_layers {
            geo = l.fwd(be, &geo, None, &img, Some(&self.pos))?;
        }
        let geo = self.geo_encode_norm.fwd(be, &geo)?;
        let geo_host = host_of(&geo, be)?;
        taps.geo = Some(geo_host.clone());

        // Prompt = [valid text tokens; geometry CLS].
        let mut prompt_host = host_of(text, be)?;
        prompt_host.extend_from_slice(&geo_host);
        let plen = prompt_host.len() / D;
        let prompt = up(be, &prompt_host, &[1, plen, D])?;

        // Fusion encoder over image tokens, cross-attending to the prompt.
        let mut mem = img;
        for l in &self.enc_layers {
            mem = l.fwd(be, &mem, Some(&self.pos), &prompt, None)?;
            taps.enc_layers.push(mem.clone());
        }
        let mem_keys = be.residual_add_forward(&mem, &self.pos)?;

        // DETR decoder.
        let mut refs: Vec<f32> = self.ref_points.iter().map(|&v| host::sigmoid(v)).collect();
        let tgt = host_of(&self.query_embed, be)?;
        let mut x_host = host_of(&self.presence_token, be)?;
        x_host.extend_from_slice(&tgt);
        let mut x = up(be, &x_host, &[1, 1 + QUERIES, D])?;
        let mut hs_last = Vec::new();
        for layer in &self.dec_layers {
            taps.ref_boxes.push(refs.clone());
            let sine = host::sine_embed_boxes(&refs);
            let qpos = self.ref_point_head.host(&sine, QUERIES);
            let mut pos = vec![0f32; D];
            pos.extend_from_slice(&qpos);
            let pos = up(be, &pos, &[1, 1 + QUERIES, D])?;
            let rpb = up(be, &self.rpb_bias(&refs), &[1, HEADS, 1 + QUERIES, TOKENS])?;
            x = layer.fwd(be, &x, &pos, &prompt, &mem, &mem_keys, &rpb)?;
            let xh = host_of(&x, be)?;
            let (presence, queries) = xh.split_at(D);
            let hs = self.dec_norm.host(queries);
            let delta = self.bbox_embed.host(&hs, QUERIES);
            refs = refs.iter().zip(&delta).map(|(&r, &d)| host::sigmoid(host::inverse_sigmoid(r) + d)).collect();
            // Upstream calls .clamp() on the presence logits and discards the result: no clamp.
            let pl = self.presence_head.host(&self.presence_out_norm.host(presence), 1)[0];
            taps.presence_logits.push(pl);
            taps.hs.push(hs.clone());
            hs_last = hs;
        }

        // Dot-product scoring on the last layer, then the joint presence score.
        let pm = self.prompt_mlp.host(&prompt_host, plen);
        let mut pooled = vec![0f32; D];
        for row in pm.chunks_exact(D) {
            pooled.iter_mut().zip(row).for_each(|(a, b)| *a += b);
        }
        pooled.iter_mut().for_each(|v| *v /= plen as f32);
        let pp = self.prompt_proj.host(&pooled, 1);
        let scale = 1.0 / (D as f32).sqrt();
        let presence = *taps.presence_logits.last().expect("six layers");
        for hs in &taps.hs {
            let proj = self.hs_proj.host(hs, QUERIES);
            taps.scores.push(proj.chunks_exact(D).map(|r| (r.iter().zip(&pp).map(|(a, b)| a * b).sum::<f32>() * scale).clamp(-12.0, 12.0)).collect());
        }
        let pp_presence = host::sigmoid(presence);
        let pred_logits: Vec<f32> = taps
            .scores
            .last()
            .expect("six layers")
            .iter()
            .map(|&s| host::inverse_sigmoid(host::sigmoid(s) * pp_presence).clamp(-10.0, 10.0))
            .collect();

        // Segmentation head.
        let t = self.seg_cross_norm.fwd(be, &mem)?;
        let ca = self.seg_cross_attn.fwd(be, &t, &prompt, &prompt, None)?;
        taps.seg_cross_attn = Some(ca.clone());
        let enc = be.residual_add_forward(&mem, &ca)?; // [1, 5184, 256]
        let mut prev = be.permute(&enc, &[0, 2, 1])?.reshape(&[1, D, FEAT, FEAT])?;
        for (i, bb) in [&fpn[1], &fpn[0]].into_iter().enumerate() {
            let upsampled = self.nearest2x(&prev)?;
            prev = be.residual_add_forward(bb, &upsampled)?;
            prev = self.pd_convs[i].fwd(be, &prev, 1)?;
            prev = be.relu_forward(&self.group_norm(&prev, &self.pd_gn[i])?)?;
        }
        taps.pixel_embed = Some(prev.clone());
        let instance = self.instance_head.fwd(be, &prev, 0)?; // [1, 256, 288, 288]
        let semantic_seg = self.semantic_head.fwd(be, &prev, 0)?;
        let mask_embed = self.mask_embed.fwd(be, &up(be, &hs_last, &[QUERIES, D])?)?;
        let inst = be.permute(&instance.reshape(&[D, MASK * MASK])?, &[1, 0])?; // [82944, 256]
        let pred_masks = be.linear_forward(&mask_embed, &inst)?.reshape(&[QUERIES, MASK, MASK])?;

        let pred_boxes = refs;
        Ok(DetectorOut { pred_logits, presence_logit: presence, pred_boxes, pred_masks, semantic_seg })
    }

    /// `TransformerDecoder._get_rpb_matrix` with `boxRPB="log"` for a 72x72
    /// map, as `[HEADS, 1 + Q, 5184]` with a zero row for the presence token.
    fn rpb_bias(&self, refs: &[f32]) -> Vec<f32> {
        let coords: Vec<f32> = (0..FEAT).map(|i| i as f32 / FEAT as f32).collect();
        let log8 = 8f32.log2();
        let enc = |d: f32| {
            let v = d * 8.0;
            v.signum() * (v.abs() + 1.0).log2() / log8
        };
        // deltas [Q * 72, 2] per axis: coord - (x0, x1) or (y0, y1).
        let (mut dx, mut dy) = (Vec::with_capacity(QUERIES * FEAT * 2), Vec::with_capacity(QUERIES * FEAT * 2));
        for b in refs.chunks_exact(4) {
            let (x0, y0, x1, y1) = (b[0] - 0.5 * b[2], b[1] - 0.5 * b[3], b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]);
            for &c in &coords {
                dx.extend([enc(c - x0), enc(c - x1)]);
                dy.extend([enc(c - y0), enc(c - y1)]);
            }
        }
        let ex = self.rpb_x.host(&dx, QUERIES * FEAT); // [Q, 72(w), 8]
        let ey = self.rpb_y.host(&dy, QUERIES * FEAT); // [Q, 72(h), 8]
        let rows = 1 + QUERIES;
        let mut out = vec![0f32; HEADS * rows * TOKENS];
        for h in 0..HEADS {
            for q in 0..QUERIES {
                let base = (h * rows + 1 + q) * TOKENS;
                for i in 0..FEAT {
                    let y = ey[(q * FEAT + i) * HEADS + h];
                    for j in 0..FEAT {
                        out[base + i * FEAT + j] = y + ex[(q * FEAT + j) * HEADS + h];
                    }
                }
            }
        }
        out
    }

    /// `F.interpolate(mode="nearest")` by exactly 2x: a broadcast into `[N, C, H, 2, W, 2]`.
    fn nearest2x(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        let be = self.be;
        let &[n, c, h, w] = x.shape() else {
            return Err(OjasError::OutOfRange { op: "st_model::nearest2x", detail: format!("rank {:?}", x.shape()) });
        };
        let zeros = be.fill(&[1, 1, 1, 2, 1, 2], DType::F32, 0.0)?;
        be.broadcast_add_forward(&x.reshape(&[n, c, h, 1, w, 1])?, &zeros)?.reshape(&[n, c, 2 * h, 2 * w])
    }

    /// `nn.GroupNorm(8, 256)` through Ojas's `group_norm_forward`.
    fn group_norm(&self, x: &Tensor, affine: &(Tensor, Tensor)) -> Result<Tensor, OjasError> {
        self.be.group_norm_forward(x, &affine.0, &affine.1, GN_GROUPS, GN_EPS)
    }
}
