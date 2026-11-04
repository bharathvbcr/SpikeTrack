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
use st_gate::Geometry;
use st_weights::TensorGroup;

use crate::host;
use crate::layers::{get, host_of, host_slice, up, Conv, Lin, Mha, Mlp, Norm};
use crate::ModelError;

pub const D: usize = 256;
pub const HEADS: usize = 8;
pub const QUERIES: usize = 200;
/// The image-token grid side, image-token count and mask-logit side at the native 1008 input
/// (`Geometry::SAM3`). The detector itself follows the [`Geometry`] it was built with; these
/// are the native values, for callers that pin that size.
pub const FEAT: usize = Geometry::SAM3.grid;
pub const TOKENS: usize = Geometry::SAM3.tokens();
pub const MASK: usize = Geometry::SAM3.low_res();
/// Every detector LayerNorm is `nn.LayerNorm(256)` (torch default eps).
/// The stage names [`Detector::forward_marked`] marks: the encoders on the device; per decoder
/// layer the host work before it (box sine embedding, query-position MLP, the box relative-position
/// bias) and its upload, the layer itself on the device, and the readback and host box update after
/// it; the host dot-product scoring; the segmentation head on the device.
pub const DETECTOR_STAGES: [&str; 6] = [
    "detector: encoders",
    "detector: decoder host prep (sine, query MLP, box RPB, upload)",
    "detector: decoder layer (device)",
    "detector: decoder readback + box update (host)",
    "detector: scoring (host)",
    "detector: segmentation head",
];

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
    fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        x: &Tensor,
        self_pos: Option<&Tensor>,
        memory: &Tensor,
        mem_pos: Option<&Tensor>,
    ) -> Result<Tensor, OjasError> {
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
    fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        x: &Tensor,
        pos: &Tensor,
        prompt: &Tensor,
        memory: &Tensor,
        mem_keys: &Tensor,
        rpb: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let qk = be.residual_add_forward(x, pos)?;
        let x = self.norm2.fwd(
            be,
            &be.residual_add_forward(x, &self.sa.fwd(be, &qk, &qk, x, None)?)?,
        )?;
        let q = be.residual_add_forward(&x, pos)?;
        let x = self.catext_norm.fwd(
            be,
            &be.residual_add_forward(&x, &self.ca_text.fwd(be, &q, prompt, prompt, None)?)?,
        )?;
        let q = be.residual_add_forward(&x, pos)?;
        let x = self.norm1.fwd(
            be,
            &be.residual_add_forward(&x, &self.ca.fwd(be, &q, mem_keys, memory, Some(rpb))?)?,
        )?;
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
    /// `[Q, M, M]` logits, `M` = [`Geometry::low_res`] (288 at 1008).
    pub pred_masks: Tensor,
    /// `[1, 1, M, M]`.
    pub semantic_seg: Tensor,
}

/// The segmentation head's inputs on the host, row-major.
pub struct HostSegInputs<'a> {
    /// `[grid^2, 256]`: the fusion encoder's image tokens.
    pub memory: &'a [f32],
    /// `[plen, 256]`: the prompt (valid text tokens, then the geometry CLS).
    pub prompt: &'a [f32],
    /// `[1, 256, M, M]` and `[1, 256, M/2, M/2]`: the detector neck's two finest levels.
    pub fpn0: &'a [f32],
    pub fpn1: &'a [f32],
    /// `[Q, 256]`: the last decoder layer's normed queries.
    pub hs: &'a [f32],
}

/// The segmentation head computed outside the Ojas backend (st-coreml's Core ML program), in place
/// of [`Detector`]'s own from the cross-attention to the prompt on. It computes what the head
/// computes, to a tolerance its own parity gate states, not bit for bit.
pub trait ExternalSegHead {
    /// What `run.json` records about it (format, precision, package hash).
    fn describe(&self) -> String;
    /// Returns `pred_masks` `[Q, M, M]` and `semantic_seg` `[1, 1, M, M]`, row-major.
    fn run(&self, x: &HostSegInputs<'_>) -> Result<(Vec<f32>, Vec<f32>), OjasError>;
}

/// The geometry and fusion encoders computed outside the Ojas backend (st-coreml's Core ML program),
/// in place of [`Detector`]'s own, from the top neck level to the encoder memory. It computes what
/// the encoders compute, to a tolerance its own parity gate states, not bit for bit.
pub trait ExternalEncoder {
    /// What `run.json` records about it (format, precision, package hash, baked adapter).
    fn describe(&self) -> String;
    /// `fpn2` `[1, 256, grid, grid]`, `pos` `[grid^2, 256]` and `text` (the valid prompt tokens,
    /// `[n, 256]`, as [`Detector::forward_inference`] receives them). Returns the encoder memory
    /// `[grid^2, 256]` and the geometry token `[256]`, row-major.
    fn run(
        &self,
        fpn2: &[f32],
        pos: &[f32],
        text: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), OjasError>;
}

impl DetectorOut {
    /// What upstream's `Sam3Processor` thresholds: `sigmoid(logit) * sigmoid(presence)`.
    pub fn probabilities(&self) -> Vec<f32> {
        let p = host::sigmoid(self.presence_logit);
        self.pred_logits
            .iter()
            .map(|&l| host::sigmoid(l) * p)
            .collect()
    }
}

pub struct Detector<'b, B: Backend + ?Sized> {
    be: &'b B,
    /// The input size the feature maps, positions and masks follow.
    geo: Geometry,
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
    /// Sine position encoding of the top (grid x grid) level, `[1, grid^2, 256]`.
    pos: Tensor,
    /// Post-decoder `[dx, dy, dw, dh]` logit shift applied strictly after the 6th decoder layer
    /// so `rpb_bias` in layers `0..5` stays untouched. `[0.0; 4]` is the bit-exact identity.
    post_box_delta: [f32; 4],
    /// Weight in `[0.0, 1.0]` blending the penultimate decoder layer (`li = 4`) into the final
    /// dot-product and presence logits. `0.0` is the bit-exact identity.
    aux_score_weight: f32,
    /// When set, the segmentation head runs here instead of on `be`.
    ext_seg: Option<Box<dyn ExternalSegHead + 'b>>,
    /// When set, the geometry and fusion encoders run here instead of on `be`, with `pos` read back
    /// once as its host input.
    ext_enc: Option<(Box<dyn ExternalEncoder + 'b>, Vec<f32>)>,
}

/// Detector weight groups, as `st_weights::Sam31Weights` holds them.
pub struct DetectorWeights<'a> {
    pub transformer: &'a TensorGroup,
    pub geometry: &'a TensorGroup,
    pub segmentation: &'a TensorGroup,
    pub scoring: &'a TensorGroup,
}

impl<'b, B: Backend + ?Sized> Detector<'b, B> {
    /// The detector for the native 1008 x 1008 input.
    pub fn new(be: &'b B, w: DetectorWeights<'_>) -> Result<Self, ModelError> {
        Self::new_at(be, w, Geometry::SAM3)
    }

    /// The detector for the input size `geo`. No weight depends on it: the sine position
    /// encoding, box-RPB coordinates and mask size are computed from the grid.
    pub fn new_at(be: &'b B, w: DetectorWeights<'_>, geo: Geometry) -> Result<Self, ModelError> {
        let (t, g, s, sc) = (w.transformer, w.geometry, w.segmentation, w.scoring);
        let gp = "detector.geometry_encoder";
        let tp = "detector.transformer";
        let sp = "detector.segmentation_head";
        let cp = "detector.dot_prod_scoring";
        let (feat, tokens) = (geo.grid, geo.tokens());
        let pos_chw = host::sine_position_map(feat, feat);
        let mut pos = vec![0f32; tokens * D];
        for c in 0..D {
            for i in 0..tokens {
                pos[i * D + c] = pos_chw[c * tokens + i];
            }
        }
        let gn = |i: usize| -> Result<(Tensor, Tensor), ModelError> {
            let wv = get(s, &format!("{sp}.pixel_decoder.norms.{i}.weight"), &[D])?;
            let bv = get(s, &format!("{sp}.pixel_decoder.norms.{i}.bias"), &[D])?;
            Ok((up(be, &wv.data, &[D])?, up(be, &bv.data, &[D])?))
        };
        Ok(Self {
            be,
            geo,
            geo_cls: up(
                be,
                &get(g, &format!("{gp}.cls_embed.weight"), &[1, D])?.data,
                &[1, 1, D],
            )?,
            geo_final_proj: Lin::load(be, g, &format!("{gp}.final_proj"), D, D)?,
            geo_norm: Norm::load(be, g, &format!("{gp}.norm"))?,
            geo_layers: (0..3)
                .map(|i| EncLayer::load(be, g, &format!("{gp}.encode.{i}")))
                .collect::<Result<_, _>>()?,
            geo_encode_norm: Norm::load(be, g, &format!("{gp}.encode_norm"))?,
            enc_layers: (0..6)
                .map(|i| EncLayer::load(be, t, &format!("{tp}.encoder.layers.{i}")))
                .collect::<Result<_, _>>()?,
            query_embed: up(
                be,
                &get(
                    t,
                    &format!("{tp}.decoder.query_embed.weight"),
                    &[QUERIES, D],
                )?
                .data,
                &[1, QUERIES, D],
            )?,
            ref_points: get(
                t,
                &format!("{tp}.decoder.reference_points.weight"),
                &[QUERIES, 4],
            )?
            .data
            .clone(),
            presence_token: up(
                be,
                &get(t, &format!("{tp}.decoder.presence_token.weight"), &[1, D])?.data,
                &[1, 1, D],
            )?,
            dec_layers: (0..6)
                .map(|i| DecLayer::load(be, t, &format!("{tp}.decoder.layers.{i}")))
                .collect::<Result<_, _>>()?,
            dec_norm: Norm::load_host(be, t, &format!("{tp}.decoder.norm"))?,
            bbox_embed: Mlp::load_host(
                be,
                t,
                &format!("{tp}.decoder.bbox_embed"),
                &[D, D, D, 4],
                false,
                false,
            )?,
            ref_point_head: Mlp::load_host(
                be,
                t,
                &format!("{tp}.decoder.ref_point_head"),
                &[2 * D, D, D],
                false,
                false,
            )?,
            rpb_x: Mlp::load_host(
                be,
                t,
                &format!("{tp}.decoder.boxRPB_embed_x"),
                &[2, D, HEADS],
                false,
                false,
            )?,
            rpb_y: Mlp::load_host(
                be,
                t,
                &format!("{tp}.decoder.boxRPB_embed_y"),
                &[2, D, HEADS],
                false,
                false,
            )?,
            presence_head: Mlp::load_host(
                be,
                t,
                &format!("{tp}.decoder.presence_token_head"),
                &[D, D, D, 1],
                false,
                false,
            )?,
            presence_out_norm: Norm::load_host(
                be,
                t,
                &format!("{tp}.decoder.presence_token_out_norm"),
            )?,
            prompt_mlp: Mlp::load_host(
                be,
                sc,
                &format!("{cp}.prompt_mlp"),
                &[D, 2048, D],
                true,
                true,
            )?,
            prompt_proj: Lin::load_host(be, sc, &format!("{cp}.prompt_proj"), D, D)?,
            hs_proj: Lin::load_host(be, sc, &format!("{cp}.hs_proj"), D, D)?,
            seg_cross_attn: Mha::load(be, s, &format!("{sp}.cross_attend_prompt"))?,
            seg_cross_norm: Norm::load(be, s, &format!("{sp}.cross_attn_norm"))?,
            pd_convs: (0..2)
                .map(|i| {
                    Conv::load(
                        be,
                        s,
                        &format!("{sp}.pixel_decoder.conv_layers.{i}"),
                        [D, D, 3, 3],
                    )
                })
                .collect::<Result<_, _>>()?,
            pd_gn: (0..2).map(gn).collect::<Result<_, _>>()?,
            instance_head: Conv::load(be, s, &format!("{sp}.instance_seg_head"), [D, D, 1, 1])?,
            semantic_head: Conv::load(be, s, &format!("{sp}.semantic_seg_head"), [1, D, 1, 1])?,
            mask_embed: Mlp::load(
                be,
                s,
                &format!("{sp}.mask_predictor.mask_embed"),
                &[D, D, D, D],
                false,
                false,
            )?,
            pos: up(be, &pos, &[1, tokens, D])?,
            post_box_delta: [0.0; 4],
            aux_score_weight: 0.0,
            ext_seg: None,
            ext_enc: None,
        })
    }

    /// Run the segmentation head on `ext` (a fixed-shape program for the native input) instead of
    /// on the backend. Refused at any other input size.
    pub fn set_external_seg_head(
        &mut self,
        ext: Box<dyn ExternalSegHead + 'b>,
    ) -> Result<(), String> {
        if self.geo != Geometry::SAM3 {
            return Err(format!(
                "an external segmentation head is built for the {}-pixel input; this detector is {} pixels",
                Geometry::SAM3.pixels(),
                self.geo.pixels()
            ));
        }
        self.ext_seg = Some(ext);
        Ok(())
    }

    /// What the external segmentation head is, when one is installed.
    pub fn external_seg_head(&self) -> Option<String> {
        self.ext_seg.as_ref().map(|e| e.describe())
    }

    /// Run the geometry and fusion encoders on `ext` (a fixed-shape program for the native input)
    /// instead of on the backend, in [`Self::forward_inference`] only: [`Self::forward`], which
    /// records [`DetectorTaps`], keeps the backend's encoders. Refused at any other input size.
    pub fn set_external_encoder(
        &mut self,
        ext: Box<dyn ExternalEncoder + 'b>,
    ) -> Result<(), String> {
        if self.geo != Geometry::SAM3 {
            return Err(format!(
                "an external encoder is built for the {}-pixel input; this detector is {} pixels",
                Geometry::SAM3.pixels(),
                self.geo.pixels()
            ));
        }
        let pos = host_slice(&self.pos, self.be)
            .map_err(|e| {
                format!("reading the position encoding back for the external encoder: {e}")
            })?
            .to_vec();
        self.ext_enc = Some((ext, pos));
        Ok(())
    }

    /// What the external encoder is, when one is installed.
    pub fn external_encoder(&self) -> Option<String> {
        self.ext_enc.as_ref().map(|(e, _)| e.describe())
    }

    /// Set a post-decoder `[dx, dy, dw, dh]` logit shift applied only to `pred_boxes` after
    /// decoder layer 5 (decoupled from `boxRPB` in layers `0..5`).
    pub fn set_post_box_delta(&mut self, delta: [f32; 4]) -> Result<(), String> {
        validate_post_box_delta(delta)?;
        self.post_box_delta = delta;
        Ok(())
    }

    pub fn post_box_delta(&self) -> [f32; 4] {
        self.post_box_delta
    }

    /// Set the blend weight in `[0.0, 1.0]` mixing penultimate decoder layer (`li = 4`) scores
    /// with the final layer (`li = 5`). `0.0` is upstream's exact final-layer-only behavior.
    pub fn set_aux_score_weight(&mut self, weight: f32) -> Result<(), String> {
        validate_aux_score_weight(weight)?;
        self.aux_score_weight = weight;
        Ok(())
    }

    pub fn aux_score_weight(&self) -> f32 {
        self.aux_score_weight
    }

    /// The input size this detector was built for.
    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    /// Weights the port reads, for the claim-everything check: every tensor
    /// in the four groups except the ones upstream's inference path never
    /// touches (geometry point/box/mask encoders: text-only prompts;
    /// `pixel_decoder.conv_layers.2`/`norms.2`: 3 stages built, 2 used).
    pub fn unused_by_text_prompts(name: &str) -> bool {
        let geo_unused = [
            "label_embed",
            "points_",
            "boxes_",
            "img_pre_norm",
            "mask_encoder",
        ];
        name.starts_with("detector.geometry_encoder.")
            && geo_unused.iter().any(|u| name.contains(u))
            || name.contains("pixel_decoder.conv_layers.2.")
            || name.contains("pixel_decoder.norms.2.")
    }

    /// `fpn`: the detector neck's three levels `[1, 256, 288|144|72, ...]`
    /// (`Backbone::neck(Neck::Detector, ..)`). `text`: the prompt's valid
    /// (non-padding) resized text tokens, `[1, L, 256]`.
    pub fn forward(
        &self,
        fpn: &[Tensor; 3],
        text: &Tensor,
        taps: &mut DetectorTaps,
    ) -> Result<DetectorOut, ModelError> {
        self.forward_impl(fpn, text, Some(taps), &mut |_| {})
    }

    /// [`Self::forward`], calling `mark` with a [`DETECTOR_STAGES`] name as each phase begins (for
    /// `track --profile`). The numerics are the same.
    pub fn forward_marked(
        &self,
        fpn: &[Tensor; 3],
        text: &Tensor,
        taps: &mut DetectorTaps,
        mark: &mut dyn FnMut(&str),
    ) -> Result<DetectorOut, ModelError> {
        self.forward_impl(fpn, text, Some(taps), mark)
    }

    /// Inference pass without recording intermediate [`DetectorTaps`]: avoids stalling the GPU on
    /// `geo`, avoids pinning intermediate encoder / pixel-decoder tensors in Metal memory, and
    /// evaluates `hs_proj` only on the final decoder layer. Numerics of [`DetectorOut`] are
    /// bit-identical to [`Self::forward`].
    pub fn forward_inference(
        &self,
        fpn: &[Tensor; 3],
        text: &Tensor,
        mark: &mut dyn FnMut(&str),
    ) -> Result<DetectorOut, ModelError> {
        self.forward_impl(fpn, text, None, mark)
    }

    fn forward_impl(
        &self,
        fpn: &[Tensor; 3],
        text: &Tensor,
        mut taps: Option<&mut DetectorTaps>,
        mark: &mut dyn FnMut(&str),
    ) -> Result<DetectorOut, ModelError> {
        mark(DETECTOR_STAGES[0]);
        let be = self.be;
        let (feat, tokens, mask) = (self.geo.grid, self.geo.tokens(), self.geo.low_res());
        for (level, side) in [(0, mask), (1, self.geo.mid_res()), (2, feat)] {
            if fpn[level].shape() != [1, D, side, side] {
                return Err(ModelError::Ojas(OjasError::OutOfRange {
                    op: "st_model::Detector::forward",
                    detail: format!("fpn[{level}] is {:?}, expected [1, {D}, {side}, {side}] for a {}-pixel input", fpn[level].shape(), self.geo.pixels()),
                }));
            }
        }
        let (mem, prompt) = match (&self.ext_enc, taps.is_none()) {
            (Some((ext, pos)), true) => {
                let text_host = host_slice(text, be)?;
                let fpn2 = host_slice(&fpn[2], be)?;
                let (mem, geo) = ext.run(fpn2.as_slice(), pos, text_host.as_slice())?;
                drop((fpn2, text_host));
                if mem.len() != tokens * D || geo.len() != D {
                    return Err(ModelError::Ojas(OjasError::OutOfRange {
                        op: "st_model::Detector::forward",
                        detail: format!(
                            "the external encoder returned {} memory and {} geometry values, want {} and {D}",
                            mem.len(),
                            geo.len(),
                            tokens * D
                        ),
                    }));
                }
                let geo = up(be, &geo, &[1, 1, D])?;
                let text_dev;
                let text = if text.device() == geo.device() {
                    text
                } else {
                    text_dev = be.upload(text)?;
                    &text_dev
                };
                let prompt = be.concat(&[text, &geo], 1)?;
                (up(be, &mem, &[1, tokens, D])?, prompt)
            }
            _ => {
                // Image tokens [1, grid^2, 256] from the grid x grid level (NCHW -> tokens).
                let img = be.permute(&fpn[2].reshape(&[1, D, tokens])?, &[0, 2, 1])?;

                // Geometry encoder, empty prompt: CLS -> final_proj -> norm -> 3 layers -> encode_norm.
                let mut geo = self
                    .geo_norm
                    .fwd(be, &self.geo_final_proj.fwd(be, &self.geo_cls)?)?;
                for l in &self.geo_layers {
                    geo = l.fwd(be, &geo, None, &img, Some(&self.pos))?;
                }
                let geo = self.geo_encode_norm.fwd(be, &geo)?;
                if let Some(t) = taps.as_deref_mut() {
                    t.geo = Some(host_of(&geo, be)?);
                }

                // Prompt = [valid text tokens; geometry CLS].
                let text_dev;
                let text = if text.device() == geo.device() {
                    text
                } else {
                    text_dev = be.upload(text)?;
                    &text_dev
                };
                let prompt = be.concat(&[text, &geo], 1)?;
                drop(geo);

                // Fusion encoder over image tokens, cross-attending to the prompt.
                let mut mem = img;
                for l in &self.enc_layers {
                    mem = l.fwd(be, &mem, Some(&self.pos), &prompt, None)?;
                    if let Some(t) = taps.as_deref_mut() {
                        t.enc_layers.push(mem.clone());
                    }
                }
                (mem, prompt)
            }
        };
        let plen = prompt.shape()[1];
        let mem_keys = be.residual_add_forward(&mem, &self.pos)?;

        // DETR decoder.
        let mut refs: Vec<f32> = self.ref_points.iter().map(|&v| host::sigmoid(v)).collect();
        let presence = if self.presence_token.shape().len() == 2 {
            self.presence_token.reshape(&[1, 1, D])?
        } else {
            self.presence_token.clone()
        };
        let query_embed = if self.query_embed.shape().len() == 2 {
            self.query_embed.reshape(&[1, QUERIES, D])?
        } else {
            self.query_embed.clone()
        };
        let mut x = be.concat(&[&presence, &query_embed], 1)?;
        drop((presence, query_embed));
        let mut hs_prev: Option<Vec<f32>> = None;
        let mut hs_last = Vec::new();
        let mut presence_prev = 0.0f32;
        let mut presence_last = 0.0f32;
        let n_dec = self.dec_layers.len();
        let use_aux = self.aux_score_weight > 0.0 && n_dec >= 2;
        for (li, layer) in self.dec_layers.iter().enumerate() {
            mark(DETECTOR_STAGES[1]);
            if let Some(t) = taps.as_deref_mut() {
                t.ref_boxes.push(refs.clone());
            }
            let sine = host::sine_embed_boxes(&refs);
            let qpos = self.ref_point_head.host(&sine, QUERIES);
            let mut pos = vec![0f32; D];
            pos.extend_from_slice(&qpos);
            let pos = up(be, &pos, &[1, 1 + QUERIES, D])?;
            let rpb = self.rpb_bias(&refs)?;
            mark(DETECTOR_STAGES[2]);
            x = layer.fwd(be, &x, &pos, &prompt, &mem, &mem_keys, &rpb)?;
            drop((pos, rpb));
            mark(DETECTOR_STAGES[3]);
            let xh = host_slice(&x, be)?;
            let (presence_tok, queries) = xh.as_slice().split_at(D);
            let hs = self.dec_norm.host(queries);
            if taps.is_some() || li + 1 == n_dec || (use_aux && li + 2 == n_dec) {
                // Upstream calls .clamp() on the presence logits and discards the result: no clamp.
                let pres = self
                    .presence_head
                    .host(&self.presence_out_norm.host(presence_tok), 1)[0];
                if use_aux && li + 2 == n_dec {
                    presence_prev = pres;
                }
                presence_last = pres;
            }
            drop(xh);
            let delta = self.bbox_embed.host(&hs, QUERIES);
            refs = refs
                .iter()
                .zip(&delta)
                .map(|(&r, &d)| host::sigmoid(host::inverse_sigmoid(r) + d))
                .collect();
            if let Some(t) = taps.as_deref_mut() {
                t.presence_logits.push(presence_last);
                t.hs.push(hs.clone());
            }
            if use_aux && li + 2 == n_dec {
                hs_prev = Some(hs.clone());
            }
            hs_last = hs;
        }
        drop((x, mem_keys));

        // Dot-product scoring on the last layer (with optional penultimate-layer blend), then the
        // joint presence score.
        mark(DETECTOR_STAGES[4]);
        let prompt_host = host_slice(&prompt, be)?;
        let pm = self.prompt_mlp.host(prompt_host.as_slice(), plen);
        drop(prompt_host);
        let mut pooled = vec![0f32; D];
        for row in pm.chunks_exact(D) {
            pooled.iter_mut().zip(row).for_each(|(a, b)| *a += b);
        }
        pooled.iter_mut().for_each(|v| *v /= plen as f32);
        let pp = self.prompt_proj.host(&pooled, 1);
        let scale = 1.0 / (D as f32).sqrt();
        let score_hs = |hs: &[f32]| -> Vec<f32> {
            let proj = self.hs_proj.host(hs, QUERIES);
            proj.chunks_exact(D)
                .map(|r| {
                    (r.iter().zip(&pp).map(|(a, b)| a * b).sum::<f32>() * scale).clamp(-12.0, 12.0)
                })
                .collect()
        };
        let mut last_scores = if let Some(t) = taps.as_deref_mut() {
            for hs in &t.hs {
                t.scores.push(score_hs(hs));
            }
            t.scores.last().expect("six layers").clone()
        } else {
            score_hs(&hs_last)
        };
        if let (true, Some(prev)) = (use_aux, hs_prev.as_deref()) {
            let prev_scores = score_hs(prev);
            blend_aux_scores(
                &mut last_scores,
                &prev_scores,
                &mut presence_last,
                presence_prev,
                self.aux_score_weight,
            );
        }
        let pp_presence = host::sigmoid(presence_last);
        let pred_logits: Vec<f32> = last_scores
            .iter()
            .map(|&s| host::inverse_sigmoid(host::sigmoid(s) * pp_presence).clamp(-10.0, 10.0))
            .collect();

        // Segmentation head.
        mark(DETECTOR_STAGES[5]);
        let mut pred_boxes = refs;
        apply_post_box_delta(&mut pred_boxes, self.post_box_delta);
        if let Some(ext) = &self.ext_seg {
            // Taps' seg_cross_attn and pixel_embed stay None: the program does not expose them.
            let (mem_h, prompt_h) = (host_slice(&mem, be)?, host_slice(&prompt, be)?);
            drop((mem, prompt));
            let (f0, f1) = (host_slice(&fpn[0], be)?, host_slice(&fpn[1], be)?);
            let (masks, semantic) = ext.run(&HostSegInputs {
                memory: mem_h.as_slice(),
                prompt: prompt_h.as_slice(),
                fpn0: f0.as_slice(),
                fpn1: f1.as_slice(),
                hs: &hs_last,
            })?;
            let n = QUERIES * mask * mask;
            if masks.len() != n || semantic.len() != mask * mask {
                return Err(ModelError::Ojas(OjasError::OutOfRange {
                    op: "st_model::Detector::forward",
                    detail: format!(
                        "the external segmentation head returned {} mask and {} semantic values, want {n} and {}",
                        masks.len(),
                        semantic.len(),
                        mask * mask
                    ),
                }));
            }
            return Ok(DetectorOut {
                pred_logits,
                presence_logit: presence_last,
                pred_boxes,
                pred_masks: Tensor::from_f32(&masks, &[QUERIES, mask, mask], be.budget())?,
                semantic_seg: Tensor::from_f32(&semantic, &[1, 1, mask, mask], be.budget())?,
            });
        }
        let t = self.seg_cross_norm.fwd(be, &mem)?;
        let ca = self.seg_cross_attn.fwd(be, &t, &prompt, &prompt, None)?;
        drop((t, prompt));
        if let Some(tp) = taps.as_deref_mut() {
            tp.seg_cross_attn = Some(ca.clone());
        }
        let enc = be.residual_add_forward(&mem, &ca)?; // [1, grid^2, 256]
        drop((mem, ca));
        let mut prev = be.permute(&enc, &[0, 2, 1])?.reshape(&[1, D, feat, feat])?;
        drop(enc);
        for (i, bb) in [&fpn[1], &fpn[0]].into_iter().enumerate() {
            let upsampled = self.nearest2x(&prev)?;
            prev = be.residual_add_forward(bb, &upsampled)?;
            prev = self.pd_convs[i].fwd(be, &prev, 1)?;
            prev = be.relu_forward(&self.group_norm(&prev, &self.pd_gn[i])?)?;
        }
        if let Some(tp) = taps {
            tp.pixel_embed = Some(prev.clone());
        }
        let instance = self.instance_head.fwd(be, &prev, 0)?; // [1, 256, M, M], M = 288 at 1008
        let semantic_seg = self.semantic_head.fwd(be, &prev, 0)?;
        drop(prev);
        let mask_embed = self.mask_embed.fwd(be, &up(be, &hs_last, &[QUERIES, D])?)?;
        let inst = be.permute(&instance.reshape(&[D, mask * mask])?, &[1, 0])?; // [M * M, 256]
        drop(instance);
        let pred_masks = be
            .linear_forward(&mask_embed, &inst)?
            .reshape(&[QUERIES, mask, mask])?;

        Ok(DetectorOut {
            pred_logits,
            presence_logit: presence_last,
            pred_boxes,
            pred_masks,
            semantic_seg,
        })
    }

    /// `TransformerDecoder._get_rpb_matrix` with `boxRPB="log"` for a grid x grid
    /// map (72 at 1008), as a device `[1, HEADS, 1 + Q, grid^2]` with a zero row for the presence
    /// token. The per-axis MLPs run on the host; the `[.., grid, grid]` sum of the two axes is a
    /// device broadcast add (one IEEE add per element, as the host loop it replaces did), so only
    /// the two `[HEADS, Q, grid]` tables are uploaded instead of the whole bias.
    fn rpb_bias(&self, refs: &[f32]) -> Result<Tensor, OjasError> {
        let feat = self.geo.grid;
        let coords: Vec<f32> = (0..feat).map(|i| i as f32 / feat as f32).collect();
        let log8 = 8f32.log2();
        let enc = |d: f32| {
            let v = d * 8.0;
            v.signum() * (v.abs() + 1.0).log2() / log8
        };
        let q_stride = feat * HEADS;
        let (mut ex, mut ey) = (
            vec![0f32; QUERIES * q_stride],
            vec![0f32; QUERIES * q_stride],
        );
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .clamp(1, QUERIES / 25);
        let q_chunk = QUERIES.div_ceil(threads);
        let (rpb_x, rpb_y) = (&self.rpb_x, &self.rpb_y);
        std::thread::scope(|s| {
            let mut ex_rest = ex.as_mut_slice();
            let mut ey_rest = ey.as_mut_slice();
            let mut q0 = 0;
            while !ex_rest.is_empty() {
                let qn = (ex_rest.len() / q_stride).min(q_chunk);
                let (ex_piece, ex_tail) = ex_rest.split_at_mut(qn * q_stride);
                let (ey_piece, ey_tail) = ey_rest.split_at_mut(qn * q_stride);
                ex_rest = ex_tail;
                ey_rest = ey_tail;
                let refs_piece = &refs[q0 * 4..(q0 + qn) * 4];
                let coords = &coords;
                s.spawn(move || {
                    let mut dx = vec![0f32; qn * feat * 2];
                    let mut dy = vec![0f32; qn * feat * 2];
                    for ((b, rx), ry) in refs_piece
                        .chunks_exact(4)
                        .zip(dx.chunks_exact_mut(feat * 2))
                        .zip(dy.chunks_exact_mut(feat * 2))
                    {
                        let (x0, x1) = (b[0] - 0.5 * b[2], b[0] + 0.5 * b[2]);
                        let (y0, y1) = (b[1] - 0.5 * b[3], b[1] + 0.5 * b[3]);
                        for ((px, py), &c) in rx
                            .chunks_exact_mut(2)
                            .zip(ry.chunks_exact_mut(2))
                            .zip(coords)
                        {
                            px[0] = enc(c - x0);
                            px[1] = enc(c - x1);
                            py[0] = enc(c - y0);
                            py[1] = enc(c - y1);
                        }
                    }
                    rpb_x.host_rows(&dx, qn * feat, ex_piece);
                    rpb_y.host_rows(&dy, qn * feat, ey_piece);
                });
                q0 += qn;
            }
        });
        rpb_from_axes(self.be, &ex, &ey, feat)
    }

    /// `F.interpolate(mode="nearest")` by exactly 2x: a broadcast into `[N, C, H, 2, W, 2]`.
    fn nearest2x(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        let be = self.be;
        let &[n, c, h, w] = x.shape() else {
            return Err(OjasError::OutOfRange {
                op: "st_model::nearest2x",
                detail: format!("rank {:?}", x.shape()),
            });
        };
        let zeros = be.fill(&[1, 1, 1, 2, 1, 2], DType::F32, 0.0)?;
        be.broadcast_add_forward(&x.reshape(&[n, c, h, 1, w, 1])?, &zeros)?
            .reshape(&[n, c, 2 * h, 2 * w])
    }

    /// `nn.GroupNorm(8, 256)` through Ojas's `group_norm_forward`.
    fn group_norm(&self, x: &Tensor, affine: &(Tensor, Tensor)) -> Result<Tensor, OjasError> {
        self.be
            .group_norm_forward(x, &affine.0, &affine.1, GN_GROUPS, GN_EPS)
    }
}

/// Validate that every component of `post_box_delta` is finite.
pub fn validate_post_box_delta(delta: [f32; 4]) -> Result<(), String> {
    if delta.iter().any(|v| !v.is_finite()) {
        return Err(format!("post_box_delta must be finite, got {delta:?}"));
    }
    Ok(())
}

/// Validate that `aux_score_weight` is finite and in `[0.0, 1.0]`.
pub fn validate_aux_score_weight(weight: f32) -> Result<(), String> {
    if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
        return Err(format!(
            "aux_score_weight must be finite in [0.0, 1.0], got {weight}"
        ));
    }
    Ok(())
}

/// Blend penultimate-layer (`li = 4`) query and presence logits into the final layer (`li = 5`).
/// Exact no-op when `weight == 0.0`.
pub fn blend_aux_scores(
    last_scores: &mut [f32],
    prev_scores: &[f32],
    presence_last: &mut f32,
    presence_prev: f32,
    weight: f32,
) {
    assert_eq!(last_scores.len(), prev_scores.len(), "aux score length");
    if weight == 0.0 {
        return;
    }
    for (s, &p) in last_scores.iter_mut().zip(prev_scores) {
        *s = ((1.0 - weight) * *s + weight * p).clamp(-12.0, 12.0);
    }
    *presence_last = (1.0 - weight) * *presence_last + weight * presence_prev;
}

/// Apply a post-decoder `[dx, dy, dw, dh]` logit shift to `cxcywh` boxes in `[0, 1]` without
/// touching decoder `boxRPB` reference points. Exact no-op when `delta == [0.0; 4]` (and per
/// coordinate when `d == 0.0`).
pub fn apply_post_box_delta(boxes_cxcywh: &mut [f32], delta: [f32; 4]) {
    if delta == [0.0; 4] {
        return;
    }
    for b in boxes_cxcywh.chunks_exact_mut(4) {
        for (coord, &d) in b.iter_mut().zip(&delta) {
            if d != 0.0 {
                *coord = host::sigmoid(host::inverse_sigmoid(*coord) + d);
            }
        }
    }
}

/// The box-RPB bias from its per-axis tables `ex` / `ey`, each `[Q, grid, HEADS]` as the MLPs
/// emit them: `bias[h, 1 + q, i * grid + j] = ey[q, i, h] + ex[q, j, h]`, row 0 (the presence
/// token) zero, as a device `[1, HEADS, 1 + Q, grid^2]`.
fn rpb_from_axes<B: Backend + ?Sized>(
    be: &B,
    ex: &[f32],
    ey: &[f32],
    feat: usize,
) -> Result<Tensor, OjasError> {
    let q = ex.len() / (feat * HEADS);
    // [Q, grid, HEADS] -> [HEADS, Q, grid] on the host (a copy).
    let by_head = |e: &[f32]| -> Vec<f32> {
        let n = q * feat;
        let mut out = vec![0f32; HEADS * n];
        for (i, row) in e.chunks_exact(HEADS).enumerate() {
            for (h, &v) in row.iter().enumerate() {
                out[h * n + i] = v;
            }
        }
        out
    };
    let ey = up(be, &by_head(ey), &[HEADS, q, feat, 1])?;
    let ex = up(be, &by_head(ex), &[HEADS, q, 1, feat])?;
    let boxes = be
        .broadcast_add_forward(&ey, &ex)?
        .reshape(&[HEADS, q, feat * feat])?;
    let presence = be.fill(&[HEADS, 1, feat * feat], DType::F32, 0.0)?;
    be.concat(&[&presence, &boxes], 1)?
        .reshape(&[1, HEADS, 1 + q, feat * feat])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::Budget;

    /// The host loop `rpb_from_axes` replaced: the oracle it must match bit for bit.
    fn rpb_host(ex: &[f32], ey: &[f32], q: usize, feat: usize) -> Vec<f32> {
        let (rows, tokens) = (1 + q, feat * feat);
        let mut out = vec![0f32; HEADS * rows * tokens];
        for h in 0..HEADS {
            for qi in 0..q {
                let base = (h * rows + 1 + qi) * tokens;
                for i in 0..feat {
                    let y = ey[(qi * feat + i) * HEADS + h];
                    for j in 0..feat {
                        out[base + i * feat + j] = y + ex[(qi * feat + j) * HEADS + h];
                    }
                }
            }
        }
        out
    }

    fn table(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
                    * [1e-4f32, 0.3, 9.0, 4e2][(s & 3) as usize]
            })
            .collect()
    }

    fn rpb_matches_the_host_loop_on<B: Backend + ?Sized>(be: &B) {
        // The native shape (200 queries, a 72 grid) and a small odd one.
        for (q, feat) in [(QUERIES, FEAT), (3, 5)] {
            let (ex, ey) = (table(q * feat * HEADS, 1), table(q * feat * HEADS, 2));
            let got = rpb_from_axes(be, &ex, &ey, feat).expect("rpb");
            assert_eq!(got.shape(), [1, HEADS, 1 + q, feat * feat]);
            let got = host_of(&got, be).expect("readback");
            let want = rpb_host(&ex, &ey, q, feat);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "({q}, {feat}) element {i}: {g} vs {w}"
                );
            }
        }
    }

    #[test]
    fn the_rpb_bias_matches_the_host_loop_bit_for_bit_on_cpu() {
        rpb_matches_the_host_loop_on(
            &ojas_cpu::CpuBackend::with_threads(Budget::new(4 << 30), 2).expect("cpu"),
        );
    }

    #[test]
    #[ignore = "needs the Metal GPU: cargo test -p st-model --lib -- --ignored rpb"]
    fn the_rpb_bias_matches_the_host_loop_bit_for_bit_on_metal() {
        rpb_matches_the_host_loop_on(
            &ojas_metal::MetalBackend::new(Budget::new(4 << 30)).expect("metal"),
        );
    }

    #[test]
    fn post_box_delta_is_bit_exact_at_zero_and_finite_on_boundaries() {
        assert!(validate_post_box_delta([0.0, -0.1, 0.2, 0.05]).is_ok());
        assert!(validate_post_box_delta([0.0, f32::NAN, 0.0, 0.0]).is_err());
        assert!(validate_post_box_delta([0.0, 0.0, f32::INFINITY, 0.0]).is_err());

        let orig = vec![0.0f32, 1.0, 0.035, 0.042, 0.5, 0.25, 0.1, 0.9999];
        let mut cur = orig.clone();
        apply_post_box_delta(&mut cur, [0.0; 4]);
        for (a, b) in cur.iter().zip(&orig) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        apply_post_box_delta(&mut cur, [0.0, 0.0, 0.15, -0.10]);
        // Unshifted cx, cy remain bit-exact; dw > 0 widens w; dh < 0 shrinks h; all stay finite in (0, 1).
        assert_eq!(cur[0].to_bits(), orig[0].to_bits());
        assert_eq!(cur[1].to_bits(), orig[1].to_bits());
        assert!(cur[2] > orig[2] && cur[2] < 1.0);
        assert!(cur[3] < orig[3] && cur[3] > 0.0);
        assert!(cur.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
    }

    #[test]
    fn aux_score_weight_validation_and_blending_preserve_zero_identity_and_bounds() {
        for ok in [0.0f32, 0.25, 1.0] {
            assert!(validate_aux_score_weight(ok).is_ok());
        }
        for bad in [-0.01f32, 1.01, f32::NAN, f32::INFINITY] {
            assert!(validate_aux_score_weight(bad).is_err());
        }

        let orig = [-3.5f32, 0.0, 4.2, 15.0];
        let prev = [-1.5f32, 2.0, 0.2, 20.0];
        let mut s0 = orig;
        let mut p0 = 1.75f32;
        blend_aux_scores(&mut s0, &prev, &mut p0, 0.25, 0.0);
        for (a, b) in s0.iter().zip(&orig) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(p0.to_bits(), 1.75f32.to_bits());

        blend_aux_scores(&mut s0, &prev, &mut p0, 0.75, 0.25);
        assert!((s0[0] - -3.0).abs() < 1e-6);
        assert!((s0[1] - 0.5).abs() < 1e-6);
        assert!((s0[2] - 3.2).abs() < 1e-6);
        assert_eq!(s0[3], 12.0);
        assert!((p0 - 1.5).abs() < 1e-6);
    }
}
