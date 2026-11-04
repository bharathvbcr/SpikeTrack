//! Semi-supervised VOS with SAM 3.1's multiplex tracker: masks on the first
//! frame, then propagation. One bucket of up to 16 objects (multiple buckets
//! are not wired yet). Mirrors upstream's `VideoTrackingMultiplexDemo` path:
//! `add_new_masks` + `propagate_in_video_preflight` on frame 0, then
//! `propagate_in_video`; see tools/reference/README.md for the configuration.

use ojas_core::{Backend, Conv2dParams, Tensor};
use st_gate::Geometry;
use st_weights::TensorGroup;

use crate::backbone::{Backbone, Neck};
use crate::host;
use crate::layers::{get, host_of, host_slice, up, Conv, Lin, Mlp};
use crate::tracker::interactive::InteractiveDecoder;
use crate::tracker::mask_decoder::{high_res, MuxMaskDecoder, MASKS_PER_OBJ, MUX};
use crate::tracker::memory_attention::{ExternalMemAttn, MemoryAttention, MemoryInputs};
use crate::tracker::memory_encoder::ExternalMemEncoder;
use crate::tracker::memory_encoder::MemoryEncoder;
use crate::ModelError;

pub const D: usize = 256;
/// The native 1008 x 1008 input's feature grid, token count, mask-logit side, image side and
/// input-mask side (`Geometry::SAM3`). Nothing in the tracker reads these: every size follows
/// the [`Geometry`] the [`Vos`] was built with ([`Vos::geometry`]). They stay for callers that
/// pin the native size.
pub const FEAT: usize = Geometry::SAM3.grid;
pub const TOKENS: usize = Geometry::SAM3.tokens();
pub const LOW: usize = Geometry::SAM3.low_res();
pub const IMAGE: usize = Geometry::SAM3.pixels();
pub const INPUT_MASK: usize = Geometry::SAM3.input_mask();
pub(crate) const NUM_MASKMEM: usize = 7;
pub(crate) const MAX_PTRS: usize = 16;
pub(crate) const NO_OBJ_SCORE: f32 = -1024.0;
pub(crate) const LN2D_EPS: f32 = 1e-6;

/// One frame's stored state (upstream's compact `current_out`).
struct FrameMem {
    /// `[1, 256, G, G]`, rounded through bf16 as upstream stores it (`G` = 72 at 1008).
    maskmem: Vec<f32>,
    /// Propagation features and positions of that frame, tokens `[G * G, 256]`.
    image: Vec<f32>,
    image_pos: Vec<f32>,
    /// Muxed object pointers `[16, 256]`.
    obj_ptr: Vec<f32>,
}

/// Masks the tracker produced for one frame.
pub struct FrameOut {
    pub frame: usize,
    /// `[N, M, M]` logits (`M` = 4 x the grid, 288 at 1008), after object-score gating.
    pub low_res: Vec<f32>,
    /// `[N, H, W]` at the video's resolution, after the non-overlap constraint.
    pub video_res: Vec<f32>,
    pub object_scores: Vec<f32>,
    /// What the frame stored for later frames: memory features `[256, G, G]`
    /// (bf16-rounded) and muxed object pointers `[16, 256]`.
    pub maskmem: Vec<f32>,
    pub obj_ptr: Vec<f32>,
}

pub struct Vos<'b, B: Backend + ?Sized> {
    pub(crate) be: &'b B,
    pub(crate) backbone: &'b Backbone<'b, B>,
    /// The backbone's input size; every tracker size is derived from it.
    pub(crate) geo: Geometry,
    pub(crate) mux_decoder: MuxMaskDecoder,
    pub(crate) interactive: InteractiveDecoder,
    pub(crate) i_conv_s0: (Tensor, Tensor),
    pub(crate) i_conv_s1: (Tensor, Tensor),
    pub(crate) mem_enc: MemoryEncoder,
    pub(crate) mem_attn: MemoryAttention,
    pub(crate) mask_downsample: Conv,
    pub(crate) pe_downscale: [(Conv, Option<(Tensor, Tensor)>); 3],
    pub(crate) not_a_point: Vec<f32>,
    pub(crate) no_mem_embed_device: Tensor,
    pub(crate) obj_ptr_proj: Mlp,
    pub(crate) interactive_obj_ptr_proj: Mlp,
    pub(crate) no_obj_ptr_linear: Lin,
    pub(crate) obj_ptr_tpos_proj: Lin,
    pub(crate) maskmem_tpos: Vec<f32>,
    pub(crate) no_obj_embed_spatial: Vec<f32>,
    pub(crate) output_valid: Vec<f32>,
    pub(crate) output_invalid: Vec<f32>,
    pub(crate) prop_pe: Tensor,
    pub(crate) inter_pe: Tensor,
    /// Sine position map of the grid x grid level, `[256, G, G]`.
    pub(crate) sine_pos: Vec<f32>,
    /// `sine_pos` token-major on the device, `[1, G * G, 256]`: memory attention's `src_pos`.
    pub(crate) pos_tokens: Tensor,
    /// Per `maskmem_tpos_enc` row `i`, `sine_pos` tokens + that row (one add per element) on the
    /// device, `[1, G * G, 256]`: the position of a stored memory `NUM_MASKMEM - 1 - i` frames back.
    pub(crate) memory_pos_tokens: Vec<Tensor>,
    /// `pos_tokens` and `memory_pos_tokens` on the host, `[G * G * 256]` each, for
    /// [`Self::with_external_mem_attn`].
    pub(crate) pos_host: Vec<f32>,
    pub(crate) memory_pos_host: Vec<Vec<f32>>,
    /// [`Self::with_external_mem_attn`]: propagation's memory attention runs there.
    pub(crate) ext_mem_attn: Option<Box<dyn ExternalMemAttn + 'b>>,
    /// [`Self::with_external_mem_encoder`]: mask memories whose masks arrive at the downsampler's
    /// input size are encoded there.
    pub(crate) ext_mem_enc: Option<Box<dyn ExternalMemEncoder + 'b>>,
    /// Memory encodes run by `ext_mem_enc` and by the Ojas encoder: [`Self::mem_encoder_calls`].
    pub(crate) mem_enc_calls: std::cell::Cell<[usize; 2]>,
}

/// SAM's `PositionEmbeddingRandom` over an `h x w` grid: `[256, h, w]`.
fn random_fourier_pe(gauss: &[f32], h: usize, w: usize) -> Vec<f32> {
    let f = gauss.len() / 2; // [2, f]
    let mut out = vec![0f32; 2 * f * h * w];
    for y in 0..h {
        for x in 0..w {
            let (cx, cy) = (
                2.0 * ((x as f32 + 0.5) / w as f32) - 1.0,
                2.0 * ((y as f32 + 0.5) / h as f32) - 1.0,
            );
            for k in 0..f {
                let v = 2.0 * std::f32::consts::PI * (cx * gauss[k] + cy * gauss[f + k]);
                out[(k * h + y) * w + x] = v.sin();
                out[((f + k) * h + y) * w + x] = v.cos();
            }
        }
    }
    out
}

/// `get_1d_sine_pe`: `[sin | cos]` halves of `pos / 10000^(2 floor(i/2) / (dim/2))`.
pub(crate) fn sine_1d(pos: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let t: Vec<f32> = (0..half)
        .map(|i| 10000f32.powf((2 * (i / 2)) as f32 / half as f32))
        .collect();
    let mut v: Vec<f32> = t.iter().map(|&d| (pos / d).sin()).collect();
    v.extend(t.iter().map(|&d| (pos / d).cos()));
    v
}

pub(crate) fn bf16_round(v: &mut [f32]) {
    for x in v.iter_mut() {
        let bits = x.to_bits();
        // Round to nearest even on the dropped 16 bits (torch's float -> bfloat16).
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
        *x = f32::from_bits(rounded);
    }
}

pub(crate) fn nchw_to_tokens(v: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * hw];
    for ch in 0..c {
        for i in 0..hw {
            out[i * c + ch] = v[ch * hw + i];
        }
    }
    out
}

/// `_apply_non_overlapping_constraints`: at each pixel keep the highest-scoring
/// object; the others are clamped to at most -10.
pub(crate) fn non_overlap(masks: &mut [f32], n: usize, px: usize) {
    if n <= 1 {
        return;
    }
    for i in 0..px {
        let best = (0..n)
            .max_by(|&a, &b| masks[a * px + i].total_cmp(&masks[b * px + i]))
            .expect("n >= 1");
        for o in 0..n {
            if o != best {
                masks[o * px + i] = masks[o * px + i].min(-10.0);
            }
        }
    }
}

impl<'b, B: Backend + ?Sized> Vos<'b, B> {
    /// The tracker for `backbone`'s input size ([`Backbone::geometry`]).
    pub fn new(
        be: &'b B,
        backbone: &'b Backbone<'b, B>,
        g: &TensorGroup,
    ) -> Result<Self, ModelError> {
        let geo = backbone.geometry();
        let feat = geo.grid;
        let p = "tracker.model";
        let host_w = |n: &str, shape: &[usize]| -> Result<Vec<f32>, ModelError> {
            Ok(get(g, &format!("{p}.{n}"), shape)?.data.clone())
        };
        let t = |n: &str, shape: &[usize]| -> Result<Tensor, ModelError> {
            Ok(up(be, &host_w(n, shape)?, shape)?)
        };
        let pe_p = format!("{p}.interactive_sam_prompt_encoder");
        let ln = |i: usize, c: usize| -> Result<Option<(Tensor, Tensor)>, ModelError> {
            Ok(Some((
                t(
                    &format!("interactive_sam_prompt_encoder.mask_downscaling.{i}.weight"),
                    &[c],
                )?,
                t(
                    &format!("interactive_sam_prompt_encoder.mask_downscaling.{i}.bias"),
                    &[c],
                )?,
            )))
        };
        let prop_gauss = host_w(
            "image_pe_layer.positional_encoding_gaussian_matrix",
            &[2, D / 2],
        )?;
        let sine_pos = host::sine_position_map(feat, feat);
        let pos_host = nchw_to_tokens(&sine_pos, D, feat * feat);
        let maskmem_tpos = host_w("maskmem_tpos_enc", &[NUM_MASKMEM, 1, 1, D])?;
        let pos_tokens = up(be, &pos_host, &[1, feat * feat, D])?;
        let memory_pos_host: Vec<Vec<f32>> = maskmem_tpos
            .chunks_exact(D)
            .map(|te| {
                pos_host
                    .chunks_exact(D)
                    .flat_map(|r| r.iter().zip(te).map(|(a, b)| a + b))
                    .collect()
            })
            .collect();
        let memory_pos_tokens = memory_pos_host
            .iter()
            .map(|h| up(be, h, &[1, feat * feat, D]))
            .collect::<Result<Vec<_>, _>>()?;
        let inter_gauss = host_w(
            "interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            &[2, D / 2],
        )?;
        Ok(Self {
            be,
            backbone,
            geo,
            mux_decoder: MuxMaskDecoder::load(be, g, &format!("{p}.sam_mask_decoder"))?,
            interactive: InteractiveDecoder::load(
                be,
                g,
                &format!("{p}.interactive_sam_mask_decoder"),
            )?,
            i_conv_s0: (
                t(
                    "interactive_sam_mask_decoder.conv_s0.weight",
                    &[D / 8, D, 1, 1],
                )?,
                t("interactive_sam_mask_decoder.conv_s0.bias", &[D / 8])?,
            ),
            i_conv_s1: (
                t(
                    "interactive_sam_mask_decoder.conv_s1.weight",
                    &[D / 4, D, 1, 1],
                )?,
                t("interactive_sam_mask_decoder.conv_s1.bias", &[D / 4])?,
            ),
            mem_enc: MemoryEncoder::load_at(be, g, &format!("{p}.maskmem_backbone"), geo)?,
            mem_attn: MemoryAttention::load_at(be, g, &format!("{p}.transformer.encoder"), geo)?,
            mask_downsample: Conv::load(
                be,
                g,
                &format!("{p}.interactive_mask_downsample"),
                [1, 1, 4, 4],
            )?,
            pe_downscale: [
                (
                    Conv::load(be, g, &format!("{pe_p}.mask_downscaling.0"), [4, 1, 2, 2])?,
                    ln(1, 4)?,
                ),
                (
                    Conv::load(be, g, &format!("{pe_p}.mask_downscaling.3"), [16, 4, 2, 2])?,
                    ln(4, 16)?,
                ),
                (
                    Conv::load(be, g, &format!("{pe_p}.mask_downscaling.6"), [D, 16, 1, 1])?,
                    None,
                ),
            ],
            not_a_point: host_w(
                "interactive_sam_prompt_encoder.not_a_point_embed.weight",
                &[1, D],
            )?,
            no_mem_embed_device: {
                let w = host_w("interactivity_no_mem_embed", &[1, 1, D])?;
                up(be, &w, &[1, D, 1, 1])?
            },
            obj_ptr_proj: Mlp::load_host(
                be,
                g,
                &format!("{p}.obj_ptr_proj"),
                &[D, D, D, D],
                false,
                false,
            )?,
            interactive_obj_ptr_proj: Mlp::load_host(
                be,
                g,
                &format!("{p}.interactive_obj_ptr_proj"),
                &[D, D, D, D],
                false,
                false,
            )?,
            no_obj_ptr_linear: Lin::load_host(be, g, &format!("{p}.no_obj_ptr_linear"), D, D)?,
            obj_ptr_tpos_proj: Lin::load_host(be, g, &format!("{p}.obj_ptr_tpos_proj"), D, D)?,
            maskmem_tpos,
            no_obj_embed_spatial: host_w("no_obj_embed_spatial", &[MUX, D])?,
            output_valid: host_w("output_valid_embed", &[MUX, D])?,
            output_invalid: host_w("output_invalid_embed", &[MUX, D])?,
            prop_pe: up(
                be,
                &random_fourier_pe(&prop_gauss, feat, feat),
                &[1, D, feat, feat],
            )?,
            inter_pe: up(
                be,
                &random_fourier_pe(&inter_gauss, feat, feat),
                &[1, D, feat, feat],
            )?,
            sine_pos,
            pos_tokens,
            memory_pos_tokens,
            pos_host,
            memory_pos_host,
            ext_mem_attn: None,
            ext_mem_enc: None,
            mem_enc_calls: std::cell::Cell::new([0, 0]),
        })
    }

    /// Propagation's memory attention runs in `ext` instead of on `be` (the VOS driver's
    /// [`MemoryAttention::fwd`] stays on `be`). The caller refuses this when an adapter or weight
    /// edit changes memory attention's weights, which `ext` does not see.
    pub fn with_external_mem_attn(mut self, ext: Box<dyn ExternalMemAttn + 'b>) -> Self {
        self.ext_mem_attn = Some(ext);
        self
    }

    /// Mask memories whose masks arrive at the downsampler's input size ([`Geometry::input_mask`],
    /// the propagation path) are encoded in `ext` instead of on `be`. A memory encoded from masks at
    /// another size (a merge at the image size) stays on `be`; [`Self::mem_encoder_calls`] counts
    /// both. No adapter or weight edit targets the memory encoder's weights today; the caller
    /// refuses one that does.
    pub fn with_external_mem_encoder(mut self, ext: Box<dyn ExternalMemEncoder + 'b>) -> Self {
        self.ext_mem_enc = Some(ext);
        self
    }

    /// What [`Self::with_external_mem_encoder`] installed, as [`ExternalMemEncoder::describe`]
    /// says it.
    pub fn external_mem_encoder(&self) -> Option<String> {
        self.ext_mem_enc.as_ref().map(|e| e.describe())
    }

    /// Memory encodes so far: `[external, Ojas]`.
    pub fn mem_encoder_calls(&self) -> [usize; 2] {
        self.mem_enc_calls.get()
    }

    /// What [`Self::with_external_mem_attn`] installed, as [`ExternalMemAttn::describe`] says it.
    pub fn external_mem_attn(&self) -> Option<String> {
        self.ext_mem_attn.as_ref().map(|e| e.describe())
    }

    /// The input size the tracker's feature maps, masks and memories follow.
    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    /// Track `n` objects through `frames` (each upstream's preprocessed
    /// `[1, 3, S, S]` image, `S` = 1008 at the native size), given masks on frame 0 at the video's
    /// resolution `[n, h, w]` (0/1). Calls `sink` with each frame's output.
    pub fn run(
        &self,
        frames: &[Tensor],
        masks0: &[f32],
        n: usize,
        h: usize,
        w: usize,
        mut sink: impl FnMut(FrameOut),
    ) -> Result<(), ModelError> {
        let be = self.be;
        let tokens = self.geo.tokens();
        assert!(
            n >= 1 && n <= MUX,
            "{n} objects: one bucket of at most {MUX} is wired"
        );
        let slot_valid: Vec<bool> = (0..MUX).map(|s| s < n).collect();
        let mut mem: Vec<Option<FrameMem>> = Vec::with_capacity(frames.len());
        for (fi, frame) in frames.iter().enumerate() {
            let trunk = self.backbone.trunk_inference(frame)?;
            let prop = self.backbone.neck(Neck::Propagation, &trunk)?;
            let (ps0, ps1) = high_res(be, &self.mux_decoder, &prop[0], &prop[1])?;
            let prop_tokens = nchw_to_tokens(&host_slice(&prop[2], be)?, D, tokens);
            let pos_tokens = nchw_to_tokens(&self.sine_pos, D, tokens);
            let out = if fi == 0 {
                let inter = self.backbone.neck(Neck::Interactive, &trunk)?;
                self.first_frame(
                    &inter,
                    &prop,
                    &prop_tokens,
                    &pos_tokens,
                    masks0,
                    n,
                    h,
                    w,
                    &slot_valid,
                    &mut mem,
                )?
            } else {
                self.propagate(
                    fi,
                    frames.len(),
                    &prop[2],
                    &ps0,
                    &ps1,
                    &prop_tokens,
                    &pos_tokens,
                    n,
                    h,
                    w,
                    &slot_valid,
                    &mut mem,
                )?
            };
            sink(out);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn first_frame(
        &self,
        inter: &[Tensor; 3],
        prop: &[Tensor; 3],
        prop_tokens: &[f32],
        pos_tokens: &[f32],
        masks0: &[f32],
        n: usize,
        h: usize,
        w: usize,
        slot_valid: &[bool],
        mem: &mut Vec<Option<FrameMem>>,
    ) -> Result<FrameOut, ModelError> {
        let be = self.be;
        let (_feat, _tokens, low_res, input_mask) = (
            self.geo.grid,
            self.geo.tokens(),
            self.geo.low_res(),
            self.geo.input_mask(),
        );
        // add_new_masks: masks to the model's input mask size (antialiased).
        let mask_in = st_video::resize_bilinear_aa_f32(masks0, n, h, w, input_mask, input_mask);
        let appearing: Vec<bool> = (0..n)
            .map(|o| masks0[o * h * w..(o + 1) * h * w].iter().any(|&v| v > 0.0))
            .collect();

        // Object pointers from the interactive path (_use_mask_as_output).
        let pix = be.broadcast_add_forward(&inter[2], &self.no_mem_embed_device)?;
        let p1 = Conv2dParams::new();
        let is0 = be.conv2d_forward(&inter[0], &self.i_conv_s0.0, Some(&self.i_conv_s0.1), p1)?;
        let is1 = be.conv2d_forward(&inter[1], &self.i_conv_s1.0, Some(&self.i_conv_s1.1), p1)?;
        let mut k4 = Conv2dParams::new();
        k4.stride = [4, 4];
        let down = be.conv2d_forward(
            &up(be, &mask_in, &[n, 1, input_mask, input_mask])?,
            &self.mask_downsample.w,
            Some(&self.mask_downsample.b),
            k4,
        )?;
        let ds = down.shape()[2];
        let prompt = if ds == low_res {
            down
        } else {
            up(
                be,
                &st_video::resize_bilinear_aa_f32(
                    &host_of(&down, be)?,
                    n,
                    ds,
                    ds,
                    low_res,
                    low_res,
                ),
                &[n, 1, low_res, low_res],
            )?
        };
        let mut x = prompt;
        let mut k2 = Conv2dParams::new();
        k2.stride = [2, 2];
        for (i, (conv, ln)) in self.pe_downscale.iter().enumerate() {
            x = be.conv2d_forward(&x, &conv.w, Some(&conv.b), if i < 2 { k2 } else { p1 })?;
            if let Some((lw, lb)) = ln {
                x = be.gelu_erf_forward(&be.channel_layer_norm_forward(&x, lw, lb, LN2D_EPS)?)?;
            }
        }
        let sparse: Vec<f32> = (0..2 * n).flat_map(|_| self.not_a_point.clone()).collect();
        let io = self
            .interactive
            .fwd(be, &pix, &self.inter_pe, &sparse, 2, &x, &is0, &is1)?;
        let mut ptrs = vec![0f32; MUX * D];
        for o in 0..n {
            let tok = &io.mask_tokens[o * 4 * D..o * 4 * D + D]; // mask token 0
                                                                 // Gated twice, as upstream does: first on the interactive decoder's own object
                                                                 // score (_forward_sam_heads), then on whether the input mask is empty
                                                                 // (_use_mask_as_output); each gate maps a pointer through no_obj_ptr_linear.
            let ptr = self.interactive_obj_ptr_proj.host(tok, 1);
            let ptr = if io.object_score_logits[o] > 0.0 {
                ptr
            } else {
                self.no_obj_ptr_linear.host(&ptr, 1)
            };
            let ptr = if appearing[o] {
                ptr
            } else {
                self.no_obj_ptr_linear.host(&ptr, 1)
            };
            ptrs[o * D..(o + 1) * D].copy_from_slice(&ptr);
        }
        let scores: Vec<f32> = appearing
            .iter()
            .map(|&a| if a { 10.0 } else { -10.0 })
            .collect();

        // Consolidation + preflight: video-res masks at +-1024 -> 288 (antialiased)
        // -> 1008 (bilinear) -> non-overlap -> memory encoder.
        let video_logits: Vec<f32> = masks0
            .iter()
            .map(|&v| if v > 0.5 { -NO_OBJ_SCORE } else { NO_OBJ_SCORE })
            .collect();
        let low = st_video::resize_bilinear_aa_f32(&video_logits, n, h, w, low_res, low_res);
        let maskmem = self.encode_memory(
            &prop[2],
            &low,
            &scores,
            n,
            &(0..n).collect::<Vec<_>>(),
            true,
            slot_valid,
        )?;
        mem.push(Some(FrameMem {
            maskmem: maskmem.clone(),
            image: prop_tokens.to_vec(),
            image_pos: pos_tokens.to_vec(),
            obj_ptr: ptrs.clone(),
        }));
        // Output: bilinear to the video's size (an upscale; upstream does not antialias it).
        let mut video_res = host_of(
            &be.interpolate_bilinear_forward(
                &up(be, &low, &[n, 1, low_res, low_res])?,
                [h, w],
                false,
            )?,
            be,
        )?;
        non_overlap(&mut video_res, n, h * w);
        Ok(FrameOut {
            frame: 0,
            low_res: low,
            video_res,
            object_scores: scores,
            maskmem,
            obj_ptr: ptrs,
        })
    }

    /// `_encode_new_memory` + bf16 storage for `n` objects' low-res logits.
    /// `non_overlap`: the consolidated conditioning frame applies the constraint
    /// first; propagation does not (`non_overlap_masks_for_mem_enc` is False).
    #[allow(clippy::too_many_arguments)]
    fn encode_memory(
        &self,
        pix72: &Tensor,
        low: &[f32],
        scores: &[f32],
        n: usize,
        conditioning: &[usize],
        apply_non_overlap: bool,
        slot_valid: &[bool],
    ) -> Result<Vec<f32>, ModelError> {
        let be = self.be;
        let (low_res, image, tokens) = (self.geo.low_res(), self.geo.pixels(), self.geo.tokens());
        let low_t = up(be, low, &[n, 1, low_res, low_res])?;
        let mut hr = host_of(
            &be.interpolate_bilinear_forward(&low_t, [image, image], false)?,
            be,
        )?;
        if apply_non_overlap {
            non_overlap(&mut hr, n, image * image);
        }
        let px = image * image;
        let mut chans = vec![0f32; 2 * MUX * px];
        for o in 0..n {
            for i in 0..px {
                chans[o * px + i] = host::sigmoid(hr[o * px + i]) * 2.0 - 1.0;
            }
            if conditioning.contains(&o) {
                chans[(MUX + o) * px..(MUX + o + 1) * px]
                    .iter_mut()
                    .for_each(|v| *v = 1.0);
            }
        }
        let masks = up(be, &chans, &[1, 2 * MUX, image, image])?;
        let enc = self.mem_enc.fwd(be, pix72, &masks)?;
        let mut feat = host_of(&enc.features, be)?;
        // no_obj_embed_spatial for every slot not appearing (padding slots score 0).
        let mut add = vec![0f32; D];
        for s in 0..MUX {
            let score = if slot_valid[s] { scores[s] } else { 0.0 };
            if score <= 0.0 {
                add.iter_mut()
                    .zip(&self.no_obj_embed_spatial[s * D..(s + 1) * D])
                    .for_each(|(a, b)| *a += b);
            }
        }
        for c in 0..D {
            feat[c * tokens..(c + 1) * tokens]
                .iter_mut()
                .for_each(|v| *v += add[c]);
        }
        bf16_round(&mut feat);
        Ok(feat)
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &self,
        fi: usize,
        num_frames: usize,
        prop72: &Tensor,
        ps0: &Tensor,
        ps1: &Tensor,
        prop_tokens: &[f32],
        pos_tokens: &[f32],
        n: usize,
        h: usize,
        w: usize,
        slot_valid: &[bool],
        mem: &mut Vec<Option<FrameMem>>,
    ) -> Result<FrameOut, ModelError> {
        let be = self.be;
        let (feat, tokens, low_res) = (self.geo.grid, self.geo.tokens(), self.geo.low_res());
        // Memory frames: the conditioning frame (0), then t_pos 1..6 -> frame fi - (7 - t_pos).
        let tpos = |t_pos: usize| -> &[f32] {
            let idx = if t_pos == 0 || t_pos >= NUM_MASKMEM {
                NUM_MASKMEM - 1
            } else {
                NUM_MASKMEM - t_pos - 1
            };
            &self.maskmem_tpos[idx * D..(idx + 1) * D]
        };
        let mut entries: Vec<(usize, usize)> = vec![(0, fi)]; // (frame, t_pos), conditioning first
        for t_pos in 1..NUM_MASKMEM {
            let prev = fi as i64 - (NUM_MASKMEM - t_pos) as i64;
            if prev >= 1 && mem.get(prev as usize).is_some_and(|m| m.is_some()) {
                entries.push((prev as usize, t_pos));
            }
        }
        let (mut memory, mut memory_pos, mut mimage, mut mimage_pos) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let maskmem_pos = nchw_to_tokens(&self.sine_pos, D, tokens);
        for &(f, t_pos) in &entries {
            let m = mem[f].as_ref().expect("stored");
            let te = tpos(t_pos);
            memory.extend(nchw_to_tokens(&m.maskmem, D, tokens));
            memory_pos.extend(
                maskmem_pos
                    .chunks_exact(D)
                    .flat_map(|r| r.iter().zip(te).map(|(a, b)| a + b)),
            );
            mimage.extend_from_slice(&m.image);
            mimage_pos.extend(
                m.image_pos
                    .chunks_exact(D)
                    .flat_map(|r| r.iter().zip(te).map(|(a, b)| a + b)),
            );
        }
        // Object pointers: the conditioning frame at |fi - 0|, then frames fi - 1 .. fi - 15 (and later ones, none yet).
        let max_ptr = num_frames.min(MAX_PTRS);
        let mut ptr_frames: Vec<(usize, usize)> = vec![(0, fi)];
        for t_diff in 1..max_ptr {
            if t_diff > fi {
                break;
            }
            let t = fi - t_diff;
            if t >= 1 && mem.get(t).is_some_and(|m| m.is_some()) {
                ptr_frames.push((t, t_diff));
            }
        }
        let t_diff_max = (max_ptr - 1).max(1) as f32;
        for &(f, pos) in &ptr_frames {
            let m = mem[f].as_ref().expect("stored");
            let pe = self
                .obj_ptr_tpos_proj
                .host(&sine_1d(pos as f32 / t_diff_max, D), 1);
            memory.extend_from_slice(&m.obj_ptr);
            for _ in 0..MUX {
                memory_pos.extend_from_slice(&pe);
            }
        }
        let n_ptr = ptr_frames.len() * MUX;
        let mtok = memory.len() / D;
        let mitok = mimage.len() / D;
        let src_pos = up(be, pos_tokens, &[1, tokens, D])?;
        let image = up(be, prop_tokens, &[1, tokens, D])?;
        let (memory, memory_pos) = (
            up(be, &memory, &[1, mtok, D])?,
            up(be, &memory_pos, &[1, mtok, D])?,
        );
        let (mimage, mimage_pos) = (
            up(be, &mimage, &[1, mitok, D])?,
            up(be, &mimage_pos, &[1, mitok, D])?,
        );
        let cond = self.mem_attn.fwd(
            be,
            &MemoryInputs {
                image: &image,
                src: &image,
                src_pos: &src_pos,
                memory_image: &mimage,
                memory_image_pos: &mimage_pos,
                memory: &memory,
                memory_pos: &memory_pos,
                num_obj_ptr_tokens: n_ptr,
            },
        )?;
        let pix = be
            .permute(&cond, &[0, 2, 1])?
            .reshape(&[1, D, feat, feat])?;

        // Multiplex decode.
        let extra: Vec<f32> = (0..MUX)
            .flat_map(|s| {
                if slot_valid[s] {
                    self.output_valid[s * D..(s + 1) * D].to_vec()
                } else {
                    self.output_invalid[s * D..(s + 1) * D].to_vec()
                }
            })
            .collect();
        let out = self
            .mux_decoder
            .fwd(be, &pix, &self.prop_pe, ps0, ps1, &extra)?;
        drop(pix);
        let masks_view = host_slice(&out.masks, be)?; // [1, 16, 3, M, M]
        let masks = masks_view.as_slice();
        let px = low_res * low_res;
        let mut low = vec![0f32; n * px];
        let mut ptrs = vec![0f32; MUX * D];
        let mut scores = vec![0f32; n];
        for o in 0..n {
            let ious = &out.iou_pred[o * MASKS_PER_OBJ..(o + 1) * MASKS_PER_OBJ];
            let best = (0..MASKS_PER_OBJ)
                .max_by(|&a, &b| ious[a].total_cmp(&ious[b]))
                .expect("3 masks");
            let score = out.object_score_logits[o];
            scores[o] = score;
            let src = &masks[(o * MASKS_PER_OBJ + best) * px..(o * MASKS_PER_OBJ + best + 1) * px];
            let dst = &mut low[o * px..(o + 1) * px];
            if score > 0.0 {
                dst.copy_from_slice(src);
            } else {
                dst.iter_mut().for_each(|v| *v = NO_OBJ_SCORE);
            }
            let tok =
                &out.sam_tokens[(o * MASKS_PER_OBJ + best) * D..(o * MASKS_PER_OBJ + best + 1) * D];
            let ptr = self.obj_ptr_proj.host(tok, 1);
            let ptr = if score > 0.0 {
                ptr
            } else {
                self.no_obj_ptr_linear.host(&ptr, 1)
            };
            ptrs[o * D..(o + 1) * D].copy_from_slice(&ptr);
        }
        drop((masks_view, out));
        let maskmem = self.encode_memory(prop72, &low, &scores, n, &[], false, slot_valid)?;
        mem.push(Some(FrameMem {
            maskmem: maskmem.clone(),
            image: prop_tokens.to_vec(),
            image_pos: pos_tokens.to_vec(),
            obj_ptr: ptrs.clone(),
        }));
        let mut video_res = host_of(
            &be.interpolate_bilinear_forward(
                &up(be, &low, &[n, 1, low_res, low_res])?,
                [h, w],
                false,
            )?,
            be,
        )?;
        non_overlap(&mut video_res, n, h * w);
        Ok(FrameOut {
            frame: fi,
            low_res: low,
            video_res,
            object_scores: scores,
            maskmem,
            obj_ptr: ptrs,
        })
    }
}
