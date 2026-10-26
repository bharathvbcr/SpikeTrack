//! Semi-supervised VOS with SAM 3.1's multiplex tracker: masks on the first
//! frame, then propagation. One bucket of up to 16 objects (multiple buckets
//! are not wired yet). Mirrors upstream's `VideoTrackingMultiplexDemo` path:
//! `add_new_masks` + `propagate_in_video_preflight` on frame 0, then
//! `propagate_in_video`; see tools/reference/README.md for the configuration.

use ojas_core::{Backend, Conv2dParams, Tensor};
use st_weights::TensorGroup;

use crate::backbone::{Backbone, Neck};
use crate::host;
use crate::layers::{get, host_of, up, Conv, Lin, Mlp};
use crate::tracker::interactive::InteractiveDecoder;
use crate::tracker::mask_decoder::{high_res, MuxMaskDecoder, MASKS_PER_OBJ, MUX};
use crate::tracker::memory_attention::{MemoryAttention, MemoryInputs};
use crate::tracker::memory_encoder::MemoryEncoder;
use crate::{ModelError, Taps};

pub const D: usize = 256;
pub const FEAT: usize = 72;
pub const TOKENS: usize = FEAT * FEAT;
pub const LOW: usize = 288;
pub const IMAGE: usize = 1008;
pub const INPUT_MASK: usize = 1152;
const NUM_MASKMEM: usize = 7;
const MAX_PTRS: usize = 16;
const NO_OBJ_SCORE: f32 = -1024.0;
const LN2D_EPS: f32 = 1e-6;

/// One frame's stored state (upstream's compact `current_out`).
struct FrameMem {
    /// `[1, 256, 72, 72]`, rounded through bf16 as upstream stores it.
    maskmem: Vec<f32>,
    /// Propagation features and positions of that frame, tokens `[5184, 256]`.
    image: Vec<f32>,
    image_pos: Vec<f32>,
    /// Muxed object pointers `[16, 256]`.
    obj_ptr: Vec<f32>,
}

/// Masks the tracker produced for one frame.
pub struct FrameOut {
    pub frame: usize,
    /// `[N, 288, 288]` logits, after object-score gating.
    pub low_res: Vec<f32>,
    /// `[N, H, W]` at the video's resolution, after the non-overlap constraint.
    pub video_res: Vec<f32>,
    pub object_scores: Vec<f32>,
}

pub struct Vos<'b, B: Backend + ?Sized> {
    be: &'b B,
    backbone: &'b Backbone<'b, B>,
    mux_decoder: MuxMaskDecoder,
    interactive: InteractiveDecoder,
    i_conv_s0: (Tensor, Tensor),
    i_conv_s1: (Tensor, Tensor),
    mem_enc: MemoryEncoder,
    mem_attn: MemoryAttention,
    mask_downsample: Conv,
    pe_downscale: [(Conv, Option<(Tensor, Tensor)>); 3],
    not_a_point: Vec<f32>,
    no_mem_embed: Vec<f32>,
    obj_ptr_proj: Mlp,
    interactive_obj_ptr_proj: Mlp,
    no_obj_ptr_linear: Lin,
    obj_ptr_tpos_proj: Lin,
    maskmem_tpos: Vec<f32>,
    no_obj_embed_spatial: Vec<f32>,
    output_valid: Vec<f32>,
    output_invalid: Vec<f32>,
    prop_pe: Tensor,
    inter_pe: Tensor,
    sine72: Vec<f32>,
}

/// SAM's `PositionEmbeddingRandom` over an `h x w` grid: `[256, h, w]`.
fn random_fourier_pe(gauss: &[f32], h: usize, w: usize) -> Vec<f32> {
    let f = gauss.len() / 2; // [2, f]
    let mut out = vec![0f32; 2 * f * h * w];
    for y in 0..h {
        for x in 0..w {
            let (cx, cy) = (2.0 * ((x as f32 + 0.5) / w as f32) - 1.0, 2.0 * ((y as f32 + 0.5) / h as f32) - 1.0);
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
fn sine_1d(pos: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let t: Vec<f32> = (0..half).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / half as f32)).collect();
    let mut v: Vec<f32> = t.iter().map(|&d| (pos / d).sin()).collect();
    v.extend(t.iter().map(|&d| (pos / d).cos()));
    v
}

fn bf16_round(v: &mut [f32]) {
    for x in v.iter_mut() {
        let bits = x.to_bits();
        // Round to nearest even on the dropped 16 bits (torch's float -> bfloat16).
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
        *x = f32::from_bits(rounded);
    }
}

fn nchw_to_tokens(v: &[f32], c: usize, hw: usize) -> Vec<f32> {
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
fn non_overlap(masks: &mut [f32], n: usize, px: usize) {
    if n <= 1 {
        return;
    }
    for i in 0..px {
        let best = (0..n).max_by(|&a, &b| masks[a * px + i].total_cmp(&masks[b * px + i])).expect("n >= 1");
        for o in 0..n {
            if o != best {
                masks[o * px + i] = masks[o * px + i].min(-10.0);
            }
        }
    }
}

impl<'b, B: Backend + ?Sized> Vos<'b, B> {
    pub fn new(be: &'b B, backbone: &'b Backbone<'b, B>, g: &TensorGroup) -> Result<Self, ModelError> {
        let p = "tracker.model";
        let host_w = |n: &str, shape: &[usize]| -> Result<Vec<f32>, ModelError> { Ok(get(g, &format!("{p}.{n}"), shape)?.data.clone()) };
        let t = |n: &str, shape: &[usize]| -> Result<Tensor, ModelError> { Ok(up(be, &host_w(n, shape)?, shape)?) };
        let pe_p = format!("{p}.interactive_sam_prompt_encoder");
        let ln = |i: usize, c: usize| -> Result<Option<(Tensor, Tensor)>, ModelError> {
            Ok(Some((t(&format!("interactive_sam_prompt_encoder.mask_downscaling.{i}.weight"), &[c])?, t(&format!("interactive_sam_prompt_encoder.mask_downscaling.{i}.bias"), &[c])?)))
        };
        let prop_gauss = host_w("image_pe_layer.positional_encoding_gaussian_matrix", &[2, D / 2])?;
        let inter_gauss = host_w("interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", &[2, D / 2])?;
        Ok(Self {
            be,
            backbone,
            mux_decoder: MuxMaskDecoder::load(be, g, &format!("{p}.sam_mask_decoder"))?,
            interactive: InteractiveDecoder::load(be, g, &format!("{p}.interactive_sam_mask_decoder"))?,
            i_conv_s0: (t("interactive_sam_mask_decoder.conv_s0.weight", &[D / 8, D, 1, 1])?, t("interactive_sam_mask_decoder.conv_s0.bias", &[D / 8])?),
            i_conv_s1: (t("interactive_sam_mask_decoder.conv_s1.weight", &[D / 4, D, 1, 1])?, t("interactive_sam_mask_decoder.conv_s1.bias", &[D / 4])?),
            mem_enc: MemoryEncoder::load(be, g, &format!("{p}.maskmem_backbone"))?,
            mem_attn: MemoryAttention::load(be, g, &format!("{p}.transformer.encoder"))?,
            mask_downsample: Conv::load(be, g, &format!("{p}.interactive_mask_downsample"), [1, 1, 4, 4])?,
            pe_downscale: [
                (Conv::load(be, g, &format!("{pe_p}.mask_downscaling.0"), [4, 1, 2, 2])?, ln(1, 4)?),
                (Conv::load(be, g, &format!("{pe_p}.mask_downscaling.3"), [16, 4, 2, 2])?, ln(4, 16)?),
                (Conv::load(be, g, &format!("{pe_p}.mask_downscaling.6"), [D, 16, 1, 1])?, None),
            ],
            not_a_point: host_w("interactive_sam_prompt_encoder.not_a_point_embed.weight", &[1, D])?,
            no_mem_embed: host_w("interactivity_no_mem_embed", &[1, 1, D])?,
            obj_ptr_proj: Mlp::load(be, g, &format!("{p}.obj_ptr_proj"), &[D, D, D, D], false, false)?,
            interactive_obj_ptr_proj: Mlp::load(be, g, &format!("{p}.interactive_obj_ptr_proj"), &[D, D, D, D], false, false)?,
            no_obj_ptr_linear: Lin::load(be, g, &format!("{p}.no_obj_ptr_linear"), D, D)?,
            obj_ptr_tpos_proj: Lin::load(be, g, &format!("{p}.obj_ptr_tpos_proj"), D, D)?,
            maskmem_tpos: host_w("maskmem_tpos_enc", &[NUM_MASKMEM, 1, 1, D])?,
            no_obj_embed_spatial: host_w("no_obj_embed_spatial", &[MUX, D])?,
            output_valid: host_w("output_valid_embed", &[MUX, D])?,
            output_invalid: host_w("output_invalid_embed", &[MUX, D])?,
            prop_pe: up(be, &random_fourier_pe(&prop_gauss, FEAT, FEAT), &[1, D, FEAT, FEAT])?,
            inter_pe: up(be, &random_fourier_pe(&inter_gauss, FEAT, FEAT), &[1, D, FEAT, FEAT])?,
            sine72: host::sine_position_map(FEAT, FEAT),
        })
    }

    /// Track `n` objects through `frames` (each upstream's preprocessed
    /// `[1, 3, 1008, 1008]` image), given masks on frame 0 at the video's
    /// resolution `[n, h, w]` (0/1). Calls `sink` with each frame's output.
    pub fn run(&self, frames: &[Tensor], masks0: &[f32], n: usize, h: usize, w: usize, mut sink: impl FnMut(FrameOut)) -> Result<(), ModelError> {
        let be = self.be;
        assert!(n >= 1 && n <= MUX, "{n} objects: one bucket of at most {MUX} is wired");
        let slot_valid: Vec<bool> = (0..MUX).map(|s| s < n).collect();
        let mut mem: Vec<Option<FrameMem>> = Vec::with_capacity(frames.len());
        for (fi, frame) in frames.iter().enumerate() {
            let trunk = self.backbone.trunk(frame, &[], &mut Taps::default())?;
            let prop = self.backbone.neck(Neck::Propagation, &trunk)?;
            let (ps0, ps1) = high_res(be, &self.mux_decoder, &prop[0], &prop[1])?;
            let prop_tokens = nchw_to_tokens(&host_of(&prop[2], be)?, D, TOKENS);
            let pos_tokens = nchw_to_tokens(&self.sine72, D, TOKENS);
            let out = if fi == 0 {
                let inter = self.backbone.neck(Neck::Interactive, &trunk)?;
                self.first_frame(&inter, &prop, &prop_tokens, &pos_tokens, masks0, n, h, w, &slot_valid, &mut mem)?
            } else {
                self.propagate(fi, frames.len(), &prop[2], &ps0, &ps1, &prop_tokens, &pos_tokens, n, h, w, &slot_valid, &mut mem)?
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
        // add_new_masks: masks to the model's input mask size (antialiased).
        let mask_in = st_video::resize_bilinear_aa_f32(masks0, n, h, w, INPUT_MASK, INPUT_MASK);
        let appearing: Vec<bool> = (0..n).map(|o| masks0[o * h * w..(o + 1) * h * w].iter().any(|&v| v > 0.0)).collect();

        // Object pointers from the interactive path (_use_mask_as_output).
        let pix = {
            let mut v = host_of(&inter[2], be)?;
            for c in 0..D {
                v[c * TOKENS..(c + 1) * TOKENS].iter_mut().for_each(|x| *x += self.no_mem_embed[c]);
            }
            up(be, &v, &[1, D, FEAT, FEAT])?
        };
        let p1 = Conv2dParams::new();
        let is0 = be.conv2d_forward(&inter[0], &self.i_conv_s0.0, Some(&self.i_conv_s0.1), p1)?;
        let is1 = be.conv2d_forward(&inter[1], &self.i_conv_s1.0, Some(&self.i_conv_s1.1), p1)?;
        let mut k4 = Conv2dParams::new();
        k4.stride = [4, 4];
        let down = be.conv2d_forward(&up(be, &mask_in, &[n, 1, INPUT_MASK, INPUT_MASK])?, &self.mask_downsample.w, Some(&self.mask_downsample.b), k4)?;
        let ds = down.shape()[2];
        let prompt = if ds == LOW { down } else { up(be, &st_video::resize_bilinear_aa_f32(&host_of(&down, be)?, n, ds, ds, LOW, LOW), &[n, 1, LOW, LOW])? };
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
        let io = self.interactive.fwd(be, &pix, &self.inter_pe, &sparse, 2, &x, &is0, &is1)?;
        let mut ptrs = vec![0f32; MUX * D];
        for o in 0..n {
            let tok = &io.mask_tokens[o * 4 * D..o * 4 * D + D]; // mask token 0
            let ptr = self.interactive_obj_ptr_proj.host(tok, 1);
            let ptr = if appearing[o] { ptr } else { self.no_obj_ptr_linear.host(&ptr, 1) };
            ptrs[o * D..(o + 1) * D].copy_from_slice(&ptr);
        }
        let scores: Vec<f32> = appearing.iter().map(|&a| if a { 10.0 } else { -10.0 }).collect();

        // Consolidation + preflight: video-res masks at +-1024 -> 288 (antialiased)
        // -> 1008 (bilinear) -> non-overlap -> memory encoder.
        let video_logits: Vec<f32> = masks0.iter().map(|&v| if v > 0.5 { -NO_OBJ_SCORE } else { NO_OBJ_SCORE }).collect();
        let low = st_video::resize_bilinear_aa_f32(&video_logits, n, h, w, LOW, LOW);
        let maskmem = self.encode_memory(&prop[2], &low, &scores, n, &(0..n).collect::<Vec<_>>(), true, slot_valid)?;
        mem.push(Some(FrameMem { maskmem, image: prop_tokens.to_vec(), image_pos: pos_tokens.to_vec(), obj_ptr: ptrs }));
        // Output: bilinear to the video's size (an upscale; upstream does not antialias it).
        let mut video_res = host_of(&be.interpolate_bilinear_forward(&up(be, &low, &[n, 1, LOW, LOW])?, [h, w], false)?, be)?;
        non_overlap(&mut video_res, n, h * w);
        Ok(FrameOut { frame: 0, low_res: low, video_res, object_scores: scores })
    }

    /// `_encode_new_memory` + bf16 storage for `n` objects' low-res logits.
    /// `non_overlap`: the consolidated conditioning frame applies the constraint
    /// first; propagation does not (`non_overlap_masks_for_mem_enc` is False).
    #[allow(clippy::too_many_arguments)]
    fn encode_memory(&self, pix72: &Tensor, low: &[f32], scores: &[f32], n: usize, conditioning: &[usize], apply_non_overlap: bool, slot_valid: &[bool]) -> Result<Vec<f32>, ModelError> {
        let be = self.be;
        let low_t = up(be, low, &[n, 1, LOW, LOW])?;
        let mut hr = host_of(&be.interpolate_bilinear_forward(&low_t, [IMAGE, IMAGE], false)?, be)?;
        if apply_non_overlap {
            non_overlap(&mut hr, n, IMAGE * IMAGE);
        }
        let px = IMAGE * IMAGE;
        let mut chans = vec![0f32; 2 * MUX * px];
        for o in 0..n {
            for i in 0..px {
                chans[o * px + i] = host::sigmoid(hr[o * px + i]) * 2.0 - 1.0;
            }
            if conditioning.contains(&o) {
                chans[(MUX + o) * px..(MUX + o + 1) * px].iter_mut().for_each(|v| *v = 1.0);
            }
        }
        let masks = up(be, &chans, &[1, 2 * MUX, IMAGE, IMAGE])?;
        let enc = self.mem_enc.fwd(be, pix72, &masks)?;
        let mut feat = host_of(&enc.features, be)?;
        // no_obj_embed_spatial for every slot not appearing (padding slots score 0).
        let mut add = vec![0f32; D];
        for s in 0..MUX {
            let score = if slot_valid[s] { scores[s] } else { 0.0 };
            if score <= 0.0 {
                add.iter_mut().zip(&self.no_obj_embed_spatial[s * D..(s + 1) * D]).for_each(|(a, b)| *a += b);
            }
        }
        for c in 0..D {
            feat[c * TOKENS..(c + 1) * TOKENS].iter_mut().for_each(|v| *v += add[c]);
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
        // Memory frames: the conditioning frame (0), then t_pos 1..6 -> frame fi - (7 - t_pos).
        let tpos = |t_pos: usize| -> &[f32] {
            let idx = if t_pos == 0 || t_pos >= NUM_MASKMEM { NUM_MASKMEM - 1 } else { NUM_MASKMEM - t_pos - 1 };
            &self.maskmem_tpos[idx * D..(idx + 1) * D]
        };
        let mut entries: Vec<(usize, usize)> = vec![(0, fi)]; // (frame, t_pos), conditioning first
        for t_pos in 1..NUM_MASKMEM {
            let prev = fi as i64 - (NUM_MASKMEM - t_pos) as i64;
            if prev >= 1 && mem.get(prev as usize).is_some_and(|m| m.is_some()) {
                entries.push((prev as usize, t_pos));
            }
        }
        let (mut memory, mut memory_pos, mut mimage, mut mimage_pos) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let maskmem_pos = nchw_to_tokens(&self.sine72, D, TOKENS);
        for &(f, t_pos) in &entries {
            let m = mem[f].as_ref().expect("stored");
            let te = tpos(t_pos);
            memory.extend(nchw_to_tokens(&m.maskmem, D, TOKENS));
            memory_pos.extend(maskmem_pos.chunks_exact(D).flat_map(|r| r.iter().zip(te).map(|(a, b)| a + b)));
            mimage.extend_from_slice(&m.image);
            mimage_pos.extend(m.image_pos.chunks_exact(D).flat_map(|r| r.iter().zip(te).map(|(a, b)| a + b)));
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
            let pe = self.obj_ptr_tpos_proj.host(&sine_1d(pos as f32 / t_diff_max, D), 1);
            memory.extend_from_slice(&m.obj_ptr);
            for _ in 0..MUX {
                memory_pos.extend_from_slice(&pe);
            }
        }
        let n_ptr = ptr_frames.len() * MUX;
        let mtok = memory.len() / D;
        let mitok = mimage.len() / D;
        let src_pos = up(be, pos_tokens, &[1, TOKENS, D])?;
        let image = up(be, prop_tokens, &[1, TOKENS, D])?;
        let (memory, memory_pos) = (up(be, &memory, &[1, mtok, D])?, up(be, &memory_pos, &[1, mtok, D])?);
        let (mimage, mimage_pos) = (up(be, &mimage, &[1, mitok, D])?, up(be, &mimage_pos, &[1, mitok, D])?);
        let cond = self.mem_attn.fwd(
            be,
            &MemoryInputs { image: &image, src: &image, src_pos: &src_pos, memory_image: &mimage, memory_image_pos: &mimage_pos, memory: &memory, memory_pos: &memory_pos, num_obj_ptr_tokens: n_ptr },
        )?;
        let pix = be.permute(&cond, &[0, 2, 1])?.reshape(&[1, D, FEAT, FEAT])?;

        // Multiplex decode.
        let extra: Vec<f32> = (0..MUX).flat_map(|s| if slot_valid[s] { self.output_valid[s * D..(s + 1) * D].to_vec() } else { self.output_invalid[s * D..(s + 1) * D].to_vec() }).collect();
        let out = self.mux_decoder.fwd(be, &pix, &self.prop_pe, ps0, ps1, &extra)?;
        let masks = host_of(&out.masks, be)?; // [1, 16, 3, 288, 288]
        let px = LOW * LOW;
        let mut low = vec![0f32; n * px];
        let mut ptrs = vec![0f32; MUX * D];
        let mut scores = vec![0f32; n];
        for o in 0..n {
            let ious = &out.iou_pred[o * MASKS_PER_OBJ..(o + 1) * MASKS_PER_OBJ];
            let best = (0..MASKS_PER_OBJ).max_by(|&a, &b| ious[a].total_cmp(&ious[b])).expect("3 masks");
            let score = out.object_score_logits[o];
            scores[o] = score;
            let src = &masks[(o * MASKS_PER_OBJ + best) * px..(o * MASKS_PER_OBJ + best + 1) * px];
            let dst = &mut low[o * px..(o + 1) * px];
            if score > 0.0 {
                dst.copy_from_slice(src);
            } else {
                dst.iter_mut().for_each(|v| *v = NO_OBJ_SCORE);
            }
            let tok = &out.sam_tokens[(o * MASKS_PER_OBJ + best) * D..(o * MASKS_PER_OBJ + best + 1) * D];
            let ptr = self.obj_ptr_proj.host(tok, 1);
            let ptr = if score > 0.0 { ptr } else { self.no_obj_ptr_linear.host(&ptr, 1) };
            ptrs[o * D..(o + 1) * D].copy_from_slice(&ptr);
        }
        let maskmem = self.encode_memory(prop72, &low, &scores, n, &[], false, slot_valid)?;
        mem.push(Some(FrameMem { maskmem, image: prop_tokens.to_vec(), image_pos: pos_tokens.to_vec(), obj_ptr: ptrs }));
        let mut video_res = host_of(&be.interpolate_bilinear_forward(&up(be, &low, &[n, 1, LOW, LOW])?, [h, w], false)?, be)?;
        non_overlap(&mut video_res, n, h * w);
        Ok(FrameOut { frame: fi, low_res: low, video_res, object_scores: scores })
    }

}
