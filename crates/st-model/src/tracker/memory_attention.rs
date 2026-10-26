//! SAM 3.1's memory attention (`decoder.py` `TransformerEncoderDecoupledCrossAttention`
//! with `DecoupledTransformerDecoderLayerv2` x4, as `_create_multiplex_transformer`
//! builds it): conditions the current frame's propagation features on stored
//! memories and object pointers.
//!
//! Per layer (pre-norm): self-attention with 2-D RoPE (72x72, head dim 32);
//! cross-attention whose queries add a projection of the raw image features and
//! whose keys add a projection of the memory frames' image features plus their
//! positions, RoPE on the memory-frame keys (the table repeated per frame, the
//! object-pointer keys left unrotated); GELU FFN. 8 heads of 32. The input is
//! `src + 0.1 * src_pos`; the output is LayerNorm'd.

use ojas_core::{axial_rope_2d_tables, AttentionOpts, Backend, OjasError, Tensor};
use st_weights::TensorGroup;

use crate::layers::{host_of, up, Lin, Norm};
use crate::ModelError;

pub const D: usize = 256;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = D / HEADS;
pub const FEAT: usize = 72;
pub const TOKENS: usize = FEAT * FEAT;

struct Layer {
    sa: [Lin; 4],
    ca_q: Lin,
    ca_k: Lin,
    ca_v: Lin,
    ca_out: Lin,
    img_q: Lin,
    img_k: Lin,
    l1: Lin,
    l2: Lin,
    n1: Norm,
    n2: Norm,
    n3: Norm,
}

pub struct MemoryAttention {
    layers: Vec<Layer>,
    norm: Norm,
    /// RoPE table for one 72x72 frame: cos and sin `[5184, 16]`, host.
    cos: Vec<f32>,
    sin: Vec<f32>,
}

/// One frame's inputs, batch-first, `B` buckets.
pub struct MemoryInputs<'a> {
    /// Current frame features, `[1, 5184, D]`.
    pub image: &'a Tensor,
    /// The same features expanded per bucket, `[B, 5184, D]`, and their positions.
    pub src: &'a Tensor,
    pub src_pos: &'a Tensor,
    /// Memory frames' image features and positions (with temporal encodings), `[B, F * 5184, D]`.
    pub memory_image: &'a Tensor,
    pub memory_image_pos: &'a Tensor,
    /// Mask memories then object pointers, `[B, F * 5184 + P, D]`, and their positions.
    pub memory: &'a Tensor,
    pub memory_pos: &'a Tensor,
    pub num_obj_ptr_tokens: usize,
}

impl MemoryAttention {
    pub fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        let lin = |l: &str, n: &str, out: usize, inp: usize| Lin::load(be, g, &format!("{l}.{n}"), out, inp);
        let layers = (0..4)
            .map(|i| -> Result<Layer, ModelError> {
                let l = format!("{p}.layers.{i}");
                Ok(Layer {
                    sa: [lin(&l, "self_attn_q_proj", D, D)?, lin(&l, "self_attn_k_proj", D, D)?, lin(&l, "self_attn_v_proj", D, D)?, lin(&l, "self_attn_out_proj", D, D)?],
                    ca_q: lin(&l, "cross_attn_q_proj", D, D)?,
                    ca_k: lin(&l, "cross_attn_k_proj", D, D)?,
                    ca_v: lin(&l, "cross_attn_v_proj", D, D)?,
                    ca_out: lin(&l, "cross_attn_out_proj", D, D)?,
                    img_q: lin(&l, "image_cross_attn_q_proj", D, D)?,
                    img_k: lin(&l, "image_cross_attn_k_proj", D, D)?,
                    l1: lin(&l, "linear1", 2048, D)?,
                    l2: lin(&l, "linear2", D, 2048)?,
                    n1: Norm::load(be, g, &format!("{l}.norm1"))?,
                    n2: Norm::load(be, g, &format!("{l}.norm2"))?,
                    n3: Norm::load(be, g, &format!("{l}.norm3"))?,
                })
            })
            .collect::<Result<_, _>>()?;
        let (cos, sin) = axial_rope_2d_tables(HEAD_DIM, FEAT, FEAT, 10000.0, 1.0, be.budget())?;
        Ok(Self { layers, norm: Norm::load(be, g, &format!("{p}.norm"))?, cos: cos.to_f32_vec()?, sin: sin.to_f32_vec()? })
    }

    /// RoPE tables for `frames` memory frames followed by `ptrs` unrotated keys.
    fn key_tables<B: Backend + ?Sized>(&self, be: &B, frames: usize, ptrs: usize) -> Result<(Tensor, Tensor), OjasError> {
        let half = HEAD_DIM / 2;
        let (mut c, mut s) = (self.cos.repeat(frames), self.sin.repeat(frames));
        c.extend(std::iter::repeat_n(1.0, ptrs * half));
        s.extend(std::iter::repeat_n(0.0, ptrs * half));
        let n = frames * TOKENS + ptrs;
        Ok((up(be, &c, &[n, half])?, up(be, &s, &[n, half])?))
    }

    /// Returns the conditioned features, `[B, 5184, D]`.
    pub fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &MemoryInputs<'_>) -> Result<Tensor, ModelError> {
        let b = x.src.shape()[0];
        let m = x.memory.shape()[1];
        let p = x.num_obj_ptr_tokens;
        let mi = x.memory_image.shape()[1];
        if m != mi + p || mi % TOKENS != 0 {
            return Err(ModelError::Ojas(OjasError::OutOfRange {
                op: "st_model::MemoryAttention::fwd",
                detail: format!("memory {m} tokens != memory_image {mi} + {p} pointers, or {mi} not a multiple of {TOKENS}"),
            }));
        }
        let frames = mi / TOKENS;
        // Pointer slots: zero image features; positions from the tail of memory_pos.
        let pad_mi = {
            let mut v = host_of(x.memory_image, be)?;
            v.extend(std::iter::repeat_n(0.0, b * p * D));
            reorder_tail(&mut v, b, mi, p);
            up(be, &v, &[b, m, D])?
        };
        let pad_mi_pos = {
            let mut v = host_of(x.memory_image_pos, be)?;
            let mp = host_of(x.memory_pos, be)?;
            // Upstream takes memory_pos[0:1, -P:] (bucket 0) for every bucket.
            let tail = &mp[(m - p) * D..m * D];
            let mut out = Vec::with_capacity(b * m * D);
            for bi in 0..b {
                out.extend_from_slice(&v[bi * mi * D..(bi + 1) * mi * D]);
                out.extend_from_slice(tail);
            }
            v.clear();
            up(be, &out, &[b, m, D])?
        };
        let (q_cos, q_sin) = (up(be, &self.cos, &[TOKENS, HEAD_DIM / 2])?, up(be, &self.sin, &[TOKENS, HEAD_DIM / 2])?);
        let (k_cos, k_sin) = self.key_tables(be, frames, p)?;

        let heads = |t: &Tensor| -> Result<Tensor, OjasError> {
            let (bb, n) = (t.shape()[0], t.shape()[1]);
            t.reshape(&[bb, n, HEADS, HEAD_DIM])
        };
        let to_bhtd = |t: &Tensor| be.permute(t, &[0, 2, 1, 3]);
        let merge = |t: &Tensor, n: usize| -> Result<Tensor, OjasError> { be.permute(t, &[0, 2, 1, 3])?.reshape(&[b, n, D]) };
        let opts = AttentionOpts::default();

        let src_pos_scaled = be.scale_forward(x.src_pos, 0.1)?;
        let mut tgt = be.residual_add_forward(x.src, &src_pos_scaled)?;
        // The raw image is per frame, shared by every bucket.
        let image_b = if b == 1 { x.image.clone() } else { up(be, &host_of(x.image, be)?.repeat(b), &[b, TOKENS, D])? };
        for l in &self.layers {
            // Self-attention with RoPE on q and k.
            let t2 = l.n1.fwd(be, &tgt)?;
            let q = be.rope_interleaved_forward(&heads(&l.sa[0].fwd(be, &t2)?)?, &q_cos, &q_sin)?;
            let k = be.rope_interleaved_forward(&heads(&l.sa[1].fwd(be, &t2)?)?, &q_cos, &q_sin)?;
            let v = heads(&l.sa[2].fwd(be, &t2)?)?;
            let a = be.attention_forward(&to_bhtd(&q)?, &to_bhtd(&k)?, &to_bhtd(&v)?, opts)?;
            tgt = be.residual_add_forward(&tgt, &l.sa[3].fwd(be, &merge(&a, TOKENS)?)?)?;
            // Decoupled cross-attention to the memories.
            let t2 = l.n2.fwd(be, &tgt)?;
            let q = be.residual_add_forward(&l.img_q.fwd(be, &image_b)?, &l.ca_q.fwd(be, &t2)?)?;
            let k = be.residual_add_forward(&l.img_k.fwd(be, &pad_mi)?, &l.ca_k.fwd(be, x.memory)?)?;
            let k = be.residual_add_forward(&k, &pad_mi_pos)?;
            let v = heads(&l.ca_v.fwd(be, x.memory)?)?;
            let q = be.rope_interleaved_forward(&heads(&q)?, &q_cos, &q_sin)?;
            let k = be.rope_interleaved_forward(&heads(&k)?, &k_cos, &k_sin)?;
            let a = be.attention_forward(&to_bhtd(&q)?, &to_bhtd(&k)?, &to_bhtd(&v)?, opts)?;
            tgt = be.residual_add_forward(&tgt, &l.ca_out.fwd(be, &merge(&a, TOKENS)?)?)?;
            // FFN (GELU).
            let t2 = l.n3.fwd(be, &tgt)?;
            let f = l.l2.fwd(be, &be.gelu_erf_forward(&l.l1.fwd(be, &t2)?)?)?;
            tgt = be.residual_add_forward(&tgt, &f)?;
        }
        Ok(self.norm.fwd(be, &tgt)?)
    }
}

/// `v` holds `b` buckets of `mi` rows followed by `b * p` zero rows; move each
/// bucket's `p` zero rows to sit right after that bucket's `mi` rows.
fn reorder_tail(v: &mut Vec<f32>, b: usize, mi: usize, p: usize) {
    if b == 1 || p == 0 {
        return;
    }
    let mut out = Vec::with_capacity(v.len());
    for bi in 0..b {
        out.extend_from_slice(&v[bi * mi * D..(bi + 1) * mi * D]);
        out.extend(std::iter::repeat_n(0.0, p * D));
    }
    *v = out;
}
