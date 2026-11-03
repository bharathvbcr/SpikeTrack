//! SAM 3.1's memory attention (`decoder.py` `TransformerEncoderDecoupledCrossAttention`
//! with `DecoupledTransformerDecoderLayerv2` x4, as `_create_multiplex_transformer`
//! builds it): conditions the current frame's propagation features on stored
//! memories and object pointers.
//!
//! Per layer (pre-norm): self-attention with 2-D RoPE (grid x grid, 72 at 1008; head dim 32);
//! cross-attention whose queries add a projection of the raw image features and
//! whose keys add a projection of the memory frames' image features plus their
//! positions, RoPE on the memory-frame keys (the table repeated per frame, the
//! object-pointer keys left unrotated); GELU FFN. 8 heads of 32. The input is
//! `src + 0.1 * src_pos`; the output is LayerNorm'd.

use ojas_core::{axial_rope_2d_tables, AttentionOpts, Backend, DType, OjasError, Tensor};
use st_gate::Geometry;
use st_weights::TensorGroup;

use crate::layers::{Lin, Norm};
use crate::ModelError;

pub const D: usize = 256;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = D / HEADS;

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
    /// Tokens per frame (5184 at 1008).
    tokens: usize,
    /// RoPE table for one grid x grid frame: cos and sin `[grid^2, 16]`, on the device.
    cos: Tensor,
    sin: Tensor,
}

/// One frame's inputs, batch-first, `B` buckets. `T` is the tokens of one frame, `grid^2`
/// (5184 at 1008).
pub struct MemoryInputs<'a> {
    /// Current frame features, `[1, T, D]`.
    pub image: &'a Tensor,
    /// The same features expanded per bucket, `[B, T, D]`, and their positions.
    pub src: &'a Tensor,
    pub src_pos: &'a Tensor,
    /// Memory frames' image features and positions (with temporal encodings), `[B, F * T, D]`.
    pub memory_image: &'a Tensor,
    pub memory_image_pos: &'a Tensor,
    /// Mask memories then object pointers, `[B, F * T + P, D]`, and their positions.
    pub memory: &'a Tensor,
    pub memory_pos: &'a Tensor,
    pub num_obj_ptr_tokens: usize,
}

/// A stored frame's memory-side projections in every layer: keys `img_k(image) + ca_k(memory)`
/// and values `ca_v(memory)`, `[1, T, D]` each, before positions, RoPE and the head split. They
/// depend on the frame alone (its temporal position is added after them), so every later frame
/// that attends it can reuse them ([`MemoryAttention::frame_kv`]).
pub struct FrameKv {
    tokens: usize,
    layers: Vec<(Tensor, Tensor)>,
}

/// One frame's inputs to [`MemoryAttention::fwd_frames`], one bucket.
pub struct FramesInputs<'a> {
    /// Current frame features, `[1, T, D]` (also the source), and their positions.
    pub image: &'a Tensor,
    pub src_pos: &'a Tensor,
    /// The memory frames in attention order: projections and positions (with temporal
    /// encodings), `[1, T, D]`.
    pub frames: &'a [(&'a FrameKv, &'a Tensor)],
    /// Object pointers after the frames, `[1, P, D]`, and their positions.
    pub ptrs: Option<(&'a Tensor, &'a Tensor)>,
}

impl MemoryAttention {
    /// The memory attention for the native 1008 x 1008 input.
    pub fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        Self::load_at(be, g, p, Geometry::SAM3)
    }

    /// The memory attention for the input size `geo`. Its RoPE tables are not stored in the
    /// checkpoint; upstream's `SimpleRoPEAttention` rebuilds them for the feature map it is
    /// given with `compute_axial_cis(end_x=w, end_y=h)` at scale 1 (`decoder.py:1122`), unlike the
    /// trunk's global blocks, which stretch the 24-token window over the grid.
    pub fn load_at<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        geo: Geometry,
    ) -> Result<Self, ModelError> {
        let lin = |l: &str, n: &str, out: usize, inp: usize| {
            Lin::load(be, g, &format!("{l}.{n}"), out, inp)
        };
        let layers = (0..4)
            .map(|i| -> Result<Layer, ModelError> {
                let l = format!("{p}.layers.{i}");
                Ok(Layer {
                    sa: [
                        lin(&l, "self_attn_q_proj", D, D)?,
                        lin(&l, "self_attn_k_proj", D, D)?,
                        lin(&l, "self_attn_v_proj", D, D)?,
                        lin(&l, "self_attn_out_proj", D, D)?,
                    ],
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
        let (cos, sin) =
            axial_rope_2d_tables(HEAD_DIM, geo.grid, geo.grid, 10000.0, 1.0, be.budget())?;
        Ok(Self {
            layers,
            norm: Norm::load(be, g, &format!("{p}.norm"))?,
            tokens: geo.tokens(),
            cos: be.upload(&cos)?,
            sin: be.upload(&sin)?,
        })
    }

    /// RoPE tables for `frames` memory frames followed by `ptrs` unrotated keys
    /// (cos 1, sin 0), built on the device.
    fn key_tables<B: Backend + ?Sized>(
        &self,
        be: &B,
        frames: usize,
        ptrs: usize,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let half = HEAD_DIM / 2;
        let table = |t: &Tensor, ptr_value: f32| -> Result<Tensor, OjasError> {
            let fill = if ptrs > 0 {
                Some(be.fill(&[ptrs, half], DType::F32, ptr_value)?)
            } else {
                None
            };
            let parts: Vec<&Tensor> = std::iter::repeat_n(t, frames)
                .chain(fill.as_ref())
                .collect();
            match parts.as_slice() {
                [only] => Ok((*only).clone()),
                _ => be.concat(&parts, 0),
            }
        };
        Ok((table(&self.cos, 1.0)?, table(&self.sin, 0.0)?))
    }

    /// A stored frame's projections in every layer, from its image features and mask memory
    /// (`[1, T, D]` each), for [`Self::fwd_frames`].
    pub fn frame_kv<B: Backend + ?Sized>(
        &self,
        be: &B,
        image: &Tensor,
        memory: &Tensor,
    ) -> Result<FrameKv, ModelError> {
        let want = [1, self.tokens, D];
        if image.shape() != want || memory.shape() != want {
            return Err(ModelError::Ojas(OjasError::OutOfRange {
                op: "st_model::MemoryAttention::frame_kv",
                detail: format!(
                    "image {:?} and memory {:?} must both be {want:?}",
                    image.shape(),
                    memory.shape()
                ),
            }));
        }
        let layers = self
            .layers
            .iter()
            .map(|l| -> Result<(Tensor, Tensor), OjasError> {
                let k =
                    be.residual_add_forward(&l.img_k.fwd(be, image)?, &l.ca_k.fwd(be, memory)?)?;
                Ok((k, l.ca_v.fwd(be, memory)?))
            })
            .collect::<Result<_, _>>()?;
        Ok(FrameKv {
            tokens: self.tokens,
            layers,
        })
    }

    /// [`Self::fwd`] for one bucket with each memory frame's projections taken from `x.frames`
    /// ([`Self::frame_kv`]) instead of recomputed: the same ops on the same rows, so the same
    /// output where the backend's linear gives a row the same bits at any row count (CPU
    /// `Numerics::Exact` does; tests/memory_attention_cache.rs). Returns `[1, T, D]`.
    pub fn fwd_frames<B: Backend + ?Sized>(
        &self,
        be: &B,
        x: &FramesInputs<'_>,
    ) -> Result<Tensor, ModelError> {
        const OP: &str = "st_model::MemoryAttention::fwd_frames";
        let tokens = self.tokens;
        let one = [1, tokens, D];
        let p = x.ptrs.map_or(0, |(ptr, _)| ptr.shape()[1]);
        let bad = |detail: String| ModelError::Ojas(OjasError::OutOfRange { op: OP, detail });
        if x.image.shape() != one || x.src_pos.shape() != one {
            return Err(bad(format!(
                "image {:?} and src_pos {:?} must both be {one:?}",
                x.image.shape(),
                x.src_pos.shape()
            )));
        }
        for (i, (kv, pos)) in x.frames.iter().enumerate() {
            if kv.tokens != tokens || kv.layers.len() != self.layers.len() || pos.shape() != one {
                return Err(bad(format!(
                    "memory frame {i}: projections of {} tokens over {} layers and positions {:?}, want {tokens} tokens over {} layers and {one:?}",
                    kv.tokens,
                    kv.layers.len(),
                    pos.shape(),
                    self.layers.len()
                )));
            }
        }
        if let Some((ptr, ptr_pos)) = x.ptrs {
            if ptr.shape() != [1, p, D] || ptr_pos.shape() != [1, p, D] {
                return Err(bad(format!(
                    "pointers {:?} and their positions {:?} must both be [1, P, {D}]",
                    ptr.shape(),
                    ptr_pos.shape()
                )));
            }
        }
        if x.frames.is_empty() && p == 0 {
            return Err(bad("no memory frames and no pointers to attend".into()));
        }
        let cat = |parts: &[&Tensor]| -> Result<Tensor, OjasError> {
            match parts {
                [only] => Ok((*only).clone()),
                _ => be.concat(parts, 1),
            }
        };
        let positions: Vec<&Tensor> = x
            .frames
            .iter()
            .map(|(_, pos)| *pos)
            .chain(x.ptrs.map(|(_, pos)| pos))
            .collect();
        let key_pos = cat(&positions)?;
        // Pointer slots: zero image features, as `fwd` pads them.
        let zeros = (p > 0)
            .then(|| be.fill(&[1, p, D], DType::F32, 0.0))
            .transpose()?;
        self.layers_fwd(
            be,
            [x.image, x.image, x.src_pos],
            &key_pos,
            (x.frames.len(), p),
            |i, l| {
                let ptr_kv = match (x.ptrs, &zeros) {
                    (Some((ptr, _)), Some(zeros)) => Some((
                        be.residual_add_forward(&l.img_k.fwd(be, zeros)?, &l.ca_k.fwd(be, ptr)?)?,
                        l.ca_v.fwd(be, ptr)?,
                    )),
                    _ => None,
                };
                let (ks, vs): (Vec<&Tensor>, Vec<&Tensor>) = x
                    .frames
                    .iter()
                    .map(|(kv, _)| (&kv.layers[i].0, &kv.layers[i].1))
                    .chain(ptr_kv.as_ref().map(|(k, v)| (k, v)))
                    .unzip();
                Ok((cat(&ks)?, cat(&vs)?))
            },
        )
    }

    /// Returns the conditioned features, `[B, T, D]`.
    pub fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        x: &MemoryInputs<'_>,
    ) -> Result<Tensor, ModelError> {
        let tokens = self.tokens;
        let b = x.src.shape()[0];
        let m = x.memory.shape()[1];
        let p = x.num_obj_ptr_tokens;
        let mi = x.memory_image.shape()[1];
        if x.src.shape()[1] != tokens || m != mi + p || mi % tokens != 0 {
            return Err(ModelError::Ojas(OjasError::OutOfRange {
                op: "st_model::MemoryAttention::fwd",
                detail: format!("src has {} tokens, memory {m} tokens != memory_image {mi} + {p} pointers, or {mi} not a multiple of {tokens} (the tokens of one frame)", x.src.shape()[1]),
            }));
        }
        let frames = mi / tokens;
        // Pointer slots: zero image features; positions from the tail of memory_pos,
        // which upstream takes from bucket 0 (memory_pos[0:1, -P:]) for every bucket.
        let (pad_mi, pad_mi_pos) = if p == 0 {
            (x.memory_image.clone(), x.memory_image_pos.clone())
        } else {
            let zeros = be.fill(&[b, p, D], DType::F32, 0.0)?;
            let tail = be.slice(&be.slice(x.memory_pos, 0, 0, 1)?, 1, m - p, p)?;
            let tails: Vec<&Tensor> = std::iter::repeat_n(&tail, b).collect();
            let tail_b = if b == 1 {
                tail.clone()
            } else {
                be.concat(&tails, 0)?
            };
            (
                be.concat(&[x.memory_image, &zeros], 1)?,
                be.concat(&[x.memory_image_pos, &tail_b], 1)?,
            )
        };
        self.layers_fwd(
            be,
            [x.image, x.src, x.src_pos],
            &pad_mi_pos,
            (frames, p),
            |_, l| {
                let k = be
                    .residual_add_forward(&l.img_k.fwd(be, &pad_mi)?, &l.ca_k.fwd(be, x.memory)?)?;
                Ok((k, l.ca_v.fwd(be, x.memory)?))
            },
        )
    }

    /// The four layers and the final norm. `memory_kv(i, layer)` gives layer `i`'s memory keys
    /// before positions and RoPE (`img_k(image) + ca_k(memory)`, pointer rows on zero image
    /// features) and its values (`ca_v(memory)`), `[B, F * T + P, D]` each; `key_pos` is the
    /// keys' positions, the same shape.
    fn layers_fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        [image, src, src_pos]: [&Tensor; 3],
        key_pos: &Tensor,
        (frames, p): (usize, usize),
        memory_kv: impl Fn(usize, &Layer) -> Result<(Tensor, Tensor), OjasError>,
    ) -> Result<Tensor, ModelError> {
        let tokens = self.tokens;
        let b = src.shape()[0];
        let (q_cos, q_sin) = (&self.cos, &self.sin);
        let (k_cos, k_sin) = self.key_tables(be, frames, p)?;

        let heads = |t: &Tensor| -> Result<Tensor, OjasError> {
            let (bb, n) = (t.shape()[0], t.shape()[1]);
            t.reshape(&[bb, n, HEADS, HEAD_DIM])
        };
        let to_bhtd = |t: &Tensor| be.permute(t, &[0, 2, 1, 3]);
        let merge = |t: &Tensor, n: usize| -> Result<Tensor, OjasError> {
            be.permute(t, &[0, 2, 1, 3])?.reshape(&[b, n, D])
        };
        let opts = AttentionOpts::default();

        let src_pos_scaled = be.scale_forward(src_pos, 0.1)?;
        let mut tgt = be.residual_add_forward(src, &src_pos_scaled)?;
        // The raw image is per frame, shared by every bucket.
        let images: Vec<&Tensor> = std::iter::repeat_n(image, b).collect();
        let image_b = if b == 1 {
            image.clone()
        } else {
            be.concat(&images, 0)?
        };
        for (i, l) in self.layers.iter().enumerate() {
            // Self-attention with RoPE on q and k.
            let t2 = l.n1.fwd(be, &tgt)?;
            let q = be.rope_interleaved_forward(&heads(&l.sa[0].fwd(be, &t2)?)?, q_cos, q_sin)?;
            let k = be.rope_interleaved_forward(&heads(&l.sa[1].fwd(be, &t2)?)?, q_cos, q_sin)?;
            let v = heads(&l.sa[2].fwd(be, &t2)?)?;
            let a = be.attention_forward(&to_bhtd(&q)?, &to_bhtd(&k)?, &to_bhtd(&v)?, opts)?;
            tgt = be.residual_add_forward(&tgt, &l.sa[3].fwd(be, &merge(&a, tokens)?)?)?;
            // Decoupled cross-attention to the memories.
            let t2 = l.n2.fwd(be, &tgt)?;
            let q = be.residual_add_forward(&l.img_q.fwd(be, &image_b)?, &l.ca_q.fwd(be, &t2)?)?;
            let (k, v) = memory_kv(i, l)?;
            let k = be.residual_add_forward(&k, key_pos)?;
            let v = heads(&v)?;
            let q = be.rope_interleaved_forward(&heads(&q)?, q_cos, q_sin)?;
            let k = be.rope_interleaved_forward(&heads(&k)?, &k_cos, &k_sin)?;
            let a = be.attention_forward(&to_bhtd(&q)?, &to_bhtd(&k)?, &to_bhtd(&v)?, opts)?;
            tgt = be.residual_add_forward(&tgt, &l.ca_out.fwd(be, &merge(&a, tokens)?)?)?;
            // FFN (GELU).
            let t2 = l.n3.fwd(be, &tgt)?;
            let f = l.l2.fwd(be, &be.gelu_erf_forward(&l.l1.fwd(be, &t2)?)?)?;
            tgt = be.residual_add_forward(&tgt, &f)?;
        }
        Ok(self.norm.fwd(be, &tgt)?)
    }
}
