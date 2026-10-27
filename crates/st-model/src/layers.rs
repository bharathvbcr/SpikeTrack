//! Building blocks shared by the detector and the tracker: checked weight
//! lookup, linear / LayerNorm / multi-head attention / MLP / conv on the
//! Ojas backend, each with a host (f32) twin for the small per-query paths.

use ojas_core::{AttentionOpts, Backend, Conv2dParams, OjasError, Tensor};
use st_weights::{TensorGroup, Weight};

use crate::detector::{D, HEADS, LN_EPS};
use crate::host;
use crate::ModelError;

pub(crate) fn get<'a>(g: &'a TensorGroup, name: &str, shape: &[usize]) -> Result<&'a Weight, ModelError> {
    let w = g.tensors.get(name).ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    if w.shape != shape {
        return Err(ModelError::WeightShape { name: name.to_string(), expected: shape.to_vec(), found: w.shape.clone() });
    }
    Ok(w)
}

pub(crate) fn up<B: Backend + ?Sized>(be: &B, data: &[f32], shape: &[usize]) -> Result<Tensor, OjasError> {
    be.upload(&Tensor::from_f32(data, shape, be.budget())?)
}

pub(crate) fn host_of<B: Backend + ?Sized>(t: &Tensor, be: &B) -> Result<Vec<f32>, OjasError> {
    t.to_host(be.budget())?.to_f32_vec()
}

pub(crate) struct Lin {
    pub(crate) w: Tensor,
    pub(crate) b: Tensor,
    /// Host copies for the per-query paths.
    pub(crate) hw: Vec<f32>,
    pub(crate) hb: Vec<f32>,
    pub(crate) out: usize,
}

impl Lin {
    pub(crate) fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str, out: usize, inp: usize) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &[out, inp])?;
        let b = get(g, &format!("{p}.bias"), &[out])?;
        Self::from_parts(be, &w.data, &b.data, out, inp)
    }

    pub(crate) fn from_parts<B: Backend + ?Sized>(be: &B, w: &[f32], b: &[f32], out: usize, inp: usize) -> Result<Self, ModelError> {
        Ok(Self { w: up(be, w, &[out, inp])?, b: up(be, b, &[out])?, hw: w.to_vec(), hb: b.to_vec(), out })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor) -> Result<Tensor, OjasError> {
        be.linear_bias_forward(x, &self.w, &self.b)
    }

    pub(crate) fn host(&self, x: &[f32], n: usize) -> Vec<f32> {
        host::linear(x, n, &self.hw, &self.hb, self.out)
    }
}

pub(crate) struct Norm {
    pub(crate) w: Tensor,
    pub(crate) b: Tensor,
    pub(crate) hw: Vec<f32>,
    pub(crate) hb: Vec<f32>,
}

impl Norm {
    pub(crate) fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &[D])?;
        let b = get(g, &format!("{p}.bias"), &[D])?;
        Ok(Self { w: up(be, &w.data, &[D])?, b: up(be, &b.data, &[D])?, hw: w.data.clone(), hb: b.data.clone() })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor) -> Result<Tensor, OjasError> {
        be.layer_norm_forward(x, &self.w, &self.b, LN_EPS)
    }

    pub(crate) fn host(&self, x: &[f32]) -> Vec<f32> {
        host::layer_norm(x, D, &self.hw, &self.hb, LN_EPS)
    }
}

/// Multi-head attention with input and output projections: `nn.MultiheadAttention`
/// / sam3 `MultiheadAttention` (packed `in_proj`, [`Mha::load`]) or SAM's
/// `Attention` (separate projections into an `internal` width, [`Mha::load_split`]).
pub(crate) struct Mha {
    pub(crate) q: Lin,
    pub(crate) k: Lin,
    pub(crate) v: Lin,
    pub(crate) out: Lin,
    pub(crate) heads: usize,
}

impl Mha {
    pub(crate) fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.in_proj_weight"), &[3 * D, D])?;
        let b = get(g, &format!("{p}.in_proj_bias"), &[3 * D])?;
        let part = |i: usize| Lin::from_parts(be, &w.data[i * D * D..(i + 1) * D * D], &b.data[i * D..(i + 1) * D], D, D);
        Ok(Self { q: part(0)?, k: part(1)?, v: part(2)?, out: Lin::load(be, g, &format!("{p}.out_proj"), D, D)?, heads: HEADS })
    }

    /// `{p}.{q,k,v}_proj` `[internal, emb]` and `{p}.out_proj` `[emb, internal]`.
    pub(crate) fn load_split<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str, emb: usize, internal: usize, heads: usize) -> Result<Self, ModelError> {
        Ok(Self {
            q: Lin::load(be, g, &format!("{p}.q_proj"), internal, emb)?,
            k: Lin::load(be, g, &format!("{p}.k_proj"), internal, emb)?,
            v: Lin::load(be, g, &format!("{p}.v_proj"), internal, emb)?,
            out: Lin::load(be, g, &format!("{p}.out_proj"), emb, internal)?,
            heads,
        })
    }

    /// `query` `[B, Tq, E]`, `key`/`value` `[B, Tk, E]`, optional additive
    /// `bias` `[1|B, heads, Tq, Tk]`. Returns `[B, Tq, E]`.
    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, query: &Tensor, key: &Tensor, value: &Tensor, bias: Option<&Tensor>) -> Result<Tensor, OjasError> {
        let internal = self.q.out;
        let split = |l: &Lin, x: &Tensor| -> Result<Tensor, OjasError> {
            let (b, t) = (x.shape()[0], x.shape()[1]);
            be.permute(&l.fwd(be, x)?.reshape(&[b, t, self.heads, internal / self.heads])?, &[0, 2, 1, 3])
        };
        let (q, k, v) = (split(&self.q, query)?, split(&self.k, key)?, split(&self.v, value)?);
        let opts = AttentionOpts { bias, ..AttentionOpts::default() };
        let a = be.attention_forward(&q, &k, &v, opts)?;
        let (b, tq) = (query.shape()[0], query.shape()[1]);
        self.out.fwd(be, &be.permute(&a, &[0, 2, 1, 3])?.reshape(&[b, tq, internal])?)
    }
}

/// `MLP` from `model_misc.py`: ReLU between layers, optional residual and out norm.
pub(crate) struct Mlp {
    pub(crate) layers: Vec<Lin>,
    pub(crate) residual: bool,
    pub(crate) out_norm: Option<Norm>,
}

impl Mlp {
    pub(crate) fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str, dims: &[usize], residual: bool, out_norm: bool) -> Result<Self, ModelError> {
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, io)| Lin::load(be, g, &format!("{p}.layers.{i}"), io[1], io[0]))
            .collect::<Result<_, _>>()?;
        let out_norm = if out_norm { Some(Norm::load(be, g, &format!("{p}.out_norm"))?) } else { None };
        Ok(Self { layers, residual, out_norm })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor) -> Result<Tensor, OjasError> {
        let mut y = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            y = l.fwd(be, &y)?;
            if i + 1 < self.layers.len() {
                y = be.relu_forward(&y)?;
            }
        }
        if self.residual {
            y = be.residual_add_forward(&y, x)?;
        }
        match &self.out_norm {
            Some(n) => n.fwd(be, &y),
            None => Ok(y),
        }
    }

    pub(crate) fn host(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = x.to_vec();
        for (i, l) in self.layers.iter().enumerate() {
            y = l.host(&y, n);
            if i + 1 < self.layers.len() {
                host::relu_in_place(&mut y);
            }
        }
        if self.residual {
            y.iter_mut().zip(x).for_each(|(a, b)| *a += b);
        }
        match &self.out_norm {
            Some(nm) => nm.host(&y),
            None => y,
        }
    }
}

pub(crate) struct Conv {
    pub(crate) w: Tensor,
    pub(crate) b: Tensor,
}

impl Conv {
    pub(crate) fn load<B: Backend + ?Sized>(be: &B, g: &TensorGroup, p: &str, shape: [usize; 4]) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &shape)?;
        let b = get(g, &format!("{p}.bias"), &[shape[0]])?;
        Ok(Self { w: up(be, &w.data, &shape)?, b: up(be, &b.data, &[shape[0]])? })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor, padding: usize) -> Result<Tensor, OjasError> {
        let mut p = Conv2dParams::new();
        p.padding = [padding, padding];
        be.conv2d_forward(x, &self.w, Some(&self.b), p)
    }
}
