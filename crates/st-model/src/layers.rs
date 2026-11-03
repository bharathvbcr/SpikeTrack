//! Building blocks shared by the detector and the tracker: checked weight
//! lookup, linear / LayerNorm / multi-head attention / MLP / conv on the
//! Ojas backend, each with a host (f32) twin for the small per-query paths.

use ojas_core::{AttentionOpts, Backend, Conv2dParams, OjasError, Tensor};
use st_weights::{TensorGroup, Weight};

use crate::detector::{D, HEADS, LN_EPS};
use crate::host;
use crate::ModelError;

pub(crate) fn get<'a>(
    g: &'a TensorGroup,
    name: &str,
    shape: &[usize],
) -> Result<&'a Weight, ModelError> {
    let w = g
        .tensors
        .get(name)
        .ok_or_else(|| ModelError::MissingWeight(name.to_string()))?;
    if w.shape != shape {
        return Err(ModelError::WeightShape {
            name: name.to_string(),
            expected: shape.to_vec(),
            found: w.shape.clone(),
        });
    }
    Ok(w)
}

pub(crate) fn up<B: Backend + ?Sized>(
    be: &B,
    data: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    be.upload(&Tensor::from_f32(data, shape, be.budget())?)
}

pub(crate) enum HostSlice<'a> {
    Leased {
        lease: Box<dyn ojas_core::HostLease>,
        start: usize,
        len: usize,
    },
    Borrowed(&'a [f32]),
    Host(Tensor),
}

impl<'a> HostSlice<'a> {
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::Leased { lease, start, len } => {
                &lease.as_f32().expect("checked f32 lease")[*start..*start + *len]
            }
            Self::Borrowed(s) => s,
            Self::Host(h) => h.f32_slice().expect("checked f32 host tensor"),
        }
    }
}

impl<'a> std::ops::Deref for HostSlice<'a> {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

pub(crate) enum HostBuf {
    Host(Tensor),
    Vec(Vec<f32>),
}

impl HostBuf {
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::Host(h) => h.f32_slice().expect("checked f32 host tensor"),
            Self::Vec(v) => v,
        }
    }
}

impl std::ops::Deref for HostBuf {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

fn view_f32_with_budget<'a>(
    t: &'a Tensor,
    budget: &ojas_core::Budget,
) -> Result<HostSlice<'a>, OjasError> {
    const OP: &str = "st_model::view_f32";
    if t.dtype() != ojas_core::DType::F32 {
        return Err(OjasError::Dtype {
            op: OP,
            expected: ojas_core::DType::F32,
            got: t.dtype(),
        });
    }
    if !t.is_contiguous()? {
        return Err(OjasError::Shape {
            op: OP,
            detail: "view is not contiguous".to_string(),
        });
    }
    if let Ok(s) = t.f32_slice() {
        return Ok(HostSlice::Borrowed(s));
    }
    if t.mapped_buffer().is_some() {
        let len = t.num_elements()?;
        let byte_off = t.byte_offset();
        if !byte_off.is_multiple_of(4) {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("byte_offset {byte_off} is not a multiple of 4"),
            });
        }
        let start = byte_off / 4;
        let lease = t.host_view()?;
        let full = lease.as_f32()?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: "element window overflows".to_string(),
            })?;
        if end > full.len() {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!(
                    "window {start}..{end} exceeds mapped f32 len {}",
                    full.len()
                ),
            });
        }
        return Ok(HostSlice::Leased { lease, start, len });
    }
    let h = t.to_host(budget)?;
    let _ = h.f32_slice()?;
    Ok(HostSlice::Host(h))
}

pub(crate) fn host_slice<B: Backend + ?Sized>(t: &Tensor, be: &B) -> Result<HostBuf, OjasError> {
    if t.mapped_buffer().is_some() {
        return Ok(HostBuf::Vec(
            view_f32_with_budget(t, be.budget())?.as_slice().to_vec(),
        ));
    }
    let h = t.to_host(be.budget())?;
    let _ = h.f32_slice()?;
    Ok(HostBuf::Host(h))
}

pub(crate) fn view_f32(t: &Tensor) -> Result<HostSlice<'_>, OjasError> {
    if let Ok(s) = t.f32_slice() {
        return Ok(HostSlice::Borrowed(s));
    }
    view_f32_with_budget(t, &ojas_core::Budget::new(u64::MAX))
}

pub(crate) fn host_of<B: Backend + ?Sized>(t: &Tensor, be: &B) -> Result<Vec<f32>, OjasError> {
    match host_slice(t, be)? {
        HostBuf::Host(h) => h.to_f32_vec(),
        HostBuf::Vec(v) => Ok(v),
    }
}

pub(crate) struct Lin {
    pub(crate) w: Tensor,
    pub(crate) b: Tensor,
    pub(crate) out: usize,
}

impl Lin {
    pub(crate) fn load<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        out: usize,
        inp: usize,
    ) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &[out, inp])?;
        let b = get(g, &format!("{p}.bias"), &[out])?;
        Self::from_parts(be, &w.data, &b.data, out, inp)
    }

    /// Host-resident linear layer for small per-query / pointer paths that only call [`Self::host`].
    pub(crate) fn load_host<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        out: usize,
        inp: usize,
    ) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &[out, inp])?;
        let b = get(g, &format!("{p}.bias"), &[out])?;
        Ok(Self {
            w: Tensor::from_f32(&w.data, &[out, inp], be.budget())?,
            b: Tensor::from_f32(&b.data, &[out], be.budget())?,
            out,
        })
    }

    pub(crate) fn from_parts<B: Backend + ?Sized>(
        be: &B,
        w: &[f32],
        b: &[f32],
        out: usize,
        inp: usize,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            w: up(be, w, &[out, inp])?,
            b: up(be, b, &[out])?,
            out,
        })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor) -> Result<Tensor, OjasError> {
        be.linear_bias_forward(x, &self.w, &self.b)
    }

    pub(crate) fn host(&self, x: &[f32], n: usize) -> Vec<f32> {
        let wv = view_f32(&self.w).expect("Lin: weight host view");
        let bv = view_f32(&self.b).expect("Lin: bias host view");
        host::linear(x, n, wv.as_slice(), bv.as_slice(), self.out)
    }
}

pub(crate) struct Norm {
    pub(crate) w: Tensor,
    pub(crate) b: Tensor,
}

impl Norm {
    pub(crate) fn load<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
    ) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &[D])?;
        let b = get(g, &format!("{p}.bias"), &[D])?;
        Ok(Self {
            w: up(be, &w.data, &[D])?,
            b: up(be, &b.data, &[D])?,
        })
    }

    /// Host-resident LayerNorm for small per-query paths that only call [`Self::host`].
    pub(crate) fn load_host<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
    ) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &[D])?;
        let b = get(g, &format!("{p}.bias"), &[D])?;
        Ok(Self {
            w: Tensor::from_f32(&w.data, &[D], be.budget())?,
            b: Tensor::from_f32(&b.data, &[D], be.budget())?,
        })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(&self, be: &B, x: &Tensor) -> Result<Tensor, OjasError> {
        be.layer_norm_forward(x, &self.w, &self.b, LN_EPS)
    }

    pub(crate) fn host(&self, x: &[f32]) -> Vec<f32> {
        let wv = view_f32(&self.w).expect("Norm: weight host view");
        let bv = view_f32(&self.b).expect("Norm: bias host view");
        host::layer_norm(x, D, wv.as_slice(), bv.as_slice(), LN_EPS)
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
    pub(crate) fn load<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
    ) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.in_proj_weight"), &[3 * D, D])?;
        let b = get(g, &format!("{p}.in_proj_bias"), &[3 * D])?;
        let part = |i: usize| {
            Lin::from_parts(
                be,
                &w.data[i * D * D..(i + 1) * D * D],
                &b.data[i * D..(i + 1) * D],
                D,
                D,
            )
        };
        Ok(Self {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: Lin::load(be, g, &format!("{p}.out_proj"), D, D)?,
            heads: HEADS,
        })
    }

    /// `{p}.{q,k,v}_proj` `[internal, emb]` and `{p}.out_proj` `[emb, internal]`.
    pub(crate) fn load_split<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        emb: usize,
        internal: usize,
        heads: usize,
    ) -> Result<Self, ModelError> {
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
    pub(crate) fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, OjasError> {
        let internal = self.q.out;
        let split = |l: &Lin, x: &Tensor| -> Result<Tensor, OjasError> {
            let (b, t) = (x.shape()[0], x.shape()[1]);
            be.permute(
                &l.fwd(be, x)?
                    .reshape(&[b, t, self.heads, internal / self.heads])?,
                &[0, 2, 1, 3],
            )
        };
        let (q, k, v) = (
            split(&self.q, query)?,
            split(&self.k, key)?,
            split(&self.v, value)?,
        );
        let opts = AttentionOpts {
            bias,
            ..AttentionOpts::default()
        };
        let a = be.attention_forward(&q, &k, &v, opts)?;
        let (b, tq) = (query.shape()[0], query.shape()[1]);
        self.out.fwd(
            be,
            &be.permute(&a, &[0, 2, 1, 3])?.reshape(&[b, tq, internal])?,
        )
    }
}

/// `MLP` from `model_misc.py`: ReLU between layers, optional residual and out norm.
pub(crate) struct Mlp {
    pub(crate) layers: Vec<Lin>,
    pub(crate) residual: bool,
    pub(crate) out_norm: Option<Norm>,
}

impl Mlp {
    pub(crate) fn load<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        dims: &[usize],
        residual: bool,
        out_norm: bool,
    ) -> Result<Self, ModelError> {
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, io)| Lin::load(be, g, &format!("{p}.layers.{i}"), io[1], io[0]))
            .collect::<Result<_, _>>()?;
        let out_norm = if out_norm {
            Some(Norm::load(be, g, &format!("{p}.out_norm"))?)
        } else {
            None
        };
        Ok(Self {
            layers,
            residual,
            out_norm,
        })
    }

    /// Host-resident MLP for per-query / pointer paths that only call [`Self::host`].
    pub(crate) fn load_host<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        dims: &[usize],
        residual: bool,
        out_norm: bool,
    ) -> Result<Self, ModelError> {
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, io)| Lin::load_host(be, g, &format!("{p}.layers.{i}"), io[1], io[0]))
            .collect::<Result<_, _>>()?;
        let out_norm = if out_norm {
            Some(Norm::load_host(be, g, &format!("{p}.out_norm"))?)
        } else {
            None
        };
        Ok(Self {
            layers,
            residual,
            out_norm,
        })
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
        if self.layers.is_empty() {
            if n == 0 {
                assert_eq!(x.len(), 0, "Mlp input");
                return Vec::new();
            }
            assert_eq!(x.len() % n, 0, "Mlp input");
            let mut y = x.to_vec();
            if self.residual {
                y.iter_mut().zip(x).for_each(|(a, b)| *a += b);
            }
            return match &self.out_norm {
                Some(nm) => nm.host(&y),
                None => y,
            };
        }
        let w_views: Vec<HostSlice<'_>> = self
            .layers
            .iter()
            .map(|l| view_f32(&l.w).expect("Mlp: weight host view"))
            .collect();
        let b_views: Vec<HostSlice<'_>> = self
            .layers
            .iter()
            .map(|l| view_f32(&l.b).expect("Mlp: bias host view"))
            .collect();
        let specs = build_mlp_specs(&self.layers, &w_views, &b_views);
        let inp0 = specs[0].inp;
        assert_eq!(x.len(), n * inp0, "Mlp input");
        if n == 0 {
            return Vec::new();
        }
        let out_dim = specs.last().expect("non-empty").out;
        let norm_views = self.out_norm.as_ref().map(|nm| {
            (
                view_f32(&nm.w).expect("Mlp norm weight"),
                view_f32(&nm.b).expect("Mlp norm bias"),
            )
        });
        let norm_slices = norm_views
            .as_ref()
            .map(|(nw, nb)| (nw.as_slice(), nb.as_slice()));
        let macs_per_row: usize = specs.iter().map(|s| s.inp * s.out).sum();
        let min_rows = if n * macs_per_row >= host::THREAD_MIN_MACS {
            1
        } else {
            n + 1
        };
        let max_hidden = specs
            .iter()
            .take(specs.len().saturating_sub(1))
            .map(|s| s.out)
            .max()
            .unwrap_or(0);
        let mut out = vec![0f32; n * out_dim];
        host::for_each_row_chunk_mut(&mut out, out_dim, min_rows, |r0, piece| {
            let chunk_rows = piece.len() / out_dim;
            let x_chunk = &x[r0 * inp0..(r0 + chunk_rows) * inp0];
            eval_mlp_chunk(
                &specs,
                self.residual,
                norm_slices,
                x_chunk,
                piece,
                max_hidden,
            );
        });
        out
    }

    /// Evaluates `n` rows into `out` (`[n, out_dim]`) on the calling thread using L1-resident
    /// 32-row tiles without spawning an inner thread pool.
    pub(crate) fn host_rows(&self, x: &[f32], n: usize, out: &mut [f32]) {
        if self.layers.is_empty() {
            assert_eq!(out.len(), x.len(), "Mlp output");
            if n == 0 {
                assert_eq!(x.len(), 0, "Mlp input");
                return;
            }
            assert_eq!(x.len() % n, 0, "Mlp input");
            out.copy_from_slice(x);
            if self.residual {
                out.iter_mut().zip(x).for_each(|(a, b)| *a += b);
            }
            if let Some(nm) = &self.out_norm {
                let nw = view_f32(&nm.w).expect("Mlp norm weight");
                let nb = view_f32(&nm.b).expect("Mlp norm bias");
                host::layer_norm_in_place(
                    out,
                    nw.as_slice().len(),
                    nw.as_slice(),
                    nb.as_slice(),
                    LN_EPS,
                );
            }
            return;
        }
        let w_views: Vec<HostSlice<'_>> = self
            .layers
            .iter()
            .map(|l| view_f32(&l.w).expect("Mlp: weight host view"))
            .collect();
        let b_views: Vec<HostSlice<'_>> = self
            .layers
            .iter()
            .map(|l| view_f32(&l.b).expect("Mlp: bias host view"))
            .collect();
        let specs = build_mlp_specs(&self.layers, &w_views, &b_views);
        let inp0 = specs[0].inp;
        let out_dim = specs.last().expect("non-empty").out;
        assert_eq!(x.len(), n * inp0, "Mlp input");
        assert_eq!(out.len(), n * out_dim, "Mlp output");
        if n == 0 {
            return;
        }
        let norm_views = self.out_norm.as_ref().map(|nm| {
            (
                view_f32(&nm.w).expect("Mlp norm weight"),
                view_f32(&nm.b).expect("Mlp norm bias"),
            )
        });
        let norm_slices = norm_views
            .as_ref()
            .map(|(nw, nb)| (nw.as_slice(), nb.as_slice()));
        let max_hidden = specs
            .iter()
            .take(specs.len().saturating_sub(1))
            .map(|s| s.out)
            .max()
            .unwrap_or(0);
        eval_mlp_chunk(&specs, self.residual, norm_slices, x, out, max_hidden);
    }
}

struct MlpLayerSpec<'a> {
    w: &'a [f32],
    b: &'a [f32],
    inp: usize,
    out: usize,
}

fn build_mlp_specs<'a>(
    layers: &[Lin],
    w_views: &'a [HostSlice<'_>],
    b_views: &'a [HostSlice<'_>],
) -> Vec<MlpLayerSpec<'a>> {
    layers
        .iter()
        .zip(w_views)
        .zip(b_views)
        .map(|((l, wv), bv)| {
            let w = wv.as_slice();
            let b = bv.as_slice();
            let out = l.out;
            let inp = w.len() / out;
            assert_eq!(w.len(), out * inp, "Mlp linear weight");
            assert_eq!(b.len(), out, "Mlp linear bias");
            MlpLayerSpec { w, b, inp, out }
        })
        .collect()
}

fn eval_mlp_chunk(
    specs: &[MlpLayerSpec<'_>],
    residual: bool,
    norm_slices: Option<(&[f32], &[f32])>,
    x: &[f32],
    piece: &mut [f32],
    max_hidden: usize,
) {
    const TILE: usize = 32;
    let inp0 = specs[0].inp;
    let out_dim = specs.last().expect("non-empty").out;
    let chunk_rows = piece.len() / out_dim;
    let mut buf_a = vec![0f32; TILE.min(chunk_rows) * max_hidden];
    let mut buf_b = if specs.len() > 2 {
        vec![0f32; TILE.min(chunk_rows) * max_hidden]
    } else {
        Vec::new()
    };
    let mut local_r = 0;
    while local_r < chunk_rows {
        let t_rows = (chunk_rows - local_r).min(TILE);
        let x_tile = &x[local_r * inp0..(local_r + t_rows) * inp0];
        let out_tile = &mut piece[local_r * out_dim..(local_r + t_rows) * out_dim];
        if specs.len() == 1 {
            let s = &specs[0];
            host::linear_rows(x_tile, s.w, s.b, s.inp, s.out, out_tile);
        } else {
            let s0 = &specs[0];
            let dst0 = &mut buf_a[..t_rows * s0.out];
            host::linear_rows(x_tile, s0.w, s0.b, s0.inp, s0.out, dst0);
            host::relu_in_place(dst0);
            for (li, s) in specs.iter().enumerate().take(specs.len() - 1).skip(1) {
                if li % 2 == 1 {
                    let src = &buf_a[..t_rows * s.inp];
                    let dst = &mut buf_b[..t_rows * s.out];
                    host::linear_rows(src, s.w, s.b, s.inp, s.out, dst);
                    host::relu_in_place(dst);
                } else {
                    let src = &buf_b[..t_rows * s.inp];
                    let dst = &mut buf_a[..t_rows * s.out];
                    host::linear_rows(src, s.w, s.b, s.inp, s.out, dst);
                    host::relu_in_place(dst);
                }
            }
            let last = specs.len() - 1;
            let sl = &specs[last];
            let src = if last % 2 == 1 {
                &buf_a[..t_rows * sl.inp]
            } else {
                &buf_b[..t_rows * sl.inp]
            };
            host::linear_rows(src, sl.w, sl.b, sl.inp, sl.out, out_tile);
        }
        if residual {
            out_tile.iter_mut().zip(x_tile).for_each(|(a, b)| *a += b);
        }
        if let Some((nw, nb)) = norm_slices {
            host::layer_norm_in_place(out_tile, out_dim, nw, nb, LN_EPS);
        }
        local_r += t_rows;
    }
}

pub(crate) struct Conv {
    pub(crate) w: Tensor,
    pub(crate) b: Tensor,
}

impl Conv {
    pub(crate) fn load<B: Backend + ?Sized>(
        be: &B,
        g: &TensorGroup,
        p: &str,
        shape: [usize; 4],
    ) -> Result<Self, ModelError> {
        let w = get(g, &format!("{p}.weight"), &shape)?;
        let b = get(g, &format!("{p}.bias"), &[shape[0]])?;
        Ok(Self {
            w: up(be, &w.data, &shape)?,
            b: up(be, &b.data, &[shape[0]])?,
        })
    }

    pub(crate) fn fwd<B: Backend + ?Sized>(
        &self,
        be: &B,
        x: &Tensor,
        padding: usize,
    ) -> Result<Tensor, OjasError> {
        let mut p = Conv2dParams::new();
        p.padding = [padding, padding];
        be.conv2d_forward(x, &self.w, Some(&self.b), p)
    }
}
