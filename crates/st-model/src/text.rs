//! SAM 3's text encoder (`text_encoder_ve.py`: `VETextEncoder` over a
//! CLIP-style `TextTransformer`), on the Ojas [`Backend`].
//!
//! 24 pre-LN blocks (width 1024, 16 heads, causal mask over the 32-token
//! context, exact GELU MLP 4096), `ln_final`, then `resizer` 1024 -> 256: the
//! per-token features the detector consumes. All 32 positions are computed;
//! the detector masks padding (token id 0) itself.

use ojas_core::{AttentionOpts, Backend, OjasError, Tensor};
use st_weights::names::{TEXT_CONTEXT, TEXT_DIM, TEXT_VOCAB};
use st_weights::{TextBlock, TextEncoder};

use crate::tokenizer::CONTEXT;

pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = TEXT_DIM / HEADS;
/// `nn.LayerNorm` default.
pub const LN_EPS: f32 = 1e-5;

struct Lin {
    w: Tensor,
    b: Tensor,
}

struct Block {
    ln1: (Tensor, Tensor),
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    ln2: (Tensor, Tensor),
    fc: Lin,
    proj: Lin,
}

/// Output of [`Text::encode`] for `n` prompts.
pub struct TextFeatures {
    /// After `ln_final`: `[n, 32, 1024]`.
    pub tokens: Tensor,
    /// After `resizer`: `[n, 32, 256]`.
    pub resized: Tensor,
    /// `true` where the token id is 0 (padding), `[n * 32]` row-major.
    pub padding: Vec<bool>,
}

pub struct Text<'b, B: Backend + ?Sized> {
    be: &'b B,
    /// `[49408, 1024]` on the host: the lookup is a gather, done before upload.
    token_embedding: Vec<f32>,
    positional: Vec<f32>,
    blocks: Vec<Block>,
    ln_final: (Tensor, Tensor),
    resizer: Lin,
}

fn up<B: Backend + ?Sized>(be: &B, data: &[f32], shape: &[usize]) -> Result<Tensor, OjasError> {
    be.upload(&Tensor::from_f32(data, shape, be.budget())?)
}

impl<'b, B: Backend + ?Sized> Text<'b, B> {
    pub fn new(be: &'b B, w: &TextEncoder) -> Result<Self, OjasError> {
        let lin = |l: &st_weights::Linear| -> Result<Lin, OjasError> {
            Ok(Lin {
                w: up(be, &l.weight.data, &l.weight.shape)?,
                b: up(be, &l.bias.data, &l.bias.shape)?,
            })
        };
        let ln = |n: &st_weights::LayerNorm| -> Result<(Tensor, Tensor), OjasError> {
            Ok((
                up(be, &n.weight.data, &[TEXT_DIM])?,
                up(be, &n.bias.data, &[TEXT_DIM])?,
            ))
        };
        // Fused in_proj rows [q; k; v], each [1024, 1024].
        let part = |b: &TextBlock, i: usize| -> Result<Lin, OjasError> {
            let r = i * TEXT_DIM..(i + 1) * TEXT_DIM;
            Ok(Lin {
                w: up(
                    be,
                    &b.in_proj.weight.data[r.start * TEXT_DIM..r.end * TEXT_DIM],
                    &[TEXT_DIM, TEXT_DIM],
                )?,
                b: up(be, &b.in_proj.bias.data[r], &[TEXT_DIM])?,
            })
        };
        let blocks = w
            .blocks
            .iter()
            .map(|b| {
                Ok(Block {
                    ln1: ln(&b.ln_1)?,
                    q: part(b, 0)?,
                    k: part(b, 1)?,
                    v: part(b, 2)?,
                    out: lin(&b.out_proj)?,
                    ln2: ln(&b.ln_2)?,
                    fc: lin(&b.c_fc)?,
                    proj: lin(&b.c_proj)?,
                })
            })
            .collect::<Result<Vec<_>, OjasError>>()?;
        Ok(Self {
            be,
            token_embedding: w.token_embedding.data.clone(),
            positional: w.positional_embedding.data.clone(),
            blocks,
            ln_final: ln(&w.ln_final)?,
            resizer: lin(&w.resizer)?,
        })
    }

    /// Encode `n` tokenized prompts (`[n][32]` ids from [`crate::tokenizer::Tokenizer::tokenize`]).
    pub fn encode(&self, ids: &[[u32; CONTEXT]]) -> Result<TextFeatures, OjasError> {
        let be = self.be;
        let n = ids.len();
        let mut x = Vec::with_capacity(n * TEXT_CONTEXT * TEXT_DIM);
        let mut padding = Vec::with_capacity(n * TEXT_CONTEXT);
        for row in ids {
            for (t, &id) in row.iter().enumerate() {
                let id = id as usize;
                if id >= TEXT_VOCAB {
                    return Err(OjasError::OutOfRange {
                        op: "st_model::Text::encode",
                        detail: format!("token id {id} >= vocabulary {TEXT_VOCAB}"),
                    });
                }
                let e = &self.token_embedding[id * TEXT_DIM..(id + 1) * TEXT_DIM];
                let p = &self.positional[t * TEXT_DIM..(t + 1) * TEXT_DIM];
                x.extend(e.iter().zip(p).map(|(a, b)| a + b));
                padding.push(id == 0);
            }
        }
        let mut x = up(be, &x, &[n, TEXT_CONTEXT, TEXT_DIM])?;
        let opts = AttentionOpts {
            causal: true,
            window: None,
            bias: None,
            key_lens: None,
            ..Default::default()
        };
        for b in &self.blocks {
            let h = be.layer_norm_forward(&x, &b.ln1.0, &b.ln1.1, LN_EPS)?;
            let heads = |l: &Lin| -> Result<Tensor, OjasError> {
                let t = be.linear_bias_forward(&h, &l.w, &l.b)?.reshape(&[
                    n,
                    TEXT_CONTEXT,
                    HEADS,
                    HEAD_DIM,
                ])?;
                be.permute(&t, &[0, 2, 1, 3])
            };
            let a = be.attention_forward(&heads(&b.q)?, &heads(&b.k)?, &heads(&b.v)?, opts)?;
            let a = be
                .permute(&a, &[0, 2, 1, 3])?
                .reshape(&[n, TEXT_CONTEXT, TEXT_DIM])?;
            let a = be.linear_bias_forward(&a, &b.out.w, &b.out.b)?;
            x = be.residual_add_forward(&x, &a)?;
            let m = be.layer_norm_forward(&x, &b.ln2.0, &b.ln2.1, LN_EPS)?;
            let m = be.gelu_erf_forward(&be.linear_bias_forward(&m, &b.fc.w, &b.fc.b)?)?;
            let m = be.linear_bias_forward(&m, &b.proj.w, &b.proj.b)?;
            x = be.residual_add_forward(&x, &m)?;
        }
        let tokens = be.layer_norm_forward(&x, &self.ln_final.0, &self.ln_final.1, LN_EPS)?;
        let resized = be.linear_bias_forward(&tokens, &self.resizer.w, &self.resizer.b)?;
        Ok(TextFeatures {
            tokens,
            resized,
            padding,
        })
    }
}
