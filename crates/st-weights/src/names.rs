//! Typed module families of `sam3.1_multiplex`, named as in the checkpoint.
//!
//! Shapes come from the official model code (external/sam3 f1647e8,
//! `model_builder.py`, `vitdet.py`, `necks.py`) and were checked against the
//! checkpoint header (`tools/reference/weights_manifest.json`). Linear weights
//! are PyTorch's `[out, in]`; conv weights `[out, in, kh, kw]`; transposed-conv
//! weights `[in, out, kh, kw]`.

use std::collections::BTreeMap;

use crate::{Source, Weight, WeightsError};

pub const VIT_DIM: usize = 1024;
pub const VIT_DEPTH: usize = 32;
pub const VIT_MLP: usize = 4736;
pub const VIT_PATCH: usize = 14;
/// 1008 / 14.
pub const VIT_GRID: usize = 72;
pub const VIT_WINDOW: usize = 24;
pub const VIT_GLOBAL_BLOCKS: [usize; 4] = [7, 15, 23, 31];
/// Complex RoPE entries per token: head dim 64 / 2.
pub const VIT_ROPE_PAIRS: usize = 32;
/// The pretrain grid of the absolute position embedding (336 / 14), plus cls.
pub const VIT_POS_TOKENS: usize = 24 * 24 + 1;
pub const NECK_DIM: usize = 256;
pub const TEXT_DIM: usize = 1024;
pub const TEXT_DEPTH: usize = 24;
pub const TEXT_MLP: usize = 4096;
pub const TEXT_CONTEXT: usize = 32;
pub const TEXT_VOCAB: usize = 49408;
pub const TEXT_PROJ: usize = 512;

const VISION: &str = "detector.backbone.vision_backbone";
const TEXT: &str = "detector.backbone.language_backbone";

#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Weight,
    pub bias: Weight,
}

impl Linear {
    fn load(src: &dyn Source, p: &str, out: usize, inp: usize) -> Result<Self, WeightsError> {
        Ok(Self {
            weight: src.tensor(&format!("{p}.weight"), &[out, inp])?,
            bias: src.tensor(&format!("{p}.bias"), &[out])?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub weight: Weight,
    pub bias: Weight,
}

impl LayerNorm {
    fn load(src: &dyn Source, p: &str, dim: usize) -> Result<Self, WeightsError> {
        Ok(Self {
            weight: src.tensor(&format!("{p}.weight"), &[dim])?,
            bias: src.tensor(&format!("{p}.bias"), &[dim])?,
        })
    }
}

/// A conv or transposed conv with bias.
#[derive(Clone, Debug)]
pub struct Conv {
    pub weight: Weight,
    pub bias: Weight,
}

impl Conv {
    fn load(src: &dyn Source, p: &str, w: [usize; 4], bias: usize) -> Result<Self, WeightsError> {
        Ok(Self {
            weight: src.tensor(&format!("{p}.weight"), &w)?,
            bias: src.tensor(&format!("{p}.bias"), &[bias])?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct VitBlock {
    /// Global attention over all 5,184 tokens; otherwise 3x3 windows of 24x24.
    pub global: bool,
    pub norm1: LayerNorm,
    /// Fused q, k, v: `[3 * 1024, 1024]`.
    pub qkv: Linear,
    pub proj: Linear,
    /// 2-D axial RoPE table, `[T, 32, 2]` (cos, sin) over interleaved pairs;
    /// T = 576 (windowed) or 5,184 (global).
    pub freqs_cis: Weight,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

#[derive(Clone, Debug)]
pub struct VitTrunk {
    /// `[1024, 3, 14, 14]`, no bias.
    pub patch_embed: Weight,
    /// `[1, 577, 1024]`: cls + a 24x24 grid, tiled 3x3 over the 72x72 grid.
    pub pos_embed: Weight,
    pub ln_pre: LayerNorm,
    pub blocks: Vec<VitBlock>,
}

impl VitTrunk {
    fn load(src: &dyn Source) -> Result<Self, WeightsError> {
        let t = format!("{VISION}.trunk");
        let patch_embed = src.tensor(
            &format!("{t}.patch_embed.proj.weight"),
            &[VIT_DIM, 3, VIT_PATCH, VIT_PATCH],
        )?;
        let pos_embed = src.tensor(&format!("{t}.pos_embed"), &[1, VIT_POS_TOKENS, VIT_DIM])?;
        let ln_pre = LayerNorm::load(src, &format!("{t}.ln_pre"), VIT_DIM)?;
        let mut blocks = Vec::with_capacity(VIT_DEPTH);
        for i in 0..VIT_DEPTH {
            let p = format!("{t}.blocks.{i}");
            let global = VIT_GLOBAL_BLOCKS.contains(&i);
            let tokens = if global {
                VIT_GRID * VIT_GRID
            } else {
                VIT_WINDOW * VIT_WINDOW
            };
            blocks.push(VitBlock {
                global,
                norm1: LayerNorm::load(src, &format!("{p}.norm1"), VIT_DIM)?,
                qkv: Linear::load(src, &format!("{p}.attn.qkv"), 3 * VIT_DIM, VIT_DIM)?,
                proj: Linear::load(src, &format!("{p}.attn.proj"), VIT_DIM, VIT_DIM)?,
                freqs_cis: src
                    .complex_pairs(&format!("{p}.attn.freqs_cis"), &[tokens, VIT_ROPE_PAIRS])?,
                norm2: LayerNorm::load(src, &format!("{p}.norm2"), VIT_DIM)?,
                fc1: Linear::load(src, &format!("{p}.mlp.fc1"), VIT_MLP, VIT_DIM)?,
                fc2: Linear::load(src, &format!("{p}.mlp.fc2"), VIT_DIM, VIT_MLP)?,
            });
        }
        Ok(Self {
            patch_embed,
            pos_embed,
            ln_pre,
            blocks,
        })
    }
}

/// One level of a simple-FPN neck. Every level ends in a 1x1 then a 3x3
/// (padding 1) conv to 256 channels.
#[derive(Clone, Debug)]
pub enum FpnLevel {
    /// x4 up to 288x288: convT 1024->512, GELU, convT 512->256 (kernel 2, stride 2).
    Up4 {
        dconv0: Conv,
        dconv1: Conv,
        conv_1x1: Conv,
        conv_3x3: Conv,
    },
    /// x2 up to 144x144: convT 1024->512.
    Up2 {
        dconv: Conv,
        conv_1x1: Conv,
        conv_3x3: Conv,
    },
    /// x1 at 72x72.
    Same { conv_1x1: Conv, conv_3x3: Conv },
}

#[derive(Clone, Debug)]
pub struct FpnNeck {
    /// Scales 4, 2, 1 (SAM 3.1's tri-neck has no x0.5 level).
    pub levels: [FpnLevel; 3],
}

impl FpnNeck {
    fn load(src: &dyn Source, name: &str) -> Result<Self, WeightsError> {
        let p = |lvl: usize, m: &str| format!("{VISION}.{name}.{lvl}.{m}");
        let d = VIT_DIM;
        let n = NECK_DIM;
        let tail = |lvl: usize, inp: usize| -> Result<(Conv, Conv), WeightsError> {
            Ok((
                Conv::load(src, &p(lvl, "conv_1x1"), [n, inp, 1, 1], n)?,
                Conv::load(src, &p(lvl, "conv_3x3"), [n, n, 3, 3], n)?,
            ))
        };
        let dconv0 = Conv::load(src, &p(0, "dconv_2x2_0"), [d, d / 2, 2, 2], d / 2)?;
        let dconv1 = Conv::load(src, &p(0, "dconv_2x2_1"), [d / 2, d / 4, 2, 2], d / 4)?;
        let (c1, c3) = tail(0, d / 4)?;
        let up4 = FpnLevel::Up4 {
            dconv0,
            dconv1,
            conv_1x1: c1,
            conv_3x3: c3,
        };
        let dconv = Conv::load(src, &p(1, "dconv_2x2"), [d, d / 2, 2, 2], d / 2)?;
        let (c1, c3) = tail(1, d / 2)?;
        let up2 = FpnLevel::Up2 {
            dconv,
            conv_1x1: c1,
            conv_3x3: c3,
        };
        let (c1, c3) = tail(2, d)?;
        let same = FpnLevel::Same {
            conv_1x1: c1,
            conv_3x3: c3,
        };
        Ok(Self {
            levels: [up4, up2, same],
        })
    }
}

/// The shared PE ViT trunk and SAM 3.1's three necks: `convs` feeds the
/// detector, `interactive_convs` the interactive (SAM 1 task) tracker path,
/// `propagation_convs` video propagation.
#[derive(Clone, Debug)]
pub struct VisionBackbone {
    pub trunk: VitTrunk,
    pub convs: FpnNeck,
    pub interactive_convs: FpnNeck,
    pub propagation_convs: FpnNeck,
}

impl VisionBackbone {
    pub fn load(src: &dyn Source) -> Result<Self, WeightsError> {
        Ok(Self {
            trunk: VitTrunk::load(src)?,
            convs: FpnNeck::load(src, "convs")?,
            interactive_convs: FpnNeck::load(src, "interactive_convs")?,
            propagation_convs: FpnNeck::load(src, "propagation_convs")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TextBlock {
    pub ln_1: LayerNorm,
    /// Fused q, k, v (`nn.MultiheadAttention` layout): `[3 * 1024, 1024]`.
    pub in_proj: Linear,
    pub out_proj: Linear,
    pub ln_2: LayerNorm,
    pub c_fc: Linear,
    pub c_proj: Linear,
}

/// CLIP-style causal text encoder plus the 1024 -> 256 resizer into the detector.
#[derive(Clone, Debug)]
pub struct TextEncoder {
    pub token_embedding: Weight,
    pub positional_embedding: Weight,
    pub blocks: Vec<TextBlock>,
    pub ln_final: LayerNorm,
    /// `[1024, 512]`, applied as `x @ text_projection` (not a Linear).
    pub text_projection: Weight,
    pub resizer: Linear,
}

impl TextEncoder {
    pub fn load(src: &dyn Source) -> Result<Self, WeightsError> {
        let e = format!("{TEXT}.encoder");
        let token_embedding = src.tensor(
            &format!("{e}.token_embedding.weight"),
            &[TEXT_VOCAB, TEXT_DIM],
        )?;
        let positional_embedding = src.tensor(
            &format!("{e}.positional_embedding"),
            &[TEXT_CONTEXT, TEXT_DIM],
        )?;
        let mut blocks = Vec::with_capacity(TEXT_DEPTH);
        for i in 0..TEXT_DEPTH {
            let p = format!("{e}.transformer.resblocks.{i}");
            blocks.push(TextBlock {
                ln_1: LayerNorm::load(src, &format!("{p}.ln_1"), TEXT_DIM)?,
                in_proj: Linear {
                    weight: src.tensor(
                        &format!("{p}.attn.in_proj_weight"),
                        &[3 * TEXT_DIM, TEXT_DIM],
                    )?,
                    bias: src.tensor(&format!("{p}.attn.in_proj_bias"), &[3 * TEXT_DIM])?,
                },
                out_proj: Linear::load(src, &format!("{p}.attn.out_proj"), TEXT_DIM, TEXT_DIM)?,
                ln_2: LayerNorm::load(src, &format!("{p}.ln_2"), TEXT_DIM)?,
                c_fc: Linear::load(src, &format!("{p}.mlp.c_fc"), TEXT_MLP, TEXT_DIM)?,
                c_proj: Linear::load(src, &format!("{p}.mlp.c_proj"), TEXT_DIM, TEXT_MLP)?,
            });
        }
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln_final: LayerNorm::load(src, &format!("{e}.ln_final"), TEXT_DIM)?,
            text_projection: src.tensor(&format!("{e}.text_projection"), &[TEXT_DIM, TEXT_PROJ])?,
            resizer: Linear::load(src, &format!("{TEXT}.resizer"), NECK_DIM, TEXT_DIM)?,
        })
    }
}

/// Every tensor under one prefix, for a module family whose typed struct
/// arrives with its port (gp-prompt-encoders, gp-detector-and-mask-head,
/// gp-tracker-multiplex). Names stay as in the checkpoint.
#[derive(Clone, Debug)]
pub struct TensorGroup {
    pub prefix: String,
    pub tensors: BTreeMap<String, Weight>,
}

/// The whole checkpoint.
#[derive(Clone, Debug)]
pub struct Sam31Weights {
    pub vision: VisionBackbone,
    pub text: TextEncoder,
    /// Fusion encoder (`.encoder`) and DETR decoder (`.decoder`).
    pub detector_transformer: TensorGroup,
    pub geometry_encoder: TensorGroup,
    pub segmentation_head: TensorGroup,
    pub dot_prod_scoring: TensorGroup,
    /// Memory encoder, memory attention, SAM mask decoders, multiplex state.
    pub tracker: TensorGroup,
}

/// Untyped families, by prefix. Their order does not matter: the prefixes are
/// disjoint and none overlaps a typed family.
pub const GROUP_PREFIXES: [&str; 5] = [
    "detector.transformer.",
    "detector.geometry_encoder.",
    "detector.segmentation_head.",
    "detector.dot_prod_scoring.",
    "tracker.model.",
];

impl Sam31Weights {
    pub fn load(src: &dyn Source) -> Result<Self, WeightsError> {
        let [t, g, s, d, k] = GROUP_PREFIXES;
        Ok(Self {
            vision: VisionBackbone::load(src)?,
            text: TextEncoder::load(src)?,
            detector_transformer: src.group(t)?,
            geometry_encoder: src.group(g)?,
            segmentation_head: src.group(s)?,
            dot_prod_scoring: src.group(d)?,
            tracker: src.group(k)?,
        })
    }
}
