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

impl Linear {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        f(&self.weight);
        f(&self.bias);
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        f(&mut self.weight);
        f(&mut self.bias);
    }
}

impl LayerNorm {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        f(&self.weight);
        f(&self.bias);
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        f(&mut self.weight);
        f(&mut self.bias);
    }
}

impl Conv {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        f(&self.weight);
        f(&self.bias);
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        f(&mut self.weight);
        f(&mut self.bias);
    }
}

impl VitTrunk {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        f(&self.patch_embed);
        f(&self.pos_embed);
        self.ln_pre.visit(f);
        for b in &self.blocks {
            b.norm1.visit(f);
            b.qkv.visit(f);
            b.proj.visit(f);
            f(&b.freqs_cis);
            b.norm2.visit(f);
            b.fc1.visit(f);
            b.fc2.visit(f);
        }
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        f(&mut self.patch_embed);
        f(&mut self.pos_embed);
        self.ln_pre.visit_mut(f);
        for b in &mut self.blocks {
            b.norm1.visit_mut(f);
            b.qkv.visit_mut(f);
            b.proj.visit_mut(f);
            f(&mut b.freqs_cis);
            b.norm2.visit_mut(f);
            b.fc1.visit_mut(f);
            b.fc2.visit_mut(f);
        }
    }
}

impl FpnNeck {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        for lvl in &self.levels {
            match lvl {
                FpnLevel::Up4 {
                    dconv0,
                    dconv1,
                    conv_1x1,
                    conv_3x3,
                } => {
                    dconv0.visit(f);
                    dconv1.visit(f);
                    conv_1x1.visit(f);
                    conv_3x3.visit(f);
                }
                FpnLevel::Up2 {
                    dconv,
                    conv_1x1,
                    conv_3x3,
                } => {
                    dconv.visit(f);
                    conv_1x1.visit(f);
                    conv_3x3.visit(f);
                }
                FpnLevel::Same { conv_1x1, conv_3x3 } => {
                    conv_1x1.visit(f);
                    conv_3x3.visit(f);
                }
            }
        }
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        for lvl in &mut self.levels {
            match lvl {
                FpnLevel::Up4 {
                    dconv0,
                    dconv1,
                    conv_1x1,
                    conv_3x3,
                } => {
                    dconv0.visit_mut(f);
                    dconv1.visit_mut(f);
                    conv_1x1.visit_mut(f);
                    conv_3x3.visit_mut(f);
                }
                FpnLevel::Up2 {
                    dconv,
                    conv_1x1,
                    conv_3x3,
                } => {
                    dconv.visit_mut(f);
                    conv_1x1.visit_mut(f);
                    conv_3x3.visit_mut(f);
                }
                FpnLevel::Same { conv_1x1, conv_3x3 } => {
                    conv_1x1.visit_mut(f);
                    conv_3x3.visit_mut(f);
                }
            }
        }
    }
}

impl VisionBackbone {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        self.trunk.visit(f);
        self.convs.visit(f);
        self.interactive_convs.visit(f);
        self.propagation_convs.visit(f);
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        self.trunk.visit_mut(f);
        self.convs.visit_mut(f);
        self.interactive_convs.visit_mut(f);
        self.propagation_convs.visit_mut(f);
    }
}

impl TextEncoder {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Weight)) {
        f(&self.token_embedding);
        f(&self.positional_embedding);
        for b in &self.blocks {
            b.ln_1.visit(f);
            b.in_proj.visit(f);
            b.out_proj.visit(f);
            b.ln_2.visit(f);
            b.c_fc.visit(f);
            b.c_proj.visit(f);
        }
        self.ln_final.visit(f);
        f(&self.text_projection);
        self.resizer.visit(f);
    }

    fn visit_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Weight)) {
        f(&mut self.token_embedding);
        f(&mut self.positional_embedding);
        for b in &mut self.blocks {
            b.ln_1.visit_mut(f);
            b.in_proj.visit_mut(f);
            b.out_proj.visit_mut(f);
            b.ln_2.visit_mut(f);
            b.c_fc.visit_mut(f);
            b.c_proj.visit_mut(f);
        }
        self.ln_final.visit_mut(f);
        f(&mut self.text_projection);
        self.resizer.visit_mut(f);
    }
}

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

    /// Visit every tensor in the checkpoint in load order: typed vision and text families first,
    /// then the five untyped groups sorted by name.
    pub fn for_each_tensor<'a>(&'a self, mut f: impl FnMut(&'a Weight)) {
        self.vision.visit(&mut f);
        self.text.visit(&mut f);
        for g in [
            &self.detector_transformer,
            &self.geometry_encoder,
            &self.segmentation_head,
            &self.dot_prod_scoring,
            &self.tracker,
        ] {
            for w in g.tensors.values() {
                f(w);
            }
        }
    }

    /// Visit every tensor mutably in the same deterministic load order as [`Self::for_each_tensor`].
    pub fn for_each_tensor_mut<'a>(&'a mut self, mut f: impl FnMut(&'a mut Weight)) {
        self.vision.visit_mut(&mut f);
        self.text.visit_mut(&mut f);
        for g in [
            &mut self.detector_transformer,
            &mut self.geometry_encoder,
            &mut self.segmentation_head,
            &mut self.dot_prod_scoring,
            &mut self.tracker,
        ] {
            for w in g.tensors.values_mut() {
                f(w);
            }
        }
    }

    /// Every tensor in deterministic load order.
    pub fn tensors(&self) -> Vec<&Weight> {
        let mut out = Vec::new();
        self.for_each_tensor(|w| out.push(w));
        out
    }

    /// Total scalar elements across every tensor in [`Sam31Weights`].
    pub fn total_numel(&self) -> usize {
        let mut total = 0usize;
        self.for_each_tensor(|w| total = total.saturating_add(w.numel()));
        total
    }

    /// Look up any tensor by its exact checkpoint name across both typed and untyped families.
    pub fn tensor(&self, name: &str) -> Option<&Weight> {
        for g in [
            &self.detector_transformer,
            &self.geometry_encoder,
            &self.segmentation_head,
            &self.dot_prod_scoring,
            &self.tracker,
        ] {
            if name.starts_with(&g.prefix) {
                return g.tensors.get(name);
            }
        }
        let mut found = None;
        self.vision.visit(&mut |w| {
            if found.is_none() && w.name == name {
                found = Some(w);
            }
        });
        if found.is_some() {
            return found;
        }
        self.text.visit(&mut |w| {
            if found.is_none() && w.name == name {
                found = Some(w);
            }
        });
        found
    }

    /// Mutably look up any tensor by its exact checkpoint name across both typed and untyped families.
    pub fn tensor_mut(&mut self, name: &str) -> Option<&mut Weight> {
        for g in [
            &mut self.detector_transformer,
            &mut self.geometry_encoder,
            &mut self.segmentation_head,
            &mut self.dot_prod_scoring,
            &mut self.tracker,
        ] {
            if name.starts_with(&g.prefix) {
                return g.tensors.get_mut(name);
            }
        }
        let mut found = None;
        if name.starts_with(VISION) {
            self.vision.visit_mut(&mut |w| {
                if found.is_none() && w.name == name {
                    found = Some(w);
                }
            });
            return found;
        }
        if name.starts_with(TEXT) {
            self.text.visit_mut(&mut |w| {
                if found.is_none() && w.name == name {
                    found = Some(w);
                }
            });
            return found;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expected, SchemaRecorder};

    #[test]
    fn tensor_lookup_reaches_every_typed_and_group_tensor_uniquely() {
        let rec = SchemaRecorder::default();
        let mut w = Sam31Weights::load(&rec).unwrap();
        let expected = rec.expected.into_inner();
        for g in [
            &mut w.detector_transformer,
            &mut w.geometry_encoder,
            &mut w.segmentation_head,
            &mut w.dot_prod_scoring,
            &mut w.tracker,
        ] {
            let name = format!("{}synthetic.weight", g.prefix);
            g.tensors.insert(
                name.clone(),
                Weight {
                    name,
                    shape: vec![2, 2],
                    data: vec![1.0, 2.0, 3.0, 4.0],
                },
            );
        }
        let mut expected_names = Vec::new();
        for e in expected {
            match e {
                Expected::Tensor { name, .. } | Expected::ComplexPairs { name, .. } => {
                    expected_names.push(name);
                }
                Expected::Group { prefix } => {
                    expected_names.push(format!("{prefix}synthetic.weight"));
                }
            }
        }
        let listed: Vec<String> = w.tensors().into_iter().map(|t| t.name.clone()).collect();
        assert_eq!(listed, expected_names);
        for (idx, name) in expected_names.iter().enumerate() {
            assert_eq!(w.tensor(name).unwrap().name, *name);
            let t = w.tensor_mut(name).unwrap();
            t.data = vec![idx as f32];
        }
        for (idx, name) in expected_names.iter().enumerate() {
            assert_eq!(w.tensor(name).unwrap().data, vec![idx as f32], "{name}");
        }
        assert!(w.tensor("no.such.tensor").is_none());
        assert!(w.tensor_mut("no.such.tensor").is_none());
    }
}
