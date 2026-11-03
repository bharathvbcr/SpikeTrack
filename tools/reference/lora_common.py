"""Shared pieces of the SAM 3.1 detector LoRA pipeline (offline PyTorch; never used by Rust).

Used by cache_detector_inputs.py, train_lora.py, lora_merge_fixture.py and lora_selftest.py.

What is here
------------
* `build_heads`: the detector's trainable-path modules only (fusion encoder + DETR decoder,
  geometry encoder, dot-product scoring head) as a real upstream `Sam3MultiplexDetector` with a
  stub backbone and no segmentation head, loaded from the converted checkpoint with
  `safe_open` (about 100 MB, not the 3.5 GB model). The backbone and text encoder are never
  needed in training: their outputs are cached (cache_detector_inputs.py).
* `LoRAMultiheadAttention`: LoRA on the attention projections. Upstream's attention passes
  `in_proj_weight` and `out_proj.weight` as raw tensors to `multi_head_attention_forward`, so
  wrapping an `nn.Linear` would not intercept anything; this module does the projections
  itself (q, k, v, out, each with its own A and B) and then runs the same masked SDPA.
  PEFT-style input dropout on the low-rank branch is applied to the branch input.
* `grounding_forward`: upstream's own `_encode_prompt`, `_run_encoder`, `_run_decoder` (same
  code as `Sam3Image.forward_grounding`) minus the segmentation head and matching.
* the adapter file (`export_adapter`, `load_adapter`, `merge_into_state`) in the layout that
  crates/st-weights/src/lora.rs reads, which is PEFT's (`<module>.lora_A.weight` [r, in],
  `<module>.lora_B.weight` [out, r], scale alpha/r), the same as ojas-model's lora.rs.
* the loss: upstream's `Sam3LossWrapper` + `BinaryHungarianMatcherV2` + `Boxes` + `IABCEMdetr`
  with the roboflow_v100_full_ft_100_images.yaml weights. `loss_fns.py` needs `torchmetrics` and
  `sigmoid_focal_loss.py` needs `triton`, neither installable here; `install_loss_shims`
  registers a pure-torch `sigmoid_focal_loss` (the formula in that file's docstring) and a
  `torchmetrics` with the one function used for a logged metric.
"""

import json
import math
import os
import pathlib
import re
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
# upstream reads SAM3_DEVICE at import time (sam3/_device.py); callers set it first (cache script: --device)
os.environ.setdefault("SAM3_DEVICE", "cpu")

import cpu_shims  # noqa: E402  (registers the triton-free sam3.model.edt before any sam3 import)

cpu_shims.install()

CKPT = pathlib.Path(os.path.expanduser("~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"))
VISEM = REPO / "data/visem/VISEM_Tracking_Train_v4/Train"
SPLIT = HERE / "visem_split.json"
WEIGHTS_MANIFEST = HERE / "weights_manifest.json"

RANK, ALPHA, DROPOUT = 16, 32.0, 0.1
FORMAT = "spiketrack-lora-v1"
METADATA_KEY = "spiketrack_lora"

# One prompt per VISEM class: the brief's "absent classes as negatives" only makes sense per class,
# and the held-out metric is class 0 (sperm). "cell" stays out of training: it is the best
# zero-shot prompt (recall 0.233) and so an unseen-prompt probe after adaptation.
CLASS_PROMPTS = {0: "sperm", 1: "sperm cluster", 2: "pinhead sperm"}
EVERYDAY_NEGATIVES = ["shoe", "chair", "dog", "cup", "bicycle", "tree"]
EVAL_SEEN_NEGATIVES = EVERYDAY_NEGATIVES[:3]
EVAL_UNSEEN_NEGATIVES = ["car", "cell nucleus"]  # the two absent prompts of zero_shot_check.py
PROBE_PROMPT = "cell"
ALL_PROMPTS = list(CLASS_PROMPTS.values()) + EVERYDAY_NEGATIVES + EVAL_UNSEEN_NEGATIVES + [PROBE_PROMPT]
# Exact non-BOS/EOS/PAD BPE token IDs of ("sperm", "sperm cluster", "pinhead sperm"):
# 1375="head</w>", 2629="pin", 15595="cluster</w>", 35781="sperm</w>" (disjoint from all negative/probe prompts).
CLASS_TOKEN_IDS = (1375, 2629, 15595, 35781)

TOKENS = 72 * 72
DIM = 256


# ------------------------------------------------------------------ data helpers

def load_split():
    return json.loads(SPLIT.read_text())


def label_boxes(label_path, cls, w=1.0, h=1.0):
    """YOLO label file -> [n, 4] float32 xyxy for one class, scaled by (w, h) (1.0 = normalized)."""
    boxes = []
    for line in pathlib.Path(label_path).read_text().split("\n"):
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != cls:
            continue
        cx, cy, bw, bh = (float(p) for p in parts[1:])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


# ------------------------------------------------------------------ loss shims

def install_loss_shims():
    """Make upstream's loss module importable without torchmetrics and triton (see module doc)."""
    if "sam3.train.loss.sigmoid_focal_loss" in sys.modules and "torchmetrics" in sys.modules:
        return
    fl = types.ModuleType("sam3.train.loss.sigmoid_focal_loss")

    def focal(inputs, targets, alpha=0.25, gamma=2):
        # sigmoid_focal_loss.py docstring: alpha_t * (1 - p_t)^gamma * bce, element-wise.
        inputs = inputs.float()
        prob = inputs.sigmoid()
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return alpha_t * ce * (1 - p_t) ** gamma

    fl.triton_sigmoid_focal_loss = focal
    fl.triton_sigmoid_focal_loss_reduce = lambda inputs, targets, alpha=0.25, gamma=2: focal(inputs, targets, alpha, gamma).sum()
    sys.modules["sam3.train.loss.sigmoid_focal_loss"] = fl

    tm = types.ModuleType("torchmetrics")
    tmf = types.ModuleType("torchmetrics.functional")

    def f1_score(preds, target, task="binary"):
        assert task == "binary"
        pred = preds > 0.5
        tgt = target.bool()
        tp = (pred & tgt).sum().float()
        return 2 * tp / (pred.sum() + tgt.sum()).clamp(min=1).float()

    tmf.f1_score = f1_score
    tm.functional = tmf
    sys.modules["torchmetrics"] = tm
    sys.modules["torchmetrics.functional"] = tmf


def build_loss(o2m=False, o2m_weight=2.0, box_weights=(5.0, 2.0)):
    """upstream roboflow_v100_full_ft_100_images.yaml, box + class + presence (optional DAC one-to-many).
    `box_weights` = (loss_bbox, loss_giou); the matcher's costs stay upstream's."""
    install_loss_shims()
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    matcher = BinaryHungarianMatcherV2(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True, alpha=0.25, gamma=2, stable=False)
    o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4) if o2m else None
    wrapper = Sam3LossWrapper(
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": box_weights[0], "loss_giou": box_weights[1]}),
            IABCEMdetr(
                weak_loss=False,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_weight=10.0, alpha=0.25, gamma=2, use_presence=True, pos_focal=False,
                pad_n_queries=200, pad_scale_pos=1.0,
            ),
        ],
        normalization="local",
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=o2m_weight,
        use_o2m_matcher_on_o2m_aux=False,
        scale_by_find_batch_size=True,
    )
    return wrapper, matcher


def make_targets(gt_xyxy, device):
    """Targets dict as `Sam3Image.back_convert` builds it. `gt_xyxy`: list of [n_i, 4] normalized."""
    from sam3.model.box_ops import box_xyxy_to_cxcywh

    bsz = len(gt_xyxy)
    counts = [len(g) for g in gt_xyxy]
    max_t = max(1, max(counts))
    padded = torch.zeros(bsz, max_t, 4)
    ids = torch.full((bsz, max_t), -1, dtype=torch.long)
    packed = []
    for i, g in enumerate(gt_xyxy):
        if len(g):
            t = torch.as_tensor(np.asarray(g, dtype=np.float32))
            c = box_xyxy_to_cxcywh(t)
            c[:, 2:] = c[:, 2:].clamp(min=1e-4)
            padded[i, : len(g)] = c
            ids[i, : len(g)] = torch.arange(len(g))
            packed.append(c)
    boxes = torch.cat(packed) if packed else torch.zeros(0, 4)
    from sam3.model.box_ops import box_cxcywh_to_xyxy

    return {
        "boxes": boxes.to(device),
        "boxes_xyxy": box_cxcywh_to_xyxy(boxes).to(device),
        "boxes_padded": padded.to(device),
        "num_boxes": torch.tensor(counts, dtype=torch.long, device=device),
        "object_ids_padded": ids.to(device),
        "object_ids_packed": torch.cat([ids[i, : c] for i, c in enumerate(counts)]).to(device) if sum(counts) else torch.zeros(0, dtype=torch.long, device=device),
        "is_exhaustive": torch.ones(bsz, dtype=torch.bool, device=device),
        "positive_map": torch.ones(len(boxes), 1, device=device),
    }


def compute_loss(wrapper, matcher, out, targets, neg_query_weight=0.0):
    """Match every decoder layer's output (final + aux) then reduce with upstream's wrapper.

    Upstream's `IABCEMdetr` masks `loss_bce` by `keep_loss = (num_boxes > 0)`, giving zero per-query
    gradient on negative prompts (`num_boxes == 0`) even when queries fire strongly on absent
    morphology classes. When `neg_query_weight > 0`, focal negative BCE (`p^2 * BCE(logit, 0)`) over
    negative batch items (final + aux layers) trains both `dot_prod_scoring` and prompt tokens to
    suppress false detections on absent concepts without perturbing quiet background queries.
    """
    out["indices"] = matcher(out, targets)
    for aux in out.get("aux_outputs", []):
        aux["indices"] = matcher(aux, targets)
    losses = wrapper.compute_loss(out, targets)
    bsz = targets["num_boxes"].shape[0]
    scale = bsz**0.5  # scale_by_find_batch_size
    losses["core_loss"] = losses["core_loss"] * scale
    if neg_query_weight > 0.0:
        neg_mask = targets["num_boxes"] == 0
        if bool(neg_mask.any()):
            n_neg = int(neg_mask.sum().item())
            neg_total = torch.zeros((), device=out["pred_logits"].device)
            for layer_out in [out, *out.get("aux_outputs", [])]:
                logits_neg = layer_out["pred_logits"][neg_mask].float()
                p_neg = logits_neg.sigmoid()
                bce = F.binary_cross_entropy_with_logits(
                    logits_neg, torch.zeros_like(logits_neg), reduction="none"
                )
                neg_total = neg_total + (p_neg.square() * bce).sum() / n_neg
            losses["neg_query_loss"] = neg_total
            losses["core_loss"] = losses["core_loss"] + (neg_query_weight * scale) * neg_total
        else:
            losses["neg_query_loss"] = torch.zeros((), device=out["pred_logits"].device)
    return losses


# ------------------------------------------------------------------ heads

def build_heads(ckpt=CKPT, device="cpu"):
    """A real `Sam3MultiplexDetector` holding transformer, geometry encoder, scoring head and text resizer."""
    from safetensors import safe_open
    from sam3 import model_builder as mb
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector

    backbone_stub = nn.Module()
    backbone_stub.language_backbone = nn.Module()
    backbone_stub.language_backbone.resizer = nn.Linear(1024, DIM)
    det = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone_stub,
        transformer=mb._create_sam3_transformer(use_fa3=False),
        segmentation_head=None,
        semantic_segmentation_head=None,
        input_geometry_encoder=mb._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=mb._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    prefixes = ("transformer.", "geometry_encoder.", "dot_prod_scoring.", "backbone.language_backbone.resizer.")
    sd = {}
    with safe_open(str(ckpt), "pt") as f:
        for k in f.keys():
            if k.startswith("detector.") and k[len("detector."):].startswith(prefixes):
                sd[k[len("detector."):]] = f.get_tensor(k)
    missing, unexpected = det.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m.startswith(prefixes)]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    det.eval()
    for p in det.parameters():
        p.requires_grad_(False)
    return det.to(device)


def build_text_encoder(ckpt=CKPT, device="cpu"):
    """Load the SAM 3.1 CLIP text backbone (`detector.backbone.language_backbone.*`) with frozen weights."""
    import pkg_resources
    from safetensors import safe_open
    from sam3 import model_builder as mb

    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    te = mb._create_text_encoder(bpe)
    prefix = "detector.backbone.language_backbone."
    with safe_open(str(ckpt), "pt") as f:
        sd = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    te.load_state_dict(sd, strict=True)
    te.eval()
    for p in te.parameters():
        p.requires_grad_(False)
    return te.to(device)


def compute_prompt_pre_resizer(prompts=ALL_PROMPTS, ckpt=CKPT):
    """Return {prompt: (pre_resizer [32, 1024] f32, mask [32] bool)} from the frozen CLIP text tower."""
    te = build_text_encoder(ckpt=ckpt, device="cpu")
    out = {}
    with torch.no_grad():
        tok = te.tokenizer(list(prompts), context_length=te.context_length)
        masks = (tok != 0).bool().ne(1)  # [N, 32], True = pad
        _, pre = te.encoder(tok)  # [N, 32, 1024]
        pre_seq = pre.transpose(0, 1).float().contiguous()  # [32, N, 1024]
        for i, p in enumerate(prompts):
            out[p] = (pre_seq[:, i].contiguous(), masks[i].bool().contiguous())
    return out


# ------------------------------------------------------------------ LoRA

_TARGET_RE = re.compile(r"^transformer\.(encoder\.layers\.\d+\.(self_attn|cross_attn_image)|decoder\.layers\.\d+\.(self_attn|cross_attn|ca_text))$")
_HEAD_LINEAR_RE = re.compile(
    r"^("
    r"dot_prod_scoring\.(prompt_mlp\.layers\.[01]|prompt_proj|hs_proj)|"
    r"transformer\.decoder\.bbox_embed\.layers\.[012]|"
    r"transformer\.decoder\.presence_token_head\.layers\.[012]|"
    r"backbone\.language_backbone\.resizer"
    r")$"
)
_FFN_LINEAR_RE = re.compile(r"^transformer\.(encoder|decoder)\.layers\.\d+\.linear[12]$")
SLOTS = (("q", "in_proj_weight", 0), ("k", "in_proj_weight", 1), ("v", "in_proj_weight", 2), ("o", "out_proj.weight", 0))


class LoRALinear(nn.Module):
    """`nn.Linear` with low-rank adaptation `W + (eff_alpha / eff_rank) * B A`.

    When `min(out_features, in_features) < rank` (e.g. `bbox_embed.layers.2` `[4, 256]` or
    `presence_token_head.layers.2` `[1, 256]`), `eff_rank = min(rank, out_features, in_features)`
    and `eff_alpha = alpha * (eff_rank / rank)` so `scale = eff_alpha / eff_rank == alpha / rank`
    stays identical across layers and satisfies `crates/st-weights/src/lora.rs`'s rank bound.
    """

    def __init__(self, base, prefix, rank=RANK, alpha=ALPHA, dropout=DROPOUT):
        super().__init__()
        assert isinstance(base, nn.Linear), (prefix, type(base))
        self.base, self.prefix = base, prefix
        out_f, in_f = base.out_features, base.in_features
        eff_rank = min(rank, out_f, in_f)
        assert eff_rank >= 1
        eff_alpha = float(alpha) * (eff_rank / float(rank))
        self.out_features, self.in_features = out_f, in_f
        self.rank, self.alpha = eff_rank, eff_alpha
        self.scale = eff_alpha / eff_rank
        self.lora_A = nn.ParameterList([nn.Parameter(torch.empty(eff_rank, in_f))])
        self.lora_B = nn.ParameterList([nn.Parameter(torch.zeros(out_f, eff_rank))])
        nn.init.kaiming_uniform_(self.lora_A[0], a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.enabled = True

    def forward(self, x):
        y = F.linear(x, self.base.weight, self.base.bias)
        if self.enabled:
            y = y + F.linear(F.linear(self.drop(x), self.lora_A[0]), self.lora_B[0]) * self.scale
        return y

    def modules_spec(self):
        return [(self.prefix, f"{self.prefix}.weight", (0, self.out_features), self.lora_A[0], self.lora_B[0])]


class LoRAMultiheadAttention(nn.Module):
    """`sam3.model.model_misc.MultiheadAttention` with rank-r LoRA on q, k, v and the output projection.

    Same call signature and the same masking as upstream's forward (the key-padding mask, a 2-D or
    3-D `attn_mask`, the box-RPB `attn_bias`), checked against it by lora_selftest.py. The wrapped
    module stays frozen and is always run as upstream's eval (no attention dropout). `self.training`
    only switches the LoRA input dropout.
    """

    def __init__(self, base, prefix, rank=RANK, alpha=ALPHA, dropout=DROPOUT):
        super().__init__()
        assert base._qkv_same_embed_dim and base.bias_k is None and not base.add_zero_attn and base.in_proj_bias is not None
        self.base, self.prefix = base, prefix  # prefix: checkpoint name of the attention module
        e = base.embed_dim
        assert base.in_proj_weight.shape == (3 * e, e) and base.out_proj.weight.shape == (e, e)
        self.embed_dim, self.num_heads, self.batch_first = e, base.num_heads, base.batch_first
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_A = nn.ParameterList([nn.Parameter(torch.empty(rank, e)) for _ in SLOTS])
        self.lora_B = nn.ParameterList([nn.Parameter(torch.zeros(e, rank)) for _ in SLOTS])
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))  # PEFT / ojas-model init: U(+-1/sqrt(in))
        self.drop = nn.Dropout(dropout)
        self.enabled = True

    def _lora(self, x, i):
        return F.linear(F.linear(self.drop(x), self.lora_A[i]), self.lora_B[i]) * self.scale

    def _proj_in(self, x, i):
        e = self.embed_dim
        y = F.linear(x, self.base.in_proj_weight[i * e:(i + 1) * e], self.base.in_proj_bias[i * e:(i + 1) * e])
        return y + self._lora(x, i) if self.enabled else y

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, attn_bias=None):
        assert not need_weights
        if self.batch_first:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        tgt_len, bsz, e = query.shape
        src_len = key.shape[0]
        h, hd = self.num_heads, e // self.num_heads
        q = self._proj_in(query, 0).contiguous().view(tgt_len, bsz * h, hd).transpose(0, 1)
        k = self._proj_in(key, 1).contiguous().view(src_len, bsz * h, hd).transpose(0, 1)
        v = self._proj_in(value, 2).contiguous().view(src_len, bsz * h, hd).transpose(0, 1)

        # Mask merging as in model_misc.multi_head_attention_forward.
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() != 3 or attn_mask.shape != (bsz * h, tgt_len, src_len):
                raise RuntimeError(f"attn_mask {tuple(attn_mask.shape)}")
        if key_padding_mask is not None:
            kpm = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, h, -1, -1).reshape(bsz * h, 1, src_len)
            if attn_mask is None:
                attn_mask = kpm
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(kpm)
            else:
                attn_mask = attn_mask.masked_fill(kpm, float("-inf"))
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            m = torch.zeros_like(attn_mask, dtype=q.dtype)
            m.masked_fill_(attn_mask, float("-inf"))
            attn_mask = m
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0) if attn_mask.size(0) == 1 else attn_mask.view(bsz, h, -1, src_len)
        if attn_bias is not None:
            assert attn_bias.shape == (bsz, h, tgt_len, src_len)
            attn_mask = attn_bias if attn_mask is None else attn_mask + attn_bias

        q = q.view(bsz, h, tgt_len, hd)
        k = k.view(bsz, h, src_len, hd)
        v = v.view(bsz, h, src_len, hd)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask, 0.0, False)
        o = o.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, e)
        out = F.linear(o, self.base.out_proj.weight, self.base.out_proj.bias)
        if self.enabled:
            out = out + self._lora(o, 3)
        out = out.view(tgt_len, bsz, e)
        return (out.transpose(1, 0) if self.batch_first else out), None

    def modules_spec(self):
        """(module name, checkpoint param, row slice, A, B) for each of q, k, v, o."""
        e = self.embed_dim
        spec = []
        for i, (name, param, block) in enumerate(SLOTS):
            module = f"{self.prefix}.in_proj_{name}" if name != "o" else f"{self.prefix}.out_proj"
            spec.append((module, f"{self.prefix}.{param}", (block * e, (block + 1) * e), self.lora_A[i], self.lora_B[i]))
        return spec


def _replace_submodule(root, name, wrapper):
    parent_name, attr = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(attr)] = wrapper
    else:
        setattr(parent, attr, wrapper)


def attach_lora(det, rank=RANK, alpha=ALPHA, dropout=DROPOUT, seed=0, include_heads=False, include_ffn=False):
    """Wrap the fusion-encoder and decoder attentions, plus optional scoring/bbox/presence/resizer heads and FFNs."""
    from sam3.model.model_misc import MultiheadAttention

    # The decoder's cross_attn is upstream's own MultiheadAttention; its self_attn and ca_text (and
    # nothing else in these paths) are torch.nn.MultiheadAttention. Both keep in_proj_weight [3e, e].
    attention_types = (MultiheadAttention, nn.MultiheadAttention)
    torch.manual_seed(seed)
    found = [(n, m) for n, m in det.named_modules() if _TARGET_RE.match(n)]
    assert len(found) == 6 * 2 + 6 * 3, [n for n, _ in found]
    wrappers = []
    for name, m in found:
        assert isinstance(m, attention_types), (name, type(m))
        w = LoRAMultiheadAttention(m, "detector." + name, rank, alpha, dropout).to(next(m.parameters()).device)
        w.eval()  # LoRA input dropout is off until set_lora_train(det, True)
        _replace_submodule(det, name, w)
        wrappers.append(w)
    if include_heads:
        found_heads = [(n, m) for n, m in det.named_modules() if _HEAD_LINEAR_RE.match(n)]
        assert len(found_heads) == 11, [n for n, _ in found_heads]
        for name, m in found_heads:
            w = LoRALinear(m, "detector." + name, rank, alpha, dropout).to(next(m.parameters()).device)
            w.eval()
            _replace_submodule(det, name, w)
            wrappers.append(w)
    if include_ffn:
        found_ffn = [(n, m) for n, m in det.named_modules() if _FFN_LINEAR_RE.match(n)]
        assert len(found_ffn) == 24, [n for n, _ in found_ffn]
        for name, m in found_ffn:
            w = LoRALinear(m, "detector." + name, rank, alpha, dropout).to(next(m.parameters()).device)
            w.eval()
            _replace_submodule(det, name, w)
            wrappers.append(w)
    return wrappers


class LoRAEmbeddingRows(nn.Module):
    """Exact rank-1 row adaptation on selected `nn.Embedding` vocabulary rows (`W[r:r+1] + scale * B_r @ A_r`).

    Each targeted token row `r` (`[1, D]`) has exact rank 1 (`B: [1, 1]`, `A: [1, D]`), so zero
    low-rank approximation error is incurred while staying 100% compatible with
    `crates/st-weights/src/lora.rs`'s disjoint row-slice merge (`rows: [r, r + 1]`).
    """

    def __init__(self, base, prefix, rows=CLASS_TOKEN_IDS, alpha=2.0):
        super().__init__()
        assert isinstance(base, nn.Embedding), (prefix, type(base))
        rows = tuple(sorted(set(int(r) for r in rows)))
        assert rows and all(0 <= r < base.num_embeddings for r in rows), rows
        self.base, self.prefix = base, prefix
        self.rows = rows
        self.embed_dim = base.embedding_dim
        self.rank, self.alpha = 1, float(alpha)
        self.scale = float(alpha)
        self.lora_A = nn.ParameterList([nn.Parameter(torch.empty(1, self.embed_dim)) for _ in rows])
        self.lora_B = nn.ParameterList([nn.Parameter(torch.zeros(1, 1)) for _ in rows])
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, tok):
        y = self.base(tok)
        if self.enabled:
            for i, r in enumerate(self.rows):
                delta = (self.lora_B[i] @ self.lora_A[i]) * self.scale  # [1, D]
                hit = (tok == r).unsqueeze(-1).to(y.dtype)
                y = y + hit * delta
        return y

    def modules_spec(self):
        return [
            (f"{self.prefix}.row_{r}", f"{self.prefix}.weight", (r, r + 1), self.lora_A[i], self.lora_B[i])
            for i, r in enumerate(self.rows)
        ]


def attach_text_lora(te, rows=CLASS_TOKEN_IDS, alpha=2.0, seed=0):
    """Wrap `te.encoder.token_embedding` on VISEM class prompt vocabulary rows and verify disjointness."""
    special = {0, 49406, 49407}
    cls_prompts = list(CLASS_PROMPTS.values())
    other_prompts = [p for p in ALL_PROMPTS if p not in CLASS_PROMPTS.values()]
    cls_tok = te.tokenizer(cls_prompts, context_length=te.context_length)
    other_tok = te.tokenizer(other_prompts, context_length=te.context_length)
    cls_ids = {int(t) for t in cls_tok.reshape(-1).tolist() if int(t) not in special}
    other_ids = {int(t) for t in other_tok.reshape(-1).tolist() if int(t) not in special}
    assert tuple(sorted(cls_ids)) == tuple(sorted(rows)), (sorted(cls_ids), sorted(rows))
    assert not (cls_ids & other_ids), f"class token IDs overlap negative/probe prompts: {cls_ids & other_ids}"

    torch.manual_seed(seed)
    emb = te.encoder.token_embedding
    dev = next(emb.parameters()).device
    w = LoRAEmbeddingRows(
        emb, "detector.backbone.language_backbone.encoder.token_embedding", rows=rows, alpha=alpha
    ).to(dev)
    w.eval()
    te.encoder.token_embedding = w
    tok_map = {p: cls_tok[i].to(dev) for i, p in enumerate(cls_prompts)}
    return w, tok_map


def encode_prompts_with_text_lora(prompts, cached_pre, te, tok_map, device):
    """Return `[32, B, 1024]` pre-resizer features, running `te.encoder` differentiably on class prompts."""
    uniq_cls = []
    for p in prompts:
        if p in tok_map and p not in uniq_cls:
            uniq_cls.append(p)
    live = {}
    if uniq_cls:
        sub_tok = torch.stack([tok_map[p] for p in uniq_cls], 0)
        _, pre = te.encoder(sub_tok)  # [M, 32, 1024]
        pre_seq = pre.transpose(0, 1).float()  # [32, M, 1024]
        for i, p in enumerate(uniq_cls):
            live[p] = pre_seq[:, i]
    return torch.stack([live[p] if p in live else cached_pre[p].to(device) for p in prompts], 1)


def _iter_roots(roots):
    if roots is None:
        return ()
    return tuple(r for r in (roots if isinstance(roots, (list, tuple)) else (roots,)) if r is not None)


def lora_wrappers(roots):
    return [
        m
        for r in _iter_roots(roots)
        for m in r.modules()
        if isinstance(m, (LoRAMultiheadAttention, LoRALinear, LoRAEmbeddingRows))
    ]


def lora_parameters(roots):
    return [p for w in lora_wrappers(roots) for p in list(w.lora_A) + list(w.lora_B)]


def set_lora_train(roots, train):
    for w in lora_wrappers(roots):
        w.train(train)


def set_lora_enabled(roots, enabled):
    for w in lora_wrappers(roots):
        w.enabled = enabled


# ------------------------------------------------------------------ adapter file

def base_sha256():
    return json.loads(WEIGHTS_MANIFEST.read_text())["safetensors_sha256"]


def adapter_entries(wrappers):
    """[(module, param, rows, rank, alpha, A, B)] in a fixed order."""
    out = []
    for w in wrappers:
        for module, param, rows, a, b in w.modules_spec():
            out.append((module, param, rows, w.rank, w.alpha, a.detach().float().cpu().contiguous(), b.detach().float().cpu().contiguous()))
    return out


def save_adapter(path, entries, extra=None):
    """Write entries (see `adapter_entries`) as the spiketrack-lora-v1 safetensors file."""
    from safetensors.torch import save_file

    tensors, modules = {}, []
    for module, param, rows, rank, alpha, a, b in entries:
        assert a.shape[0] == rank and b.shape == (rows[1] - rows[0], rank), (module, a.shape, b.shape)
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), module
        tensors[f"{module}.lora_A.weight"] = a
        tensors[f"{module}.lora_B.weight"] = b
        modules.append({"module": module, "param": param, "rows": list(rows), "rank": rank, "alpha": float(alpha)})
    meta = {"format": FORMAT, "base_sha256": base_sha256(), "modules": modules}
    meta.update(extra or {})
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={METADATA_KEY: json.dumps(meta)})


def load_adapter(path):
    """-> (meta dict, [(module, param, rows, rank, alpha, A, B)])."""
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()[METADATA_KEY])
        assert meta["format"] == FORMAT, meta["format"]
        entries = [
            (m["module"], m["param"], tuple(m["rows"]), m["rank"], m["alpha"],
             f.get_tensor(f"{m['module']}.lora_A.weight"), f.get_tensor(f"{m['module']}.lora_B.weight"))
            for m in meta["modules"]
        ]
    return meta, entries


def load_adapter_into(roots, path, strict=True):
    """Copy a saved adapter into the attached wrappers (`strict=False` warm-starts matching subset)."""
    meta, entries = load_adapter(path)
    spec = {s[0]: s for w in lora_wrappers(roots) for s in w.modules_spec()}
    if strict:
        assert len(spec) == len(entries), (len(spec), len(entries))
    with torch.no_grad():
        for module, param, rows, rank, alpha, a, b in entries:
            assert module in spec, f"adapter module {module!r} missing from attached LoRA wrappers"
            _, want_param, want_rows, pa, pb = spec[module]
            assert (param, tuple(rows)) == (want_param, want_rows) and pa.shape == a.shape and pb.shape == b.shape, module
            pa.copy_(a)
            pb.copy_(b)
    return meta


def merge_into_state(state, entries):
    """W[rows] += (alpha / rank) * (B @ A) on `state[param]` (f32), for every entry. In place.

    The same formula crates/st-weights/src/lora.rs applies. torch's matmul sums in a different
    order than Rust's loop, so the two agree to f32 rounding, not bit for bit."""
    for module, param, rows, rank, alpha, a, b in entries:
        w = state[param]
        assert w.dtype == torch.float32 and w.shape[1] == a.shape[1] and rows[1] <= w.shape[0], module
        w[rows[0]:rows[1]] += (alpha / rank) * (b.float() @ a.float())
    return state


# ------------------------------------------------------------------ forward

def grounding_forward(det, feats, pos, txt, txt_mask, aux=True, act_ckpt=False, dac=False, txt_pre_resizer=None):
    """Detector outputs from cached inputs; `Sam3Image.forward_grounding` without segmentation/matching.

    feats [B, 256, 72, 72] (the 72x72 FPN level of the detector neck), pos [1 or B, 256, 72, 72],
    txt [32, B, 256] (used when `txt_pre_resizer` is None), `txt_pre_resizer` [32, B, 1024] (projected
    through `det.backbone.language_backbone.resizer` so LoRA on `resizer` receives gradients),
    txt_mask [B, 32] (True = pad). `aux` also returns the 5 earlier decoder layers as `aux_outputs`;
    `dac` enables the decoder's one-to-many (DAC) queries during training (`aux=True`).
    `act_ckpt` turns on upstream's own encoder activation checkpointing (a flag on the encoder module).
    """
    from sam3.model.data_misc import FindStage, NestedTensor

    bsz = feats.shape[0]
    dev = feats.device
    if txt_pre_resizer is not None:
        txt = det.backbone.language_backbone.resizer(txt_pre_resizer)
    bo = {
        "backbone_fpn": [NestedTensor(feats, None)],
        "vision_pos_enc": [pos.expand(bsz, -1, -1, -1)],
        "language_features": txt,
        "language_mask": txt_mask,
    }
    ids = torch.arange(bsz, device=dev)
    fs = FindStage(img_ids=ids, text_ids=ids, input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                   input_points=None, input_points_mask=None)
    prompt, prompt_mask, bo = det._encode_prompt(bo, fs, det._get_dummy_prompt(bsz))
    enc = det.transformer.encoder
    saved_enc = enc.training
    enc.training = bool(act_ckpt)
    try:
        bo, enc_out, _ = det._run_encoder(bo, fs, prompt, prompt_mask)
    finally:
        enc.training = saved_enc
    out = {"encoder_hidden_states": enc_out["encoder_hidden_states"]}
    dec = det.transformer.decoder
    saved = (det.training, dec.dac)
    det.training, dec.dac = bool(aux), bool(dac)
    try:
        out, _ = det._run_decoder(memory=out["encoder_hidden_states"], pos_embed=enc_out["pos_embed"],
                                  src_mask=enc_out["padding_mask"], out=out, prompt=prompt,
                                  prompt_mask=prompt_mask, encoder_out=enc_out)
    finally:
        det.training, dec.dac = saved
    return out


def scores(out):
    """(joint prob [B,Q], presence prob [B], processor-style prob [B,Q]) from `grounding_forward` outputs.

    `pred_logits` already is inverse_sigmoid(sigmoid(cls) * sigmoid(presence)) (supervise_joint_box_scores,
    sam3_image.py:345-353); that joint probability is what the video path thresholds. Sam3Processor
    (`_forward_grounding`) multiplies the presence in a second time; zero_shot_check.py used the
    processor, so its numbers correspond to `proc`."""
    joint = out["pred_logits"].squeeze(-1).sigmoid()
    pres = out["presence_logit_dec"].reshape(joint.shape[0], -1)[:, 0].sigmoid()
    return joint, pres, joint * pres[:, None]
