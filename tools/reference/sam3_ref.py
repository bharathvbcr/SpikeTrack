"""Build the official SAM 3.1 multiplex stack on CPU (offline reference only).

Mirrors sam3.model_builder.build_sam3_multiplex_video_predictor (external/sam3
f1647e8) with three changes so it runs on a Mac: no FlashAttention 3, complex
RoPE instead of the real-valued compile variant, and no `.cuda()`. Never
imported by any Rust crate.
"""

import os
import sys

import torch

import cpu_shims

cpu_shims.install()

CKPT = os.path.expanduser("~/Code/research/models/sam3.1/sam3.1_multiplex.pt")


def use_fp32_mlp():
    """Upstream's ViT MLP runs fc1+GELU through `perflib.fused.addmm_act`, which
    casts bias, input and weight to bf16 (fc2 then sees bf16 under CUDA autocast,
    and a dtype error on CPU). fp32 fixtures replace it with the same math in
    fp32: exact-erf GELU of the biased fc1 output (on CPU `_addmm_activation`'s
    GELU is the erf form; see tools/reference/README.md)."""
    import torch.nn.functional as F
    from sam3.model import vitdet

    def addmm_act_fp32(activation, linear, mat1):
        y = F.linear(mat1, linear.weight, linear.bias)
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(y)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    vitdet.addmm_act = addmm_act_fp32


def build_reference(checkpoint_path=CKPT, multiplex_count=16, max_num_objects=16):
    import pkg_resources  # noqa: F401  (sam3 resolves its BPE path through it)
    from sam3 import model_builder as mb
    from sam3.model.sam3_multiplex_base import Sam3MultiplexPredictorWrapper
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    bpe_path = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
    )
    tracker_model = mb.build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        multiplex_count=multiplex_count,
        use_fa3=False,
        use_rope_real=False,
        compile=False,
        strict_state_dict_loading=False,
        device="cpu",
    )
    del tracker_model.backbone
    tracker_model.backbone = None
    sam2_predictor = Sam3MultiplexPredictorWrapper(
        model=tracker_model,
        per_obj_inference=False,
        fill_hole_area=0,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )
    tri_neck = mb._create_multiplex_tri_backbone(
        compile_mode=None, use_fa3=False, use_rope_real=False
    )
    text_encoder = mb._create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    detector = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=mb._create_sam3_transformer(use_fa3=False),
        segmentation_head=mb._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=mb._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=mb._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    # Same heuristics as the official predictor builder (not config.json, which
    # is the Hugging Face port's file and differs: 0.5/0.7 there vs 0.4/0.65 here).
    demo_model = Sam3MultiplexTrackingWithInteractivity(
        tracker=sam2_predictor,
        detector=detector,
        score_threshold_detection=0.4,
        det_nms_thresh=0.1,
        det_nms_use_iom=True,
        assoc_iou_thresh=0.1,
        new_det_thresh=0.65,
        hotstart_delay=15,
        hotstart_unmatch_thresh=8,
        hotstart_dup_thresh=8,
        suppress_unmatched_only_within_hotstart=False,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=True,
        fill_hole_area=0,
        recondition_every_nth_frame=16,
        use_iom_recondition=True,
        iom_thresh_recondition=0.5,
        masklet_confirmation_enable=True,
        reconstruction_bbox_iou_thresh=-1,
        reconstruction_bbox_det_score=0.8,
        max_num_objects=max_num_objects,
        postprocess_batch_size=16,
        use_batched_grounding=True,
        batched_grounding_batch_size=16,
        max_num_kboxes=0,
        sprinkle_removal_area=0,
        is_multiplex=True,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    missing, unexpected = demo_model.load_state_dict(ckpt, strict=False)
    demo_model.eval()
    return demo_model, list(missing), list(unexpected)


def load_image(path, size=1008):
    """Official preprocessing: RGB, straight resize to 1008x1008, /255, mean/std 0.5.

    Returns (uint8 HWC original, float32 1x3xHxW normalized).
    """
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    raw = np.asarray(im)
    resized = im.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255.0
    x = (x - 0.5) / 0.5
    return raw, x[None]


if __name__ == "__main__":
    import time

    use_fp32_mlp()
    t0 = time.time()
    model, missing, unexpected = build_reference()
    print(f"built+loaded in {time.time() - t0:.1f}s")
    print(f"missing {len(missing)}: {missing[:20]}")
    print(f"unexpected {len(unexpected)}: {unexpected[:20]}")
    if len(sys.argv) > 1:
        _, x = load_image(sys.argv[1])
        trunk = model.detector.backbone.vision_backbone.trunk
        with torch.inference_mode():
            t0 = time.time()
            out = trunk(x)
            print(f"trunk forward {time.time() - t0:.1f}s ->", [tuple(o.shape) for o in out])
