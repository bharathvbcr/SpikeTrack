# tools/reference — offline PyTorch reference (Python lives only here)

Nothing in this folder is built, imported or run by any Rust crate. It produces fixtures and
facts the Rust port is checked against.

Labels: **[V]** verified this session by a run or by reading the code/checkpoint (source named),
**[I]** inferred, **[U]** unverified.

## Setup

```sh
uv venv --python 3.12 tools/reference/.venv
uv pip install --python tools/reference/.venv/bin/python torch==2.14.1 torchvision==0.29.1 \
  'numpy<2' timm tqdm ftfy==6.1.1 regex iopath huggingface_hub safetensors pillow \
  opencv-python-headless pycocotools einops scipy scikit-image psutil 'setuptools<81'
git clone https://github.com/facebookresearch/sam3 ~/Code/research/external/sam3   # f1647e8
git -C ~/Code/research/external/sam3 switch -c spiketrack-cpu
git -C ~/Code/research/external/sam3 am tools/reference/sam3-cpu.patch
uv pip install --python tools/reference/.venv/bin/python --no-deps -e ~/Code/research/external/sam3
```

- A separate venv: upstream pins `numpy<2`, and the shared `~/.venvs/ml-2026-10` (tessl's Qwen
  reference) must not change. (scikit-image pulled numpy 2.5.4 back in; the reference runs on it.)
- Upstream's `pyproject.toml` leaves out deps the model code imports: einops, torchvision, scipy,
  scikit-image, psutil, and `pkg_resources` (setuptools < 81) [V, import errors on first run].
- `setuptools<81` provides `pkg_resources`, which upstream uses to find its BPE vocabulary.

## Making upstream run on a Mac CPU

The official code states CUDA as a requirement. Three things stop it on CPU [V]:

1. `sam3.model.edt` imports `triton` at module level. `cpu_shims.py` registers a replacement
   module whose `edt_triton` raises if called. A fake `triton` does not work, because
   `torch._dynamo` probes it.
2. `sam3_multiplex_base.py` calls `torch.cuda.get_device_properties(0)` at import time to
   enable TF32. `cpu_shims.py` answers that one probe during the import.
3. About 30 hardcoded CUDA placements (`.cuda()`, `device="cuda"`, `torch.device("cuda")`).
   `make_cpu_patch.py` routes them through `SAM3_DEVICE` (default `cuda`, so CUDA runs are
   unchanged); the result is `sam3-cpu.patch`. Autocast contexts are left alone: on CPU-only torch
   they disable themselves, which is what fp32 fixtures want.

Run everything with `SAM3_DEVICE=cpu`.

## Scripts

| Script | Does |
|---|---|
| `sam3_ref.py` | Builds the SAM 3.1 multiplex stack exactly as `build_sam3_multiplex_video_predictor` does (FA3 off, complex RoPE, no `.cuda()`). Loads with **0 missing, 0 unexpected** keys [V]. `python sam3_ref.py IMAGE` times one trunk forward (6.6 s fp32 CPU, M5 Pro, one run, machine busy). |
| `convert_checkpoint.py` | `.pt` → `~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors` (SHA-256 `6f0e46e4…b160`), plus the committed `weights_manifest.json` (names/shapes/dtypes, no values). It checks the source SHA-256 first. The 32 complex64 RoPE tables become f32 `[T,32,2]` and are listed in the file metadata under `complex64_as_f32_pairs`. |
| `dump_fixtures.py` | fp32 backbone fixtures for 5 images → `fixtures/backbone/*.safetensors` and `fixtures/MANIFEST.json` (shape, op, SHA-256, command). Image 0 (VISEM 22 frame 0) also gets every block output, the per-op internals of block 0 (windowed) and block 7 (global), and the neck-level-0 chain. |
| `dump_backbone_size.py` | Trunk + three necks at another input size (a multiple of 336; E7 uses 672) from the official code: builds the PE ViT with `img_size=SIZE` (the builder hard-codes 1008, so `model_builder.ViT` is wrapped for that one argument), loads the checkpoint minus the four global blocks' `freqs_cis` (upstream rebuilds them for the grid: `scale_pos = 24 / grid`), and writes `fixtures/backbone672/visem_22_f0.safetensors` for `backbone_parity`'s 672 gate (`tools/jobs/input_size_gates.sh`). |
| `check_fixtures.py` | Recomputes fixture tensors from their recorded inputs with independent torch math. **27/27 pass** [V]. |
| `positive_control.py` | Detector sanity check on upstream's images. Upstream's notebook oracle: `test_image.jpg` with "shoe" finds **12** objects at threshold 0.5 (SAM 3 image model). This CPU 3.1 pipeline also finds **12** [V]. `truck.jpg`: "truck" 0.78, "wheel" 4 detections at 0.88–0.94, "zebra" 0. |
| `zero_shot_check.py` | gp-zero-shot-check, run locally through upstream's `Sam3Processor` (no API key). Results go to `bench/results/zero-shot/`. |
| `dump_text_fixtures.py` | Tokenizer ids and text-encoder outputs for 13 prompts → `fixtures/text/prompts.safetensors`; the ids are also committed as `tokenizer_cases.json` (the Rust BPE oracle). |

### Preprocessing: two upstream paths
- The image path (`Sam3Processor`): `torchvision.transforms.v2.Resize((1008, 1008))` on a uint8
  image tensor, then /255 and mean/std 0.5 [V sam3_image_processor.py]. That is v2's default
  antialiased bilinear on a tensor, not PIL. The backbone fixtures were made with PIL bilinear. That
  does not affect backbone parity (Rust gets the same input tensor), but the Rust preprocessor must
  match v2.Resize for end-to-end parity [U, not yet compared].
- The video path (`io_utils.load_video_frames` -> `_load_img_as_tensor`): `TF.resize` on the PIL image, which is PIL bilinear (identical output, checked on 3 images), then `to_tensor`, **float16 storage**, and mean/std 0.5 applied in float16 [V io_utils.py]. Frames are cast back with `.float()` at use.
- The two resamplers disagree slightly [V, 2026-10-26: 640x480 VISEM, 854x480 DAVIS and 1800x1200 truck, resized to 1008]. PIL bilinear vs torchvision v2 on a uint8 tensor: max difference 1 grey level, in 0.10-0.41% of pixels. A Rust port of Pillow's fixed-point bilinear matches the video path bit for bit and the image path to within that.
- The tracker stores memories as **bfloat16** each frame (gaps.jsonl GAP-UPSTREAM-BF16-MEMORY).

## Architecture, from the checkpoint and upstream code

Sources: `sam3.1_multiplex.pt` header (1,623 tensors, 874,365,676 params, 1,591 f32 + 32 complex64)
and `external/sam3` f1647e8 `model_builder.py`, `vitdet.py`, `necks.py`. Rows marked
*check_fixtures* were recomputed numerically.

### Vision trunk (PE ViT, 468 M params incl. the three necks)
| Fact | Value | Evidence |
|---|---|---|
| Input | 1008×1008, patch 14 → 72×72 = 5,184 tokens | V builder |
| Patch embed | conv 3→1024, k14 s14, **no bias**, output NHWC | V check_fixtures |
| Abs pos embed | `pos_embed [1,577,1024]` = cls + 24×24 pretrain grid, **tiled 3×3** (not interpolated), cls dropped, added before `ln_pre` | V check_fixtures |
| `ln_pre` | LayerNorm **eps 1e-5** (the builder's `norm_layer="LayerNorm"` -> `partial(nn.LayerNorm, eps=1e-5)`, vitdet.py:868-869; shared with the blocks) | V code |
| Blocks | 32; dim 1024, 16 heads, **head dim 64**; MLP 4736; qkv bias; no layer scale; no QK-norm | V checkpoint |
| Attention layout | 28 windowed (3×3 windows of 24×24 = batch 9 × 576) + global at 7, 15, 23, 31 (1 × 5,184) | V check_fixtures |
| Block norms | LayerNorm **eps 1e-5**, same norm layer as ln_pre (vitdet.py:868-869, :907). Corrected 2026-10-27: this row said 1e-6 and check_fixtures' tolerance hid the difference (found by ojas-78) | V code + backbone_parity `vit_layer_norm_eps_matches_upstream` |
| GELU | exact erf in MLP and neck | V check_fixtures |
| RoPE | 2-D axial, **interleaved pairs** (2i, 2i+1); first 16 pairs x, next 16 y; freq_i = 10000^(−4i/64). Windowed: integer grid 24×24. Global: 72×72 grid × (24/72), i.e. fractional positions. Tables stored per block (`attn.freqs_cis`, complex64); all windowed blocks share one table, all global blocks another | V check_fixtures (max diff 6e-8 vs formula) |
| Upstream numerics | ViT MLP fc1+GELU runs in **bf16** in upstream inference (`perflib/fused.addmm_act` casts). On CPU `_addmm_activation`'s GELU is erf | V fused.py; V CPU test |
| Neck | **three** simple-FPN necks of identical shape: `convs` (detector), `interactive_convs`, `propagation_convs` (tracker); scales 4, 2, 1 (**no 0.5 level** in 3.1). ×4: convT k2s2 1024→512, GELU, convT 512→256; ×2: convT 1024→512; each then conv 1×1 → 256 and conv 3×3 pad 1 | V check_fixtures (level 0) |

### Text encoder (354 M)
CLIP-style: 24 resblocks, width 1024, 16 heads (head dim 64), MLP 4096, context 32, vocab 49,408,
`text_projection [1024,512]`, then `resizer` Linear 1024→256 into the detector [V checkpoint].
Tokenizer: upstream ships `assets/bpe_simple_vocab_16e6.txt.gz` (CLIP's SimpleTokenizer) [V builder].

### Detector and tracker (typed later)
- DETR encoder/decoder: 6 + 6 layers, d 256, 8 heads (head dim 32), presence token [V checkpoint and HF config].
- Multiplex bucket size **16** (`multiplex_count=16` default; `max_num_objects=16` in the predictor) [V builder].
- Memory: `num_maskmem=7`, `max_cond_frames_in_attn=4`, `max_obj_ptrs_in_encoder=16` [V builder].
- Mask downsampler input has 32 channels: (16, 32, 3, 3) → 16 → 64 → 256 → 1024, then 1×1 → 256 [V checkpoint].
- Memory fuser: 2 ConvNeXt blocks (depthwise 7×7, LN, 256→1024→256, gamma) [V checkpoint].
- Video heuristics come from the official predictor builder, **not** `config.json` (the HF port's
  file): `score_threshold_detection=0.4`, `new_det_thresh=0.65` (config.json says 0.5 / 0.7),
  `det_nms_use_iom=True`, `fill_hole_area=0`, `recondition_every_nth_frame=16` [V builder].
- The tracker has no backbone of its own; it uses the shared trunk through the tri-neck [V builder].

## Not yet done (see gaps.jsonl)
Text, detector, mask and video fixtures; CoreML and PyTorch-MPS baselines; an fp32 vs bf16
reference comparison; DAVIS / SA-V clips.

### Tracker (multiplex VOS), from the live model and the checkpoint [V]
- Config read from the built model: `num_maskmem` 7, `max_cond_frames_in_attn` 4, `memory_temporal_stride_for_eval` 1, no memory selection, `max_obj_ptrs_in_encoder` 16 (pointers from frames before *and after* the current frame: `only_obj_ptrs_in_the_past_for_eval` False), unsigned temporal positions, sine tpos encoding projected by `obj_ptr_tpos_proj`, `use_maskmem_tpos_v2`, image features saved per memory frame, object-score threshold 0.0, no stability attenuation, multimask on in tracking (0-1 points), pointer from the chosen multimask token, linear no-object pointer, memory mask = `sigmoid(logit) * 2 - 1`, conditioning channel 1.0 / 0.0, output-suppression embeddings, output non-overlap on, no hole filling, `input_mask_size` 1152, no shared necks.
- The interactive decoder uses dynamic multimask (stability) and the multiplex decoder does not. In VOS the interactive decoder only supplies frame 0's pointers, which always come from mask token 0.
- Multiplex decoder: 16 slots x 3 masks (48 mask tokens), 16 IoU and 16 object-score tokens, two-way transformer depth 2 (ReLU MLP 2048, cross-attention width 128). Memory attention: 4 layers, 8 heads of 32 (config.json says 1), RoPE 72x72, GELU FFN. Memory encoder: 32-channel input (16 masks + 16 condition flags) resized to 1152, 4 stride-2 3x3 convs to 1024 channels, 1x1 to 256, plus `pix_feat_proj`, 2 ConvNeXt blocks. Memories are stored in bf16.
- Padding slots: their object score is 0, which is not above the 0.0 threshold. So `no_obj_embed_spatial` for every padding slot is added to each frame's memory.
- `get_1d_sine_pe` concatenates `[sin, cos]` halves (not interleaved).
- Upstream's standalone demo fails on a feature-cache miss (`video_tracking_multiplex.py:1095` at f1647e8). `sam3_ref.TrackerFeatureCache` supplies features the way the full predictor does.
- VOS reference J&F on DAVIS 2017 val, upstream on CPU: `bench/results/vos-reference/`.

### Memory attention head count: 1 head (config.json) or 8 heads of 32 (SpikeTrack)? Decision record, 2026-11-01
One softmax over a 256-wide head and eight softmaxes over 32-wide slices are different functions, so this was checked, not assumed. Finding: **8 heads of 32 is what upstream's multiplex model computes**; `config.json` describes a different model. Evidence, each item read in the files named (nothing was run for this record):
- **Source of the 8 [verified].** `external/sam3/sam3/model_builder.py:865` `_create_multiplex_transformer`, which `build_sam3_multiplex_video_model` calls (`:968`), builds `SimpleRoPEAttention(d_model=256, num_heads=8)` for self-attention (`:867-869`) and cross-attention (`:877-879`) and `DecoupledTransformerDecoderLayerv2(d_model=256, num_heads=8)` (`:888-891`). `git blame` attributes those three lines to Meta's upstream commit `f1647e8` (2026-10-23), not to SpikeTrack's local commit `7af056f`, which only changed device placement. The RoPE is built for `d_model // num_heads` (`decoder.py:1074+`), so upstream's rotary width is 32 as well; SpikeTrack builds its tables with `HEAD_DIM` (`memory_attention.rs:97`).
- **Source of the 1 [verified].** `models/sam3.1/config.json:633` `memory_attention_num_attention_heads = 1`. That file is the Hugging Face `transformers` port's `Sam3VideoModel` config, which has no multiplex fields and disagrees with the official builder elsewhere too (`gaps.jsonl:5`, GAP-HF-CONFIG-NOT-UPSTREAM). It is not what builds the checkpoint's model.
- **Intent [verified that it was recorded; the author's reasoning beyond that is unknown].** `memory_attention.rs` came in with `77c8ef0` (2026-10-26) with "8 heads of 32" in its module doc (`memory_attention.rs:10`) and `HEADS = 8` (`:21`). The same day `gaps.jsonl:5` was opened, saying config.json has 1 head and the official builder builds 8. I found no commit message or note that reasons about it further.
- **The weights cannot settle it [verified].** All 122 `tracker.model.transformer.encoder.*` tensors in the `sam3.1_multiplex.safetensors` header were listed (header only, no model load): in each of the 4 layers the self-attention, cross-attention and image cross-attention q/k/v/out projections are `[256, 256]` with `[256]` biases, `linear1` is `[2048, 256]`, `linear2` is `[256, 2048]`, and the norms are `[256]`. So the head count is not encoded in any shape. SpikeTrack's loader (`memory_attention.rs:77-88`) loads them as plain `[D, D]` matrices and reshapes activations `[B, N, 256] -> [B, N, 8, 32]` at run time (`:144-149`). It does not reinterpret stored weights; upstream does the same split at run time.
- **A parity test exists and passed [verified from the logs; not re-run].** `crates/st-model/tests/tracker_parity.rs:131-182` (`memory_attention_matches_upstream`, `#[ignore]`d, needs the checkpoint and fixtures) feeds the inputs recorded from upstream's own `transformer.encoder` forward hook (`tools/reference/dump_tracker_fixtures.py:80`; model from `sam3_ref.build_tracker_vos`, the official builder with the checkpoint loaded, no unexpected keys) and compares outputs on 7 frames, asserting worst cosine > 0.999. Logged: CPU worst cosine 1.0000000 (`bench/results/module-gates/cpu-memory_attention_matches_upstream.log`, 2026-10-26) and Metal worst cosine 1.0000000, max abs 1.1e-5 to 2.6e-5 (`bench/results/ojas-ea3cf39/metal-memory_attention_matches_upstream.log`, 2026-10-29, ojas `ea3cf39`). `memory_attention.rs` has not changed since 2026-10-28 (`cfdbe16`), before the later log.
- **Inference, not measured:** a 1x256 implementation would not reproduce upstream's output to cosine 1.0000000 and 1e-5 absolute error over 7 frames, since the attention weights differ. No negative control (the same test with `HEADS = 1`) was run, so the test's sensitivity to the head count is argued, not shown.
- **Not verified:** the gate was last logged at ojas `ea3cf39`; ojas `main` has moved since and it has not been re-run against it. The fixtures are one clip (`judo`, 8 frames) from DAVIS, not the VISEM data.

Decision: keep 8 heads of 32. Where a document quotes `config.json`'s 1 head for the memory attention, it is quoting the Hugging Face config, not the model this checkpoint is.
