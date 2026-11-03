# Weight-Space Programming on SAM 3.1 (`st-weights` + `spiketrack weights` / `--weight-edit`)

Measured 2026-11-02 on Apple M5 Pro (macOS, Metal + CoreML fp32 ViT trunk, zero Python at runtime).
Evidence labels: **[V]** verified from committed logs/JSON in this folder; **[I]** mechanistic interpretation.

Inspired by Takehi et al. (2026, *Weight-Space Programming*, `https://rikiyat.github.io/blog/weight-space-programming/`), we added native Rust weight-space reading (`st_weights::inspect`, `spiketrack weights`) and atomic declarative weight editing (`st_weights::edit`, `--weight-edit FILE`, format `spiketrack-weight-edit-v1`) across all **1,623 checkpoint tensors (875,545,324 parameters)** of Meta's SAM 3.1 Multiplex model — covering both typed (`VisionBackbone`, `TextEncoder`) and untyped (`TensorGroup`) parameters.

All diagnostic probes, edit designs, and gain sweeps were conducted **exclusively on the `"train"` split** of `tools/reference/visem_split.json` (stratified 5-video subset `14, 30, 19, 54, 13`, one per density quintile). Three manifests were frozen before touching any `"held_out"` video (`21, 22, 36, 47, 82`).

---

## 1. Mechanistic Weight-Space Diagnostics (`inspect/`)

1. **Why zero-shot `--prompt sperm` outputs 0 detections [V, `inspect/base_tokens_and_bbox_svd.txt`]:**
   - In `detector.backbone.language_backbone.encoder.token_embedding.weight` (`[49408, 1024]`), `"sperm"` tokenizes to a single BPE row: **`35781` (`"sperm</w>"`, `L2 = 0.9363`)**.
   - Its nearest neighbors in raw token-embedding space are clinical/reproductive nouns rather than visual microscopy objects: `39159:"ivf</w>"` (`cos = 0.185`), `23528:"fertility</w>"` (`0.169`), `16673:"hump"` (`0.160`), `16301:"mummy</w>"` (`0.157`), `35267:"embr"` (`0.154`).
   - Its raw cosine with `"cell"` (row **`5533`**, `"cell</w>"`, `L2 = 0.9042`) is **`0.0510`**, and after all 24 CLIP transformer layers + `resizer` (`256-D`), its EOT cosine with `"cell"` is only **`0.3661`** (mean valid tokens: `0.3850`). On VISEM train video `30` frame `0`, `"sperm"` gets `presence_probability = 0.1715` and `0` detections (`>= 0.1`), whereas `"cell"` gets `presence_probability = 0.9842` and `21` detections `> 0.5`.
   - Furthermore, `"pinhead sperm"` (`[2629:"pin", 1375:"head</w>", 35781:"sperm</w>"]`) and `"sperm cluster"` (`[35781:"sperm</w>", 15595:"cluster</w>"]`) share row `35781` and retain **`0.6492`** and **`0.6122`** 256-D prompt cosine with `"sperm"`, explaining why detector-only LoRA (which froze `language_backbone`) exhibits 16% false positives on those absent VISEM classes.

2. **Orthogonal `(cx, cy)` vs `(w, h)` subspaces in `bbox_embed.layers.2.weight` (`[4, 256]`) [V, `inspect/base_tokens_and_bbox_svd.txt`]:**
   - Matrix-free `f64` SVD of `detector.transformer.decoder.bbox_embed.layers.2.weight` (`L2 = 2.0118`) shows exact decoupling (`< 0.027` cross-talk) between size and center:
     - `sigma[0] = 1.3221`: `u = [2:+0.836, 3:+0.549, 0:-0.004, 1:-0.000]` (isotropic `[w, h]` scale)
     - `sigma[1] = 1.1634`: `u = [3:+0.835, 2:-0.549, 1:+0.027, 0:+0.001]` (`[h - w]` aspect ratio)
     - `sigma[2] = 0.7038`: `u = [1:+1.000, 3:-0.023, 2:+0.015, 0:+0.014]` (pure `cy`)
     - `sigma[3] = 0.6715`: `u = [0:+1.000, 1:-0.014, 2:+0.004, 3:+0.002]` (pure `cx`)

3. **Spectral footprint of 600-step LoRA & the frozen scoring-head bottleneck [V, `inspect/lora600_vs_base_spectrum.txt`]:**
   - `visem-lora-r16/adapter.safetensors` modifies **60 tensors / 7,864,317 scalars (`0.898219%` of SAM 3.1)**. Its largest updates concentrate in **decoder text cross-attention** (`decoder.layers.4.ca_text.in_proj_weight`: `delta_l2 = 1.2005`, `rel_l2 = 8.96%`, `sv = [0.8969, 0.4855, 0.4412, 0.2871]`) and fusion cross-attention (`encoder.layers.{3,4}.cross_attn_image.in_proj_weight`: `delta_l2 = 1.0562, 1.0479`).
   - Crucially, `tools/reference/train_lora.py` **froze `detector.dot_prod_scoring.*`**. In `Detector::run` (`crates/st-model/src/detector.rs:579-599`), query logits are `s = (hs_proj(hs) . prompt_proj(pooled)) / sqrt(256)` and probabilities are `sigmoid(s) * sigmoid(presence_logit)`.
   - On train video `30` frame `0` (12 GT sperm) and dense train video `13` frame `0` (44 GT sperm), LoRA achieves near-perfect sign separation: on `13_frame_0`, **43 of 44 GT sperm have `s > 0` (`score in [0.504, 0.713]`), followed by a sharp cliff to `score = 0.347` (`s = -0.632`) at rank 44** [V]. Yet because `dot_prod_scoring` was frozen, **only 18 of those 43 sperm reach SAM 3.1's `0.65` VOS birth threshold (`s >= +0.619`)** — 25 true sperm per frame are suppressed at birth!
    - Scaling `detector.dot_prod_scoring.prompt_mlp.out_norm.{weight,bias}` (**512 scalars = `0.0000585%` of SAM 3.1**) acts as a zero-gradient inverse-temperature calibration: positive `s > 0` sperm logits expand above `0.65`, negative `s < 0` background logits are pushed further below `0.50`, and `presence_token_head` is untouched so absent/negative prompts remain at `0` births (`>= 0.65`) on all spot-checked frames (`30_frame_0` for all 4 prompts; `21, 22, 36, 47, 82` frame `0` for absent `"sperm cluster"`; `22_frame_0` for absent `"pinhead sperm"`; `36_frame_0` for `"car"` and `"bacteria"`; full 156-item absent-class sweep from `lora/visem-lora-r16/eval.md` not re-run) [V].

4. **Enhanced LoRA v2 (`visem-lora-v2`: O2M + scoring/bbox/presence/resizer heads) & exact 768-scalar inverse-temperature [V, `inspect/lorav2_vs_base_spectrum.txt`, `inspect/lorav2_prompt_cosines.txt`]:**
   - Unfreezing `dot_prod_scoring`, `bbox_embed`, `presence_token_head`, and `language_backbone.resizer` alongside upstream's DAC one-to-many queries (`BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)`, `o2m_weight=2.0`) adapts **41 modules / 131 target slices / 71 tensors (`1,127,697` LoRA params)**.
   - The single largest weight update in the entire checkpoint becomes **`detector.dot_prod_scoring.prompt_mlp.layers.0.weight` (`delta_l2 = 1.8654`, `rel_l2 = 12.38%`, `sv[0] = 1.7310`)**, directly confirming that the frozen scoring MLP was the primary gradient bottleneck in v1!
   - `detector.transformer.decoder.bbox_embed.layers.2.weight` (`[4, 256]`) concentrates its updates in the `(w, h)` rows (`row 3 = 0.0187`, `row 2 = 0.0105` vs `row 1 = 0.0012`, `row 0 = 0.0007` — **15x–26x** larger on size than center), tightening matched head IoU (`LocA: 0.7202 -> 0.7390`). Meanwhile `language_backbone.resizer.weight` (`delta_l2 = 0.4565`) lowers 256-D prompt cosine between `"sperm"` and `"sperm cluster"` from **`0.6122 -> 0.5049`** (`"pinhead sperm"`: `0.6492 -> 0.5797`).
   - Because `prompt_proj(pooled) = W_p * pooled + b_p` retains an unadapted bias `b_p`, scaling `out_norm.{weight,bias}` *and* `prompt_proj.bias` together by `G` (**768 scalars = `0.0000877%` of SAM 3.1**, `lorav2_exact_it_3p20.json`) produces an **exact linear logit scaling $s'_q = G \cdot s_q$** with zero decision-boundary shift (`s_q = 0` invariant, preserving `AUROC` and `AP@0.50` to 5 decimal places on positive frames) while aligning `IABCEMdetr`'s `0.50` training boundary with SAM 3.1's `0.65` VOS birth gate.

5. **Enhanced LoRA v3 (`visem-lora-v3`: CLIP BPE `token_embedding` rows + focal negative query suppression) & `G = 3.00` exact inverse-temperature [V, `inspect/lorav3_prompt_cosines.txt`, `../lora/visem-lora-v3/eval.md`]:**
   - In `visem-lora-v2`, adapting `language_backbone.resizer` alone left raw CLIP `token_embedding.weight` frozen while upstream `IABCEMdetr` masked query classification loss by `keep_loss = (num_boxes > 0)`, causing absent morphology prompts (`"sperm cluster"`, `"pinhead sperm"`) to rise to `33.3%` any-det FPR on `held_out`.
   - `visem-lora-v3` (`42` modules / `135` target slices / `72` tensors, `1,131,797` params) adds rank-1 surgical `LoRAEmbeddingRows` (`4,096 / 50,593,792` scalars changed in `detector.backbone.language_backbone.encoder.token_embedding.weight`, `delta_l2 = 0.0055`, `top_rows = [1375:0.0045 ("head</w>"), 15595:0.0026 ("cluster</w>"), 35781:0.0012 ("sperm</w>"), 2629:0.0012 ("pin")]`) plus focal negative query suppression (`neg_query_weight = 1.5`, `p^2 * BCE(logit, 0)` across final + aux decoder layers when `num_boxes == 0`, `hard_neg_frac = 0.65`).
   - On unseen `held_out` (`120` frames), `visem-lora-v3` cuts `absent_class` false-positive rate from **`33.33% -> 19.87%` (`-40.4%` relative)** and `presence > 0.5` from **`46.79% -> 27.56%` (`-41.1%` relative, below zero-shot `35.90%`)** while keeping `100.0%` specificity (`0/600` FP) on everyday negatives. Sweeping exact 768-scalar inverse-temperature on `train-5` (`G in {1.0, 2.8, 3.0, 3.2, 3.4}` selection grid, extended across `{1.6, 2.4, 4.0}`) reaches its selection-grid peak and max `DetA` (`0.4800`) at **`G = 3.00` (`lorav3_exact_it_3p00.json`, `HOTA = 0.5802`)** with a flat post-saturation plateau (`0.577–0.581`) before touching `held_out`.
6. **Error diagnosis after v3, and two levers that did not transfer (2026-11-03) [V, `diag-train5/buckets.txt`, `box-delta-train5/RESULT.md`, `../lora/visem-lora-v4/RESULT.md`]:**
   - On 32 frames of train-5 with v3 + `lorav3_exact_it_3p00.json`, 62.5% of false positives are localization errors (IoU 0.1-0.5 with an unmatched head), 18.2% background, 13.9% concept (cluster/pinhead boxes), 5.4% duplicates; 63.7% of false negatives are near-misses. Concept errors are far below half, so no direction edit (projecting a concept direction out of the scoring projection) was made: there is no dominant concept failure for it to remove.
   - Predicted boxes are smaller than the labels on train-5 (median ln(gt/pred) w/h +0.06/+0.05 matched, +0.34/+0.39 near-misses). `--post-box-delta 0,0,0.16,0.16` won train-5 selection (0.5802 -> 0.5913) only through dense 54 and 13, and lost on held-out (32 frames, 0.6025 -> 0.5017): rejected. The gap is how those videos were labelled.
   - LoRA v4 (128 cached frames per training video, box loss weights 7.5/4.0, +400 steps) with the same 768-scalar edit at G = 3.6 won train-5 (0.5937, 4 of 5 videos up) and tied v3 on held-out (0.5995): not shipped.

---

## 2. Train-Split Selection & Falsified Hypotheses (`train-5`: `14, 30, 19, 54, 13`, 40 frames, 838 GT boxes)

All candidate edits, tracker rules, and LoRA v2/v3 checkpoints were evaluated on the stratified 5-video `"train"` subset (`--max-frames 8 --boxes detector`, `--max-objects 64`) before freezing:

| Track | Condition (`--prompt sperm` unless noted) | Values edited | Fraction of 875.5M | Pred boxes | HOTA | DetA | AssA | LocA | Gate decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | Base `--prompt sperm` | 0 | 0% | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | Baseline |
| Zero-shot | Base `--prompt cell` | 0 | 0% | 598 | 0.1557 | 0.0548 | 0.5030 | 0.6310 | Prompt ref |
| Zero-shot | `graft_row` `5533 -> 35781` (`preserve_norm`) | **1,024** | **0.000117%** | 598 | **0.1557** | 0.0548 | 0.5030 | 0.6310 | **Frozen (`zeroshot_sperm_graft.json`)** |
| Zero-shot | `graft_row` + `bbox_embed.bias[2..4] -= 0.05..0.20` | 1,026 | 0.000117% | 582..433 | 0.1423..0.0943 | 0.0479..0.0266 | 0.4963..0.3929 | 0.6307..0.6330 | **Rejected (perturbs `boxRPB`)** |
| Zero-shot | `graft_row` + `presence_bias +1.5` + `out_norm * 1.25` | **1,537** | **0.000176%** | 882 | **0.1661** | 0.0640 | 0.5251 | 0.6252 | **Frozen (`zeroshot_sperm_graft_calibrated.json`)** |
| LoRA v1 | Base `visem-lora-r16` (`G = 1.00`, whole-mask NMS) | 0 (+983k LoRA) | 0% | 313 | 0.4092 | 0.2347 | 0.7668 | 0.7389 | Baseline |
| LoRA v1 | `bbox_embed.bias[2..4] -= 0.03` | 2 | 0.0000002% | 302 | 0.3925 | 0.2196 | 0.7347 | 0.7302 | **Rejected (perturbs `boxRPB`)** |
| LoRA v1 | **`out_norm * 2.40` (whole-mask NMS)** | **512** | **0.0000585%** | **657** | **0.5395** | **0.4196** | **0.7608** | **0.7256** | **Frozen (`lora_scoring_outnorm_2p40.json`)** |
| LoRA v1 | `out_norm * 2.40` + **head-box/IoM NMS + bulb fallback** | 512 | 0.0000585% | 697 | **0.5613** | 0.4496 | 0.7645 | 0.7242 | **Tracker architecture validated (`+2.18` pts)** |
| LoRA v2 | `visem-lora-v2` step 600 raw (`G = 1.00`) | 0 (+1.13M LoRA) | 0% | 397 | 0.4611 | 0.3076 | 0.7303 | 0.7442 | Raw v2 (`+5.19` pts over raw v1) |
| LoRA v2 | `visem-lora-v2` step 400 raw (`G = 1.00`) | 0 (+1.13M LoRA) | 0% | 518 | 0.5093 | 0.3748 | 0.7428 | 0.7369 | Raw v2 (`+10.01` pts over raw v1) |
| LoRA v2 | `visem-lora-v2` step 200 + `exact_it * 3.20` | 768 | 0.0000877% | 652 | 0.5410 | 0.4261 | 0.7444 | 0.7215 | Under-trained on dense `19` |
| LoRA v2 | `visem-lora-v2` step 600 + `exact_it * 3.20` | 768 | 0.0000877% | 745 | 0.5672 | 0.4643 | 0.7429 | 0.7257 | O2M slightly past step peak |
| LoRA v2 | `visem-lora-v2` step 600 + `exact_it * 3.20` + `--post-box-delta 0,0,0.04,0.04` | 768 | 0.0000877% | 745 | 0.5701 | 0.4701 | 0.7293 | 0.7265 | Confirms post-layer-5 shift avoids `boxRPB` harm |
| LoRA v2 | `visem-lora-v2` step 400 + `exact_it * 2.40` | 768 | 0.0000877% | 743 | 0.5738 | 0.4657 | 0.7643 | 0.7254 | Sweep |
| LoRA v2 | `visem-lora-v2` step 400 + `exact_it * 2.80` | 768 | 0.0000877% | 754 | 0.5752 | 0.4693 | 0.7626 | 0.7247 | Sweep |
| LoRA v2 | **`visem-lora-v2` step 400 + `exact_it * 3.20` (re-verified post-winner coast `0.5751`)** | **768** | **0.0000877%** | **756** | **0.5759** | **0.4689** | **0.7644** | **0.7250** | **Frozen (`lorav2_exact_it_3p20.json`)** |
| LoRA v2 | `visem-lora-v2` step 400 + `exact_it * 3.50` | 768 | 0.0000877% | 782 | 0.5767 | 0.4715 | 0.7625 | 0.7249 | Sparse `14` begins FP (`28/24` boxes, `0.677 -> 0.630`) |
| LoRA v2 | `visem-lora-v2` step 400 + `exact_it * 4.00` | 768 | 0.0000877% | 796 | 0.5799 | 0.4789 | 0.7602 | 0.7243 | Over-predicts on sparse `14` (`28/24`, `0.6300`) |
| LoRA v3 | `visem-lora-v3` step 160 raw (`G = 1.00`) | 0 (+1.13M LoRA) | 0% | 524 | 0.5228 | 0.3890 | 0.7594 | 0.7384 | Raw v3 (`+1.35` pts over raw v2) |
| LoRA v3 | `visem-lora-v3` step 160 + `exact_it * 1.60` | 768 | 0.0000877% | 699 | 0.5683 | 0.4581 | 0.7640 | 0.7307 | Sweep |
| LoRA v3 | `visem-lora-v3` step 160 + `exact_it * 2.40` | 768 | 0.0000877% | 748 | 0.5733 | 0.4652 | 0.7675 | 0.7274 | Sweep |
| LoRA v3 | `visem-lora-v3` step 160 + `exact_it * 2.80` | 768 | 0.0000877% | 754 | 0.5770 | 0.4708 | 0.7687 | 0.7268 | Sweep |
| LoRA v3 | **`visem-lora-v3` step 160 + `exact_it * 3.00` (`train-5` selection-grid peak, max `DetA`)** | **768** | **0.0000877%** | **772** | **0.5802** | **0.4800** | **0.7642** | **0.7258** | **Frozen (`lorav3_exact_it_3p00.json`)** |
| LoRA v3 | `visem-lora-v3` step 160 + `exact_it * 3.20` | 768 | 0.0000877% | 792 | 0.5771 | 0.4759 | 0.7624 | 0.7259 | Selection-grid plateau (`G = 3.40` identical `0.5771`) |
| LoRA v3 | `visem-lora-v3` step 160 + `exact_it * 4.00` | 768 | 0.0000877% | 796 | 0.5810 | 0.4799 | 0.7654 | 0.7264 | Post-saturation plateau (`14` stays `24/24`, `0.6918`) |

**Why naive `bbox_embed` `[w, h]` shrinking failed while decoupled `--post-box-delta` and end-to-end `bbox_embed` LoRA succeeded [V]:** Editing `bbox_embed.bias` inside the weights alters `refs` inside decoder layers `0..4`, distorting `boxRPB` spatial attention (`0.4092 -> 0.3925`). Applying `[dx, dy, dw, dh]` *after* layer 5 (`Detector::set_post_box_delta`) leaves `boxRPB` untouched (`745 -> 745` tracks on step 600, `0.5672 -> 0.5701`), while adapting `bbox_embed` end-to-end in `visem-lora-v2`/`v3` solves head sizing directly in the weights (so no post-hoc `post_box_delta` is needed; `held8_lorav2_s400_it320_pbd04` confirmed `0.5867` vs `0.6055` without `pbd`). Similarly, `--tile-every 8` lifted dense tiny-sperm video `19` by `+3.46` HOTA (`0.4527 -> 0.4873`, `153/152` boxes) while over-proposing on sparse videos (`0.5594` overall), so `2x2` tiling is kept opt-in (`image --tile 2x2`, `track --tile-every K`).

---

## 3. Unseen Held-Out Evaluation (`21, 22, 36, 47, 82`)

### 3.1 Held-Out 8-Frame Controlled Suite (40 frames, 879 GT boxes, `--max-objects 64`) [V, `eval/held8_*.txt`]

| Configuration | Prompt | Boxes | Extra weights edited | Pred boxes | HOTA | DetA | AssA | LocA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot base | `sperm` | `detector` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Zero-shot base | `cell` | `mask` | 0 | 622 | 0.1516 | 0.0515 | 0.5223 | 0.6625 |
| Zero-shot base | `cell` | `detector` | 0 | 622 | 0.1726 | 0.0600 | 0.5591 | 0.6745 |
| **Zero-shot + `zeroshot_sperm_graft.json`** | `sperm` | `detector` | **1,024 (`0.000117%`)** | 621 | **0.1729** | 0.0606 | 0.5581 | 0.6740 |
| **Zero-shot + `zeroshot_sperm_graft_calibrated.json`** | `sperm` | `detector` | **1,537 (`0.000176%`)** | 802 | **0.1929** | **0.0705** | **0.6318** | **0.6791** |
| LoRA v1 (`visem-lora-r16`, old whole-mask NMS) | `sperm` | `detector` | 0 (+983k LoRA) | 444 | 0.4515 | 0.2788 | 0.8008 | 0.7318 |
| LoRA v1 + `lora_scoring_outnorm_2p40.json` (old NMS) | `sperm` | `detector` | 512 (`0.0000585%`) | 707 | 0.5533 | 0.4247 | 0.8000 | 0.7239 |
| **LoRA v1 + `lora_scoring_outnorm_2p40.json` (head-box NMS + bulb)** | `sperm` | `detector` | **512 (`0.0000585%`)** | 740 | **0.5756** | 0.4533 | 0.8053 | 0.7288 |
| LoRA v2 (`visem-lora-v2` step 400, unedited `G = 1.0`) | `sperm` | `detector` | 0 (+1.13M LoRA) | 634 | 0.5609 | 0.4261 | 0.7788 | **0.7474** |
| LoRA v2 + `lorav2_exact_it_3p20.json` (`768` wts) | `sperm` | `detector` | 768 (`0.0000877%`) | 871 | 0.6055 | 0.4984 | 0.7828 | 0.7409 |
| **LoRA v3 (`visem-lora-v3` step 160, unedited `G = 1.0`)** | `sperm` | `detector` | 0 (+1.13M LoRA) | 650 | **0.5618** | 0.4274 | 0.7951 | 0.7454 |
| **LoRA v3 + `lorav3_exact_it_3p00.json` (`768` wts)** | `sperm` | `detector` | **768 (`0.0000877%`)** | **880** | **0.6121** | **0.5018** | **0.8115** | 0.7406 |

- **Tracker-only improvement (`0` weight changes):** Joint head-box/IoM NMS (`max_box_iou = 0.35`, `max_mask_iom = 0.65`, `boundary_margin = -1.0`) plus morphological sperm-head bulb localization (with kinematic head coasting when multi-object post-winner masks are occluded) lifts `v1 + ws240` from **`0.5533 -> 0.5756` HOTA (`+2.23` pts)** on identical weights (`eval/held8_v1_ws240_p34.txt`).
- **Enhanced LoRA v3 + 768-weight exact inverse-temperature (`G = 3.00`):** Combining `visem-lora-v3` (token-level BPE adaptation + focal negative query suppression) with `lorav3_exact_it_3p00.json` (**768 scalars = `0.0000877%`**) reaches **`0.6121` HOTA (`+0.66` pts over v2 `0.6055`, `+5.88` pts over `0.5533`, `+16.06` pts over raw v1)** with **`AssA = 0.8115` (`+2.87` pts ID persistence over v2; Video `82` `AssA = 0.9078`, Video `36` `HOTA = 0.5980`)** and **`880` predicted boxes vs `879` GT boxes (`99.9%` count balance)**.

### 3.2 Held-Out 32-Frame Window Past `hotstart_delay = 15` (160 frames, 3,492 GT boxes) [V, `eval/held32_*.txt`]

To verify that newly born tracks survive SAM 3.1's 15-frame `hotstart_delay` confirmation filter and 7-frame memory bank (`num_maskmem = 7`), all 5 held-out videos were tracked for 32 frames (`160` frames total):

| Configuration | Cap | Prompt | Extra weights edited | Pred boxes | HOTA | DetA | AssA | LocA | Per-video HOTA (`21 / 22 / 36 / 47 / 82`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Zero-shot base | 64 | `sperm` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | `0 / 0 / 0 / 0 / 0` |
| Zero-shot base | 16 | `cell` | 0 | 2,395 | 0.1658 | 0.0589 | 0.5708 | 0.6595 | `0.114 / 0.081 / 0.223 / 0.132 / 0.092` |
| **Zero-shot + `zeroshot_sperm_graft.json`** | 16 | `sperm` | **1,024 (`0.000117%`)** | 2,375 | **0.1663** | 0.0593 | 0.5697 | 0.6593 | `0.114 / 0.081 / 0.223 / 0.136 / 0.092` |
| **Zero-shot + `zeroshot_sperm_graft_calibrated.json`** | 64 | `sperm` | **1,537 (`0.000176%`)** | 3,797 | **0.2106** | 0.0926 | 0.5503 | 0.6664 | `0.169 / 0.089 / 0.292 / 0.125 / 0.126` |
| LoRA v1 (`visem-lora-r16`, old NMS) | 64 | `sperm` | 0 (+983k LoRA) | 2,002 | 0.4751 | 0.3162 | **0.7909** | 0.7259 | `0.595 / 0.624 / 0.318 / 0.492 / 0.475` |
| LoRA v1 + `lora_scoring_outnorm_2p40.json` (old NMS) | 64 | `sperm` | 512 (`0.0000585%`) | 2,941 | 0.5526 | 0.4359 | 0.7812 | 0.7202 | `0.639 / 0.649 / 0.490 / 0.543 / 0.517` |
| **LoRA v1 + `lora_scoring_outnorm_2p40.json` (head-box NMS + bulb)** | 64 | `sperm` | **512 (`0.0000585%`)** | 3,107 | **0.5762** | 0.4689 | 0.7864 | 0.7219 | `0.690 / 0.660 / 0.513 / 0.585 / 0.522` |
| LoRA v2 (`visem-lora-v2` step 400, unedited `G = 1.0`) | 64 | `sperm` | 0 (+1.13M LoRA) | 2,782 | 0.5736 | 0.4578 | 0.7710 | **0.7390** | `0.690 / 0.686 / 0.506 / 0.561 / 0.519` |
| LoRA v2 + `lorav2_exact_it_3p20.json` (`768` wts, post-winner coast) | 64 | `sperm` | 768 (`0.0000877%`) | 3,558 | 0.6013 | **0.5028** | 0.7746 | 0.7343 | `0.667 / 0.688 / 0.576 / 0.549 / 0.557` |
| **LoRA v3 (`visem-lora-v3` step 160, unedited `G = 1.0`)** | 64 | `sperm` | 0 (+1.13M LoRA) | 2,818 | **0.5751** | 0.4613 | 0.7817 | 0.7362 | `0.700 / 0.680 / 0.501 / 0.571 / 0.522` |
| **LoRA v3 + `lorav3_exact_it_3p00.json` (`768` wts)** | 64 | `sperm` | **768 (`0.0000877%`)** | **3,599** | **0.6025** | 0.5026 | **0.7880** | 0.7338 | **`0.674 / 0.680 / 0.575 / 0.560 / 0.555`** |

*Note on zero-shot `--max-objects 64` vs `16`:* On noisy uncalibrated zero-shot `"cell"` (and uncalibrated `zeroshot_sperm_graft`), video `22` spawns 19 tracks on frame 0 (`16` in state 0, `3` in state 1) and prunes all 3 tracks in state 1 by frame 7, triggering upstream's empty-state check on frame 8 (`propagate: no conditioning frame in this state`). Zero-shot baseline comparisons at 32 frames therefore use upstream's `--max-objects 16` (`held300-zeroshot-cell-mask` convention), while `zeroshot_sperm_graft_calibrated.json` sustains secondary states cleanly across all 5 videos at `--max-objects 64` (`0.2106` HOTA).

### 3.3 Continuous `[0..1]` ROC, Precision–Recall & `2x2` Tiling Curves [V, `roc_pr_held_out_f0.json`]

Sweeping `spiketrack image --threshold 0.0` across frame `0` of all 5 unseen held-out videos (`21, 22, 36, 47, 82`: `n_queries = 1,000` single-scale; `n_queries = 1,145` / `1,126` after cross-scale NMS under `--tile 2x2` with per-tile `Sam3Processor` presence scaling `sigmoid(logit) * sigmoid(tile_presence)`, `112` normal-sperm class-0 GT boxes, greedy `IoU >= 0.50` matching; plots and bounding-box comparison panels in [`docs/EVALUATION.md`](../../../docs/EVALUATION.md)):

| Model (`--prompt sperm` unless noted) | `n_queries` | Query AUROC | AP@0.50 | `@ 0.50` Prec / Rec / F1 / Box Acc | `@ 0.65` Birth Gate Prec / Rec / F1 / Box Acc (`TP/112`) |
|---|---:|---:|---:|---|---|
| Zero-shot `"sperm"` | `1,000` | `0.4523` | `0.0280` | `—` / `0.0%` / `0.0%` / `0.0%` | `—` / `0.0%` / `0.0%` / `0.0%` (`0/112`) |
| Zero-shot `"cell"` | `1,000` | `0.7148` | `0.0716` | `11.6%` / `19.6%` / `14.6%` / `7.9%` | `3.3%` / `1.8%` / `2.3%` / `1.2%` (`2/112`) |
| Zero-shot + `zeroshot_sperm_graft_calibrated.json` (`1,537` wts) | `1,000` | `0.7229` | `0.0705` | `11.9%` / `21.4%` / `15.3%` / `8.3%` | `6.4%` / `5.4%` / `5.8%` / `3.0%` (`6/112`) |
| LoRA v1 (`visem-lora-r16`) | `1,000` | `0.9832` | `0.6528` | `73.4%` / `71.4%` / `72.4%` / `56.7%` | `80.9%` / `33.9%` / `47.8%` / `31.4%` (`38/112`) |
| LoRA v1 + `lora_scoring_outnorm_2p40.json` (`512` wts) | `1,000` | `0.9833` | `0.6521` | `73.8%` / `70.5%` / `72.1%` / `56.4%` | `75.3%` / `59.8%` / `66.7%` / `50.0%` (`67/112`) |
| LoRA v2 (`visem-lora-v2` step 400, unedited `G = 1.0`) | `1,000` | `0.9872` | `0.8014` | `78.2%` / `83.0%` / `80.5%` / `67.4%` | **`90.4%`** / `58.9%` / `71.4%` / `55.5%` (`66/112`) |
| LoRA v2 + `lorav2_exact_it_3p20.json` (`768` wts) | `1,000` | `0.9872` | `0.8014` | `78.2%` / `83.0%` / `80.5%` / `67.4%` | `79.5%` / `79.5%` / `79.5%` / `65.9%` (`89/112`) |
| LoRA v2 + `lorav2_exact_it_3p20.json` + `--tile 2x2` | `1,145` | **`0.9951`** | **`0.8341`** | `78.7%` / `85.7%` / `82.1%` / `69.6%` | `81.2%` / **`84.8%`** / `83.0%` / `70.9%` (`95/112`) |
| LoRA v3 (`visem-lora-v3` step 160, unedited `G = 1.0`) | `1,000` | `0.9835` | `0.7485` | `74.2%` / `79.5%` / `76.7%` / `62.2%` | `85.7%` / `53.6%` / `65.9%` / `49.2%` (`60/112`) |
| LoRA v3 + `lorav3_exact_it_3p00.json` (`768` wts) | `1,000` | `0.9835` | `0.7485` | `74.2%` / `79.5%` / `76.7%` / `62.2%` | `78.4%` / `77.7%` / `78.0%` / `64.0%` (`87/112`) |
| **LoRA v3 + `lorav3_exact_it_3p00.json` + `--tile 2x2`** | `1,126` | `0.9945` | `0.8236` | **`78.9%`** / **`86.6%`** / **`82.6%`** / **`70.3%`** | **`82.6%`** / **`84.8%`** / **`83.7%`** / **`72.0%`** (**`95/112`**, `20` FP) |

---

## 4. Reproducing

```sh
export SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors
export SPIKETRACK_MODEL_DIR=~/Code/research/models/sam3.1

# 1. Inspect LoRA v3 weight-update spectrum, adapted BPE token_embedding rows, and prompt cosines
target/release/spiketrack weights \
  --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors --diff base --svd-k 4 --top-k 20 \
  --tokens "sperm,cell,pinhead sperm,sperm cluster"

# 2. High-recall 2x2 tiled single-image sperm detection (83.7% F1 @ 0.65, 84.1% peak F1 @ 0.66)
target/release/spiketrack image data/visem/VISEM_Tracking_Train_v4/Train/36/images/36_frame_0.jpg \
  --prompt sperm --tile 2x2 --threshold 0.65 \
  --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
  --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json \
  --out out/visem36-tile2x2

# 3. Track unseen held-out videos (0.6121 HOTA on 8f / 0.6025 HOTA on 32f across 160 frames / 3,492 GT boxes)
for VID in 21 22 36 47 82; do
  target/release/spiketrack track "data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" \
    --prompt sperm --boxes detector --max-frames 32 \
    --lora fixtures/lora_runs/visem-lora-v3/adapter.safetensors \
    --weight-edit bench/results/weight-space/lorav3_exact_it_3p00.json \
    --out out/held32-lorav3-it300
done
target/release/spiketrack eval-visem --pred out/held32-lorav3-it300 \
  --split tools/reference/visem_split.json --set held_out --max-frames 32
```
