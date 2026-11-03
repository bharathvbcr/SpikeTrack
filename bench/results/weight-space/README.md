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

---

## 2. Train-Split Selection & Falsified Hypotheses (`train-5`: `14, 30, 19, 54, 13`, 40 frames, 838 GT boxes)

All candidate edits were evaluated on the stratified 5-video `"train"` subset (`--max-frames 8 --boxes detector`, `--max-objects 64`) before freezing:

| Track | Condition (`--prompt sperm` unless noted) | Values edited | Fraction of 875.5M | Pred boxes | HOTA | DetA | AssA | LocA | Gate decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot | Base `--prompt sperm` | 0 | 0% | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | Baseline |
| Zero-shot | Base `--prompt cell` | 0 | 0% | 598 | 0.1557 | 0.0548 | 0.5030 | 0.6310 | Prompt ref |
| Zero-shot | `graft_row` `5533 -> 35781` (`preserve_norm`) | **1,024** | **0.000117%** | 598 | **0.1557** | 0.0548 | 0.5030 | 0.6310 | **Frozen (`zeroshot_sperm_graft.json`)** |
| Zero-shot | `graft_row` + `bbox_embed.bias[2..4] -= 0.05` | 1,026 | 0.000117% | 582 | 0.1423 | 0.0479 | 0.4963 | 0.6307 | **Rejected (falsified)** |
| Zero-shot | `graft_row` + `bbox_embed.bias[2..4] -= 0.10` | 1,026 | 0.000117% | 514 | 0.1225 | 0.0373 | 0.4791 | 0.6274 | **Rejected (falsified)** |
| Zero-shot | `graft_row` + `bbox_embed.bias[2..4] -= 0.20` | 1,026 | 0.000117% | 433 | 0.0943 | 0.0266 | 0.3929 | 0.6330 | **Rejected (falsified)** |
| Zero-shot | `graft_row` + `presence_bias +1.5` + `out_norm * 1.25` | **1,537** | **0.000176%** | 882 | **0.1661** | 0.0640 | 0.5251 | 0.6252 | **Frozen (`zeroshot_sperm_graft_calibrated.json`)** |
| Post-LoRA | Base `visem-lora-r16` (`G = 1.00`) | 0 (+LoRA) | 0% | 313 | 0.4092 | 0.2347 | 0.7668 | 0.7389 | Baseline |
| Post-LoRA | `bbox_embed.bias[2..4] -= 0.03` | 2 | 0.0000002% | 302 | 0.3925 | 0.2196 | 0.7347 | 0.7302 | **Rejected (perturbs `boxRPB`)** |
| Post-LoRA | `out_norm * 1.35` | 512 | 0.0000585% | 499 | 0.4980 | 0.3607 | 0.7514 | 0.7276 | Sweep |
| Post-LoRA | `out_norm * 1.60` | 512 | 0.0000585% | 562 | 0.5059 | 0.3727 | 0.7531 | 0.7238 | Sweep |
| Post-LoRA | `prompt_proj * 1.50` (65,792 values) | 65,792 | 0.007514% | 572 | 0.5056 | 0.3728 | 0.7529 | 0.7225 | Redundant (`out_norm` is 128x smaller) |
| Post-LoRA | `out_norm * 1.85` | 512 | 0.0000585% | 601 | 0.5152 | 0.3831 | 0.7584 | 0.7248 | Sweep |
| Post-LoRA | `out_norm * 2.10` | 512 | 0.0000585% | 626 | 0.5267 | 0.4001 | 0.7585 | 0.7258 | Sweep |
| Post-LoRA | **`out_norm * 2.40` (unimodal peak)** | **512** | **0.0000585%** | **657** | **0.5395** | **0.4196** | **0.7608** | **0.7256** | **Frozen (`lora_scoring_outnorm_2p40.json`)** |
| Post-LoRA | `out_norm * 3.00` (past knee) | 512 | 0.0000585% | 681 | 0.5315 | 0.4084 | 0.7628 | 0.7203 | Turns over |
| Post-LoRA | `out_norm * 4.00` (past knee) | 512 | 0.0000585% | 706 | 0.5316 | 0.4087 | 0.7630 | 0.7187 | Past knee |

**Why naive `bbox_embed` `[w, h]` shrinking failed [V]:** Zero-shot `"cell"` boxes enclose the entire 80–95 px spermatozoon (head + flagellum), placing `(cx, cy)` halfway down the tail rather than on the 20 px head. Subtracting `[w, h]` bias contracts the box around the tail midpoint and narrows `boxRPB` spatial attention in decoder layers `1..5`, reducing IoU with the GT head (`0.1557 -> 0.0943`). Similarly, post-LoRA `bbox_embed` bias edits perturb `boxRPB` on dense scenes (`0.4092 -> 0.3925`).

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
| LoRA step-600 (`visem-lora-r16`) | `sperm` | `detector` | 0 (+LoRA) | 444 | 0.4515 | 0.2788 | 0.8008 | 0.7318 |
| **LoRA step-600 + `lora_scoring_outnorm_2p40.json`** | `sperm` | `detector` | **512 (`0.0000585%`)** | **707** | **0.5533** | **0.4247** | **0.8000** | **0.7239** |

- **Zero-data weight-space programming (`0` gradients):** Editing 1,024 weights (`0.000117%`) lifts `--prompt sperm` from `0.0000` to `0.1729` HOTA; adding 513 birth-gate calibration weights (`1,537` total = `0.000176%`) reaches **`0.1929` HOTA** (`+2.03` points / `+11.8%` relative over unedited `--prompt cell`).
- **Post-LoRA frozen-head calibration (`0` extra gradients):** Editing **512 weights (`0.0000585%`)** improves 600-step LoRA from **`0.4515` to `0.5533` HOTA (`+10.18` points / `+22.5%` relative)** and **`DetA` from `0.2788` to `0.4247` (`+52.3%` relative)** with `AssA` unchanged (`0.8008 -> 0.8000`), improving **5 of 5 unseen videos** (`21`: `0.5793 -> 0.6355`, `22`: `0.5932 -> 0.6238`, `36`: `0.2736 -> 0.4895`, `47`: `0.4966 -> 0.5980`, `82`: `0.4555 -> 0.5193`).

### 3.2 Held-Out 32-Frame Window Past `hotstart_delay = 15` (160 frames, 3,492 GT boxes) [V, `eval/held32_*.txt`]

To verify that newly born tracks survive SAM 3.1's 15-frame `hotstart_delay` confirmation filter and 7-frame memory bank (`num_maskmem = 7`), all 5 held-out videos were tracked for 32 frames (`160` frames total):

| Configuration | Cap | Prompt | Extra weights edited | Pred boxes | HOTA | DetA | AssA | LocA | Per-video HOTA (`21 / 22 / 36 / 47 / 82`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Zero-shot base | 64 | `sperm` | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | `0 / 0 / 0 / 0 / 0` |
| Zero-shot base | 16 | `cell` | 0 | 2,395 | 0.1658 | 0.0589 | 0.5708 | 0.6595 | `0.114 / 0.081 / 0.223 / 0.132 / 0.092` |
| **Zero-shot + `zeroshot_sperm_graft.json`** | 16 | `sperm` | **1,024 (`0.000117%`)** | 2,375 | **0.1663** | 0.0593 | 0.5697 | 0.6593 | `0.114 / 0.081 / 0.223 / 0.136 / 0.092` |
| Zero-shot + `zeroshot_sperm_graft_calibrated.json` | 16 | `sperm` | 1,537 (`0.000176%`) | 2,456 | 0.1580 | 0.0562 | 0.5307 | 0.6584 | `0.118 / 0.081 / 0.211 / 0.107 / 0.092` |
| **Zero-shot + `zeroshot_sperm_graft_calibrated.json`** | 64 | `sperm` | **1,537 (`0.000176%`)** | 3,797 | **0.2106** | **0.0926** | 0.5503 | **0.6664** | `0.169 / 0.089 / 0.292 / 0.125 / 0.126` |
| LoRA step-600 (`visem-lora-r16`) | 64 | `sperm` | 0 (+LoRA) | 2,002 | 0.4751 | 0.3162 | 0.7909 | 0.7259 | `0.595 / 0.624 / 0.318 / 0.492 / 0.475` |
| **LoRA step-600 + `lora_scoring_outnorm_2p40.json`** | 64 | `sperm` | **512 (`0.0000585%`)** | **2,941** | **0.5526** | **0.4359** | **0.7812** | **0.7202** | **`0.639 / 0.649 / 0.490 / 0.543 / 0.517`** |

*Note on zero-shot `--max-objects 64` vs `16`:* On noisy uncalibrated zero-shot `"cell"` (and uncalibrated `zeroshot_sperm_graft`), video `22` spawns 19 tracks on frame 0 (`16` in state 0, `3` in state 1) and prunes all 3 tracks in state 1 by frame 7, triggering upstream's empty-state check on frame 8 (`propagate: no conditioning frame in this state`). Zero-shot baseline comparisons at 32 frames therefore use upstream's `--max-objects 16` (`held300-zeroshot-cell-mask` convention), while `zeroshot_sperm_graft_calibrated.json` sustains secondary states cleanly across all 5 videos at `--max-objects 64` (`0.2106` HOTA).

### 3.3 Continuous `[0..1]` ROC, Precision–Recall & Birth-Gate Curves [V, `roc_pr_held_out_f0.json`]

Sweeping `spiketrack image --threshold 0.0` across frame `0` of all 5 unseen held-out videos (`21, 22, 36, 47, 82`: `1,000` DETR queries/model, `112` normal-sperm class-0 GT boxes, greedy `IoU >= 0.50` matching; plots and bounding-box comparison panels in [`docs/EVALUATION.md`](../../../docs/EVALUATION.md)):

| Model (`--prompt sperm` unless noted) | Query AUROC | AP@0.50 | `@ 0.50` Prec / Rec / F1 / Box Acc | `@ 0.65` Birth Gate Prec / Rec / F1 / Box Acc (`TP/112`) |
|---|---:|---:|---|---|
| Zero-shot `"sperm"` | `0.4523` | `0.0280` | `—` / `0.0%` / `0.0%` / `0.0%` | `—` / `0.0%` / `0.0%` / `0.0%` (`0/112`) |
| Zero-shot `"cell"` | `0.7148` | `0.0716` | `11.6%` / `19.6%` / `14.6%` / `7.9%` | `3.3%` / `1.8%` / `2.3%` / `1.2%` (`2/112`) |
| Zero-shot + `zeroshot_sperm_graft_calibrated.json` (`1,537` wts) | `0.7229` | `0.0705` | `11.9%` / `21.4%` / `15.3%` / `8.3%` | `6.4%` / `5.4%` / `5.8%` / `3.0%` (`6/112`) |
| LoRA step-600 (`visem-lora-r16`) | **`0.9832`** | **`0.6528`** | `73.4%` / **`71.4%`** / **`72.4%`** / **`56.7%`** | **`80.9%`** / `33.9%` / `47.8%` / `31.4%` (`38/112`) |
| **LoRA step-600 + `lora_scoring_outnorm_2p40.json` (`512` wts)** | **`0.9833`** | **`0.6521`** | **`73.8%`** / `70.5%` / `72.1%` / `56.4%` | `75.3%` / **`59.8%`** / **`66.7%`** / **`50.0%`** (**`67/112`**) |

---

## 4. Reproducing

```sh
export SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors
export SPIKETRACK_MODEL_DIR=~/Code/research/models/sam3.1

# 1. Inspect token-embedding neighborhood, post-24-layer Text+resizer cosines, and bbox_embed SVD
target/release/spiketrack weights \
  --tokens "sperm,cell,cells,spot,dot,head,pinhead sperm,sperm cluster" \
  --tensor detector.transformer.decoder.bbox_embed.layers.2.weight --svd-k 4 --top-k 8

# 2. Inspect LoRA weight-update spectrum against base checkpoint
target/release/spiketrack weights \
  --lora fixtures/lora_runs/visem-lora-r16/adapter.safetensors --diff base --svd-k 4 --top-k 15

# 3. Track unseen held-out videos with frozen 512-parameter weight-space calibration
for VID in 21 22 36 47 82; do
  target/release/spiketrack track "data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" \
    --prompt sperm --boxes detector --max-frames 32 \
    --lora fixtures/lora_runs/visem-lora-r16/adapter.safetensors \
    --weight-edit bench/results/weight-space/lora_scoring_outnorm_2p40.json \
    --out out/held32-lora-ws240
done
target/release/spiketrack eval-visem --pred out/held32-lora-ws240 \
  --split tools/reference/visem_split.json --set held_out --max-frames 32
```
