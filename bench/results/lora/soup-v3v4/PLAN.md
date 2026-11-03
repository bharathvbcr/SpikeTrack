# v3/v4 adapter soup (pre-stated 2026-11-03, before any run)

## Why

LoRA v4 was warm-started from v3 and trained 400 more steps on 2,040 frames. On the 32-frame
held-out gate the two tie: v3 + edit scored 0.6025 and v4 s300 + G 3.6 scored 0.5995
(`../visem-lora-v4/RESULT.md`). Two fine-tunes from one basin that tie are the textbook case for
weight interpolation (model soups, WiSE-FT).

This is the one weight edit not yet tried:
- the 768-scalar exact inverse temperature is shipped;
- the box-size logit shift (the same as editing `bbox_embed.layers.2.bias`) failed held-out.

The expected prior is small: same-basin soups typically move HOTA by a few thousandths to about
0.01.

## The edit

`tools/reference/lora_soup.py v3 v4 c`, per module:
- `delta = c * s3 * B3 A3 + (1 - c) * s4 * B4 A4`, formed in f64;
- re-factored by thin SVD at rank `min(r3 + r4, rows, in)`, which holds it exactly;
- written as `spiketrack-lora-v1` with `alpha = rank`.

The script refuses if any module's re-factored delta is off by more than 1e-5 relative.
- v3: `fixtures/lora_runs/visem-lora-v3/adapter.safetensors` (step 160).
- v4: `fixtures/lora_runs/visem-lora-v4/adapter_step300.safetensors`.
- Both have 135 modules with identical param, rows, rank and alpha.

## Protocol (same as every selection since v3)

Prompt `sperm`, `--boxes detector`, cap 64, Metal. Selection uses the first 8 frames of train-5
(14 30 19 54 13). Held-out uses the first 32 frames of 21 22 36 47 82.

## Pre-stated rules

0. **Parity.** The c = 1.0 soup (v3 re-factored) with G 3.0 must give train-5 HOTA within 0.002
   of v3's 0.5802. Otherwise the soup file is wrong and nothing below runs.
1. **Grid.** c in {0.75, 0.5, 0.25}, at v3's edit G 3.0 (`lorav3_exact_it_3p00.json`).
2. **Selection.** Pick the c with the highest train-5 "all" row, but only among c values with
   at least 4 of 5 videos at or above v3's per-video train-5 score:
   14 0.6918, 30 0.6973, 19 0.5062, 54 0.4660, 13 0.6256.
   - If no c qualifies, stop: negative result.
   - This clause is the one the box-size shift would have failed.
3. **Temperature.** At the selected c only, also try G in {3.3, 3.6}, under the same rule. A
   winner at 3.6 extends to 4.2 once.
4. **Held-out gate.** One 32-frame run. Adopt only if HOTA ≥ 0.6125 (v3 + 0.010) and at least
   4 of 5 videos are above v3's held-out per-video scores:
   21 0.6744, 22 0.6799, 36 0.5752, 47 0.5596, 82 0.5552.
5. **The README claim.** Only if rule 4 passes: a 1,500-frame run, which must beat 0.5700 with
   at least 4 of 5 videos above v3's 1,500-frame per-video scores:
   21 0.669, 22 0.669, 36 0.533, 47 0.484, 82 0.529.

Known bias: v4 trained on 128 frames per training video against v3's 64, so train-5 selection
leans toward small c (more v4). Rule 4 is what catches that.

**Stop time:** 2026-11-04 10:00 CT. Anything not through rule 5 by then is reported as is, and v3
stays shipped.
