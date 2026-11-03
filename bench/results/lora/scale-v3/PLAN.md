# Scaled v3 task vector (pre-stated 2026-11-03, before any run)

## Why

The shipped adapter v3 (`fixtures/lora_runs/visem-lora-v3/adapter.safetensors`) was trained in two
stages: v2 for 400 steps, then 160 more steps for v3. Its full update from the base checkpoint, `delta_v3`, has
never been scaled. Two outcomes are plausible:
- **Scaling down** (`lambda < 1`, WiSE-FT toward the base) often helps under distribution
  shift, and the held-out videos are a shift: other donors, other labelling density.
- **Scaling up** (`lambda > 1`, task-vector extrapolation) helps if the run stopped
  short of its direction.

The v3/v4 soup (`../soup-v3v4/RESULT.md`) moved along another axis, v3 toward v4, and failed.
The prior is again small: typically a few thousandths.

## The edit

`tools/reference/lora_soup.py v3 zero lambda` writes v3 with every module's `alpha` multiplied
by `lambda`. The factors are unchanged, and the loader's scale `alpha / r` is exact, so the
merged update is exactly `lambda * delta_v3`. This covers all 135 modules, the four
`token_embedding` rows included. The exact inverse-temperature edit (G 3.0,
`../../weight-space/lorav3_exact_it_3p00.json`) is applied after the merge, as it is when
shipped.

## Pre-stated rules

The protocol is the same as `../soup-v3v4/PLAN.md`: prompt `sperm`, `--boxes detector`, cap 64,
Metal, first 8 frames of train-5 for selection.

1. **Grid.** `lambda` in {0.8, 0.9, 1.1, 1.2}. A winner at 0.8 or 1.2 extends the grid once, to
   0.7 or 1.35.
2. **Selection.** Pick the highest train-5 "all" row, but only among values with at least 4 of 5
   videos at or above v3's per-video train-5 score:
   14 0.6918, 30 0.6973, 19 0.5062, 54 0.4660, 13 0.6256.
   It must also beat v3's 0.5802 by at least 0.005. If no value qualifies, stop: negative result.
3. **Held-out gate.** One 32-frame held-out run. Adopt only if HOTA ≥ 0.6125 and at least 4 of
   5 videos are above v3's per-video scores:
   21 0.6744, 22 0.6799, 36 0.5752, 47 0.5596, 82 0.5552.
4. **The README claim.** Only after rule 3 passes: the 1,500-frame run must beat 0.5700 with
   at least 4 of 5 videos above v3:
   21 0.669, 22 0.669, 36 0.533, 47 0.484, 82 0.529.

**Stop time:** 2026-11-04 10:00 CT. v3 stays shipped unless rule 4 passes by then.
