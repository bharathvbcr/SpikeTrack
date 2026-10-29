# E7: held-out HOTA at `--input-size 672`

Measured 2026-10-28, Metal. Binary from branch `speed/e7-geometry` at 45e5d63, built against ojas 8b88328
and tessl c9adc87 (`versions.txt`). The setup is the same as `../held300-lora600-sperm-detector` (cap 16)
and `../held300-lora600-sperm-detector-max64` (cap 64): 300 frames per held-out video, prompt "sperm",
`--boxes detector`, LoRA `fixtures/lora_runs/visem-lora-r16`. The only change is `--input-size 672`
(48x48 trunk tokens instead of 72x72). This directory holds cap 64; `../held300-lora600-sperm-detector-672-max16`
holds cap 16.

Pre-stated acceptance (docs/SPEED_PLAN.md E7): held-out HOTA >= 0.47 at both caps.
**Result: refuted at both caps.**

| cap | 1008: HOTA / DetA / AssA / LocA | 672: HOTA / DetA / AssA / LocA |
|---|---|---|
| 64 | 0.4868 / 0.3632 / 0.6948 / 0.7219 | 0.4645 / 0.3581 / 0.6206 / 0.7127 |
| 16 | 0.4695 / 0.3198 / 0.7250 / 0.7294 | 0.4479 / 0.3203 / 0.6428 / 0.7166 |

Per video, cap 64, HOTA at 1008 vs 672:

| video | 1008 | 672 |
|---|---|---|
| 21 | 0.6204 | 0.5604 |
| 22 | 0.6285 | 0.6296 |
| 36 | 0.3518 | 0.3499 |
| 47 | 0.5149 | 0.3903 |
| 82 | 0.4860 | 0.4879 |

- DetA is nearly unchanged; the loss is in association, with AssA 0.69 → 0.62 at cap 64. Most of it comes
  from videos 47 and 21. Detection holds at the lower resolution while identities hold less well. That reading
  comes from the DetA/AssA split and was not checked per track.
- The LoRA was trained at 1008 and has not been retrained or checked at 672. Whether a 672-trained adapter would
  close the gap is untested.
- Mean s/frame at 672 was 1.60-2.00 (cap 64) and 1.54-1.67 (cap 16), against 3.70-5.03 at 1008. The 1008 runs
  were made on a different day under different load and were not paired, so the ratio (~2.5x) is indicative
  only. The paired 8-frame smoke on the branch measured 1008 at 3.10 / 3.02 s/frame and 672 at 1.36 / 1.34
  (2.27x).
- One run per cap. The pipeline is deterministic (session parity reproduces id sets exactly), so a rerun would
  not change these numbers. Five videos are a small sample of the held-out set, but this is the set the gate
  was stated on.

Scoring uses `eval-visem --max-frames 300`, the window the runs covered. A first scoring pass without
`--max-frames` scored all 1470 labelled frames against 300 frames of predictions and gave HOTA ~0.13. That
pass was a scoring mistake, so it was discarded.

Files: `run/*.json` (tracks), `run/*.run.json` (settings and timing, with `input_size`), `logs/*.log` (per-frame
telemetry), `eval.{txt,json}`, `versions.txt`. Masks are not committed.
