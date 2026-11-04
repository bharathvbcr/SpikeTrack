# E27: the CoreML necks on the trunk pipeline's worker (2026-11-04)

Setup:
- `job.sh`, with the gates in `RULES.md` stated before any run;
- VISEM 36 (held-out, dense), first 32 frames, shipped configuration;
- 5 arms, 3 rounds, order rotated per round; one binary built from the main tree (`versions.txt`).

Since this build, the trunk pipeline is on by default whenever the trunk is a CoreML program, so every
arm here is pipelined. Checks before the run: `checks.log` (clippy -D warnings clean; 23 test binaries
ok, 0 failed). A 6-frame smoke run (`smoke/36.run.json`) shows both neck programs as "pipelined".

"wall" is `(total - load) / frames` from run.json. The table is `summary.md` (`tools/analysis/e25_frame.py`,
steady frames >= 8).

| arm | wall ms, rounds 1-3 | wall vs default | wall vs the same precision without CoreML necks |
|---|---|---|---|
| default (coreml-fp32 trunk, pipelined) | 2961, 2945, 2791 | 1 | |
| m16 (+ `--mem-attn coreml-fp16`) | 2440, 2246, 2173 | 0.824, 0.763, 0.779 | |
| m16-n32 (+ `--necks coreml-fp32`) | 2397, 2318, 2216 | 0.809, 0.787, 0.794 | **0.982, 1.032, 1.020** |
| fast16 (fp16 trunk + fp16 memory attention) | 1776, 1676, 1666 | 0.600, 0.569, 0.597 | |
| fast16-n16 (+ `--necks coreml-fp16`) | 1500, 1474, 1596 | 0.507, 0.500, 0.572 | **0.845, 0.879, 0.958** |

Tracks of each necks arm against its own precision's arm, the same round. This was computed on a copy
of the run folders, so `summary.md` is unchanged:

| arm vs reference | equal id sets | mean box IoU | min box IoU |
|---|---|---|---|
| m16-n32 vs m16 | 32/32, every round | 1.0000 | 0.9951 |
| fast16-n16 vs fast16 | 32/32, every round | 0.9999 | 0.9907 |

## Verdicts

**N1 fails [V].**
- The gate was 32/32 equal id sets with min box IoU >= 0.999 for m16-n32 against m16.
- The id sets all match, but one box falls to IoU 0.9951.
- [I] fp32 neck programs (G1 rel L2 ~1e-6) are not bit-identical to Ojas's necks, and a 1e-6 change
  is enough to move one box at the tracker's thresholds.
- E25's necks32 reached min IoU 1.0000 against an unpipelined default. The two are not the same
  comparison.

**N2 fails for both precisions [V].**
- fp32 necks are no faster in the frame: 0.98-1.03x. Running them on the worker removed E25's 6%
  penalty but bought nothing.
- fp16 necks help in 2 of 3 rounds (0.85, 0.88), but round 3 is 0.958, above the pre-stated 0.95 in
  all 3 rounds.

**Outcome, per RULES.md:**
- The necks stay on Ojas, and `--necks coreml-*` remains opt-in.
- G3 (`../track/held300-coreml/`) runs `default` and `fast16`, not `fast16-n16`.
- The fp16 necks are a candidate for a new experiment with its own pre-stated gates (more rounds and
  an accuracy gate). That experiment would cover the fp16 necks only.

**Kept from this change [V]:**
- The trunk pipeline is now on by default with a CoreML trunk, byte-identical in E26.
- The neck programs can run on the trunk's worker when asked for.
