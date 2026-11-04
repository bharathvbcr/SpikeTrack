# E30 rules, written before any run (2026-11-04)

Two levers were left unresolved earlier today. Both are re-run here under rules written down first.

## Lever A: fp16 CoreML necks on today's default (`--necks coreml-fp16`)

- **History.** E27 (`../e27-pipe-necks/README.md`) measured fast16-n16 against fast16 at 0.845,
  0.879 and 0.958 wall. That missed its N2 gate (< 0.95 in all 3 rounds) by one round.
- **Accuracy gate.** E27's N1 asked fp32 necks for min box IoU >= 0.999 against Ojas's necks. No
  fp16 placement was ever held to that: the fp16 trunk was judged by the task metric (G2, G3 of
  `../e25-coreml-frame/RULES.md`), and fp16 necks are held to the same gates here.
- **Arm.** `t16m16n16` is the default (fp16 trunk pipelined + fp16 memory attention) plus
  `--necks coreml-fp16`.

Gates, all required for adoption:
- **A-G2:** held-out HOTA over the first 32 frames of the 5 held-out videos >= 0.6005, with no video
  more than 0.01 below t16m16's (`../track/held32-coreml/`).
- **A-G3:** held-out HOTA over 1,500 frames >= 0.567, with no video more than 0.01 below t16m16's
  0.5694 per-video values (`../track/held300-coreml/`).
- **A-speed:** a fresh speed run, on a machine whose 1-minute load stays under 8 throughout (the job
  logs the load per arm). t16m16n16's wall is below 0.95x t16m16's in all 3 rounds, and the median
  of the 3 ratios is below 0.92. That is stricter than N2, because E27 already took one look at this
  lever. If the load passes 8 during an arm, the speed result is "not read", not a pass or a fail.

If A passes all three, `--necks` defaults to `coreml-fp16` beside the default fp16 trunk, with the
same guards as `Placement::default_mem_attn`. If it passes the accuracy gates but not the speed gate,
nothing changes.

## Lever B: unwaited commits every N dispatches (`--overlap-dispatches`), re-run quietly

E29 (`../e29-overlap/README.md`) was unreadable: the load rose from 7 to 16 and its default arm
spread 22%. The arms are o256 (the default) and o32. Of E29's arms, 32 was the least slow in its
worst round. The rules are E29's O1 and O2 (byte-identical predictions, wall < 0.95x in all 3
rounds), under the same load condition as A-speed. If o32 passes, `open_metal` sets 32.

## Common setup for the speed runs

One binary, run from `target/release` (not copied into the results folder, so no freshly written
unsigned executable for macOS to assess mid-run). The shipped configuration as in E27: LoRA v3 +
`lorav3_exact_it_3p00.json`, prompt sperm, `--boxes detector`, cap 64. Video 36, first 32 frames,
3 rounds with the order rotated. Arms: t16m16 (the default, no flags), t16m16n16, o32. The noise
floor is t16m16's spread across rounds.

## What an adopted lever A changes in what was already shipped (decided before the result)

fp16 necks change predictions by design, so adopting lever A means:
- re-rendering the demo clips, the README figure and the demo video (`../demo/fp16-job.sh`,
  `../demo/video/make.sh`);
- re-wording the video's tracking and speed segments.

**Wording.** If t16m16n16's 1,500-frame HOTA is at least 0.5684 (within 0.001 of t16m16's 0.5694),
the narration may still say accuracy is "unchanged". Below that, it states the number.
