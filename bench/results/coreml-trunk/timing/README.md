# E10 frame time: Ojas trunk vs CoreML trunk (2026-10-29)

49 frames of VISEM 22, prompt "sperm", r16 LoRA, `--boxes detector`, `--max-objects 64`, Metal. This
is the setup of `../../ojas-ea3cf39`. The binary is a frozen build of b11bb69. Three arms are chosen
by `--trunk-precision`, interleaved over three rounds, and each arm takes each position once
(`../job-timing.sh`). The machine was shared: load 9-17 and GPU utilization 80-92% at every arm's
start (`timing.txt`). The ratios are for that contended machine; idle-machine ratios were not measured.

| arm | s/frame, rounds 1 / 2 / 3 | mean | vs f32 | model load s |
|---|---|---|---|---|
| f32 (the Ojas trunk) | 3.3528 / 3.3649 / 3.4746 | 3.397 | 1.00x | 1.2 / 0.6 / 0.7 |
| coreml-fp32 | 2.6496 / 2.7445 / 2.7132 | 2.702 | **0.80x** | 12.7 / 9.0 / 7.7 |
| coreml-fp16 | 1.8553 / 1.9601 / 1.8140 | 1.877 | **0.55x** | 8.9 / 4.1 / 3.9 |

- Every CoreML round is faster than every f32 round, in every position [V].
- Each arm's predictions are byte-identical across its three rounds [V, `timing.txt`].
- coreml-fp32 against f32, all 49 frames: the id sets are identical, with max |box| 3.0e-6 and max
  |score| 1.6e-6. The files differ in their last digits only.
- coreml-fp16 against f32: the id sets are identical, with max |box| 3.8e-3 and max |score| 2.9e-4.
- Cold start: the CoreML load adds 3-12 s against the Ojas trunk's ~1 s. Once compiled, the package is
  reused; its first compile took 15 s for fp32 (`../gate1/parity-fp32.log`). At ~0.7 s saved per frame,
  coreml-fp32 breaks even after ~10-17 frames. On shorter runs the Ojas trunk is faster end to end.
- Pre-stated (docs/SPEED_PLAN.md E10): fp32 frame <= 0.90x is met, against a refute line of > 0.95x.
  fp16 <= 0.70x is met on time; its HOTA gate is in `../README.md`.

Files: `timing.txt`, `versions.txt`, `track-<arm>-<round>.log` (per-frame telemetry) and
`track-<arm>-<round>/22.{json,run.json}`. Masks are not committed.
