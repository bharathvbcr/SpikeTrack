# E31: the detector's geometry + fusion encoders as one fp16 Core ML program (2026-11-04)

The rules are in `RULES.md`, written before any run. The arm `e16` is today's default plus
`--det-encoder coreml-fp16`, using the package exported for LoRA v3
(`sam3.1_detenc_fp16_db6a8308ccdb.mlpackage`). Both arms run on one binary from the build copy
(`SpikeTrack-pipe`).

**Order deviation from RULES.md.** The rules put E-speed first. The machine's load was 11-14 when the
gates started (system `mediaanalysisd` and another session's test run). So E-G2 ran first: accuracy
does not depend on load. E-G3 and E-speed run after it, in that order (`accuracy.sh 300`, `job.sh`).

| gate | result | verdict |
|---|---|---|
| E-G2, 32 frames x 5 videos (`../track/held32-coreml/t16m16n16e16/`, `g2.log`) | HOTA **0.6024** against t16m16n16's 0.6025; per video -0.0001, -0.0002, -0.0002, +0.0000, +0.0000; predicted box counts equal on every video | **pass** |
| E-G3, 1,500 frames (`../track/held300-coreml/t16m16n16e16/`, `g3.log`) | HOTA **0.5693** against t16m16n16's 0.5697; per video +0.0000, -0.0002, -0.0008, +0.0002, -0.0001 | **pass** |
| E-speed, run 1 (`job.log`) | wall e16/base 0.918, 0.907, 0.939 (median 0.918); only base r1 had start and end load under 8 (the others ended at 8.0-8.9) | **not read** |
| E-speed, run 2 (`run2/`, `job-run2.log`) | wall 1.112, 0.878, 0.811; the end load rose to 8.3-20.6 in 5 of 6 arms (another session's `rustc` jobs) | **not read** |
| E-speed, run 3 (`run3/`, `job-run3.log`) | wall e16/base **0.818, 0.912, 0.920** (median 0.912); push 0.837, 0.939, 0.918. Each arm started below load 6 (`E31_START_MAX=6`); start and end load 3.4-7.2, every arm read. base spread 1,228-1,360 ms (10%) | **pass** |

**E31 passes every gate, and the flag is adopted as opt-in**, as RULES.md decided before the result.
The code moved from the build copy into the main tree unchanged: one patch, checked to apply before
the gates read. After the move, the flag's input-size check was removed, because the shared
placement parser already refuses any CoreML placement at another size (the 2-frame check below shows
that refusal).

How close it was: the median wall ratio, 0.912, is under the 0.92 line by 0.008. Round 1's ratio,
0.818, comes from a base arm that ran 100 ms slower than in the other two rounds. Without round 1,
the ratios are 0.912 and 0.920.

Main-tree checks (`main-checks.sh`, output in `main-checks/`):
- fmt clean on every file E31 changed. The one diff was an import line in `tracker/mod.rs`, from
  before E31; `cargo fmt -p st-model` fixed it, and `cargo fmt -p st-model -p st-coreml -p spiketrack
  --check` is now clean. clippy `-D warnings` clean (re-run after both edits, with `Checking
  st-model` and `Checking spiketrack` in its output); spiketrack + st-model + st-coreml tests 157
  passed, 0 failed.
- 2-frame runs: the default records `det_encoder: ojas`; `--det-encoder coreml-fp16` with LoRA v3
  records the package and its sha256; `coreml-fp32` with LoRA v3, which has no export, is refused
  with the export command; `--input-size 672` is refused.

**Speed run 1 is not read.** Every ratio was under 0.95, with a median of 0.918, but 5 of its 6 arms
started or ended with the load at or above 8. Under the rule, no verdict comes from those rounds.
Run 2 repeats them under the same rule. The only procedural change is that the job waits, up to
10 minutes, for a quiet machine before each arm.

Tracks against base (run 1, each round): 32/32 frames with equal id sets, mean box IoU 0.9994, min
0.9656. They are not byte-identical, which is expected for fp16 encoders; the accuracy gates judge
them.

## Smoke run (not a gate)

Video 36, 6 frames, load 13. Steady frames (2-5): base 1.19, 1.44, 1.23, 1.32 s; e16 1.08, 1.30,
1.12, 1.10 s. Dispatches per steady frame fall from 3,344 to 2,488. `run.json` records
`det_encoder: coreml-fp16` and the package's sha256. Both arms bore 39 tracks. Those frame times were
taken under load 13, so they are not a speed reading.

## Checks in the build copy

- `cargo test --release -p spiketrack -p st-model -p st-coreml --lib --bins`: 157 passed, 0 failed,
  including `the_det_encoder_package_is_named_by_the_adapter_folded_into_it` and the encoder rows of
  `coreml_placements_refuse_edits_of_the_weights_their_packages_bake_in`.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- Not tested: no unit test runs the hook against the real package, because that needs the gated
  checkpoint. The accuracy gates are what judge the numbers. The package's parity against the torch
  mirror is in `../coreml-detenc/fp16-CPU_AND_GPU-db6a8308ccdb.json`.
