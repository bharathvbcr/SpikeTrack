# E30: fp16 CoreML necks by the task metric, and overlap commits on a quiet machine (2026-11-04)

The rules are in `RULES.md`, written before any run. The speed table is `job.log` (from
`tools/analysis/e25_frame.py`, steady frames >= 8, VISEM 36, 32 frames). The accuracy runs are
`../track/held32-coreml/t16m16n16/` and `../track/held300-coreml/t16m16n16/`.

## Lever A: `--necks coreml-fp16` beside the default fp16 trunk + memory attention

| gate | result | verdict |
|---|---|---|
| A-G2, 32 frames x 5 videos | HOTA 0.6025, the same as t16m16; every video within +0.0004 / -0.0001 | **pass** |
| A-G3, 1,500 frames | HOTA **0.5697** against t16m16's 0.5694; per video +0.0000, +0.0000, +0.0006, +0.0001, +0.0000 | **pass** |
| A-speed, run 1 | wall/t16m16 0.892, 0.942, 0.866 (median 0.892); only round 1 was under the load limit | **not read** |
| A-speed, run 2 (`run2/`, `job-run2.log`) | wall/t16m16 **0.916, 0.898, 0.886** (median 0.898); load 3.8-7.6 throughout, every round read; t16m16 spread 1,338-1,365 ms (2%) | **pass** |

**Lever A passes every gate and is adopted.** `Placement::default_necks` (`track_cmd.rs`) gives
`--necks coreml-fp16` whenever `default_mem_attn` would pick CoreML fp16 and the trunk is pipelined.
Otherwise the necks stay on Ojas: under `--trunk-precision coreml-fp32`, `--trunk-pipeline off`,
another backend or another input size. That pipelined condition is the only one E30 measured.

Checks (`default-checks.sh`, output in `default-checks/`):
- fmt clean, clippy `-D warnings` clean, spiketrack + st-model tests 149 passed, 0 failed (including
  `the_necks_default_to_coreml_fp16_only_beside_the_default_pipelined_fp16_trunk`);
- 2-frame runs read the choice back from run.json: the default picks `coreml-fp16` necks; the
  coreml-fp32 trunk, `--trunk-pipeline off` and `--necks ojas` each pick `ojas`.

Job scripts whose arms meant the earlier defaults now pin them: `--necks ojas` in the t16m16 arms;
the coreml-fp32 trunk and pipeline setting in E25/E26. So re-running an old result reproduces it.

**Revert:** in `Placement::default_necks`, return `Self::Ojas` (one line).

The tracks against t16m16 have 32/32 equal id sets in every round, mean box IoU 0.9999 and min
0.9907. They are not byte-identical, which is expected for fp16 necks; the accuracy gates judge them.

**Speed run 1 is not read.** The 1-minute load was under 8 for round 1 (5.4-7.2) and passed 8 in
rounds 2 and 3 (8.7-13.8). Under the rule, no verdict comes from rounds that are not read. Every round
was under 0.95x and the median under 0.92, but that does not count. Run 2 repeats the speed rounds
under the same rule (`run2/`).

The 1,500-frame run started once at 14:21 UTC and was stopped at frame 288 of video 21. The
`research` folder was moved out from under it, and the machine lock's memory guard could not find its
helper, so it stopped the job (exit 71) and voided the result. After the folder was restored, the
run was repeated in full from 14:53 UTC (`../track/held300-coreml/job-t16m16n16.log`). The table
above is that second run.

## Lever B: `--overlap-dispatches 32`

- **O1 passes:** predictions byte-identical to t16m16 in all 3 rounds.
- **O2 fails in the one round that was read:** round 1, under load 5.4-7.2, ran 1.060x wall (push 0.984).
- **The lever is dropped.** E29 (contended) and this run both fail to show a gain. `--overlap-dispatches`
  stays a diagnostic flag, and `MetalBackend::set_overlap_dispatches` stays as the documented setter.

## Clippy (run with the speed job)

- **SpikeTrack:** `cargo clippy --workspace --all-targets -- -D warnings` is clean (`clippy-spiketrack.log`).
- **ojas fails on code this work did not touch.** `ojas-cpu/src/quant_sparse.rs:468` (committed in 1f97685,
  2026-11-02) trips `clippy::chunks_exact_to_as_chunks`, a lint new in this toolchain (rustc 1.99.1)
  (`clippy-ojas.log`). So ojas's HEAD does not pass its own clippy gate on this machine. That is a
  separate fix.
- **ojas-metal is clean with that one lint allowed:**
  `cargo clippy -p ojas-metal --all-targets -- -D warnings -A clippy::chunks_exact_to_as_chunks`,
  covering `set_overlap_dispatches` and its test.
