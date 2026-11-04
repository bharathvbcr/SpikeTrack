# The fp16 Metal default: checks of the change (2026-11-04)

This change makes `track` on `--backend metal` default to the fp16 CoreML trunk (`TrunkPrecision::default_for`).
Memory attention defaults to its fp16 CoreML program (`Placement::default_mem_attn`), but only beside
that trunk on Metal at the native 1008 input. In every other case it stays on Ojas.

The gates the pair passed are recorded in:
- `../e27-pipe-necks/README.md`: speed, 0.57-0.60x;
- `../track/held32-coreml/README.md`: 32 frames;
- `../track/held300-coreml/README.md`: 1,500 frames.

Job: the script recorded in `job.log`, run under the mac_heavy lock.

| check | result | log |
|---|---|---|
| clippy `-D warnings` on st-coreml, st-model, spiketrack, all targets | exit 0 | `clippy.log` |
| tests, release, the same three packages | exit 0; 23 binaries ok, 182 passed, 0 failed | `tests.log` |
| new tests ran | `the_default_trunk_is_coreml_fp16_on_metal_and_the_ojas_trunk_on_cpu`, `memory_attention_defaults_to_coreml_fp16_only_beside_the_default_fp16_trunk` | `tests.log` |
| 6 frames of VISEM 36, shipped LoRA v3 + edit, no placement flags | run.json: `trunk_precision coreml-fp16`, `mem_attn coreml-fp16`, `trunk_pipeline true`, both fp16 packages named | `noflags/36.run.json` |
| the same with `--trunk-precision coreml-fp16 --mem-attn coreml-fp16` | `36.json` byte-identical to the no-flag run (`cmp`) | `explicit/`, `job.log` |
| the same with `--trunk-precision coreml-fp32` only | run.json: `coreml-fp32` trunk, `mem_attn ojas`, which is the fp32 reference configuration | `fp32ref/36.run.json` |

The per-frame times in these 6-frame runs include warm-up. They are not a speed result.

After the checks above, `--mem-attn`'s help text gained a clause: an adapter or edit of memory attention
is refused under the default too. That change is string-only. Clippy and the tests were re-run after it:
exit 0 and exit 0, 23 binaries ok (`clippy-2.log`, `tests-2.log`).

To revert to the fp32 default, change two lines in `crates/spiketrack/src/track_cmd.rs`:
- `TrunkPrecision::default_for`: `BackendChoice::Metal => Self::Coreml("fp32")`;
- the `mem-attn` default passed to `placement`: `Placement::Ojas`.
