# Gate for merging main (E10) into speed/e7-geometry (2026-10-29)

The merge is a1a4c08: main at d9a137f (E10, the CoreML fp32 trunk as the Metal default) into
`speed/e7-geometry` at 45e5d63 (E7, one runtime input geometry, `track --input-size`). `job.sh`
ran it under mac_heavy on the committed merge (dirty 0, `versions.txt`).

| step | result |
|---|---|
| build, workspace unit tests | 244 passed, 0 failed (`unit-tests.log`) |
| backbone vs upstream at 1008, Metal | worst cosine 1.0000000 over 95 stages |
| backbone vs upstream at 672, Metal | worst cosine 1.0000000 over 17 stages |
| generated RoPE tables vs the checkpoint's | pass |
| CoreML fp32 trunk + Ojas necks vs upstream | worst rel L2 2.737e-5, as on main |
| 49 frames of VISEM 22, `--trunk-precision f32` | 22.json byte-identical to the b11bb69 binary's f32 run |
| 49 frames of VISEM 22, default trunk (coreml-fp32) | 22.json byte-identical to the b11bb69 binary's coreml-fp32 run |

After the fast-forward, on main 3ee54cd (`gating-parity-cpu.log`): st-model's `gating_parity`, ignored
tests included, 6 passed, 0 failed. That covers `trunk_gated`, which the merge resolved by hand and which
`track` never calls. On the real VISEM frames all 9 windows change, so the partial-recompute path is
exercised only by the synthetic test. The 2 dirty files were documentation.

The two track runs compare against `bench/results/coreml-trunk/timing/track-{f32,coreml-fp32}-1`.
Their s/frame (3.55 and 2.53) are single unpaired runs on a shared machine, not a benchmark.
