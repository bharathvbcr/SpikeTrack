# E11: where the production frame goes with the CoreML fp32 trunk (2026-10-29)

This profiles `spiketrack track` itself, the frame that ships, not the judo `frame_profile` test.
Setup: Metal, the default CoreML fp32 trunk, prompt "sperm", r16 LoRA, `--boxes detector`,
`--max-objects 64`, the first 16 frames. VISEM 22 holds 11-12 tracks, which fit one tracker state;
VISEM 82 holds 17-19 from frame 2, which take two.

The binary was built from 1ff9670 plus this change, while the tree was dirty (sha in `versions.txt`).
`job.sh` ran under mac_heavy on a shared machine (load 8-20). For each video it ran an unprofiled
control, `--profile stages`, then a second control, and finally `--profile ops` on 8 frames of 82.

**Validity [V].** The stages run's in-Session sum is within 10% of the controls, which was the
pre-stated condition (docs/SPEED_PLAN.md E11): 2390 against 2366 / 2373 ms on 22, and 2889 against
2913 / 2801 ms on 82. With the marker on, predictions are byte-identical to the controls.

## Steady frames (1-15), ms per frame, `--profile stages`

| stage | 22 (1 state) | share | 82 (2 states) | share |
|---|---|---|---|---|
| trunk (CoreML, incl. image download and output upload) | 925 | 38.5% | 928 | 32.0% |
| detector neck + detector | 786 | 32.7% | 821 | 28.3% |
| tracker propagate, state 0 | 344 | 14.3% | 337 | 11.6% |
| tracker propagate, state 1 | — | | 284 | 9.8% |
| memory encode | 158 | 6.6% | 258 | 8.9% |
| propagation neck | 137 | 5.7% | 141 | 4.9% |
| births, interactive neck, outputs, host association | 41 | 1.7% | 119 | 4.2% |
| decode + upload, progress line + collector | 13 | 0.5% | 14 | 0.5% |
| sum | 2403 | | 2903 | |

Frame 0, which runs two visits and every birth, is in each `*.profile-stages.md`. A "-0.0" cell in
those files is an empty stage: the binary that wrote them printed an empty f64 sum as -0.0, which is
fixed since.

## Inside the stages, video 82, `--profile ops` (each op synced)

Per steady frame (`tools/analysis/profile_ops_by_kind.py v82-ops/82.profile-ops.md 7`):

| stage | op time | wall (stages) | wall minus ops | largest ops |
|---|---|---|---|---|
| detector neck + detector | 525 | 821 | ~296 | conv2d 197 (two 3x3 256->256 at 288x288: 113), linear 97 (182 calls), attention 64, permute 40 (152 calls) |
| tracker state 0 | 268 | 337 | ~69 | attention 131 (memory cross-attention over up to 36,400 keys), linear 61 |
| tracker state 1 | 159 | 284 | ~125 | attention 77, linear 38 |
| memory encode | 145 | 258 | ~113 | conv2d 83, upload [1, 32, 1152, 1152] 23 |
| propagation neck | 138 | 141 | ~3 | conv2d 100 |
| trunk | 1.3 | 928 | ~927 | the CoreML call itself, not a backend op |

Reading [V for the numbers, I for the causes]:

- **No single stage reaches the pre-stated 40%.** On 22 the trunk is 38.5%, on 82 32%, and the
  tracker states on 82 are 21.4%. So the next lever is not one stage.
- **The detector is nearly as large as the trunk**: 786-821 ms against 925. That is where the f32
  Ojas path spends most of the frame now that the trunk is CoreML.
- **About 610 ms per frame on 82 is not backend op time** (2889 wall, minus 928 trunk, minus about
  1350 of ops). It is host work and device-to-host reads, which `Profiled` does not time; it times
  `upload` but not reads back.
  - It sits where large tensors come back to the host [I]: the detector reads back `pred_masks`
    (200 queries x 288 x 288 f32, ~66 MB) every frame, and memory encode reads back the tracks'
    1152 x 1152 masks (~95 MB at 18 tracks) and runs a host loop over them.
  - The tracker states read back their low-res masks and scores.
  - Which reads and loops cost what is not yet measured.
- The second tracker state costs ~284 ms of wall time against the first's 337. It holds 2 objects to
  the first's 16, and its attention is 77 ms against 131. Its wall-minus-ops gap (~125 ms) is the
  larger of the two [I: per-state host work that does not shrink with the object count].
- The "wall minus ops" column mixes windows: the ops run covers frames 1-7 of 82, the stage times
  frames 1-15. Read it as a size, not to the millisecond.

Files: `v*-stages/*.profile-stages.md`, `v82-ops/82.profile-ops.md`, and `*.log` (per-frame
telemetry). Run JSON and predictions are in each run directory; masks are not committed.
