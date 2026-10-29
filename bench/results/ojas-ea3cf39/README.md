# Ojas pin 8b88328 -> ea3cf39, and the 4 GiB Metal memory cap in `spiketrack track`

Measured 2026-10-29, Metal, tessl c9adc87 (`versions.txt`). The bump has the owner's approval of 2026-10-28:
one bump, to an ojas main that has `set_mem_cap` as public API. ojas-78 sent ea3cf39 after its full workspace
check passed.

What changes in SpikeTrack:
- `spiketrack track` on Metal opens its backend through `open_metal()` (crates/spiketrack/src/track_cmd.rs). That
  raises the memory cap from ojas's default min(1 GiB, budget / 4) to `METAL_MEM_CAP` = 4 GiB.
- `run.json` records the cap the device reports as `metal_mem_cap_bytes`, or `null` on CPU.
- The cap is the knee of the footprint sweep in `../speed-baseline/memcap-footprint`.
- `image` keeps ojas's default cap. It runs one detector pass, and the cap was measured only on tracking.

## Gate (stated in the job before it ran)

| check | result |
|---|---|
| build (release, workspace, tests and bins) | ok (one pre-existing dead-code warning: `eval_visem::evaluate`) |
| workspace unit tests | ok, 0 failed (`unit-tests.log`) |
| `a_track_runs_metal_backend_has_the_raised_memory_cap` (GPU) | ok: ojas opens at 1 GiB, `open_metal` reports 4 GiB |
| `visem_22_frames_0_to_7_on_metal` (GPU, now also asserts `metal_mem_cap_bytes`) | ok |
| session parity VISEM 22, Metal, ojas's default cap | 0 id-set misses, mean mask IoU 0.99855 over 470 pairs (8b88328: 0.99855) |
| session parity VISEM 22, Metal, 4 GiB cap (new test) | 0 id-set misses, mean mask IoU 0.99855 over 470 pairs |
| Metal module gates: text, backbone, the four tracker modules | ok |
| Metal detector gate | not run: `fixtures/detector` is not on disk (GAP-FIXTURES-DELETED-FOR-DISK) |
| CPU gates | not run; production is Metal. The range touches the CPU softmax row kernel (8cb6c5d, 82d8c2c) |
| `cargo fmt --check` | fails across the whole repo (1192 hunks in files this change does not touch). SpikeTrack has never been rustfmt-clean and no tool runs fmt. Recorded, not gated |
| clippy `-D warnings` | cannot lint this change: `st-video` and `st-model` fail on lints new in clippy 1.99 (`chunks_exact_to_as_chunks`, `too_many_arguments`, ...), none in changed lines (`clippy*.log`, GAP-CLIPPY-199) |

`Cargo.lock` did not change. SpikeTrack does not depend on gusset, and ojas's gusset/libc lock bump (2318ccb) does not reach it.

## Timing: what ships, before vs after

`job-timing.sh` runs 49 frames of VISEM 22 with prompt "sperm", the r16 LoRA, `--boxes detector` and
`--max-objects 64`, interleaved over three rounds. The two arms:
- old: the SpikeTrack-e7 45e5d63 binary on ojas 8b88328 at the default cap. At 1008 its output is byte-identical
  to main's before the bump.
- new: this change.

| round | old s/frame | new s/frame | new / old |
|---|---|---|---|
| 1 (old, new) | 3.0469 | 2.8854 | 0.947 |
| 2 (new, old) | 3.0829 | 2.8676 | 0.930 |
| 3 (old, new) | 3.0948 | 2.8675 | 0.927 |

- Median ratio 0.930: about 0.21 s/frame saved. Arm spread (max/min) is 1.016 for old and 1.006 for new. All
  three rounds have the same sign.
- Load was 2.7-4.0 and the GPU was 75-92% busy before each arm, including other sessions' work.
- The gain is smaller than the frame profile's ~0.87x at 4 GiB. That profile timed the tracker frames of a 4-frame
  judo clip; this run is a whole tracking run with 12 tracks. The difference is not decomposed.
- Predictions (`22.json`) are byte-identical between old and new in all three rounds, and across the three new
  rounds. Neither the bump nor the cap changes what is tracked on this clip.

Files: `versions.txt`; `*.log` per gate step; `timing.txt`; `track-{old,new}-{1,2,3}/` (predictions and
run.json, masks not kept) with their logs; `job-gate.sh` and `job-timing.sh`.
