# E20: the memory encode without duplicate host copies (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E20, d88ee88) before the run. The code is cd237ab on branch
`e20-copy-elim`, merged as 9b3a628. One mac_heavy hold, 15:46-16:45Z, load 8.9-58.5 (a spike to 58
came from outside the job, during the CPU parity test). Binary shas are in `versions.txt`; the
script is `job.sh`. The control is the E19 binary.

## The change

- Mask channels: only the n object planes are built on the host. Each is written once, as
  `host::sigmoid(x) * 2 - 1`, into a fresh tensor that is then uploaded. The 170 MB zeroed
  32 x 1152^2 buffer is gone.
- The 32 - n constant planes are made on the device with `fill` and joined in slot order with
  `slice` and `concat`, which are bit copies.
- Memory update: the readback's host tensor is the working copy, with no `to_f32_vec`.
- Each state reads its rows from the shared logits by index, instead of gathering its own copy.

## Run 1 (957c66d) failed two gates [V, `run1-957c66d/`]

- **CPU detect_track_parity panicked at frame 0**: "contiguous write requires a uniquely owned
  allocation". On a host backend, `to_host` returns a clone that shares storage, so
  `f32_slice_mut` refused.
  - That is a real E20 bug. cd237ab drops `upsampled` before the in-place area rule.
  - That CPU parity test is the regression test: it fails at 957c66d and passes at cd237ab.
- **The Metal table diff flagged 128 lines.** Every one was a frame line's wall-clock time. With
  those times masked, the tables were identical. That was a gate-script flaw.
  - The script now masks the times.
  - It also fails a table under 100 lines, which is what a panic leaves.
- Run 1's predictions were byte-identical to control-1 in all 8 comparisons.

## Gates, run 2 [V]

- st-model lib tests: 46 pass.
- tracker_parity: 8 of 8 pass.
- detect_track_parity on video 22, Metal and CPU:
  - both pass, with a mean per-id mask IoU of 0.99855 (Metal) and 0.99854 (CPU);
  - **the visit and yield tables, 135 lines each, are identical to E16's apart from frame times.**
- **Every E20 run's predictions are byte-identical to control-1** (8 of 8).

## Validity: within 10% of both controls [V]

| video | control-1 | control-2 | E19 stage sum | E20 stage sum |
|---|---|---|---|---|
| 82 | 2728.7 ms | 2937.3 ms | 2497.8 ms (-8.5%, **-15.0%**): **invalid** | 2648.9 ms (-2.9%, -9.8%): valid |
| 22 | 2024.7 ms | 2215.3 ms | 2161.1 ms (+6.7%, -2.4%): valid | 2174.5 ms (+7.4%, -1.8%): valid |

- E19's 82 table fails a second time. On 82, only E20's own table is read.
- 22 has the pre-stated before-and-after comparison.

## Sub-phases on 22, same hold, mean ms per steady frame [V]

| sub-phase | E19 | E20 | pre-stated direction | verdict |
|---|---|---|---|---|
| upsampled readback | 8.7 | 7.6 | falls by one copy | falls |
| area rule (host) | 15.4 | 15.9 | untouched | +0.5, within noise [I] |
| mask channels (host sigmoid + fill) | 29.5 | 19.3 | falls by the zero-fill share | falls |
| mask upload | 17.1 | 14.1 | falls | falls |
| encoder (device) + feature readback | 69.5 | 67.8 | — | — |
| no-object add + bf16 + token upload | 6.6 | 4.8 | — | — |

- No sub-phase E20 changed rose, so nothing refutes it.
- The memory-update sub-phases sum to about 14 ms per frame less on 22, out of ~2160 ms (<1%)
  [I: n = 1 per arm].
- **The unprofiled frame time is not claimed.**
  - On 82, E20's 2407 ms is under both controls (2729 and 2937 ms).
  - On 22, E20's 2016 ms sits beside control-1's 2025 ms.
  - But on 82 the controls themselves differ by 7.6%, and the load fell from 17 to 9 between them.

## Video 82, E20's valid table [V], against the invalid E19 table [I]

| sub-phase | state 0 | state 1 | E19 (invalid), state 0 / 1 |
|---|---|---|---|
| mask channels | 32.1 | 3.4 | 25.8 / 3.2 |
| mask upload | 18.8 | 9.4 | 16.6 / 14.4 |
| upsampled readback | 13.0 (per frame) | | 12.7 |
| area rule | 32.3 (per frame) | | 25.9 |

**The E19 column is not a "before".** It comes from a table that failed validity under different
load. So the pair is not a before-and-after, and nothing on 82 is confirmed or refuted by it. The
mask-channels pair (25.8 against 32.1) is not a rise E20 caused. On 22, where both tables are
valid, mask channels fell.

Three 82 readings of the two qualifying sub-phases:

| reading | area rule | mask channels |
|---|---|---|
| E19 run (invalid) | 30.0 | 33.5 |
| E20 run 1 (valid) | 28.7 | 31.3 |
| E20 run 2 (valid) | 32.3 | 35.5 |

All three readings clear 20 ms.

- Under E20, mask upload follows the state's object count: 18.8 ms for state 0 and 9.4 ms for
  state 1, which has few objects. Under E19 both states paid 14-20 ms [I: the E19 tables are
  invalid].
- What is left in a few-object upload (~9 ms) is, by inference, mostly the `Wait::Upload` drain
  plus the device fill and concat [I: not split].

## Memory, frame 15 [V for this hold: both controls equal]

Later evidence (E21, `../e21-host-threads/README.md`):

- On 82, five of six runs there read 8386-8391 MiB. That reproduces the 8382 below; one outlier
  read 8000.
- On 22, the same code read 7944-8134 MiB, so the footprint there is not deterministic. The 22
  fall below is still larger than that spread.
- E19 was not rerun, so both falls are unrepeated.

| video | footprint, control | footprint, E20 | Metal allocated, control / E20 |
|---|---|---|---|
| 82 | 8712 MiB | 8382 MiB (-330) | 5664 / 5667 MiB |
| 22 | 8503 MiB | 7934 MiB (-569) | 5555 / 5558 MiB |

- The fall is [V]. The cause, that the 170 MB zeroed mask buffer and the per-state gather are gone,
  is [I]. 82, with two states, drops less than 22 with one, which a simple per-state sum does not
  explain.
- The fix's regression guard is the ignored, fixture-gated CPU detect_track_parity test, which took
  48 minutes here. The `to_host` + drop + `f32_slice_mut` sequence is inline in `Session`, with no
  seam for a lib test.
- Metal allocated rises 3 MiB, for the cached constant planes.

## Next, against the pre-stated rule [V]

On E20's valid 82 table, both of these are at 20 ms or more:

- mask channels: 32.1 + 3.4 = 35.5 ms, summed over states;
- the area rule: 32.3 ms.

**So E21 is exact threading of the sigmoid fill and the area rule**, with the same f32 op per
element and E12's shape.

Read for E21 [V, `crates/st-model/src/tracker/session.rs:254`]: `suppress_area_shrinkage` has no
float sum.

- Its per-pixel winner is a strict-greater argmax over objects, so the first index wins ties.
- Its before and after areas are `usize` counts.
- Its clamp and score are per element or `any`.

So threading it over pixels (the winner) and over objects or pixel chunks (the counts) is exact in
any split order. The sigmoid fill is elementwise, so the same holds for it.

Files: the run directories and logs (masks removed), `run1-957c66d/` (the failed first run), the
test and build logs, the table diffs (empty for run 2), `job.sh` and `versions.txt`.
