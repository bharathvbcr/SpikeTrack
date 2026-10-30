# E21: exact host threading of the sigmoid fill and the area rule (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E21, c4abe93) before the code. The code is 869540f on branch
`e21-host-threads`. The read below is attempt 3: one mac_heavy hold, 23:12-23:43Z, load 4.4-15.9.
Binary shas are in `versions.txt`; the script is `job.sh`.

The control is E20 (cd237ab, the `e20-copy-elim` worktree), rebuilt in its own target dir with the
same Metal toolchain as E21 (27B47, installed after the E20 binary was built). The old E20 binary
(4041d319) runs once per video as a toolchain check. That check is not an E21 gate.

## The change

`host::for_each_chunk_mut` splits a slice into contiguous pieces and hands them to up to
`MAX_THREADS` (8) scoped threads. It has E12's shape.

- **The sigmoid fill** (`fill_sigmoid_planes`, `tracker/state.rs`):
  - Each object-plane element is written once, as `host::sigmoid(x) * 2 - 1`, from the same source
    element.
  - Pieces of at least 16384 elements.
- **The area rule** (`suppress_area_shrinkage`, `tracker/session.rs`):
  - The per-pixel winner is a strict-greater argmax, so the first index wins ties. It runs over
    pixel chunks.
  - The counts, the clamp and the score run per object plane. The counts are `usize`, and the clamp
    and `any` are per element.
  - No float is summed, so every output bit is independent of the split.

## Attempts 1 and 2 (kept, not read)

**Attempt 1** (`attempt1-no-metallib/`) failed at the build. The Xcode 27.2 update had removed the
Metal toolchain, so `xcrun metallib` was missing.

**Attempt 2** (`attempt2-self-control/`) is invalid as an A/B: its "E20 control" was E21.

- **What happened:** the control binary and the E21 binary had the same sha (94a3f82). `nm`
  showed E21's `fill_sigmoid_planes` in the control.
- **Cause:** the main tree and the E20 worktree are two checkouts of one workspace, and they shared
  a `CARGO_TARGET_DIR`. Their crates get the same unit hashes, so cargo judged the worktree's older
  `st-model` sources fresh against E21's rlib and linked E21 into the "E20" binary.
- **Fix:** attempt 3 builds the control in its own target dir. Before any test or timing, a binary
  gate checks each binary for code only one side has.
- **E20's committed results are not affected.** Its E20 binary (4041d319) has no E21 code; the
  gate below checks it too.
- **A flawed gate, removed:** attempt 2 also ran `cargo fmt --check`. It fails across the crate,
  including files E21 never touched. The repo uses long lines and has no `rustfmt.toml`, so that is
  the existing state, not an E21 failure.

Attempt 2's gates did run on E21's code and all passed. Its timings compare E21 with itself, and
are superseded by attempt 3.

## Gates, attempt 3 [V]

- **Binary gate:**
  - the rebuilt E20 control has no E21 code and does have E20's slot-assembly string;
  - the E21 binary has both;
  - the old E20 binary has E20's code only.
  - The shas differ: E21 a931e9f7, control 2453a5ac, old E20 4041d319.
- **st-model lib tests: 49 pass.** They include the three new bit-identity tests:
  - `chunked_pieces_cover_every_index_once_at_their_start`;
  - `threaded_sigmoid_fill_is_bit_identical_to_the_serial_one`;
  - `threaded_area_rule_is_bit_identical_to_the_serial_one`. This one covers ties, -0.0, n of
    1, 2, 3, 5 and 18, and pixel counts that do not divide by the thread count.
- **tracker_parity:** 8 of 8.
- **detect_track_parity on 22**, Metal and CPU:
  - both pass, with a mean per-id mask IoU of 0.99855 (Metal) and 0.99854 (CPU);
  - **the 135-line visit and yield tables are identical to E16's apart from frame times.**
- **Predictions are byte-identical to control-1 in 8 of 8 runs.**
- **Toolchain check:** the old E20 binary's predictions equal the rebuilt E20's on both videos.

## Validity: within 10% of both controls [V]

| video | control-1 | control-2 | E21 stage sum | E20 stage sum |
|---|---|---|---|---|
| 82 | 2559.3 ms | 2449.3 ms | 2497.2 ms (-2.4%, +2.0%): valid | 2566.5 ms (+0.3%, +4.8%): valid |
| 22 | 1967.3 ms | 1850.0 ms | 1843.7 ms (-6.3%, -0.3%): valid | 1881.3 ms (-4.4%, +1.7%): valid |

All four tables are read.

## Sub-phases, same hold, mean ms per steady frame [V]

| video | sub-phase | E20 | E21 | ratio | refute if >= 0.7 |
|---|---|---|---|---|---|
| 82 | area rule (host) | 27.1 | 5.8 | 0.21 | holds |
| 82 | mask channels, state 0 + state 1 | 28.3 + 2.8 = 31.1 | 5.0 + 0.7 = 5.7 | 0.18 | holds |
| 22 | area rule (host) | 13.5 | 3.3 | 0.24 | holds |
| 22 | mask channels, state 0 | 16.8 | 3.3 | 0.20 | holds |

**Neither sub-phase reaches 0.7x on either video, so E21 holds.**

- The two sub-phases fall by about 46.7 ms per frame on 82 and 23.7 ms on 22.
  - n = 1 per arm, so this is [I]. But every ratio is 0.18-0.24, far from the 0.7 line.
  - With 8 threads on 6 P-cores and 12 E-cores, a 4-5.5x fall is in the expected range [I].
- Untouched sub-phases stay within noise. On 82, E20 then E21:
  - upsampled readback 11.2 then 10.4 ms;
  - mask upload, state 0, 16.2 then 16.1 ms;
  - mask upload, state 1, 9.1 then 9.5 ms.

## What is not claimed

- **The unprofiled frame time.** E21 ran at 2416.0 ms on 82 and 1849.3 ms on 22. Its controls
  ran at 2559.3 and 2449.3 ms on 82, and 1967.3 and 1850.0 ms on 22.
  - On 82, E21 is under both controls. On 22, it ties control-2.
  - But the controls themselves differ by 4.5% on 82 and 6.3% on 22, as load fell from 11 to 4.5.
  - The expected 1.9% (82) and 1.3% (22) falls are below that spread.
- **Footprint.** Metal allocated at frame 15 is equal in every run: 5667 MiB on 82, and 5557-5558
  MiB on 22. E21 does not change the host footprint either:
  - on 82, E21's two runs read 8386 and 8387 MiB, inside the E20 code's cluster;
  - on 22, they read 8042 and 8038 MiB, inside its 7944-8134 MiB range.

## Later evidence for E20's footprint claim [V]

E20 called its host footprint deterministic, because its two controls were equal. At frame 15 in
this hold:

- **On 82,** five of six runs read 8386-8391 MiB: E20 code in three, E21 in two.
  - With E20's own 8382, that is a 9 MiB cluster across two binaries and two holds.
  - One run is an outlier: control-1 read 8000 MiB. It was the first run after the 18-minute CPU
    parity test.
  - So E20's reading reproduces. Its -330 MiB against E19 is unrepeated, not contradicted: E19
    was not rerun here.
- **On 22,** the same E20 code reads 7944-8134 MiB across four runs, with no cluster. So the
  footprint is not deterministic there.
  - E20's -569 MiB fall (to 7934) is larger than that 190 MiB spread, so it stands [I: E19 not
    rerun].

## Next, by the plan

The detector's decoder host prep (sine, query MLP, box RPB, upload) is next. It is 52.0 ms per
frame on E21's 82 table.

Files: the run directories and logs (masks removed), `attempt1-no-metallib/`,
`attempt2-self-control/`, the test and build logs, the table diffs (empty), `job.sh` and
`versions.txt`.
