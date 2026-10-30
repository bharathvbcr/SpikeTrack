# E17: E15's footprint creep, by category and at full length (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E17, 20cdefc) before the run. One mac_heavy hold, 08:18-09:18Z
(it waited 16 min behind ojas-78's `78-gate2`), load 2.7-4.5. Binary shas are in `versions.txt`;
the script is `job.sh`.

The runs:
- E13 (49a92571) over 300 frames of held-out video 47: the control.
- E15 (8567e60a) over all 1,470 frames of video 47.

Both used the committed held-out settings. Each run took `footprint --pid` and `vmmap -summary`
snapshots of the live process (`*/snap/`). Frame times from these runs are not read.

## It levels off [V]

E15's per-frame footprint (progress lines), mean MiB by window:

| 16-99 | 100-299 | 300-499 | 500-699 | 700-899 | 900-1099 | 1100-1299 | 1300-1469 |
|---|---|---|---|---|---|---|---|
| 8919 | 9039 | 9296 | 9409 | 9327 | 9337 | 9342 | 9334 |

- **The slope over frames 700-1469 is -0.005 MiB per frame.** That is under the pre-stated 0.2, so
  the creep settles. It is not a leak, and no fix is due under the pre-stated rule.
- **E15 rises ~420 MiB over its first ~500 frames, then holds at ~9.33 GB.** The peak after frame 16
  is 9.71 GB.
- **The plateau is below E13.** E13's mean over frames 16-299 on 47 is ~9.9 GB (E16), so E15 still
  uses less memory than the build before it at every length measured.
- The E13 300-frame predictions are byte-identical to the committed held-out 47.json.

## Where it is [V for the sizes, I for the reading]

Dirty MiB by `footprint` category:

| run, frame | Malloc Large | IOAccelerator (graphics) | owned unmapped (graphics) | Malloc Small | total |
|---|---|---|---|---|---|
| E13, 30 | 5305 | 3914 | 534 | 450 | 10251 |
| E13, 150 | 4650 | 5115 | 440 | 418 | 10671 |
| E13, 290 | 5319 | 2813 | 628 | 417 | 9225 |
| E15, 30 | 4396 | 3026 | 628 | 451 | 8548 |
| E15, 150 | 4589 | 4952 | 628 | 456 | 10673 |
| E15, 290 | 4670 | 3268 | 2053 | 457 | 10497 |
| E15, 700 | 4800 | 3065 | 628 | 464 | 9006 |
| E15, 1100 | 4894 | 3118 | 628 | 468 | 9157 |
| E15, 1460 | 4894 | 3118 | 628 | 468 | 9157 |

- **The pre-stated 60% rule picks the wrong owner, because snapshots catch transients.**
  - E15's frame-30-to-290 snapshot delta is +1949 MiB. 73% of it is "owned unmapped (graphics)"
    (628 -> 2053), and that is back to 628 by frame 700.
  - IOAccelerator swings by ~2 GB between snapshots in both builds, with where in the frame the
    snapshot lands.
  - By the letter of the rule the owner is the graphics driver. Read against the full table,
    that category is a within-frame transient, not the creep.
- **The steady growth is in Malloc Large [I].** It goes 4396 -> 4894 MiB from frame 30 to 1100,
  then holds, which matches the ~420 MiB rise and plateau in the per-frame lines.
  - E13's Malloc Large swings 4650-5319 MiB between snapshots, so six samples cannot pin the
    growth to that category.
  - The rise and plateau are consistent with Malloc Large and with nothing else in the table.
  - E15 drops the host token copies, so it starts lower. The large-allocation zone may then fill
    back toward a steady size over ~500-700 frames.
  - That reads as the allocator's free-region retention settling, not a structure that grows. It
    is not instrumented further.
- Metal device allocations (`currentAllocatedSize`, which includes Ojas's capped pool) stayed flat
  in E16. The creep is host malloc, not the GPU heap.

GAP-E15-HOST-FOOTPRINT-CREEP is answered: bounded, ~9.33 GB on 47, below E13.

Files: the per-run logs, `snap/` (footprint and vmmap at each frame), predictions (masks removed),
`job.sh` and `versions.txt`.
