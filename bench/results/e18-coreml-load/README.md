# E18: the CoreML trunk's load, by phase (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E18, 443236b) before the run. One mac_heavy hold, 12:57-13:01Z.
The load was high, 13.9-28.1, because peers were active. Binary shas are in `versions.txt`; the
script is `job.sh`.

`track` ran on 2 frames of VISEM 22, interleaving the E15 binary ("old", total only) and 443236b
("new", with the split). Each run reused the compiled model, so compile is 0.

| run | load (1 min) | trunk load, s | hash | model load | `track` model load, s |
|---|---|---|---|---|---|
| old-1 | 16.5 | 11.6 | - | - | 12.2 |
| new-1 | 17.0 | 22.6 | 7.61 | 14.97 | 23.4 |
| old-2 | 20.3 | 8.6 | - | - | 9.3 |
| new-2 | 23.3 | 7.7 | 3.68 | 4.06 | 8.1 |
| old-3 | 21.7 | 5.1 | - | - | 5.4 |
| new-3 | 20.3 | 11.3 | 6.59 | 4.67 | 12.2 |
| new-16 | 28.1 | 4.3 | 3.80 | 0.48 | 4.7 |

- **The hash is the lever [V].** Its minimum is 3.68 s, over the pre-stated 1 s, and it is 47-88%
  of every split load.
  - The hash reads all 1.72 GB of `weight.bin` on every start, to name a compiled model that never
    changes unless the package is re-exported.
  - Under this load it ran 3.7-7.6 s, about 0.2-0.5 GB/s.
- **Core ML's own model load swings 0.48-14.97 s** with machine load and with what is already
  resident [unexplained in detail]. The cache key cannot reduce it.
- The 1.7 GB package's page-cache state was not controlled. new-1 is the slowest on both phases.
- The old and new totals are not compared: the instrumentation adds three clock reads, and the
  spread is several seconds.
- **Identity:** new-16's predictions are byte-identical to `closing-e15/speed/v22-new-1/22.json`.
  The st-coreml unit tests pass (4 of 4).

Next, by the pre-stated rule: a cheaper cache key with its own gate (SPEED_PLAN, E18b).

Files: the run directories and logs (masks removed), the build and unit-test logs, `job.sh` and
`versions.txt`.
