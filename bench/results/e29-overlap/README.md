# E29: unwaited Metal commits every N dispatches (2026-11-04): not adopted

The gates were set in `RULES.md` before the run. `job.sh` ran under the machine lock with one binary
(`versions.txt`). The checks passed (`checks.log`): the flag's refusal test, plus ojas-metal's
`set_overlap_dispatches_refuses_zero_and_keeps_the_bits` and
`overlap_commits_from_many_threads_match_a_serial_run_bit_for_bit`. The table is `summary.md`
(`tools/analysis/e25_frame.py`, steady frames >= 8).

| arm | wall ms, rounds 1-3 | wall / o256, rounds 1-3 |
|---|---|---|
| o256 (Ojas's default) | 1445, 1742, 1765 | 1 |
| o32 | 1526, 1512, 1597 | 1.056, 0.868, 0.905 |
| o8 | 1734, 1341, 1546 | 1.200, 0.770, 0.876 |
| o1 | 1663, 1688, 1663 | 1.151, 0.969, 0.943 |

## Verdicts

- **O1 passes [V].** Every arm's predictions are byte-identical to the same round's o256, in all 3
  rounds. Committing sooner does not change what is computed.
- **O2 fails for every arm [V].** No arm is under 0.95x in all 3 rounds; each is slower than o256 in
  round 1. Under the rules nothing changes: the default stays at 256 and `--overlap-dispatches`
  stays a diagnostic flag.
- **The run was contended, and this is a negative, not a refutation.**
  - The load average rose from about 7 to 16 during the run (`job.log`), and o256 alone spread
    1,445-1,765 ms (22%) across rounds. E28's quiet-machine controls on the same video were
    1,308-1,331 ms.
  - With that floor, a real gain of 5-10% cannot be told apart from noise, in either direction.
  - [I] Rounds 2 and 3 lean towards 32 and 8 dispatches (0.77-0.91x). A rerun on a quiet machine
    is the next step before the lever is dropped.

**Follow-up (E30, `../e30-quiet/README.md`):** the quiet rerun of 32 dispatches ran 1.060x in its one
read round, with predictions byte-identical. The lever is dropped.
