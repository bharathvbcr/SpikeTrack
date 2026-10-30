# E19: the memory update and encode split into sub-phases (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E19, 0dd25eb) before the run. The code is 0dd25eb (markers only,
no numerics). One mac_heavy hold, 15:22-15:29Z. Binary shas are in `versions.txt`; the script is
`job.sh`. The control is the E18b binary.

## Gate [V]

- Unit tests pass, none failed (`unit-tests.log`): spiketrack's binary tests 75, st-model's lib
  tests 46.
- **Every E19 run's predictions are byte-identical to control-1**, on 82 and 22, profiled and not.

## Validity [V]

The rule: the table is read only if its stage sum is within 10% of the controls.

| video | control-1 | E19 | E19 `--profile stages` (stage sum) | control-2 | 1-min load, control-1 -> control-2 |
|---|---|---|---|---|---|
| 82 | 3308.7 ms | 3108.0 ms | 3078.0 (3092.5) ms | 2340.0 ms | 21.0 -> 10.9 |
| 22 | 1893.3 ms | 1992.7 ms | 1970.0 (1982.8) ms | 2044.7 ms | 9.6 -> 9.8 |

(Mean of steady frames 1-15, unprofiled from the frame lines, profiled from the table.)

- **82 fails the rule.** Its stage sum is 6.5% under control-1 but 32% over control-2. The two
  controls differ by 30% because the load fell from 21 to 11 during the hold, as ojas-78's jobs
  drained. **The 82 table is not read.** Its sub-phase numbers below are context only, labelled [I].
- **22 passes**: +4.7% against control-1 and -3.0% against control-2.
- The 82 re-read is folded into E20's hold: the E19 binary with `--profile stages`, beside E20's
  own, under the same load. That avoids a second lock wait.

## Sub-phases, mean ms per steady frame

| sub-phase | kind | 22 [V] | 82, state 0 + state 1 [I: invalid table] |
|---|---|---|---|
| upsampled readback | transfer | 8.3 | 14.8 |
| area rule (host) | host compute | 17.0 | 30.0 |
| mask channels (host sigmoid + fill) | host compute | **21.3** | 29.7 + 3.8 = 33.5 |
| mask upload | transfer | 17.1 | 19.3 + 16.7 = 36.0 |
| encoder (device) + feature readback | device + transfer | 61.9 | 82.3 + 71.1 = 153.4 |
| no-object add + bf16 + token upload (host) | host + transfer | 4.6 | 5.2 + 4.3 = 9.5 |

Video 22 has one tracker state; 82 has two. Waits per steady frame (progress lines): 82, 23
(upload 4, read 19); 22, 18 (upload 3, read 15).

## Decision, against the pre-stated rule

- On the valid table (22), exactly one sub-phase reaches 20 ms: **mask channels, 21.3 ms, host
  compute**. The rule says E20 threads it **or drops its redundant work**, exactly.
- E20 (branch `e20-copy-elim`, 957c66d) drops the redundant work:
  - the zeroed 32 x 1152^2 host buffer (170 MB);
  - the 32 - n constant planes, which are now made on the device with `fill` and joined with
    `concat`, a bit copy.
  - It also covers the transfer sub-phases: only n planes are uploaded, and the readback's own host
    buffer is used in place of a third copy.
- Under the invalid 82 table, the largest item would be mask upload (36.0 ms). That is inside E20's
  scope too. So E20's content is the same under either table.
- The area rule (17.0 ms on 22) is below 20 ms on the valid table. E20 does not touch it.
- **Threading does not join E20.** It becomes E21 if E20's valid tables still show:
  - mask channels at 20 ms or more, summed over states; or
  - the area rule at 20 ms or more on a valid 82 table.

## Observation [I: the 82 table is invalid]

- On 82, mask channels scales with the state's object count: 29.7 ms for state 0 and 3.8 ms for
  state 1.
- Mask upload does not scale: 19.3 ms and 16.7 ms.
- That fits a fixed 170 MB upload per state, the cost E20 removes.

Files: the run directories and logs (masks removed), `build.log`, `unit-tests.log`, `job.sh` and
`versions.txt`.
