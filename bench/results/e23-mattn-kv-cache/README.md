# E23: memory attention reuses stored frames' projections (pre-stated 2026-11-03; bit-neutral; tracker step −6.4% on a pre-stated paired run, see Paired speed run: result)

## The change

Memory attention projected every stored memory frame on every frame: per layer, `img_k(feats)`,
`ca_k(maskmem)` and `ca_v(maskmem)` over all ~36,400 memory tokens (12 linears per frame,
~39 ms of the ~405 ms tracker step at 16 objects, `../ab-coreml-tracker/run2-ops/`).

Those projections depend on the stored frame alone. Its temporal position is added after them,
and RoPE after that. So:
- `MemoryAttention::frame_kv` computes a frame's projections once per layer.
- `MaskMem` keeps them (`kv`, filled by the first frame that attends it, shared by clones).
- `MemoryAttention::fwd_frames` concatenates the stored ones with the pointer block's, which is
  projected fresh each frame.
- `TrackerState::prune` drops them once a non-conditioning frame is before the next frame's
  7-frame memory.

`fwd` (the VOS driver and the upstream parity test) keeps the uncached path. Both now share one
layer loop (`layers_fwd`).

Files: `crates/st-model/src/tracker/memory_attention.rs`, `tracker/mod.rs` (exports),
`tracker/state.rs` (three hunks: the `MaskMem` field and `kv`, `compute_propagation`'s call,
`prune`), `tests/memory_attention_cache.rs`. `state.rs` also carries another agent's uncommitted
host-buffer refactor (with `layers.rs` and `vos.rs`), whose owner could not be identified (no
Claude Desktop transcript holds it). These hunks do not touch its lines.

## Gates, fixed before the run

1. **`fwd` unchanged** (CPU, `Numerics::Exact`): `fwd_is_unchanged` holds the uncached forward to
   output-bit hashes taken on the code before the change (run 2026-11-03 03:18 UTC, the test
   failing on placeholder zeros and printing the hashes).
2. **Cached = uncached, bitwise** (CPU Exact): `cached_equals_uncached` at 1, 3 and 7 memory
   frames, with 0, 16 and 32 pointer tokens.
3. **Same on Metal** (`cached_equals_uncached_on_metal`). It holds only if Metal's linear gives a
   row the same bits at any row count, which is not established. Its pass or fail is recorded
   either way. A fail moves the production gate to 6's fallback.
4. **Prune boundary**: `prune_drops_projections_before_the_next_frames_memory_only`.
5. **Lint**: `cargo clippy -p st-model -p spiketrack --all-targets -- -D warnings`.
   - **Narrowed before run 2.** Run 1 (03:23 UTC, `job.log` overwritten by run 2) stopped at
     clippy on `manual_checked_ops` at `crates/st-model/src/tracker/session.rs:1045`. That line
     is not in HEAD; it belongs to the other agent's uncommitted edits.
   - With `-D warnings` that error stops the lint before any target that depends on the library,
     this change's test included.
   - So clippy now runs without `-D`, and the gate fails on any finding in a file this change
     touches: `memory_attention.rs`, `tracker/mod.rs`, `state.rs` (a finding there is then read
     against its line's owner), `tests/memory_attention_cache.rs`, `spiketrack/src/track_profile.rs`.
   - Findings elsewhere are listed in `clippy-findings.txt` and are not this change's.
6. **Production predictions**:
   - Primary: the new binary's `track` predictions (VISEM 82, cap 16, 34 frames, the A/B's
     settings) are byte-identical to the old binary's (`../ab-coreml-tracker/run2/spiketrack`)
     in every round.
   - Fallback, used only if gate 3 failed: `detect_track_parity`
     `session_matches_upstream_on_metal` on VISEM 22 passes as it did at E22 (32/32 frames
     id-equal, mean mask IoU >= 0.9985), plus `tracker_parity`.
   - The summary says which gate was met. The Metal session parity runs either way.
7. **The cached path ran**: a `--profile ops` run of the new binary on 12 frames shows no
   `linear_bias_forward` over more than 5,184 memory rows inside memory attention. The old
   profile has 12 per full-depth frame.

## Speed and memory, fixed before the run

- Old and new binaries, 3 interleaved rounds (old, new / new, old / old, new), `--profile
  stages`, plus one unprofiled run of each.
- Read: the tracker step (`tracker propagate*` + `memory encode*`), steady frames >= 8, median
  per round; ratio new / old per round; noise = the larger same-binary round-to-round spread.
- Claim a speedup only if every round's ratio < 1 - noise.
- Expected, not a gate: about -35 ms per frame. The new frame's projections still cost one
  frame's worth (~6 ms), and the concats add a little.
- Memory: record `metal allocated` and the process footprint at frames 8 and 33 for both.
  Expected: about +300 MB (7 frames x 4 layers x K and V x 5,184 x 256 x 4 bytes ~ 297 MB at
  one tracker state).

## Result (run 2, 2026-11-03 03:37–04:05 UTC): production gate NOT MET as pre-stated

`job.log`, `summary.md`, `versions.txt`. The job exited 1 because predictions differ.

What the tree was: `38e0a35` with nothing tracked left uncommitted. That commit was made at
21:25:57 local time (03:25:57 UTC), during run 1, and it holds this change together with the other
agent's E22 host-residency, tiling and weight-edit work. This session did not make it. So the old
binary (`../ab-coreml-tracker/run2/spiketrack`, built 20:21 local) and the new one differ by more
than this change. The old binary's `versions.txt` records `7f8db18` with 30 dirty files, so much of
that work was already in the tree at 20:21, but which of it is not recorded.

| Gate | Result |
|---|---|
| 1 `fwd` unchanged (CPU Exact) | pass (`cache-cpu.log`) |
| 2 cached = uncached, CPU bitwise | pass |
| 3 same on Metal (synthetic, grid 8 = 64 tokens a frame) | pass (`cache-metal.log`) |
| 4 prune boundary | pass (`unit-tests.log`, 34 passed) |
| 5 lint | pass: clippy reported no findings at all (`clippy-findings.txt` empty) |
| 6 production predictions byte-identical | **FAIL**: every new run differs from old-control; every old run is identical to it |
| 6 session parity (recorded either way) | pass: 32 frames, 0 id-set misses, mean mask IoU 0.99853, the same as E22's 0.99853 |
| `tracker_parity` | pass, 8 tests |
| 7 the cached path ran | pass: no tracker linear over 5,184 rows in `new-ops` |

The difference in predictions (`new-control` vs `old-control`, 34 frames, 16 tracks):
- Frames 0 and 1 are identical. The first difference is at frame 2, the first frame whose memory
  holds two frames.
- Largest absolute difference in any box coordinate or score: 0.0057. No id-set differences.
- All four new runs are identical to each other.

Why gate 3 did not carry to production (inferred, not yet tested): the old path projects all
memory frames in one linear (10,368 rows at two frames), the new one projects 5,184 rows per frame.
At one memory frame both project 5,184 rows, and frame 1 is identical. That fits a Metal linear
whose bits for a row depend on the total row count at production sizes, which the 64-row
synthetic test does not reach. The other agent's changes in `38e0a35` are the other candidate. But
frame 1 is identical, and frame 1 runs the whole detector and tracker path with one memory frame,
so a change outside the multi-frame projection would have to leave all of that untouched.

## Speed and memory (run 2): no speedup claimed

| Round | Old tracker step | New | New / old |
|---|---|---|---|
| 1 | 1043.0 ms | 960.8 ms | 0.921 |
| 2 | 1179.2 ms | 929.4 ms | 0.788 |
| 3 | 1123.3 ms | 1001.8 ms | 0.892 |

- Same-binary spread: old 13.1%, new 7.8%. Noise is 13.1%, so the rule (every ratio < 0.869) is
  not met.
- The ratios are confounded too. The new binary carries whatever of `38e0a35` the old binary
  lacked, and the other agent's E22 work is described as latency optimization.
- The machine was about 2x slower than in the CoreML A/B. The same old binary took 3.84 s at frame
  8 here against 1.95 s in `../ab-coreml-tracker/run2/st-control-1.log`, with load averages of 5–8.
  The cause is not identified. The numbers are not comparable with the A/B's ~405 ms.
- Memory at frame 8: `metal allocated` 3,859 → 4,139 MiB (+280 MiB, against the expected
  ~300 MB). At frame 33: 3,950 → 4,230 MiB (+280 MiB). Footprint moved by +1,015 MiB at frame 8
  and +198 MiB at frame 33, which is too noisy to read.

## Row-count hypothesis: refuted (2026-11-03)

`cached_equals_uncached_on_metal_at_tracker_size` repeats gate 3 at the tracker's 72 x 72 grid
(5,184 tokens a frame): 1 frame; 2 frames (10,368 rows in one uncached projection); and 2 frames
with 16 pointers. All three are bitwise equal on Metal (`cache-metal-tracker-size.log`). The CPU
tests still pass, and the `fwd` golden is unchanged. So the per-frame projection does not move
bits at production size, and the prediction difference needs another explanation.

## Same-tree control (pre-stated 2026-11-03, before the run)

- `../../../../SpikeTrack-e23ctl` is `git archive 38e0a35` with `control_tree.py` applied.
  `compute_propagation` goes back to 7f8db18's concatenation and the uncached `fwd`, and nothing
  else changes (the `MaskMem` cache field stays, unfilled).
- `control-job.sh` builds it and runs one unprofiled control, then 3 interleaved profiled rounds
  against the E23 new binary (`ctl, new / new, ctl / ctl, new`).
- Reading:
  - Control predictions byte-identical to `new-control`: the cache is bit-neutral in production.
    The difference from the old binary comes from the rest of `38e0a35`, which the old binary
    (`7f8db18` + 30 dirty files) did not fully hold. The byte-identity gate is then met against
    the same-tree control. That is a gate revised after run 2, and it is recorded as such.
  - Byte-identical to `old-control`: the cache changes production predictions although every
    synthetic test is bitwise. Find it before anything else.
  - Neither: report as found.
- Speed: new / control per round, with the same rule (every ratio < 1 - noise). This is the
  clean speed comparison; run 2's old / new was confounded.

### Same-tree control: result (2026-11-03 04:13–04:30 UTC)

`control-job.log`, `control-summary.md`, `control-versions.txt`. Run 1 of this job
(`control-job-run1-no-weights.log`) ran nothing: the script did not set the weights path. The
script now refuses a run with no predictions instead of reporting it as "differs".

- **Predictions:** all 7 runs (control and new, profiled and not) are byte-identical to
  `new-control`, and all differ from `old-control`. So the cache is bit-neutral in production.
  Run 2's difference comes from the rest of `38e0a35` against what the old binary was built from,
  not from this change. The byte-identity gate is met against the same-tree control. That gate was
  revised after run 2 (above).
- **Speed:** SPEEDUP by the pre-stated rule. The one-tree confirmation below withdrew this claim.

  | Round | Control tracker step | New | New / control |
  |---|---|---|---|
  | 1 | 1191.0 ms | 955.1 ms | 0.802 |
  | 2 | 1229.1 ms | 955.3 ms | 0.777 |
  | 3 | 1157.6 ms | 966.5 ms | 0.835 |

  - Same-binary spread: control 6.2%, new 1.2%. Every ratio is below 0.938.
  - The machine was still about 2x slower than in the CoreML A/B (load 6–19). So the absolute
    times are not comparable with its ~405 ms, and the measured saving (~200–270 ms, ~20%) is
    larger than the ~35 ms (~9%) expected from the quiet-machine ops profile.
  - Unexplained, not studied: the 36,400-row projections may slow down more than the rest under
    contention.
  - A quiet-machine rerun is what would put a number on the saving at the A/B's load.
- **Memory:** +280 MiB `metal allocated` (run 2, above).

### Caveats on this result

- **Speed was provisional.** The new binary was built in `SpikeTrack/target` and the control in
  `SpikeTrack-e23ctl/target`. The one-tree confirmation below withdrew the claim; the paired
  run later measured −6.4% (see Paired speed run: result).
- **Lint after the geometry test:** `cargo clippy -p st-model --test memory_attention_cache -- -D warnings`
  is clean (`clippy-tests-after-geometry.log`). Gate 5's log predates that test change.
- This folder holds three ~12 MB binaries (`spiketrack`, `spiketrack-ctl`, `spiketrack-1t`); `spiketrack` and `spiketrack-1t` are byte-identical. Leave them out if the
  folder is committed.

## One-tree confirmation (pre-stated 2026-11-03, before the run)

- `one-tree-job.sh` (under the lock):
  - Restores `state.rs` in `../../../../SpikeTrack-e23ctl` from `38e0a35`.
  - Refuses unless every file under `crates/`, `Cargo.toml` and `Cargo.lock` matches `38e0a35`.
  - Builds into the same `target/` as `spiketrack-ctl`, and copies the result to `spiketrack-1t`.
  - Both arms then come from one tree and one target directory and differ only in the
    `compute_propagation` call.
- Runs:
  - One unprofiled run of `spiketrack-1t`.
  - Then 3 interleaved profiled rounds against `spiketrack-ctl` (`ctl, new / new, ctl / ctl, new`),
    named `1t-ctl-r*` and `1t-new-r*`.
- Predictions: every run must be byte-identical to `new-control`. A difference means the main
  tree's build is not what the one-tree build is, and the speed result is not read.
- Speed: same rule (every new / control ratio < 1 − noise, with noise the larger same-binary
  spread).
  - Met: the ~20% saving is confirmed on one tree.
  - Not met: the speed claim above is withdrawn and the result reported as found.
- Machine load is recorded at every step. As before, absolute times are not comparable with the
  CoreML A/B's.

### One-tree confirmation: result (2026-11-03 13:18–13:28 UTC): speed claim withdrawn

`one-tree-job.log`, `one-tree-summary.md`, `one-tree-versions.txt`.

- **Tree:** the control tree matched `38e0a35` in all 166 files under `crates/`, `Cargo.toml`
  and `Cargo.lock` before the build.
- **Build:** the one-tree build is byte-identical to the main tree's `spiketrack` (SHA-256 prefix
  `596ae93d57261abf` for both).
  - So the earlier same-tree control already compared these exact two binaries
    (`spiketrack-ctl` is `e69a0a02fcecb4b8` in both runs).
  - The cross-tree concern does not explain anything. The two runs differ in machine load, not in
    the binaries.
- **Predictions:** all 7 runs are byte-identical to `new-control`. The cache is bit-neutral, a
  second time.
- **Speed:** not met, so the ~20% claim is withdrawn.

  | Round | Control tracker step | New | New / control |
  |---|---|---|---|
  | 1 | 435.2 ms | 489.0 ms | 1.124 |
  | 2 | 527.1 ms | 488.1 ms | 0.926 |
  | 3 | 459.1 ms | 420.9 ms | 0.917 |

  - Same-binary spread: control 21.1%, new 16.2%, so noise 21.1%. One round has the new binary
    slower.
  - Load was 3.7 at the start and rose to 11.3 during the rounds.
  - Absolute times were about 2.3x lower than in the same-tree control (load 6–19 then). They are
    near the CoreML A/B's ~405 ms.
  - At this noise, the expected ~35 ms (~9%) saving cannot be resolved either way. The mean of
    round means is 473.8 ms for the control and 466.0 ms for the new binary (−1.6%). That is not
    a claim.
- **Reading:**
  - The ~20% saving was measured on these same two binaries under heavy contention (load 6–19).
    It did not reproduce at lower load.
  - Unverified: whether the cache saves more under contention, or the earlier gap came from load
    changing between rounds.
  - A speed number needs a quiet machine (load near 0) and more rounds than 3.
- **Memory:** unchanged, +280 MiB `metal allocated` (run 2).

## What the existing logs already show (read 2026-11-03, no new runs)

From the per-frame lines of all 22 `track` logs (frames ≥ 8):
- **Dispatches:** every control-path run (`old-*`, `ctl-*`, `1t-ctl-*`) dispatches 2,960 a frame.
  Every cached-path run (`new-*`, `cnew-*`, `1t-new-*`) dispatches 3,058. The cached path ran in
  every run, and it issues 98 more dispatches a frame (the concatenations), not fewer.
- **Memory:**
  - Same-tree, unprofiled: `metal allocated` at frame 33 is 3,950 MiB for the control
    (`ctl-control`) and 4,230 MiB for the cache (`1t-new-control`), so +280 MiB, as run 2 found.
  - The frame peak falls by about 250 MiB (6,236–6,237 to 5,987–5,997 MiB) in every pair. The
    uncached path's transient 36,400-row projections no longer exist.
- **Swapping:** it does not explain last night's slow control rounds.
  - Steady-frame swap-ins were 97 / 100 / 315 for the control (`ctl-r*`) against
    140 / 192 / 163 for the cache (`cnew-r*`).
  - Page-outs were 8 / 43 / 0 against 0 / 20 / 2.
  - Neither arm swapped consistently more.
- **Lock:** `mac_heavy` serialises the heavy jobs, so no two `spiketrack` runs overlap under it.
  The load in these runs came from processes outside the lock.

## Paired speed run (pre-stated 2026-11-03, before the run): `paired/`

Why: 3 rounds against a max/min spread rule could not resolve a ~9% effect at 16–21% noise.
Also, a max/min spread over more rounds only grows, so that rule cannot be reused at larger n.

- **Binaries:** `spiketrack-ctl` (A, `e69a0a02fcecb4b8`) and `spiketrack-1t` (B, `596ae93d57261abf`),
  one tree, one target directory. The job refuses to run if either hash differs. Nothing is built.
- **Mechanism, run first:** `--profile ops` on 20 frames, in the order A, B, B, A.
  - Read the `tracker propagate, state 0: memory attention (device)` stage, in ms per frame
    (frames 1–19).
  - Read, in A, the summed ms of the `linear_bias_forward` calls over more than 5,184 rows in
    that stage. In B that sum must be 0.
  - A's big-linear time is the most the cache can remove. The stage difference, A − B, is what
    it removes net of the concatenations.
  - Each op is synced, so this is a mechanism measurement, not a wall-clock claim.
- **Wall clock:** 16 pairs of 34-frame `--profile stages` runs, in this fixed order:
  `AB AA BA BB AB BA AA AB BB BA AB AA BA BB AB BA`.
  - That is 10 A/B pairs (5 each order) and 6 same-binary pairs (3 A/A, 3 B/B).
  - Before each pair the job waits, for at most 10 minutes, for the 1-minute load to fall to 5 or
    below. A pair that starts above 5 is flagged.
  - The load is sampled every 5 s during every run (`*.load`).
- **Metric:** a run's median tracker step (`tracker propagate*` + `memory encode*`, frames ≥ 8),
  as before. A pair's value is ln(B/A) for an A/B pair, and ln(second/first) for a same-binary
  pair.
- **Decision**, on all 16 pairs; the same analysis without the flagged pairs is reported beside it:
  - Valid only if the same-binary pairs' 95% t-interval on the mean includes 0, so pair position
    has no bias. Otherwise: invalid, reported as such.
  - Speedup if the A/B pairs' 95% t-interval on the mean ln(B/A) lies wholly below 0, and at least
    8 of 10 pairs are below 0.
  - Slowdown: the mirror image.
  - Otherwise no difference resolved. The interval is then reported as the bound on the effect.
- **Descriptive, not a gate:** each A/B pair's ratio against its mean sampled load (Spearman).
  Contention is not induced on purpose (machine-safety rules), so whether the cache saves more
  under load is read only from natural variation.
- **Predictions:** every 34-frame run must be byte-identical to `new-control`. The 20-frame ops
  runs must be byte-identical to each other.

### Paired speed run: result (2026-11-03 15:27–16:34 UTC): SPEEDUP, −6.4%

`paired-job.log`, `paired/summary.md`, `paired/versions.txt`, per-run `paired/*.log` and `*.load`.

- **Binaries:** the hashes matched (`e69a0a02fcecb4b8`, `596ae93d57261abf`). Nothing was built.
- **Predictions:** all 32 34-frame runs are byte-identical to `new-control`. The 3 later ops runs
  are byte-identical to `ops-1-ctl`. The cache is bit-neutral, a third time.
- **Mechanism** (`--profile ops`, each op synced, frames 1–19):

  | Run | Memory attention ms / frame | Linears over 5,184 rows ms / frame |
  |---|---|---|
  | `ops-1-ctl` (A) | 298.8 | 28.3 |
  | `ops-2-new` (B) | 277.1 | 0.0 |
  | `ops-3-new` (B) | 272.0 | 0.0 |
  | `ops-4-ctl` (A) | 299.9 | 28.3 |

  - In B the big-linear sum is 0, so the cached path ran.
  - The most the cache can remove is A's 28.3 ms a frame. It removes 24.8 ms net (A − B), so the
    concatenations cost about 3.5 ms a frame.
- **Load:** pairs 1–15 started at a 1-minute load of 4.3–5.0. Pair 16 was flagged (load 8.4 after
  the 10-minute wait). Mean sampled load per pair was 3.6–8.8.
- **Same-binary pairs** (validity): mean ln ratio −0.0037, 95% CI [−0.0334, +0.0260], which
  includes 0. Pair position carries no detectable bias, so the run is valid.
- **A/B pairs** (decides, all 10): mean ln(B/A) −0.0663, 95% CI [−0.0973, −0.0354], so B/A is
  −6.4% (−9.3% to −3.5%). 9 of 10 pairs are below 0 (8 needed). **Verdict: SPEEDUP.**
  - Without the flagged pair (9 A/B pairs): −6.5% (−9.7% to −3.2%), 8 of 9 below 0 (7 needed).
    Also SPEEDUP.
  - In ms: the A/B pairs' median tracker steps average 429.0 ms for A and 401.6 ms for B, a
    27.4 ms difference. That is consistent with the 24.8 ms the mechanism runs say the cache
    removes net. Those runs average frames 1–19, including frames before the memory bank is full,
    so 24.8 ms is a lower bound on the steady-state removal (frames ≥ 8).
  - Descriptive, not pre-stated: by order, the 5 A-first pairs average ln(B/A) −0.053 and the 5
    B-first pairs −0.079. Both are negative, so the saving does not depend on which binary ran
    first.
- **Load, descriptive only:** Spearman ρ of the A/B ln ratio against mean load is +0.59 (n = 10).
  The saving is smaller, not larger, in the more loaded pairs. That is against the idea that the
  cache saves more under contention. With n = 10 and load not controlled, it is not a finding.
- **What this settles:**
  - The cache makes the tracker step about 6% faster (27 ms a frame at 16 objects, 34 frames),
    on one tree, at moderate load, with a pre-stated paired test.
  - The earlier ~20% (heavy load) and −1.6% (3 rounds) readings are consistent with noise around
    this value (inferred, not tested).
  - The pre-run expectation of ~35 ms (~9%) was high. The net removal is 24.8 ms, because the
    concatenations cost back about 3.5 ms a frame.
  - Memory, from the same-tree logs: +280 MiB `metal allocated`, about −250 MiB frame peak.
- **Caveat:** `mac_heavy` stopped one leftover process (pid 17750) when the job exited. Inferred:
  the orphaned `sleep 5` of the last run's load sampler, since `kill` stops the sampler subshell
  but not its `sleep` child. The process was gone before it could be checked.
