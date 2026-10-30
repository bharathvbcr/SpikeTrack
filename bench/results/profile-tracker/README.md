# E13: each tracker state's propagation, split (2026-10-30)

The run is pre-stated in docs/SPEED_PLAN.md (E13), and its setup is E11's:
- Metal, the default CoreML fp32 trunk.
- Prompt "sperm", r16 LoRA, `--boxes detector`, `--max-objects 64`.
- 16 frames of VISEM 82 (two tracker states, 16 and 2 objects) and 22 (one state).
- `--profile ops` on 8 frames of 82.

The binary is 6ba37d6 with a clean tree. The control is the E12 B binary (5e13d74), not rebuilt;
both shas are in `versions.txt`. `job.sh` ran it in one mac_heavy hold, with load swinging 11-41
(recorded per run).

**Validity [V].**
- The stages runs sit within the 10% rule:
  - 82: 2498 against controls of 2879 and 2463 (-6.4% of their mean).
  - 22: 1960 against 1980 and 2353 (-9.6%, at the edge; the controls themselves differ by 19%).
- Predictions are byte-identical to control-1 for the E13 binary, with and without the marker, on
  both videos.
- Tests: spiketrack 75 and st-model 46 passed, and the Metal RPB test passed.

## Tracker phases, steady frames (1-15), ms per frame, `--profile stages`

| phase | 82 state 0 (16 objects) | 82 state 1 (2 objects) | 22 state 0 (12 objects) |
|---|---|---|---|
| memory + pointer tokens (host) + upload | 63.6 | 50.1 | 55.5 |
| memory attention (device) | 249.5 | 203.1 | 246.0 |
| multiplex decoder | 22.5 | 19.2 | 21.4 |
| mask readback + per-object pick + pointers (host) | 2.5 | 1.7 | 1.7 |
| state total | 338.1 | 274.1 | 324.6 |

The op time per steady frame, from the ops run on 82 (frames 1-7, `tools/analysis/profile_ops_by_kind.py`):

| phase | state 0 op ms | state 1 op ms |
|---|---|---|
| tokens + upload | 6.9 (uploads) | 2.6 |
| memory attention | 222.2 | 133.7 |
| multiplex decoder | 28.2 | 17.2 |

## Reading [V for the numbers, I where marked]

- **Memory attention is the tracker.** It is 74% of each state's time. It does not shrink with the
  object count:
  - 16 objects take 250 ms and 2 objects 203 ms, because the 16 multiplex slots go through it
    together.
  - What differs between the states is the memory length [I]: state 1 was born later, so it holds
    fewer memory frames. In the ops run, state 0's longest cross-attention has 36,400 keys and
    state 1's 26,000.
  - So the per-state fixed cost the rule looks for is memory attention. Every state pays it in
    full.
- **Its ops are Ojas kernels**, about 1,100 dispatches per state-frame across 7 frames. Measured
  per call:
  - Cross-attention of 5184 queries over 36,400 keys: 42.6 ms, about 4.5 TFLOP/s.
  - Self-attention over 5184 tokens: 6.7 ms, about 4.1 TFLOP/s.
  - The `[5184, 256] x [256, 256]` projections: 0.57 ms each, about 1.2 TFLOP/s, 28 per
    state-frame.
  - The FFN `[5184, 256] x [2048, 256]`: 3.3 ms, about 1.6 TFLOP/s.
- **Host phases, against the pre-stated rule (100 ms per state):** tokens plus readback come to
  66 ms on state 0 and 52 on state 1. That is under 100, so by the rule this is not an E12-shape
  SpikeTrack target.
  - It is still about 100 ms of host work per frame on 82 across the two states, and 56 on 22: the
    tokens phase minus its upload ops.
  - It rebuilds, every frame and for every state, the token-major copies of up to 7 stored
    memories and their image features, then uploads them [V, read from
    `TrackerState::compute_propagation`]. The memories, the features and the 7 possible position
    rows (sine position + `maskmem_tpos[t]`) do not change once stored [I, read from the code].
  - Keeping them on the device per stored frame and concatenating there would be an exact copy.
- **Device time minus op time, per state [I: the windows differ].** The stages table covers frames
  1-15 and the ops run frames 1-7. On 82, frames 8-15 run about 300 ms slower than 1-7 as the
  memory fills.
  - State 0: 338 wall against 257 ops, so about 80 ms.
  - State 1: 274 against 154, so about 120 ms. That is inflated by the window, since state 1's
    memory grows most after frame 7.
- **Pipeline gate (E14), same window [I].**
  - On frames 1-7 of 82 the stages run takes 2337 ms per frame, and the ops run, op-synced, takes
    2333. So the per-op sync costs little.
  - Wall minus op time over the non-trunk frame is 2337 - ~911 (trunk) - 1172 (non-trunk ops), so
    about 254 ms.
  - That falls between the pre-stated lines (>= 400 build, < 250 shelve), at the shelve edge.
  - Overlapping frame t+1's trunk could hide at most ~250 ms of GPU idle on a 2.3-2.6 s frame.

Files: `v*-stages/*.profile-stages.md`, `v82-ops/82.profile-ops.md`, the run logs (per-frame
telemetry), `job.sh` and `versions.txt`. Masks are not committed.
