# Tracker step against the CoreML port at 16 objects (pre-stated 2026-11-02, run 2026-11-03)

Each run writes its own folder (`AB_RUN=run2 tools/locked.sh ab-coreml bash
tools/jobs/ab_coreml_tracker.sh`); run 1's outputs are in `run1/`. Run 2 repeats run 1 unchanged,
same rule and threshold, because run 1's verdict was not read.

## Result: NOT MET against either port precision (run 2, 02:20-02:28 UTC, `run2/`)

Read under the pre-stated rule [V]. Every precondition held:
- Stage runs were -6.5%, -7.1% and -0.6% against the controls.
- 16 active objects on every steady frame, one tracker state, no births on steady frames.
- Every SpikeTrack run was byte-identical to control 1.

| round | SpikeTrack step | port fp32 | port fp16 | ours / fp32 | ours / fp16 |
|---|---|---|---|---|---|
| 1 | 407.8 ms | 204.0 | 70.6 | 2.00 | 5.78 |
| 2 | 403.2 | 206.0 | 68.6 | 1.96 | 5.88 |
| 3 | 425.0 | 201.2 | 82.8 | 2.11 | 5.14 |

Noise: SpikeTrack 5.4%, port fp32 2.4%, port fp16 20.6%. Every round is beyond the noise against
both arms. At equal precision our tracker step is about 2x the port's; the rest of the ~5.5x
fp16 gap is precision (the port's fp32 step is ~2.9x its fp16 step) [I for that split].

Where our step goes, median ms per steady frame (`run2/st-r*/82.profile-stages.md`):

| sub-stage | r1 | r2 | r3 |
|---|---|---|---|
| tracker propagate: memory attention (device) | 285.4 | 284.0 | 296.8 |
| memory encode: encoder (device) + feature readback | 56.4 | 55.5 | 60.2 |
| tracker propagate: multiplex decoder | 14.6 | 14.1 | 14.8 |
| memory encode: mask upload | 13.4 | 13.1 | 13.8 |
| tracker propagate: memory + pointer tokens (host) + upload | 11.4 | 10.9 | 11.5 |
| everything else (readbacks, area rule, host fills) | ~25 | ~24 | ~27 |

Memory attention alone (~285 ms) is longer than the port's whole fp32 step (~204 ms). The E22
flash cross-attention bench was 5184 queries x 5200 keys; at 7 memory frames this step attends
up to ~36,400 keys, so whether that path dispatches here, and what else memory attention spends,
is not measured [I].

Builds: SpikeTrack 7f8db18 with 30 dirty files (binary 061fc4326821803d; another session's
edits moved the count from run 1's 27), ojas 286a444 clean, tessl 52bdd19 clean, coremltools 9.0.
Run 2's job also exited 0 on a read verdict, as it should.

## Where memory attention goes (ops profile, 2026-11-03 02:35 UTC, `run2-ops/`)

`ops-job.sh`: run 2's binary, the same settings, 12 frames, `--profile ops` (every op synced, so
the per-op times sum to ~339 ms where the unsynced stage was ~285). Frames 7-11 attend the full
memory (36,400-36,464 keys; 5 frames); ops on the current frame's 5,184 tokens are averaged over
frames 1-11. Per full-depth frame [V for the times]:

| ms | what (4 layers) |
|---|---|
| 194.9 | cross-attention `[1,8,5184,32] x [1,8,~36400,32]`, 4 calls of ~49 ms |
| 38.7 | `img_k`, `ca_k`, `ca_v` projections over the ~36,400 memory tokens, 12 calls |
| 31.4 | the other linears on 5,184 tokens (self-attention q/k/v/out, `img_q`, `ca_q`, `ca_out`, FFN) |
| 23.2 | self-attention `[1,8,5184,32]`, 4 calls (tessl `encoder_attn`) |
| 18.9 | permutes, residual adds, RoPE and slices on the 36,400-token tensors |
| ~32 | GELU, layer norms, the 5,184-token permutes, adds, RoPE, concat, fill |

- **Kernel [I]:** at 7 memory frames this call should take the flash kernel (`ojas_attn_fwd_d32`).
  - The routing source admits it: no bias or lengths, head dim 32, 1,296 threadgroups.
  - ojas `bench/results/2026-11-01-sam-xattn-route` reports the same: one or two memory frames
    route to tessl's pad-q `encoder_attn`, three or more to flash.
  - Its evidence is a test at the smallest flash-regime shape (3 x 5184 + 32 keys,
    `ojas-metal/tests/attention_flash.rs`). Its log is not in that folder, and this run did not
    re-check the kernel at 36,400 keys.
  - The flash-vs-split-K timing (that folder's Q2) was pre-registered and never ran.
- **Throughput:** 49 ms per call is 4 TFLOP/s on the useful work (8 x 5184 x 36416 x 64 x 2 =
  193 GFLOP). The flash bench's f32 GEMM control row is 6.3 TFLOP/s [V, `attention_flash_bench.rs`
  `CONTROL_TFLOPS`].
- **Scaling:** across this profile's depths the cost grows roughly linearly in keys, over two
  kernels: pad-q at 5,200 and 10,400 keys, flash from 15,600 up.
- **Only earlier split-K number:** ojas's 2026-10-27 split-K bench measured 81-94 ms at 36,320
  keys (`2026-10-27-attention-stage2`). That is another day and other conditions, not a
  comparison, but it does not suggest flash is the slower kernel here.
- **E22's 34.5x** was flash against the scalar kernel, not against split-K.

Levers, largest first [I: none is measured]:
1. **The cross-attention kernel (~195 ms):** at the GEMM control's 6.3 TFLOP/s it would be
   ~120 ms. Faster still needs lower precision: the kernel has a bf16 accumulation mode, and
   half-precision K/V operands would be a kernel change. Either needs a parity gate, the trade
   that fp16 and bf16 lost on the trunk.
2. **Cache the stored frames' projections (~45 ms):** `img_k(mi) + ca_k(mem)` and `ca_v(mem)` for
   a stored frame depend only on that frame and the layer. Positions are added after the
   projection and RoPE after that, so 6 of 7 frames' projections (and their permutes) could be
   kept per layer instead of recomputed each frame (~300 MB at 7 frames x 4 layers x K and V,
   f32). This is a SpikeTrack change in `memory_attention.rs`, with no kernel work. The port
   recomputes them too. Bit-identity depends on the Metal GEMM giving a row the same bits at any
   M, which is not established; the tracker and session parity gates would decide.
3. **Fusing the per-layer small ops (~30-50 ms):** q/k/v as one 768-wide linear, and the
   elementwise chains on the 36k-token tensors.

Outside memory attention, the step's other ~120 ms is the memory encoder (~56 ms) and the rest
of the host work and copies (~60 ms).

## Result: verdict not read (run 1, 01:20-01:29 UTC, `run1/`)

The pre-stated rule leaves this run unread. Round 2's stage run was +10.4% against the controls,
past the 10% precondition. Rounds 1 and 3 were +9.7% and +9.6%. The threshold is not moved after
the fact; a rerun is the way to get a read verdict. Numbers below are recorded, not claimed [V for
the measurements, not for the verdict]:

- Preconditions that held: 16 active objects on every steady frame of every round, one tracker
  state, no births on steady frames, and every SpikeTrack run byte-identical to control 1.
- SpikeTrack tracker step, steady median: 523 / 520 / 513 ms (spread 1.9%).
- Port fp32 step: 216 / 274 / 276 ms. Port fp16 step: 80 / 99 / 104 ms. Both port arms ran ~25%
  slower in rounds 2-3 than round 1 at similar load (7.4-10.5); the cause is not recorded. Within
  a round the port's step cost is flat across bank depth (77-88 ms in round 1).
- Ratios, ours / port: vs fp32 2.42 / 1.89 / 1.86, vs fp16 6.55 / 5.23 / 4.94.
- The port's fp32 step is 2.6-2.8x its fp16 step, so precision accounts for about that much of the
  fp16 gap [I].
- Backbone (context only): ours (CoreML fp32 trunk + Ojas propagation neck) 1257-1296 ms; the
  port's fp16 backbone 323-438 ms.
- Builds: SpikeTrack 7f8db18 with 27 dirty files, binary 90a5437ef4f062f5. `versions.txt`'s ojas
  line is empty because the job pointed at the removed `../ojas-st`; SpikeTrack builds against
  `../ojas`, which was at 286a444, clean, when this session started (read after the run). The job
  now records `../ojas`.
- The job exited 0 although the verdict was not read. `tools/analysis/ab_coreml_tracker.py` now
  exits non-zero when the verdict is not read or the fp32 arm is missing.

The speed target restated on 2026-10-27 (HANDOFF.md) is parity-or-better with the CoreML port
(`sdkv2/sam3.1-coreml` @ 9668c49) on the tracker path at 16 objects. The "~8x slower per frame"
in SUMMARY.md compares our whole frame (trunk, text, detector, tracker, births) with the port's
backbone + `track_step`, which has no detector or text encoder. It was never measured on the
same work under the same load. This run measures only the tracker.

## What each arm times

Both arms start from features that are already computed and end with full-resolution masks and
an updated memory on the host.

| | CoreML port `track_step` | SpikeTrack |
|---|---|---|
| Work | memory attention, multiplex mask decoder, memory encoder, object pointer bank update [I: inferred from its inputs, outputs and state banks, not observed] | stages `tracker propagate, state 0[...]` + `memory encode[...]` |
| Objects | dense multiplex 16 | `--max-objects 16` on VISEM 82 (17-19 detections from frame 2, so the cap binds and one state holds 16) |
| Precision | fp16 (deployed) and fp32 (the port's reference package) | f32 (bf16 failed the D2/D3 gates) |
| Inputs | random tensors of the spec's shapes, one state over 34 calls | real frames, `track` as shipped (prompt "sperm", r16 LoRA, `--boxes detector`) |
| Pays and the other doesn't | numpy -> MLMultiArray conversion of ~27 MB inputs, 16 x 1008 x 1008 fp16 masks out | its own low-res mask and score readbacks, the host area rule |
| Timer | wall clock per `predict` call | wall clock per stage, Metal synced at each stage boundary |
| Memory attended | 1 conditioning frame (the port does not wire more) + 6 recent, 15 pointers | up to 4 conditioning frames + recent frames, 7 in all; 16 pointers |

Left out of the comparison: the trunk, the propagation neck (both in the port's backbone,
reported as context only), the detector, text, association, births, recondition and the collector.
SpikeTrack may attend more conditioning frames than the port. That makes our step do more work,
so it biases against parity, not toward it [I].

## Steady state

The memory bank holds 7 frames from frame 7 on (`NUM_MASKMEM`, `crates/st-model/src/tracker/vos.rs`).
On VISEM 82, frames 8-15 ran ~300 ms slower than frames 1-7 (`profile-tracker/README.md`). Only frames / steps >= 8 count: SpikeTrack runs 34 frames (8..33, 26
samples) and the port 2 untimed + 32 timed calls (steps 8..33, 26 samples). Every port step is
kept, so the summary also shows whether its dense graph costs the same at every bank depth.

## Schedule (one `tools/locked.sh` hold)

1. Build, then `track_profile` unit tests (the per-frame tsv block this run reads).
2. Unprofiled SpikeTrack control 1.
3. Three rounds, order rotated: (fp16, fp32, SpikeTrack), (SpikeTrack, fp16, fp32),
   (fp32, SpikeTrack, fp16). SpikeTrack runs use `--profile stages`.
4. Unprofiled SpikeTrack control 2.
5. Every SpikeTrack run's predictions must be byte-identical to control 1.

Load and GPU utilization are logged before each step. The machine is shared: nothing is gated idle.

## Decision rule, fixed before the run

- **Precondition:** each stage run's in-Session time per steady frame is within 10% of the two
  controls' mean push_frame time (E11's rule). If not, the stage times are not read.
- **Precondition:** every steady frame of every stage run holds 16 active objects (the `track`
  log's "K active track(s)"). A removed object's slot is never reused, so the count can drop;
  if it does, that run is not the port's 16-object step and the verdict is not read.
- **Precondition:** every steady frame carries `tracker propagate`, `memory encode`, `trunk` and
  `propagation neck` rows, and no `state 1` row. A renamed stage stops the summary rather than
  dropping out of the sum.
- **Ratio per round:** SpikeTrack's median tracker step / the port's median step, steady only.
- **Noise:** the larger of the two arms' round-to-round spread of medians (max / min - 1).
- **Verdict, per port precision:** MET if every round's ratio <= 1 + noise; NOT MET if every
  round's ratio > 1 + noise; otherwise INCONCLUSIVE.
- The fp32 port arm is the precision-matched verdict. If its package is absent, that verdict is
  reported as not run, never as passed. The fp16 verdict alone does not separate kernel quality
  from precision.

What a result would mean: NOT MET against fp32 means the tracker kernels, readbacks or host
work are slower at equal precision. MET against fp32 but NOT MET against fp16 means the gap is
mostly precision.

## Files

- `tools/jobs/ab_coreml_tracker.sh` (the job), `tools/analysis/ab_coreml_tracker.py` (summary)
- `tools/baselines/coreml_bench.py --only/--tracker-precision/--out` (the port arm)
- `st-control-{1,2}/`, `st-r{1,2,3}/` (`82.profile-stages.md` carries a per-frame tsv block),
  `coreml-fp{16,32}-r{1,2,3}.json`, `versions.txt`, `summary.md`
