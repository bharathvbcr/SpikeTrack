# Speed plan (2026-10-28)

Goal: close the gap to the CoreML port (fp16, tracker only) on the M5 Pro without giving up the
exact path. The f32 path stays the default and the parity reference. Faster numerics ship as a
`--fast` tier, gated on D3 kept-sets, VISEM 22 session IoU and held-out HOTA. Every speed number
is paired, interleaved, min-of-N, with the GPU utilization from other sessions recorded.

Labels: [V] measured or read in code, [I] inferred, [U] unverified.

## Where a frame goes (profile-cbd5565, the pin before 8b88328; refresh in `speed-baseline/`)

One tracker frame at F = 7 memory frames, every op synced: 2380 ms over 961 calls [V].

| op class | calls | ms | share |
|---|---|---|---|
| linear_bias_forward (all f32) | 276 | 1198 | 50% |
| attention_forward | 47 | 553 | 23% |
| permute | 255 | 144 | 6% |
| conv2d_forward | 17 | 134 | 6% |
| gelu_erf_forward | 45 | 105 | 4% |
| residual_add + layer_norm + rope | 281 | 152 | 6% |

The trunk alone is 1430 ms of it [V]. Its matmuls are about 4.6 TFLOP per frame [I].

## The machine

- M5 Pro: 20 GPU cores, each with a matrix unit ("neural accelerator") that only TensorOps / MPP
  reach; 307 GB/s; 32 KB threadgroup memory [V, Apple].
- Exact f32 runs on the SIMD ALUs. tessl's exact-f32 NT GEMM is 6.3 TFLOP/s at our shapes, its
  ceiling [V, ojas `bench/results/2026-10-27-linear-gemm`].
- bf16 NT at our shapes reaches 16-19 TFLOP/s [V]; tessl's bf16 NN reaches 24.8-26.6 TFLOP/s at
  4096³ and mlp shapes [V, tessl `docs/benchmarking.md`]. Community data on base M5 puts fp16 and
  bf16 at the same rate, int8 at 2x [C]. The 20-core peak is unmeasured, plausibly ~30 [I].
- tessl emits a device barrier after every dispatch by default (`ab_flags::hazard_barriers`,
  off) [V], so independent ops never overlap on the GPU.
- The ANE is not worth chasing for a ViT-L at 1008: community reports GPU 4.6x faster than the
  ANE on M5 Max and allocation failures at >= 768 px [C].

## What CoreML does that we do not

fp16 everywhere on the matrix units [I: 4.6 TFLOP in 0.272 s is ~17 TFLOP/s]. Its output matches
upstream to cosine 0.99995 [V], so it is not exact either.

## Experiments (each with the result that would refute it)

| id | change | repo | prediction | refuted if |
|---|---|---|---|---|
| E0 | baseline: occupancy, profile f32 and bf16, CoreML CPU_AND_GPU / ALL / CPU_AND_NE | SpikeTrack | — | — |
| E1 | count Autocast's per-call input/output rounding in the bf16 profile | ojas (read) | >= 100 ms per trunk pass | < 30 ms: the bf16 gap is elsewhere |
| E2 | one fused qkv linear `[3072, 1024]` instead of three | SpikeTrack | -10% f32, -30% bf16 on those calls | < 3% |
| E3 | bf16 GEMM at our shapes: NN on pre-transposed weights vs NT, tile and split-K sweep | tessl | >= 1.3x on fc1 | < 1.1x: tessl's bf16 is at its ceiling |
| E4 | bias (+GELU) fused in `gemm_epilogue` (NN only: pre-transpose weights at load) | ojas | -8% f32 fc1, -20% bf16 fc1 | < 3% |
| E5 | hazard-tracked barriers; detector and tracker as independent work; frame t+1 trunk under frame t tracker | ojas/tessl | >= 1.2x per frame if steady-state GPU occupancy < 70% | occupancy > 90%: skip |
| E6 | permute and elementwise passes at bandwidth (permute measures ~70 GB/s of 307); fold head split into attention strides | ojas/tessl | permute >= 3x faster, most of 255 calls gone | < 1.5x |
| E7 | 672 input (48x48 tokens, 2304 vs 5184): VISEM is 640x480 | SpikeTrack | trunk >= 2x faster; held-out HOTA >= 0.47 at 16 and 64 objects | HOTA < 0.47 |
| E8 | memory cross-attention at F = 7 on 8b88328 | ojas | read the fresh profile first | — |
| E9 | bf16 trunk as `--fast` once E1-E4 land | SpikeTrack | trunk <= 500 ms | D3 or session gate fails |
| E10 | the ViT trunk as a CoreML ML Program (`--trunk-precision coreml-fp32 / coreml-fp16`); necks, detector and tracker stay on Ojas | SpikeTrack | fp32: frame <= 0.90x of the ea3cf39 main with the same tracks; fp16: frame <= 0.70x | fp32: frame > 0.95x, or any stage's rel L2 >= 1e-4, or VISEM 22 ids differ from the Ojas trunk on 8 frames. fp16 as `--fast`: held-out HOTA below 0.47 at cap 64 or 16, or more than 0.005 below the f32 run at the same cap (0.4868 / 0.4695) |

| E11 | profile the production frame with the CoreML fp32 trunk (`track --profile stages\|ops`), VISEM 22 (one tracker state) and 82 (two), 16 frames, cap 64 | SpikeTrack | names the next lever: trunk >= 40% of the steady frame means the trunk is still it (fp16 is refuted, so not "use fp16"); tracker states >= 40% on 82 means the tracker is | the stage table is read literally only if its in-Session sum is within 10% of the unprofiled controls' push_frame time; predictions with the marker must be byte-identical to the controls' |

## Results so far (2026-10-28, ojas 8b88328, tessl c9adc87; contended machine, load recorded)

- E0 [V]: CoreML backbone 0.2715 s on CPU_AND_GPU, 0.2754 s on ALL (stays on the GPU), 1.41 s on
  CPU_AND_NE; the stateful tracker refuses the ANE (`speed-baseline/coreml-*.json`, 2ab09c4).
  Our 16-frame LoRA track: 3.21 s/frame, GPU busy 86% (about 17% of that is other sessions),
  0.50 CPU cores. Profile f32, unprofiled pass 3: tracker frames 2210 / 2241 / 2492 ms at
  F = 1 / 3 / 7; ~9,100 dispatches and ~97 waited commits per frame (7ce7a71).
- E1 [V]: the bf16 lane dispatches (fc1: 9 dispatches per call against f32's 5) and forces no waits.
  bf16 trunk linears: 946 -> 687 ms profiled, frames 2205 / 2295 / 2425 ms, no end-to-end gain.
  Whole-model Autocast is slower than f32 (2320 / 2450 / 2759 ms); every op it touches slows,
  gelu_erf 94 -> 343 ms. The cost is the passes around the GEMM inside a frame (5b1c0cd).
- E2: dropped on reading. RoPE and attention take contiguous q/k/v, so splitting a fused output
  costs a full extra copy, which cancels the saving.
- E3 [V, refuted]: at our shapes NT bf16 already runs 22-25 TFLOP/s; NN, either tile and f16 are
  within noise of it; exact f32 is 5.2-6.2 (tessl daff7db, `sam` lane). The GEMM is not the
  bottleneck, and the bf16 trunk GEMMs would be ~200 ms per frame.
- E4: the NN bias epilogue costs 5-18% over the plain GEMM and still beats GEMM + separate bias
  add; held until the in-frame pass cost is split (ojas-78, O1).
- E4b [V kernel, I frame]: tessl `gemm_nt_bf16_bias(a, b, c, bias, round_bf16)` on branch
  `gp/w34-nt-bias-round` (f535dd3, not on main). The epilogue adds the bias and rounds the output
  to bf16 in registers. Its output is bit-identical to gemm_nt_bf16 + bias_add + the two casts
  (`tests/gemm_nt_bf16_bias.rs`); a deliberate break makes 4 of 5 tests fail. Timings: one bench
  at M=5184, 6 rounds x 8 calls, load ~4; bias + round, separate steps -> fused:
  q/k/v/proj 742 -> 473 us, fc1 3567 -> 2191 us, fc2 2083 -> 1897 us. Against the bare GEMM the
  epilogue costs 0.86x (bias) and 0.76x (bias + round) at q/k/v/proj.
  The ~85 ms per trunk pass this implies overstates ojas's saving. The bench's separate-steps arm
  is four dispatches (gemm, bias_add, and two cast passes; bench_gemm_routes.rs:847-853). ojas's
  emit is one in-register `ojas_round_bf16`. Fable puts the ojas-side saving at ~2.5-3% of a frame.
  **Fable's ruling: parked.** f535dd3 stays on its branch, off tessl main. Its base daff7db is off
  main too, and its bench exited 3 on the qkv-fused self-spread (11.1%).
  It reopens once the epilogue is within spread of the bare GEMM. Then it needs an ojas-side
  interleaved A/B: linear_bias_forward + emit against fused, at q/k/v/proj, fc1 and fc2, with
  n >= 6, self-spread reported, load < 5, committed. Wiring needs >= 5% of a whole frame, outside
  the A-vs-A spread. Parity would be checked against ojas's own broadcast add + `ojas_round_bf16`.
  Round-on-store needs a signal from the Autocast seam: a Backend change, an owner gate, via O1.
- E5: deferred: GPU 86% busy leaves under the 1.2x bar.
- E6: permutes are 87 ms of a 2492 ms frame (3.5%), under ojas-78's 5% rule; parked.
- Waits [V]: ojas's allocated-between-waits cap (min(1 GiB, budget/4)) forces 231 of f32's 387
  waits per 4-frame pass (1d2f906). Raising it in a throwaway build against ojas main
  (`set_mem_cap`) makes every mode ~20-28% faster at 8 GiB (a9e9407); the footprint sweep puts the
  knee at 4 GiB: ~0.87x frame time for +3 GiB peak footprint (11.0 -> 14.0 GiB; 327fc92). With the
  cap raised, bf16 trunk linears are still level with f32: the cap is not what hides bf16's gain.
- Fable's ruling (via ojas-78): ojas's default cap stays; SpikeTrack sets its own (4 GiB).
  Owner, 2026-10-28: promote `set_mem_cap` to public API (ojas-78), and bump SpikeTrack's ojas pin
  once, to an ojas main that has it (Lane M: the plan set-aside and the promotion). Then SpikeTrack
  sets the cap at startup and records it in run.json, re-verified. Not `DType::Bf16` weights: a
  Bf16-dtype operand is never bf16-exact, so it takes the exact-f32 lane after a widen (ojas-78,
  link.rs:202-208); `with_trunk_linears_bf16` already pre-casts the weights to f32-stored
  bf16-exact once at load, which is the right form. The O1 ojas weight slice is not needed.
- E7 [V, refuted at the gate]: on `speed/e7-geometry` (45e5d63) the 672 trunk + necks match upstream
  at 672 (cosine 1.0000000), and 1008 is byte-identical to before. Held-out HOTA at 672 with the
  1008-trained LoRA is 0.4645 at cap 64 and 0.4479 at cap 16, against 0.4868 / 0.4695 at 1008
  (gate 0.47). The 1008 cap-64 number is stale: 0.4922 on the full re-run (E10). The gate was the
  0.47 floor, so the verdict stands. DetA holds; AssA falls (0.69 -> 0.62) and LocA falls on all five videos; videos 47
  and 21 carry most of it (`bench/results/track/held300-lora600-sperm-detector-672-max64`). Speed:
  the paired 8-frame smoke on the branch is 2.27x (1008 3.10 / 3.02 s/frame, 672 1.36 / 1.34); the
  held-out runs look faster still, but they are unpaired against the 1008 runs.
  Fable's ruling: no `--fast` 672, no 672 LoRA retrain this week (DetA, the part an adapter would
  most plausibly fix, did not drop), and the branch stays parked unmerged. Its only tested
  non-default size fails the gate, and `image`, GatedStats flops and the 192² threshold
  calibration still assume 1008. It reopens if the owner changes the gate or a 672-trained
  adapter is scheduled. It is the largest single speed lever found.
- Mem cap shipped [V]: ojas pin 8b88328 -> ea3cf39. `spiketrack track` on Metal sets a 4 GiB cap
  and records it in run.json (`metal_mem_cap_bytes`). Session parity on VISEM 22 is unchanged at
  the default cap and at 4 GiB (IoU 0.99855). 49 frames of VISEM 22 (LoRA, 12 tracks),
  interleaved over 3 rounds: 0.930x frame time (0.947 / 0.930 / 0.927), with byte-identical
  predictions (`bench/results/ojas-ea3cf39`).
- E10 (stated 2026-10-29, before the runs) [V for the export, `bench/results/coreml-trunk`]: the
  sdkv2 port's backbone fuses trunk + propagation neck, so the detector cannot use it. Its trunk is
  exported alone instead (`tools/baselines/coreml_trunk.py`).
  - fp32 measures 0.9-1.2 s per call at rel L2 1-3e-5. That is the same exactness class as Ojas f32:
    block-31 max abs 6.2e-2 against Ojas Metal's 2.5e-2.
  - fp16 measures 0.25-0.29 s at 0.5-2% rel L2. Every fp16 op split lands at the same error, so the
    linears themselves carry it.
  - Owner, 2026-10-29: new dependencies approved (`objc2-core-ml`, `sha2`). The Ojas trunk stays in
    the code: CPU parity, st-gate and the block fixtures need it.
  - With `coreml-*` the trunk is no longer Ojas/tessl. It is still native and still zero Python at
    runtime.
  - Owner, 2026-10-29: make coreml-fp32 the default once held-out HOTA confirms it. Stated before
    its run produced an eval (the fp16 holds were running): it confirms if HOTA at cap 64 is within
    0.005 of the Ojas f32 run's 0.4868. The default then flips on `--backend metal` only; `--backend
    cpu` keeps the Ojas trunk, the upstream-parity oracle. A missing package is an error, not a
    fallback to the Ojas trunk.
  - Baseline replaced, 2026-10-29, written before its result was read. The 0.4868 / 0.4695 runs
    came from an older binary: their run.json has no `trunk_precision`, `max_objects` or
    `metal_mem_cap_bytes`. The ea3cf39 byte-identity check covered VISEM 22 only. Every E10 HOTA
    differs from them on video 21 alone, the other four videos agreeing to 1e-4. The comparison
    baseline is therefore a fresh `--trunk-precision f32` run on the same frozen binary at caps 64
    and 16, with the same 0.005 lines. The 0.47 floor does not move. "Same tracks" for the fp32
    default is checked on all five videos at cap 64, not only VISEM 22.
  - Results [V, `bench/results/coreml-trunk`]. Gate 1 passes: worst rel L2 2.7e-5, and VISEM 22 ids
    are the same on 8 frames.
  - Frame time on a contended machine, 49 frames interleaved over 3 rounds: coreml-fp32 0.80x and
    coreml-fp16 0.55x the Ojas trunk's frame. Every CoreML round beat every f32 round. The CoreML load
    adds 3-12 s, a net loss below ~10-17 frames.
  - The fresh f32 baseline is 0.4922 at cap 64 and 0.4695 at cap 16. The old cap-64 run had copied
    video 21 from the cap-16 run, though video 21 hits the 16 cap at frame 0.
  - **coreml-fp32: confirmed.** HOTA is 0.4922 at cap 64, equal to f32 on every video. The id sets are
    equal on all 1500 frames. Since 2026-10-29 it is the default trunk on `--backend metal`.
  - **coreml-fp16: refuted as `--fast`.** It scores 0.4943 at cap 64 but 0.4638 at cap 16, below the
    0.47 floor and 0.0057 under f32. One near-threshold detection on video 21's first frame drops
    under fp16, and that renumbers the whole video. `--trunk-precision coreml-fp16` stays an explicit,
    unrecommended flag.
- E11 [V, `bench/results/profile-coreml`]: the profile is valid. Its stage sums are within 1.1% of
  the unprofiled controls, and predictions are byte-identical with the marker on.
  - Steady frames: trunk 925 / 928 ms (38.5% on 22, 32.0% on 82); detector neck + detector 786 / 821
    (32.7% / 28.3%); tracker states 344 on 22, and 337 + 284 on 82 (21.4%); memory encode 158 / 258;
    propagation neck ~140.
  - Neither pre-stated reading holds: no stage reaches 40%.
  - ~610 ms per frame on 82 is not accounted for by backend ops, and readback sizes do not explain
    it. At unified-memory copy rates the 66 MB `pred_masks` readback is ~5-10 ms, and memory encode's
    transfers ~30 ms. The candidates are the `to_host` path itself, host dispatch encoding, and GPU
    idle between dependent dispatches.
  - Split pending (E11b). Pre-stated before the run: if `detector readback` is <= 30 ms and the
    detector's device stage stays ~780 ms, the gap is dispatch or GPU-side, and ojas-78's. If the
    readback is >= 200 ms, the gap is SpikeTrack's: on-device post-processing or a cheaper
    `to_host`. The same split applies to memory encode.
- E11b [V, `bench/results/profile-coreml/split`]: video 82 is valid (-4% against the controls), and
  predictions are byte-identical. Video 22 failed the 10% rule at load 75-95 (+85%) and is not read.
  - Neither branch of the rule fits, because the readback is small (`pred_masks` 7.7 ms) and the
    device detector is 330 ms, not ~780.
  - The detector's missing time is **host compute in SpikeTrack's DETR decoder**, 337 ms per frame
    (13%), more than the device detector itself:
    - decoder host prep: 264 ms. This is the box-RPB MLPs over 200 x 72 rows per axis,
      `ref_point_head`, the 33 MB `[8, 201, 5184]` bias, and its upload, done 6 times.
    - readback + box update: 48 ms.
    - scoring: 25 ms.
  - The six decoder layers take 23 ms on the GPU.
  - Memory encode is mostly device: encode 179 ms, host part 42 ms.
- Ownership: ojas-78 holds O1 (prepared weights, fused rounding, the in-frame pass cost,
  ft-73a0631d / ft-580e6c40) and O3; f16 as an Autocast mode is the owner's call.

Not this week: replacing or distilling the backbone (no parity gate for a different trunk, and no
time to train one), int8 W8A8, the ANE.

E7 scope [V, read]: windowed blocks keep their 576-token RoPE tables; the 4 global blocks load a
5184-row `freqs_cis` from the checkpoint and need it regenerated for 48x48; 118 places in
`crates/` hard-code 72 / 288 / 1008 / 5184.
