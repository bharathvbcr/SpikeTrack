# SAM 3.1's ViT trunk as a CoreML ML Program (E10 input, 2026-10-29)

`tools/baselines/coreml_trunk.py` exports upstream's ViT trunk alone. The input is `image`
[1,3,1008,1008], normalized. The output is `trunk` [1,72,72,1024] NHWC, the layout `Backbone::trunk`
returns. The script checks the output against each backbone fixture's `trunk.blocks.31.out` (upstream
fp32, CPU). Compute units CPU_AND_GPU, macOS 27.0, M5 Pro, timed under mac_heavy with other sessions'
GPU work running (load in each JSON).

The sdkv2 port's backbone cannot stand in for the trunk: it fuses trunk + propagation neck and does not
produce what the detector neck reads.

| precision | median s (min, max), n = 10 | worst cosine (5 images) | rel L2 | max abs, visem_22 | repeat call byte-identical |
|---|---|---|---|---|---|
| fp32 | first run 0.905 (0.889, 0.913); rerun 1.213 (1.105, 1.464) | 1.0000000 | 1.1e-5 .. 2.7e-5 | 0.062 | yes |
| fp16 (every op) | first run 0.252 (0.249, 0.253); rerun 0.289 (0.279, 0.317) | 0.9997793 | 5.5e-3 .. 2.1e-2 | 31.2 | yes |
| mixed (linear, attention, patch conv fp16) | 0.351 | 0.9998226 | 6.0e-3 .. 1.9e-2 | 19.7 | not checked |
| mixed-lin (linear, patch conv fp16) | 0.428 | 0.9998049 | 4.8e-3 .. 2.0e-2 | 18.1 | not checked |
| mixed-res (fp16 except residual adds, layer norms) | 0.269 | 0.9997957 | 5.3e-3 .. 2.0e-2 | 21.0 | not checked |

For scale, on visem_22:
- Ojas's Metal f32 trunk reaches block-31 max abs 2.5e-2 (`../ojas-ea3cf39/metal-backbone.log`).
- The torch export wrapper against the fixture reaches 1.2e-2.

Each JSON holds the rerun. The first-run numbers above are from the export logs. Between the two runs the
GPU load from other sessions differed; the frame-level interleaved A/B is the speed number, not these.
`fp16-CPU_AND_GPU.json` has since been overwritten by the GPU side of the Neural Engine comparison below
(2026-11-03, macOS 27.2, n = 5, median 0.260 s), so its numbers differ from the fp16 rerun row above.

Reading [V for the numbers, I for the cause]:
- fp32 CoreML is in the same exactness class as the Ojas f32 path. It became the Metal default after
  the tracker gates below.
- Every fp16 split lands at the same 0.5-2% rel L2, keeping fp16 on the linears alone included. So the
  error is the fp16 rounding of the linears' operands. No op selector recovers it.
- fp16 is a `--fast` candidate only. Its gate is held-out HOTA, D3 kept sets and VISEM 22 session IoU.
  D2's cosine > 0.999 is too weak to decide it, since fp16 passes that.
- The mixed variants are kept here as the evidence that the split was tried. They are not carried
  into Rust.

## In the tracker: frame time and held-out HOTA (2026-10-29)

Gate 1 [V, `gate1/`]: the CoreML trunk + Ojas CPU necks against upstream on the 5 fixtures. fp32
passes the 1e-4 rel L2 gate (worst 2.7e-5) and fp16 passes D2 (worst cosine 0.99978). Both refuse
block taps, gating and bf16 linears. On VISEM 22, coreml-fp32 tracks the same ids as the Ojas trunk on
8 frames, and dispatches fall 6284 -> 4007 per frame.

Frame time [V, `timing/README.md`], 49 frames, interleaved over 3 rounds on a contended machine:
coreml-fp32 is 0.80x the Ojas trunk's frame and coreml-fp16 0.55x. The CoreML load adds 3-12 s.

Held-out HOTA, the first 300 frames of the 5 held-out VISEM videos: prompt "sperm", r16 LoRA,
`--boxes detector`. Every run is the same frozen binary (built from b11bb69; `job-hota.sh`), chosen
by `--trunk-precision`. The tree was dirty with the staged default change during the f32 and coreml
runs, which does not matter because the binary was frozen before it; its sha is in each
`versions.txt`.

| trunk | cap 64: HOTA / DetA / AssA | cap 16: HOTA / DetA / AssA | s/frame, cap 64 (21 / 22 / 36 / 47 / 82) |
|---|---|---|---|
| f32, the Ojas trunk (fresh) | 0.4922 / 0.3734 / 0.6903 | 0.4695 / 0.3198 / 0.7251 | 4.32 / 3.30 / 3.60 / 3.12 / 3.45 |
| coreml-fp32 | **0.4922 / 0.3734 / 0.6903** | not run | 2.89 / 2.90 / 3.17 / 2.67 / 3.47 |
| coreml-fp16 | 0.4943 / 0.3743 / 0.6940 | **0.4638** / 0.3139 / 0.7218 | 2.22 / 1.82 / 2.45 / 1.83 / 2.24 |

The s/frame column comes from sequential runs on a shared machine, not paired ones; `timing/` is the
speed number.

- **coreml-fp32 confirms [V].** Its every per-video row equals the fresh f32 run's. Its id sets equal
  f32's on all 1500 frames (`tools/analysis/cmp_tracks.py`), with max |score| 3.5e-6. The largest box
  difference is 1.1e-2, on one video 36 box; elsewhere it is <= 5.4e-5.
- **coreml-fp16 is refuted as `--fast` [V].** At cap 16 it scores 0.4638, below the 0.47 floor and
  0.0057 under f32. At cap 64 it passes (+0.0021). The cause is a single near-threshold detection on
  video 21's first frame: fp16 finds 14 objects there against f32's 15. That renumbers the video's
  ids, so its id sets differ on all 300 frames. Video 36 at cap 64 diverges from frame 40, and 82
  from frame 211. A 1e-2 trunk error decides close births, in either direction.
- **The old cap-64 baseline (0.4868) was stale [V].** `../track/held300-lora600-sperm-detector-max64`
  re-ran only videos 36 and 82 and copied 21, 22 and 47 from the cap-16 run. But video 21 already has
  16 active tracks at frame 0 at cap 16, so its row was capped too. Re-run at 64, it peaks at 23 active
  tracks and scores 0.6422 against 0.6204. The cap-16 baseline reproduces exactly (0.4695).

The packages are derived from gated weights. They live in `$SPIKETRACK_MODEL_DIR/coreml/` and are
never committed.

**Superseded 2026-11-04.** The fp16 refutation above is for LoRA v1 at cap 16. On the shipped
configuration (LoRA v3 + `lorav3_exact_it_3p00.json`, cap 64), the fp16 trunk with fp16 memory
attention passes G2 and G3 of `../e25-coreml-frame/RULES.md`: 0.5694 against 0.5700 over 1,500
frames (`../track/held300-coreml/README.md`). It became the Metal default.

## The Neural Engine (2026-11-03)

`check --precision fp16 --units CPU_AND_NE -n 5` (`check-fp16-ane-2026-11-03.log`, `fp16-CPU_AND_NE.json`) [V]:
- median 2.39 s per call, against 0.26 s on CPU_AND_GPU the same evening;
- rel L2 2.4e-2 .. 4.4e-2 against upstream, about twice the GPU fp16 error.

Running the trunk on the Neural Engine, in parallel with the Metal tracker, is refuted as a speed path. The trunk stays on CPU_AND_GPU.
