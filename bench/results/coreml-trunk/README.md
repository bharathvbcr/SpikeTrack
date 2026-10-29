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

Reading [V for the numbers, I for the cause]:
- fp32 CoreML is in the same exactness class as the shipped f32 path. It is the default candidate.
- Every fp16 split lands at the same 0.5-2% rel L2, keeping fp16 on the linears alone included. So the
  error is the fp16 rounding of the linears' operands. No op selector recovers it.
- fp16 is a `--fast` candidate only. Its gate is held-out HOTA, D3 kept sets and VISEM 22 session IoU.
  D2's cosine > 0.999 is too weak to decide it, since fp16 passes that.
- The mixed variants are kept here as the evidence that the split was tried. They are not carried
  into Rust.

The packages are derived from gated weights. They live in `$SPIKETRACK_MODEL_DIR/coreml/` and are
never committed.
