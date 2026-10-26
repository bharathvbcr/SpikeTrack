# Detector parity vs upstream, from upstream's detector inputs (gp-detector-and-mask-head)

- 2026-10-26, M5 Pro, ojas d2922fb CpuBackend (8 threads). Test crates/st-model/tests/detector_parity.rs, job tools/jobs/d3_detector.sh; fixtures tools/reference/dump_detector_fixtures.py (upstream Sam3Processor, fp32).
- Inputs are upstream's own FPN levels and resized text features, so this isolates the detector. The end-to-end gate (our backbone and text) is bench/results/d3-gate.
- Result: **pass**. All 96 stage checks clear their cosine gates; worst cosine 0.9999998. Kept detections at 0.5 are identical to upstream's in all 3 cases.
- Detector-only CPU time is one run per case, not a benchmark.
- Not read on the text path: 19 detector tensors (point/box/mask prompt encoders, the unused 3rd pixel-decoder stage).

| case | upstream kept @0.5 | ours kept | only upstream (q:ref/ours) | only ours | mean mask IoU (upstream's kept) | detector s (CPU) |
|---|---|---|---|---|---|---|
| visem22_cell | 28 | 28 |  |  | 0.9998 | 2.3 |
| testimage_shoe | 12 | 12 |  |  | 1.0000 | 2.1 |
| truck_wheel | 4 | 4 |  |  | 1.0000 | 2.2 |

| case | stage | cosine | max abs | max abs / max ref | gate |
|---|---|---|---|---|---|
| visem22_cell | sine pos 72x72 | 1.0000000 | 5.960e-8 | 5.960e-8 | pass |
| visem22_cell | geometry CLS | 1.0000000 | 9.537e-7 | 9.117e-8 | pass |
| visem22_cell | encoder layer 0 | 1.0000000 | 2.861e-6 | 4.518e-7 | pass |
| visem22_cell | encoder layer 1 | 1.0000000 | 8.166e-6 | 1.022e-6 | pass |
| visem22_cell | encoder layer 2 | 1.0000000 | 1.132e-5 | 1.230e-6 | pass |
| visem22_cell | encoder layer 3 | 1.0000000 | 2.718e-5 | 2.316e-6 | pass |
| visem22_cell | encoder layer 4 | 1.0000000 | 4.673e-5 | 2.781e-6 | pass |
| visem22_cell | encoder layer 5 | 1.0000000 | 9.060e-5 | 4.342e-6 | pass |
| visem22_cell | decoder 0 queries | 1.0000000 | 1.135e-4 | 1.658e-5 | pass |
| visem22_cell | decoder 0 ref boxes in | 1.0000000 | 5.960e-8 | 9.531e-8 | pass |
| visem22_cell | decoder 0 scores | 1.0000000 | 4.035e-5 | 1.055e-5 | pass |
| visem22_cell | decoder 1 queries | 1.0000000 | 8.008e-4 | 1.211e-4 | pass |
| visem22_cell | decoder 1 ref boxes in | 1.0000000 | 1.585e-5 | 1.603e-5 | pass |
| visem22_cell | decoder 1 scores | 1.0000000 | 1.388e-4 | 3.704e-5 | pass |
| visem22_cell | decoder 2 queries | 1.0000000 | 9.429e-4 | 1.306e-4 | pass |
| visem22_cell | decoder 2 ref boxes in | 1.0000000 | 2.029e-5 | 2.051e-5 | pass |
| visem22_cell | decoder 2 scores | 1.0000000 | 2.213e-4 | 6.240e-5 | pass |
| visem22_cell | decoder 3 queries | 1.0000000 | 1.003e-3 | 1.317e-4 | pass |
| visem22_cell | decoder 3 ref boxes in | 1.0000000 | 2.876e-5 | 2.906e-5 | pass |
| visem22_cell | decoder 3 scores | 1.0000000 | 3.569e-4 | 9.280e-5 | pass |
| visem22_cell | decoder 4 queries | 1.0000000 | 1.923e-3 | 2.490e-4 | pass |
| visem22_cell | decoder 4 ref boxes in | 1.0000000 | 8.392e-5 | 8.482e-5 | pass |
| visem22_cell | decoder 4 scores | 1.0000000 | 7.877e-4 | 2.166e-4 | pass |
| visem22_cell | decoder 5 queries | 1.0000000 | 2.212e-3 | 3.285e-4 | pass |
| visem22_cell | decoder 5 ref boxes in | 1.0000000 | 1.180e-4 | 1.192e-4 | pass |
| visem22_cell | decoder 5 scores | 1.0000000 | 1.201e-3 | 3.192e-4 | pass |
| visem22_cell | seg prompt cross-attn | 1.0000000 | 2.623e-5 | 6.562e-6 | pass |
| visem22_cell | pixel embed | 0.9999998 | 9.178e-3 | 1.690e-3 | pass |
| visem22_cell | semantic seg | 1.0000000 | 5.168e-2 | 1.745e-3 | pass |
| visem22_cell | pred_logits | 1.0000000 | 1.197e-3 | 3.167e-4 | pass |
| visem22_cell | pred_boxes | 1.0000000 | 1.817e-4 | 1.835e-4 | pass |
| visem22_cell | pred_masks | 1.0000000 | 5.190e-1 | 2.048e-3 | pass |
| testimage_shoe | sine pos 72x72 | 1.0000000 | 5.960e-8 | 5.960e-8 | pass |
| testimage_shoe | geometry CLS | 1.0000000 | 1.311e-6 | 1.307e-7 | pass |
| testimage_shoe | encoder layer 0 | 1.0000000 | 2.384e-6 | 3.599e-7 | pass |
| testimage_shoe | encoder layer 1 | 1.0000000 | 1.621e-5 | 1.884e-6 | pass |
| testimage_shoe | encoder layer 2 | 1.0000000 | 1.621e-5 | 1.549e-6 | pass |
| testimage_shoe | encoder layer 3 | 1.0000000 | 2.408e-5 | 1.920e-6 | pass |
| testimage_shoe | encoder layer 4 | 1.0000000 | 4.387e-5 | 2.532e-6 | pass |
| testimage_shoe | encoder layer 5 | 1.0000000 | 5.341e-5 | 2.492e-6 | pass |
| testimage_shoe | decoder 0 queries | 1.0000000 | 3.386e-5 | 5.188e-6 | pass |
| testimage_shoe | decoder 0 ref boxes in | 1.0000000 | 5.960e-8 | 9.531e-8 | pass |
| testimage_shoe | decoder 0 scores | 1.0000000 | 1.860e-5 | 2.357e-6 | pass |
| testimage_shoe | decoder 1 queries | 1.0000000 | 2.101e-4 | 2.823e-5 | pass |
| testimage_shoe | decoder 1 ref boxes in | 1.0000000 | 8.523e-6 | 9.233e-6 | pass |
| testimage_shoe | decoder 1 scores | 1.0000000 | 1.135e-4 | 2.273e-5 | pass |
| testimage_shoe | decoder 2 queries | 1.0000000 | 5.885e-4 | 7.311e-5 | pass |
| testimage_shoe | decoder 2 ref boxes in | 1.0000000 | 1.389e-5 | 1.489e-5 | pass |
| testimage_shoe | decoder 2 scores | 1.0000000 | 5.931e-5 | 1.240e-5 | pass |
| testimage_shoe | decoder 3 queries | 1.0000000 | 5.907e-4 | 7.503e-5 | pass |
| testimage_shoe | decoder 3 ref boxes in | 1.0000000 | 1.109e-5 | 1.192e-5 | pass |
| testimage_shoe | decoder 3 scores | 1.0000000 | 6.008e-5 | 1.136e-5 | pass |
| testimage_shoe | decoder 4 queries | 1.0000000 | 7.882e-4 | 9.358e-5 | pass |
| testimage_shoe | decoder 4 ref boxes in | 1.0000000 | 2.384e-5 | 2.556e-5 | pass |
| testimage_shoe | decoder 4 scores | 1.0000000 | 2.642e-4 | 5.168e-5 | pass |
| testimage_shoe | decoder 5 queries | 1.0000000 | 4.126e-3 | 6.302e-4 | pass |
| testimage_shoe | decoder 5 ref boxes in | 1.0000000 | 9.400e-5 | 1.004e-4 | pass |
| testimage_shoe | decoder 5 scores | 1.0000000 | 7.644e-4 | 1.488e-4 | pass |
| testimage_shoe | seg prompt cross-attn | 1.0000000 | 8.821e-6 | 1.906e-6 | pass |
| testimage_shoe | pixel embed | 0.9999999 | 1.830e-2 | 1.801e-3 | pass |
| testimage_shoe | semantic seg | 1.0000000 | 1.483e-1 | 1.758e-3 | pass |
| testimage_shoe | pred_logits | 1.0000000 | 7.629e-4 | 1.442e-4 | pass |
| testimage_shoe | pred_boxes | 1.0000000 | 4.205e-4 | 4.488e-4 | pass |
| testimage_shoe | pred_masks | 1.0000000 | 5.517e-1 | 2.071e-3 | pass |
| truck_wheel | sine pos 72x72 | 1.0000000 | 5.960e-8 | 5.960e-8 | pass |
| truck_wheel | geometry CLS | 1.0000000 | 9.537e-7 | 9.468e-8 | pass |
| truck_wheel | encoder layer 0 | 1.0000000 | 2.503e-6 | 3.464e-7 | pass |
| truck_wheel | encoder layer 1 | 1.0000000 | 1.979e-5 | 2.219e-6 | pass |
| truck_wheel | encoder layer 2 | 1.0000000 | 3.147e-5 | 2.957e-6 | pass |
| truck_wheel | encoder layer 3 | 1.0000000 | 3.910e-5 | 2.864e-6 | pass |
| truck_wheel | encoder layer 4 | 1.0000000 | 5.579e-5 | 3.055e-6 | pass |
| truck_wheel | encoder layer 5 | 1.0000000 | 8.297e-5 | 3.581e-6 | pass |
| truck_wheel | decoder 0 queries | 1.0000000 | 7.749e-5 | 1.133e-5 | pass |
| truck_wheel | decoder 0 ref boxes in | 1.0000000 | 5.960e-8 | 9.531e-8 | pass |
| truck_wheel | decoder 0 scores | 1.0000000 | 5.245e-5 | 5.492e-6 | pass |
| truck_wheel | decoder 1 queries | 1.0000000 | 2.105e-4 | 2.887e-5 | pass |
| truck_wheel | decoder 1 ref boxes in | 1.0000000 | 2.694e-5 | 2.851e-5 | pass |
| truck_wheel | decoder 1 scores | 1.0000000 | 5.674e-5 | 9.819e-6 | pass |
| truck_wheel | decoder 2 queries | 1.0000000 | 6.462e-4 | 8.063e-5 | pass |
| truck_wheel | decoder 2 ref boxes in | 1.0000000 | 4.235e-5 | 4.507e-5 | pass |
| truck_wheel | decoder 2 scores | 1.0000000 | 2.327e-4 | 4.619e-5 | pass |
| truck_wheel | decoder 3 queries | 1.0000000 | 1.420e-3 | 1.718e-4 | pass |
| truck_wheel | decoder 3 ref boxes in | 1.0000000 | 6.622e-5 | 7.036e-5 | pass |
| truck_wheel | decoder 3 scores | 1.0000000 | 8.979e-4 | 1.670e-4 | pass |
| truck_wheel | decoder 4 queries | 1.0000000 | 1.108e-3 | 1.300e-4 | pass |
| truck_wheel | decoder 4 ref boxes in | 1.0000000 | 1.383e-4 | 1.469e-4 | pass |
| truck_wheel | decoder 4 scores | 1.0000000 | 7.193e-4 | 1.315e-4 | pass |
| truck_wheel | decoder 5 queries | 1.0000000 | 3.548e-3 | 5.252e-4 | pass |
| truck_wheel | decoder 5 ref boxes in | 1.0000000 | 1.507e-4 | 1.595e-4 | pass |
| truck_wheel | decoder 5 scores | 1.0000000 | 1.037e-3 | 1.692e-4 | pass |
| truck_wheel | seg prompt cross-attn | 1.0000000 | 1.851e-5 | 4.136e-6 | pass |
| truck_wheel | pixel embed | 0.9999999 | 1.093e-2 | 1.467e-3 | pass |
| truck_wheel | semantic seg | 1.0000000 | 7.440e-2 | 1.287e-3 | pass |
| truck_wheel | pred_logits | 1.0000000 | 1.037e-3 | 1.689e-4 | pass |
| truck_wheel | pred_boxes | 1.0000000 | 7.716e-5 | 8.215e-5 | pass |
| truck_wheel | pred_masks | 1.0000000 | 3.634e-1 | 1.624e-3 | pass |
