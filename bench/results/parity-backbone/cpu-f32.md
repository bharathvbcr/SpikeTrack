# Backbone parity, Ojas CPU backend vs PyTorch fp32 (gp-vision-backbone, D2 gate)

- Date: 2026-10-26. Machine: M5 Pro. Commit: SpikeTrack 297a4d8; ojas d2922fb (CpuBackend, single thread); fixtures from tools/reference/dump_fixtures.py (external/sam3 f1647e8 + sam3-cpu.patch, fp32 MLP).
- Command: `tools/locked.sh d1 bash tools/jobs/d1_build_and_parity.sh` (test `crates/st-model/tests/backbone_parity.rs`). Raw log: cpu-f32.log.
- Gate: cosine > 0.999 at every recorded stage on all 5 images.
- Result: **pass.** worst cosine 1.0000000 over 95 stages (gate > 0.999). The worst max-abs / max-ref is 9.605e-4.
- Timing (not a benchmark; one run, single-threaded CPU, machine busy): visem_22_f0 136.7 s;visem_47_f300 100.4 s;visem_82_f600 142.0 s;sam3_truck 145.1 s;sam3_groceries 232.8 s.
- Not covered yet: the Metal backend (conv2d/convT Metal paths land D2 from ojas-78), the bf16 path, latency.

| image | stage | cosine | max abs | max abs / max ref | gate |
|---|---|---|---|---|---|
| visem_22_f0 | patch_embed | 1.0000000 | 0.000e0 | 0.000e0 | pass |
| visem_22_f0 | ln_pre | 1.0000000 | 1.335e-5 | 6.092e-7 | pass |
| visem_22_f0 | block 0 | 1.0000000 | 1.097e-4 | 6.370e-6 | pass |
| visem_22_f0 | block 1 | 1.0000000 | 8.821e-5 | 3.593e-6 | pass |
| visem_22_f0 | block 2 | 1.0000000 | 1.049e-4 | 3.022e-6 | pass |
| visem_22_f0 | block 3 | 1.0000000 | 9.155e-5 | 2.244e-6 | pass |
| visem_22_f0 | block 4 | 1.0000000 | 1.030e-4 | 2.439e-6 | pass |
| visem_22_f0 | block 5 | 1.0000000 | 1.144e-4 | 2.479e-6 | pass |
| visem_22_f0 | block 6 | 1.0000000 | 1.335e-4 | 2.699e-6 | pass |
| visem_22_f0 | block 7 | 1.0000000 | 1.488e-4 | 3.009e-6 | pass |
| visem_22_f0 | block 8 | 1.0000000 | 2.327e-4 | 3.967e-6 | pass |
| visem_22_f0 | block 9 | 1.0000000 | 3.204e-4 | 4.116e-6 | pass |
| visem_22_f0 | block 10 | 1.0000000 | 3.853e-4 | 4.738e-6 | pass |
| visem_22_f0 | block 11 | 1.0000000 | 5.245e-4 | 6.504e-6 | pass |
| visem_22_f0 | block 12 | 1.0000000 | 1.282e-3 | 1.382e-5 | pass |
| visem_22_f0 | block 13 | 1.0000000 | 2.327e-3 | 2.011e-5 | pass |
| visem_22_f0 | block 14 | 1.0000000 | 5.123e-3 | 3.844e-5 | pass |
| visem_22_f0 | block 15 | 1.0000000 | 1.523e-2 | 9.943e-5 | pass |
| visem_22_f0 | block 16 | 1.0000000 | 6.521e-2 | 3.050e-4 | pass |
| visem_22_f0 | block 17 | 1.0000000 | 1.371e-1 | 5.404e-4 | pass |
| visem_22_f0 | block 18 | 1.0000000 | 2.460e-1 | 8.687e-4 | pass |
| visem_22_f0 | block 19 | 1.0000000 | 2.636e-1 | 9.277e-4 | pass |
| visem_22_f0 | block 20 | 1.0000000 | 2.757e-1 | 9.605e-4 | pass |
| visem_22_f0 | block 21 | 1.0000000 | 2.717e-1 | 9.411e-4 | pass |
| visem_22_f0 | block 22 | 1.0000000 | 2.661e-1 | 9.219e-4 | pass |
| visem_22_f0 | block 23 | 1.0000000 | 2.548e-1 | 8.774e-4 | pass |
| visem_22_f0 | block 24 | 1.0000000 | 2.422e-1 | 8.346e-4 | pass |
| visem_22_f0 | block 25 | 1.0000000 | 2.368e-1 | 8.177e-4 | pass |
| visem_22_f0 | block 26 | 1.0000000 | 2.334e-1 | 7.990e-4 | pass |
| visem_22_f0 | block 27 | 1.0000000 | 2.195e-1 | 7.550e-4 | pass |
| visem_22_f0 | block 28 | 1.0000000 | 2.065e-1 | 7.195e-4 | pass |
| visem_22_f0 | block 29 | 1.0000000 | 1.871e-1 | 6.755e-4 | pass |
| visem_22_f0 | block 30 | 1.0000000 | 1.834e-1 | 6.852e-4 | pass |
| visem_22_f0 | block 31 | 1.0000000 | 1.603e-1 | 7.502e-4 | pass |
| visem_22_f0 | convs[0] | 1.0000000 | 1.432e-4 | 7.522e-4 | pass |
| visem_22_f0 | convs[1] | 1.0000000 | 6.902e-3 | 7.036e-4 | pass |
| visem_22_f0 | convs[2] | 1.0000000 | 4.484e-3 | 7.335e-4 | pass |
| visem_22_f0 | interactive_convs[0] | 1.0000000 | 7.585e-4 | 9.557e-4 | pass |
| visem_22_f0 | interactive_convs[1] | 1.0000000 | 1.834e-3 | 5.973e-4 | pass |
| visem_22_f0 | interactive_convs[2] | 1.0000000 | 1.903e-3 | 7.119e-4 | pass |
| visem_22_f0 | propagation_convs[0] | 1.0000000 | 3.849e-4 | 3.977e-4 | pass |
| visem_22_f0 | propagation_convs[1] | 1.0000000 | 1.171e-3 | 3.904e-4 | pass |
| visem_22_f0 | propagation_convs[2] | 1.0000000 | 3.131e-3 | 5.592e-4 | pass |
| visem_47_f300 | block 7 | 1.0000000 | 1.564e-4 | 3.015e-6 | pass |
| visem_47_f300 | block 15 | 1.0000000 | 5.695e-3 | 3.280e-5 | pass |
| visem_47_f300 | block 23 | 1.0000000 | 4.495e-2 | 1.438e-4 | pass |
| visem_47_f300 | block 31 | 1.0000000 | 3.413e-2 | 1.455e-4 | pass |
| visem_47_f300 | convs[0] | 1.0000000 | 9.325e-6 | 4.134e-5 | pass |
| visem_47_f300 | convs[1] | 1.0000000 | 6.759e-4 | 7.121e-5 | pass |
| visem_47_f300 | convs[2] | 1.0000000 | 4.543e-4 | 7.784e-5 | pass |
| visem_47_f300 | interactive_convs[0] | 1.0000000 | 4.470e-5 | 5.564e-5 | pass |
| visem_47_f300 | interactive_convs[1] | 1.0000000 | 1.380e-4 | 4.690e-5 | pass |
| visem_47_f300 | interactive_convs[2] | 1.0000000 | 2.348e-4 | 8.122e-5 | pass |
| visem_47_f300 | propagation_convs[0] | 1.0000000 | 3.375e-5 | 3.773e-5 | pass |
| visem_47_f300 | propagation_convs[1] | 1.0000000 | 7.647e-5 | 2.542e-5 | pass |
| visem_47_f300 | propagation_convs[2] | 1.0000000 | 2.092e-4 | 3.519e-5 | pass |
| visem_82_f600 | block 7 | 1.0000000 | 1.450e-4 | 2.717e-6 | pass |
| visem_82_f600 | block 15 | 1.0000000 | 1.026e-2 | 6.028e-5 | pass |
| visem_82_f600 | block 23 | 1.0000000 | 9.914e-2 | 3.169e-4 | pass |
| visem_82_f600 | block 31 | 1.0000000 | 6.941e-2 | 2.956e-4 | pass |
| visem_82_f600 | convs[0] | 1.0000000 | 4.287e-5 | 1.996e-4 | pass |
| visem_82_f600 | convs[1] | 1.0000000 | 2.301e-3 | 2.193e-4 | pass |
| visem_82_f600 | convs[2] | 1.0000000 | 1.360e-3 | 2.444e-4 | pass |
| visem_82_f600 | interactive_convs[0] | 1.0000000 | 2.185e-4 | 2.588e-4 | pass |
| visem_82_f600 | interactive_convs[1] | 1.0000000 | 5.286e-4 | 1.661e-4 | pass |
| visem_82_f600 | interactive_convs[2] | 1.0000000 | 8.837e-4 | 3.480e-4 | pass |
| visem_82_f600 | propagation_convs[0] | 1.0000000 | 1.892e-4 | 2.532e-4 | pass |
| visem_82_f600 | propagation_convs[1] | 1.0000000 | 4.185e-4 | 1.441e-4 | pass |
| visem_82_f600 | propagation_convs[2] | 1.0000000 | 1.060e-3 | 1.932e-4 | pass |
| sam3_truck | block 7 | 1.0000000 | 1.869e-4 | 3.525e-6 | pass |
| sam3_truck | block 15 | 1.0000000 | 6.762e-3 | 4.688e-5 | pass |
| sam3_truck | block 23 | 1.0000000 | 3.228e-2 | 1.138e-4 | pass |
| sam3_truck | block 31 | 1.0000000 | 2.020e-2 | 9.562e-5 | pass |
| sam3_truck | convs[0] | 1.0000000 | 1.837e-5 | 7.631e-5 | pass |
| sam3_truck | convs[1] | 1.0000000 | 1.285e-3 | 1.078e-4 | pass |
| sam3_truck | convs[2] | 1.0000000 | 1.204e-3 | 1.958e-4 | pass |
| sam3_truck | interactive_convs[0] | 1.0000000 | 1.680e-4 | 1.875e-4 | pass |
| sam3_truck | interactive_convs[1] | 1.0000000 | 3.593e-4 | 1.026e-4 | pass |
| sam3_truck | interactive_convs[2] | 1.0000000 | 4.805e-4 | 1.841e-4 | pass |
| sam3_truck | propagation_convs[0] | 1.0000000 | 5.865e-5 | 4.332e-5 | pass |
| sam3_truck | propagation_convs[1] | 1.0000000 | 2.412e-4 | 6.671e-5 | pass |
| sam3_truck | propagation_convs[2] | 1.0000000 | 5.930e-4 | 1.337e-4 | pass |
| sam3_groceries | block 7 | 1.0000000 | 2.022e-4 | 4.068e-6 | pass |
| sam3_groceries | block 15 | 1.0000000 | 8.129e-3 | 4.941e-5 | pass |
| sam3_groceries | block 23 | 1.0000000 | 1.228e-1 | 4.024e-4 | pass |
| sam3_groceries | block 31 | 1.0000000 | 8.619e-2 | 3.746e-4 | pass |
| sam3_groceries | convs[0] | 1.0000000 | 3.783e-5 | 1.924e-4 | pass |
| sam3_groceries | convs[1] | 1.0000000 | 1.739e-3 | 1.462e-4 | pass |
| sam3_groceries | convs[2] | 1.0000000 | 1.146e-3 | 2.107e-4 | pass |
| sam3_groceries | interactive_convs[0] | 1.0000000 | 1.263e-4 | 1.465e-4 | pass |
| sam3_groceries | interactive_convs[1] | 1.0000000 | 3.870e-4 | 1.093e-4 | pass |
| sam3_groceries | interactive_convs[2] | 1.0000000 | 6.321e-4 | 2.484e-4 | pass |
| sam3_groceries | propagation_convs[0] | 1.0000000 | 1.096e-4 | 1.226e-4 | pass |
| sam3_groceries | propagation_convs[1] | 1.0000000 | 2.683e-4 | 8.726e-5 | pass |
| sam3_groceries | propagation_convs[2] | 1.0000000 | 5.195e-4 | 1.307e-4 | pass |
