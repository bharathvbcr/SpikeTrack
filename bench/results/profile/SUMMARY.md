# Per-op profile of one Metal frame (2026-10-27, contended)

SpikeTrack f902513, Ojas d5d33cc, tessl 789a132 (versions-metal.txt). `tools/jobs/frame_profile.sh metal`
under mac_heavy, single run, other sessions active. Every op synced after the call (wall time incl.
dispatch; an upper bound per op). Frame: judo frame 2 after 2 warm-up frames, 2 objects; prompt "cell".

Unprofiled: detector run (trunk + detector neck + detector) 3748 ms; tracker frame (trunk + propagation
neck + tracker step) 4874 ms. Profiled (synced): 3558 ms and 4403 ms, so sync inflation is small here;
the profiled rows sum to 8014 ms including model upload (271 ms) and the text encode (187 ms, once per prompt).

Stage totals (profiled): trunk 1368 ms; detector neck + detector 1835 ms; tracker frame 4352 ms, which
includes its own trunk (~1370 ms, same ops and shapes), so propagation neck + tracker step ~2980 ms.
The production Session runs the trunk once per frame.

By op class (both trunks counted; frame_profile's unlabelled and text rows excluded):

| op class | ms | share |
|---|---|---|
| attention, head dim 32 (Ojas scalar kernel) | 4013 | 53.1% |
| linear_bias (f32 GEMM) | 1994 | 26.4% |
| attention, head dim 64 (routed to tessl encoder_attn) | 353 | 4.7% |
| conv2d | 277 | 3.7% |
| permute | 263 | 3.5% |
| gelu_erf | 181 | 2.4% |
| layer_norm | 111 | 1.5% |
| residual_add | 100 | 1.3% |
| attention D32 with bias (detector decoder cross-attention) | 71 | 0.9% |
| rope_interleaved | 69 | 0.9% |

Top rows: memory cross-attention [1,8,5184,32] x [1,8,10400,32] 4 calls 1716 ms (429 ms each); detector
encoder self-attention [1,8,5184,32] 6 calls 1371 ms (228 ms each); memory self-attention [1,8,5184,32]
4 calls 905 ms (226 ms each). For comparison, routed D64 global attention [1,16,5184,64] is ~20 ms per call.
ViT MLP GEMMs [5184,1024]x[4736,1024] ~12.5 ms and [5184,4736]x[1024,4736] ~7.9 ms per block.

CoreML on the same machine (bench/results/baselines): trunk + propagation neck 0.272 s, track step 0.0775 s.
