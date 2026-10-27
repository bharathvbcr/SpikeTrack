# Ojas work SpikeTrack is waiting on

Ojas is a pre-existing library (owner: the ojas-78 session; API changes are its and Bharath's call).
This lists what SpikeTrack asked for, what ojas-78 committed to, and the SpikeTrack call sites
that switch when each lands. Update a row when the commit arrives; delete it once switched.

| Op | Status (2026-10-27) | SpikeTrack call sites to switch | Notes |
|---|---|---|---|
| Metal `rope_interleaved_forward` | implemented on gp/78-metal (4897d82), merging with the tessl attention routing fix | backbone.rs block_forward; memory_attention.rs | unblocks the Metal backbone gate; move ojas-st once, to that commit |
| Metal conv2d any stride/padding (groups 1, dilation 1) | queued, Lane A batch 1 | memory_encoder.rs mask downsampler (3x3 s2 p1) | im2col already handles it; refusal was narrow |
| Metal depthwise conv2d (groups == C) | queued, Lane A batch 1 | memory_encoder.rs ConvNeXt fuser (7x7 p3, 256 ch, 72^2) | direct kernel |
| Metal `interpolate_bilinear_forward` (both align_corners) | queued, Lane A batch 1 | memory_encoder.rs (to 1152), vos.rs and tracker/state.rs (to 1008 and to video size) | must match CPU exactly |
| Metal `relu_forward` | queued, Lane A batch 1 (CPU already on main 105e203) | layers.rs `relu` (host round trip today) | then `relu` becomes `be.relu_forward` |
| `group_norm_forward` (CPU f64 two-pass, then Metal multi-threadgroup) | queued, Lane B2 then Lane A | detector.rs `group_norm` (layer_norm composition + host ones/zeros) | fixtures: fixtures/groupnorm/pixel_decoder.safetensors (env OJAS_SAM_GROUPNORM_FIXTURES) |
| `concat` (one axis, same rank) and `slice` (narrow one axis) | queued, Lane B2 after group_norm; Metal in Lane A | memory_attention.rs memory/pointer assembly (host_of + Vec + upload); vos.rs and tracker/state.rs memory build | bit-exact |
| `fill(shape, dtype, value)` | queued, Lane B2 after concat/slice; Metal in Lane A | memory_attention.rs pointer-slot zero padding; detector.rs nearest2x zeros | group_norm's ones/zeros go away with the native op |
| tessl `encoder_attn` routing for D=32 self-attention | Lane C2 bench decides (threshold: win beyond the A-vs-A spread) | none (inside Ojas Metal) | 14 self-attention layers per frame |
| Bias-capable tiled attention kernel | built only if bench row (b) > 10% of the CoreML frame time (provisional 2.75 s -> 275 ms) | none (inside Ojas Metal) | denominator updates when the M5 Pro CoreML timing lands |

Not requested: `max_pool2d` (no call site in SpikeTrack).
