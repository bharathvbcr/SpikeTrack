# Module parity gates, CPU and Metal (2026-10-27)

SpikeTrack 76d9b91, Ojas d5d33cc (ojas-st), tessl 789a132 (clean). `tools/jobs/module_gates.sh both`
under mac_heavy. Every gate is against upstream SAM 3.1 (PyTorch fp32 reference policy) on the same
fixtures and thresholds for both backends; logs beside this file.

| Module | Fixture | Gate | CPU | Metal (Ojas on tessl) |
|---|---|---|---|---|
| Text encoder | 13 prompts | cosine > 0.9999 | 1.000000000 (3.20 s) | 1.000000000 (0.24 s) |
| Detector + mask head (stage by stage) | 3 images | cosine > 0.999 per stage, mask IoU >= 0.95 | worst 1.0000000, IoU 1.0000 | worst 1.0000000, IoU 1.0000, kept sets identical (~2.0 s/image) |
| PE ViT trunk + 3 necks | 5 images, 95 stages | cosine > 0.999 | worst 1.0000000 (17.3-18.1 s/image) | worst 1.0000000 (1.7-1.8 s/image) |
| Multiplex mask decoder | judo, 7 frames | cosine > 0.999 | worst 1.0000000 | worst 1.0000000 |
| Memory encoder | judo, 8 frames | cosine > 0.999 | worst 1.0000000 | worst 1.0000000 |
| Memory attention | judo, 7 frames | cosine > 0.999 | worst 1.0000000 | worst 1.0000000 |
| Interactive decoder | judo frame 0 | cosine > 0.999 | 1.0000000 | 1.0000000 |

Times are single runs inside parity tests (not benchmarks: no warm-up, no min-of-N, CPU at 8
threads, other sessions active). The CoreML port's published backbone figure (~2.05 s/frame) is from
a different Mac; the M5 Pro CoreML baseline is queued (tools/jobs/coreml_bench.sh), and no speed
claim is made until it and an interleaved min-of-N bench exist.
