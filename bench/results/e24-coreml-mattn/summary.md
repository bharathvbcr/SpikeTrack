# E24 summary: --mem-attn arms, VISEM 82, 26 frames, 16 objects (steady frames >= 8)

| round | ojas tracker ms (median) | coreml-fp32 tracker ms (median) | coreml-fp16 tracker ms (median) | fp32 / ojas | fp16 / ojas |
|---|---|---|---|---|---|
| 1 | 455.9 | 374.0 | 223.8 | 0.820 | 0.491 |
| 2 | 399.0 | 486.0 | 190.2 | 1.218 | 0.477 |
| 3 | 398.3 | 314.2 | 196.4 | 0.789 | 0.493 |

## Tracks against the Ojas arm (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU |
|---|---|---|---|---|
| 1 | coreml-fp32 | 26/26 | 1.0000 | 1.0000 |
| 1 | coreml-fp16 | 26/26 | 1.0000 | 1.0000 |
| 2 | coreml-fp32 | 26/26 | 1.0000 | 1.0000 |
| 2 | coreml-fp16 | 26/26 | 1.0000 | 1.0000 |
| 3 | coreml-fp32 | 26/26 | 1.0000 | 1.0000 |
| 3 | coreml-fp16 | 26/26 | 1.0000 | 1.0000 |
