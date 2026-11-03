class-0 detection at IoU 0.5, processor score (comparable to zero_shot_check.py), 120 held_out frames

| prompt | thr | zero-shot recall | zero-shot precision | LoRA recall | LoRA precision |
|---|---|---|---|---|---|
| sperm | 0.3 | 0.000 | - | 0.798 | 0.585 |
| sperm | 0.4 | 0.000 | - | 0.788 | 0.669 |
| sperm | 0.5 | 0.000 | - | 0.744 | 0.733 |
| cell | 0.3 | 0.425 | 0.117 | 0.536 | 0.181 |
| cell | 0.4 | 0.337 | 0.127 | 0.531 | 0.239 |
| cell | 0.5 | 0.212 | 0.122 | 0.523 | 0.319 |

class-0 detection at IoU 0.5, joint score (video path), 120 held_out frames

| prompt | thr | zero-shot recall | zero-shot precision | LoRA recall | LoRA precision |
|---|---|---|---|---|---|
| sperm | 0.3 | 0.000 | 0.000 | 0.798 | 0.585 |
| sperm | 0.4 | 0.000 | 0.000 | 0.788 | 0.669 |
| sperm | 0.5 | 0.000 | - | 0.744 | 0.732 |
| cell | 0.3 | 0.429 | 0.116 | 0.536 | 0.181 |
| cell | 0.4 | 0.343 | 0.126 | 0.531 | 0.239 |
| cell | 0.5 | 0.225 | 0.123 | 0.523 | 0.319 |

absent-concept prompts: fraction of (frame, prompt) with any detection >= 0.5 (processor score) / presence head > 0.5

| role | items | zero-shot any-det | LoRA any-det | zero-shot presence>0.5 | LoRA presence>0.5 |
|---|---|---|---|---|---|
| absent_class | 156 | 0.0 | 0.1987 | 0.359 | 0.2756 |
| seen_negative | 360 | 0.0 | 0.0 | 0.0 | 0.0 |
| unseen_negative | 240 | 0.0 | 0.0 | 0.0 | 0.0 |

ECE (10 bins): presence head zero-shot 0.1769, LoRA 0.0675; class-0 detections (joint) zero-shot 0.0717, LoRA 0.1097
