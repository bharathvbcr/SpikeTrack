class-0 detection at IoU 0.5, processor score (comparable to zero_shot_check.py), 2 held-out frames

| prompt | thr | zero-shot recall | zero-shot precision | LoRA recall | LoRA precision |
|---|---|---|---|---|---|
| sperm | 0.3 | 0.000 | - | 0.000 | 0.000 |
| sperm | 0.4 | 0.000 | - | 0.000 | 0.000 |
| sperm | 0.5 | 0.000 | - | 0.000 | 0.000 |
| cell | 0.3 | 0.227 | 0.049 | 0.273 | 0.055 |
| cell | 0.4 | 0.227 | 0.064 | 0.227 | 0.062 |
| cell | 0.5 | 0.091 | 0.035 | 0.091 | 0.034 |

class-0 detection at IoU 0.5, joint score (video path), 2 held-out frames

| prompt | thr | zero-shot recall | zero-shot precision | LoRA recall | LoRA precision |
|---|---|---|---|---|---|
| sperm | 0.3 | 0.000 | - | 0.000 | 0.000 |
| sperm | 0.4 | 0.000 | - | 0.000 | 0.000 |
| sperm | 0.5 | 0.000 | - | 0.000 | 0.000 |
| cell | 0.3 | 0.227 | 0.048 | 0.273 | 0.055 |
| cell | 0.4 | 0.227 | 0.062 | 0.227 | 0.062 |
| cell | 0.5 | 0.091 | 0.035 | 0.091 | 0.034 |

absent-concept prompts: fraction of (frame, prompt) with any detection >= 0.5 (processor score) / presence head > 0.5

| role | items | zero-shot any-det | LoRA any-det | zero-shot presence>0.5 | LoRA presence>0.5 |
|---|---|---|---|---|---|
| absent_class | 4 | 0.0 | 1.0 | 0.75 | 1.0 |
| seen_negative | 6 | 0.0 | 0.0 | 0.0 | 0.0 |
| unseen_negative | 4 | 0.0 | 0.0 | 0.0 | 0.0 |

ECE (10 bins): presence head zero-shot 0.1889, LoRA 0.2344; class-0 detections (joint) zero-shot 0.0952, LoRA 0.1322
