# Zero-shot diagnosis: looser matching

Score threshold 0.3; recall over all GT sperm (class 0) in the 50 frames.

| prompt | preds | IoU>=0.5 | IoU>=0.3 | IoU>=0.1 | centre <=10 px | median pred w x h | median GT w x h |
|---|---|---|---|---|---|---|---|
| sperm | 3 | 0.000 | 0.000 | 0.002 | 0.000 | 79x25 | 21x20 |
| sperm cell | 663 | 0.015 | 0.073 | 0.267 | 0.058 | 41x38 | 21x20 |
| spermatozoon | 0 | 0.000 | 0.000 | 0.000 | 0.000 | - | 21x20 |
| cell | 3840 | 0.233 | 0.519 | 0.814 | 0.542 | 18x19 | 21x20 |
| car | 0 | 0.000 | 0.000 | 0.000 | 0.000 | - | 21x20 |
| cell nucleus | 0 | 0.000 | 0.000 | 0.000 | 0.000 | - | 21x20 |
