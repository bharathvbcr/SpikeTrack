# `visem-lora-v4`: plan, written before any v4 number exists (2026-11-03)

What v4 changes against v3, all at once (so a win is "the v4 recipe beats the v3 recipe", never
"more data helped" on its own):

1. **Data:** the frozen-feature cache at 128 frames per training video (15 videos, 1,920 frames),
   against v3's 64 (960). Held-out videos stay at 24 cached frames and are not read in training
   (`--no-eval`). Job: `cache-job.sh`, log `cache.log` here.
2. **Steps:** warm start from `fixtures/lora_runs/visem-lora-v3/adapter.safetensors` (step 160), up to
   400 more steps with v3's flags, checkpoints every 100.
3. **Box loss weights:** raised from `loss_bbox 5 / loss_giou 2` only if the train-5 diagnosis
   (`../../weight-space/diag-train5/`) shows near-misses (IoU 0.1-0.5) are at least half of the false
   negatives. The values are fixed in the train job's header before it runs.
   **Decided 2026-11-03, before training:** the diagnosis (`../../weight-space/diag-train5/buckets.txt`)
   has near-misses at 63.7% of FN and boxes smaller than the labels, so v4 trains with
   `--box-loss-weights 7.5 4.0` (bbox x1.5, GIoU x2; GIoU is the term that penalises a too-small box
   around a small head most). The Hungarian matcher's costs stay upstream's.

**Selection (train data only):** each checkpoint (100/200/300/400) with the exact inverse-temperature
edit at G in {2.4, 3.0, 3.6}, prompt `sperm`, `--boxes detector`, cap 64, first 8 frames of train-5
(14 30 19 54 13). Pick the highest HOTA; if the best G is at a grid edge, extend the grid one step
on that side. v3's own best on the same protocol is 0.5802 (G = 3.0).

Added 2026-11-03 after the box-size correction failed held-out (`../../weight-space/box-delta-train5/RESULT.md`),
before any v4 number: the selection rule is unchanged, but the per-video train-5 HOTA of the chosen
setting is reported beside v3's, with how many of the 5 videos improve.

**Held-out gate, one run:** the selected (checkpoint, G) on the 5 held-out videos, 32 frames
(`track_visem.sh 32`). v4 replaces v3 only if HOTA >= 0.6025 + 0.010 = **0.6125** (v3's 32-frame
result, `../../weight-space/README.md`). Otherwise v3 stays the shipped adapter and this folder
records the negative result.

**Only if the gate passes:** the 1,500-frame rerun (`track_visem.sh 300`), reported beside v3's
0.570, and the README updated to name all three changes.
