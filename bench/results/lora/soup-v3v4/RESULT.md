# v3/v4 adapter soup: negative, stopped at selection (2026-11-03)

The rules were written in [PLAN.md](PLAN.md) before any run. No blend passed rule 2, so there
was no held-out run, and v3 + `lorav3_exact_it_3p00.json` stays shipped.

Train-5 results, first 8 frames, prompt `sperm`, `--boxes detector`, cap 64, all at G 3.0.
"c" is the share of v3; `c 0` is plain v4 step 300 (`../visem-lora-v4/select/eval-300-3.0.txt`).

| c (share of v3) | 14 | 30 | 19 | 54 | 13 | all | videos ≥ v3 |
|---|---|---|---|---|---|---|---|
| 1.0, parity: v3 re-factored | 0.6918 | 0.6973 | 0.5062 | 0.4660 | 0.6256 | **0.5802** | 5 of 5 (identical) |
| 0.75 | 0.6864 | 0.6935 | 0.5022 | 0.4668 | 0.6266 | 0.5793 | 2 of 5 |
| 0.5 | 0.6821 | 0.6966 | 0.5104 | 0.4790 | 0.6368 | 0.5878 | 3 of 5 |
| 0.25 | 0.6917 | 0.6935 | 0.5068 | 0.4794 | 0.6428 | 0.5898 | 3 of 5 |
| 0, v4 s300 | 0.6967 | 0.6922 | 0.5080 | 0.4842 | 0.6450 | 0.5921 | 4 of 5 |

- **Parity (rule 0) passes.** The c = 1.0 soup re-factors v3's update by SVD at rank 32, with a
  worst relative error of 4.4e-8. It reproduces v3's train-5 table digit for digit, so
  `lora_soup.py` and the Rust loader agree.
- **Rule 2 fails at every c.** No blend has 4 of 5 videos at or above v3. Every blend loses on
  14 and 30 (c 0.25 misses 14 by 0.0001 and 30 by 0.0038), while all the gains come from the
  dense videos 54 and 13. That is the same per-video pattern as the rejected box-size shift.
- **No soup bonus.** The "all" row rises almost monotonically from v3 (0.5802) to v4 (0.5921), and
  no blend beats v4. A blend that worked would score above both ends. These scores are what
  sliding from v3 toward v4 gives, and v4 already tied v3 on held-out (0.5995 against 0.6025). So
  a held-out run of a blend is expected to tie as well, and the pre-stated rule does not allow
  one.

Not done, because rule 2 stopped the run:
- the G 3.3 and 3.6 temperature step (rule 3);
- the held-out 32-frame gate (rule 4);
- the 1,500-frame run (rule 5).

Files:
- `select-job.sh`, `select-job.log` and `select/` (run JSON, evals, `soup.log`);
- the builder, `tools/reference/lora_soup.py`;
- the soup adapters, in `fixtures/lora_runs/soup-v3v4/` (git-ignored, 9.1 MB each, rebuilt by the
  job).
