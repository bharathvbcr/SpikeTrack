# Scaled v3 task vector: negative, stopped at selection (2026-11-03)

The rules were written in [PLAN.md](PLAN.md) before any run. No value passed rule 2, so there was
no held-out run, and v3 + `lorav3_exact_it_3p00.json` stays shipped.

Train-5 results, first 8 frames, prompt `sperm`, `--boxes detector`, cap 64, G 3.0. Each run
merges `lambda * delta_v3` exactly: `alpha` is scaled and the factors are unchanged. The
`lambda 1.0` row is v3 itself (`../../weight-space/box-delta-train5/eval-0a.txt`).

| lambda | 14 | 30 | 19 | 54 | 13 | all | videos ≥ v3 |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.6871 | **0.7146** | 0.5056 | 0.4313 | 0.5789 | 0.5573 | 1 of 5 |
| 0.9 | 0.6813 | **0.7119** | 0.4894 | 0.4530 | 0.6021 | 0.5680 | 1 of 5 |
| 1.0 (v3) | 0.6918 | 0.6973 | 0.5062 | 0.4660 | 0.6256 | **0.5802** | - |
| 1.1 | 0.6738 | 0.6720 | 0.5049 | **0.4919** | **0.6316** | 0.5818 | 2 of 5 |
| 1.2 | 0.6739 | 0.6549 | 0.5056 | **0.4966** | **0.6301** | 0.5802 | 2 of 5 |

- **Rule 2 fails at every lambda.** No value holds 4 of 5 videos. The best "all" score, at
  `lambda` 1.1, is +0.0016, short of the required +0.005. The winner is not at a grid edge, so
  the grid does not extend.
- **v3 is a peak on train-5.** Moving `lambda` either way lowers the "all" score or leaves it
  flat.
- **Each direction trades the same videos.**
  - Scaling down helps the sparse video 30 and hurts the dense videos 54 and 13.
  - Scaling up does the reverse.

  The box-size shift and the v3/v4 soup showed the same split. The selection videos disagree
  with each other: the sparse ones (14 with 24 boxes, 30 with 96) and the dense ones (54 with
  216, 13 with 350) want opposite moves. So every global edit tried so far buys one group with
  the other.

Not done, because rule 2 stopped the run: the held-out 32-frame gate (rule 3) and the 1,500-frame
run (rule 4).

Files:
- `select-job.sh`, `select-job.log` and `select/` (run JSON, evals, `scale.log`);
- the builder, `tools/reference/lora_soup.py`, run as `v3 zero lambda`;
- the adapters, in `fixtures/lora_runs/scale-v3/` (git-ignored, rebuilt by the job).
