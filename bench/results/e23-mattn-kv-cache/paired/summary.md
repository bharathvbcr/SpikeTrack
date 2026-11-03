# E23 paired speed run

## Mechanism (`--profile ops`, each op synced)

| run | memory attention ms / frame | linears over 5,184 rows ms / frame |
|---|---|---|
| ops-1-ctl | 298.8 | 28.3 |
| ops-2-new | 277.1 | 0.0 |
| ops-3-new | 272.0 | 0.0 |
| ops-4-ctl | 299.9 | 28.3 |

Mean A − B, memory attention: 24.8 ms / frame; A's big linears: 28.3 ms / frame; B's: 0.0.

## Wall clock (median tracker step, frames >= 8)

| pair | kind | order | first ms | second ms | value | mean load | flagged |
|---|---|---|---|---|---|---|---|
| 1 | A/B | ctl,new | 435.0 | 404.9 | -0.0717 | 5.6 |  |
| 2 | ctl/ctl | ctl,ctl | 443.8 | 433.1 | -0.0245 | 4.5 |  |
| 3 | A/B | new,ctl | 390.1 | 435.9 | -0.1110 | 4.2 |  |
| 4 | new/new | new,new | 401.7 | 400.5 | -0.0031 | 4.3 |  |
| 5 | A/B | ctl,new | 435.1 | 403.9 | -0.0746 | 6.1 |  |
| 6 | A/B | new,ctl | 418.7 | 445.0 | -0.0608 | 6.5 |  |
| 7 | ctl/ctl | ctl,ctl | 454.1 | 439.7 | -0.0321 | 6.7 |  |
| 8 | A/B | ctl,new | 441.5 | 389.1 | -0.1264 | 5.5 |  |
| 9 | new/new | new,new | 382.8 | 401.8 | +0.0485 | 3.6 |  |
| 10 | A/B | new,ctl | 380.5 | 411.0 | -0.0772 | 4.8 |  |
| 11 | A/B | ctl,new | 411.5 | 406.1 | -0.0132 | 5.0 |  |
| 12 | ctl/ctl | ctl,ctl | 425.2 | 421.2 | -0.0096 | 4.7 |  |
| 13 | A/B | new,ctl | 383.8 | 420.3 | -0.0910 | 5.1 |  |
| 14 | new/new | new,new | 383.7 | 383.3 | -0.0010 | 5.0 |  |
| 15 | A/B | ctl,new | 427.7 | 436.3 | +0.0201 | 6.8 |  |
| 16 | A/B | new,ctl | 402.9 | 426.8 | -0.0574 | 8.8 | yes |

### all pairs (decides): 10 A/B pairs, 6 same-binary pairs
- Same-binary mean ln ratio -0.0037, 95% CI [-0.0334, +0.0260] (-3.3% to +2.6%).
- A/B mean ln(B/A) -0.0663, 95% CI [-0.0973, -0.0354]: B/A -6.4% (-9.3% to -3.5%); 9 of 10 pairs below 0 (needed 8).
- Verdict: **SPEEDUP**.

### unflagged pairs only: 9 A/B pairs, 6 same-binary pairs
- Same-binary mean ln ratio -0.0037, 95% CI [-0.0334, +0.0260] (-3.3% to +2.6%).
- A/B mean ln(B/A) -0.0673, 95% CI [-0.1025, -0.0321]: B/A -6.5% (-9.7% to -3.2%); 8 of 9 pairs below 0 (needed 7).
- Verdict: **SPEEDUP**.

Descriptive: Spearman rho of A/B ln ratio against mean load = +0.59 (n = 10).
