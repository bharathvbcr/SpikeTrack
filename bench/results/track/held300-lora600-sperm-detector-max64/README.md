# `--max-objects 64` on the dense held-out videos

**Superseded 2026-10-29.** Video 21 was capped too. It has 16 active tracks at frame 0 at cap 16, but
its row here was copied from the cap-16 run. All five videos re-run at 64 score **0.4922** / DetA
0.3734 / AssA 0.6903; video 21 alone moves, 0.6204 -> 0.6422
(`../held300-lora600-sperm-detector-f32-max64`, `../../coreml-trunk/README.md`). The 0.4868 below is
kept as the record of this run.

Measured 2026-10-28, Metal, ojas cbd5565. Same setup as `../held300-lora600-sperm-detector`:
300 frames per video, prompt "sperm", `--boxes detector`, LoRA `fixtures/lora_runs/visem-lora-r16`.
The only change is `--max-objects 64` (upstream's builder default, which that run used, is 16).
Only videos 36 and 82 were rerun. In that run they were the two videos that reached 16 active tracks.
The 21, 22 and 47 run files are copied from that run, so their rows are identical.

Before these runs, the chain re-ran the Metal session parity test on VISEM 22 (default cap). The
result was unchanged: 32/32 id sets equal, mean mask IoU 0.99855
(`../../parity-tracker/metal-f32-detect-track-visem22.log`).

| video | cap 16: HOTA / DetA / AssA | cap 64: HOTA / DetA / AssA | peak active tracks (16 → 64 cap) | pred boxes |
|---|---|---|---|---|
| 36 | 0.3261 / 0.1839 / 0.5923 | 0.3518 / 0.2371 / 0.5352 | 16 → 28 | 4718 → 6338 |
| 82 | 0.4412 / 0.2788 / 0.7978 | 0.4860 / 0.3726 / 0.7443 | 16 → 24 | 4760 → 6515 |
| all 5 | 0.4695 / 0.3198 / 0.7250 | 0.4868 / 0.3632 / 0.6948 | | 18738 → 22113 |

- The 16 cap was binding on both videos. With the 64 cap neither video reached even 32 tracks.
- Detection accuracy rose: more sperm in each frame got a track. Association accuracy fell. This fits
  the extra tracks being shorter or more often swapped. That is inferred from the DetA/AssA split and
  was not checked per track.
- Mean s/frame: 36 went from 4.62 to 4.45, and 82 from 3.79 to 4.15. The runs were hours apart under
  different machine load and were not paired, so these numbers say nothing about what extra tracks
  cost in time.

Files: `run/*.json` (tracks), `run/*.run.json` (run settings and timing), `logs/{36,82}.log` (per-frame
telemetry), `eval.{txt,json}`. Masks are not committed, the same as in the other held-out runs.
