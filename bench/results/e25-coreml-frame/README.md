# E25: whole-frame time of the CoreML placements (2026-11-03)

E25 measured whole-frame time with CoreML placements against the shipped default. The setup:
- `job.sh`; gates in `RULES.md`, stated before the runs;
- VISEM 36 (held-out, dense), first 32 frames, shipped configuration (LoRA v3 + `lorav3_exact_it_3p00.json`, cap 64);
- 3 interleaved rounds; table in `summary.md` (`tools/analysis/e25_frame.py`).

| arm | push_frame / default, per round | wall / default, per round | tracks vs default |
|---|---|---|---|
| necks32 (`--necks coreml-fp32`) | 0.988, 1.087, 0.949 | 1.066, 1.065, 0.970 | 32/32 id sets, IoU 1.0000 |
| mattn16 (+ `--mem-attn coreml-fp16`) | 0.731, 0.971, 0.705 | 0.712, 1.003, 0.746 | 32/32 id sets, min IoU 0.9919 |
| fast16 | failed: `sam3.1_trunk_fp16.mlpackage` was missing (re-export queued) | | |

**necks32 fails G4 [V].** The neck program alone is 2x faster than Ojas's necks (56 vs ~110 ms).
In the frame it buys nothing, and the wall time is about 6% worse.
- [I] The likely cause is the host round trip around each neck: trunk readback, the copy into Core
  ML's input, three level readbacks (111 MB) and their re-upload, twice per frame.
- It is not offered.

**mattn16 fails G4 as stated, on round 2 [V].**
- Rounds 1 and 3 are 0.71-0.75x on both measures.
- In round 2, frames 21-24 took 3.2-5.2 s against ~1.9 s elsewhere in the run. The load average was
  16.6 when the round started, so another process held the machine [I].
- The rule is "all 3 rounds", so it is not waived. E26 re-measures mattn16 (`../e26-pipeline/`).
