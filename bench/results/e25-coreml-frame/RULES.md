# Adopting CoreML placements (pre-stated 2026-11-03, before any held-out run of them)

The candidates are:
- `--necks coreml-fp32|coreml-fp16`: the detector and propagation necks;
- `--mem-attn coreml-fp16|coreml-fp32`: propagation's memory attention;
- `--trunk-precision coreml-fp16`: the trunk.

Each runs as an exported CoreML ML Program from `tools/baselines/coreml_{necks,mattn,trunk}.py`. The
shipped configuration is LoRA v3 + `lorav3_exact_it_3p00.json`, prompt `sperm`, `--boxes detector`,
cap 64, `--trunk-precision coreml-fp32`, with necks and memory attention on Ojas.

**G1, module parity** (each package against upstream's fixture, `bench/results/coreml-{mattn,necks}/`).
An fp32 package must reach relative L2 ≤ 2.7e-5, the worst the fp32 trunk package reached when it
was adopted. fp16 packages are recorded, not gated here.

**G2, held-out accuracy at 32 frames** (`tools/jobs/track_visem.sh 32`, all 5 held-out videos).
- An fp32 placement becomes a default only if HOTA is within ±0.002 of v3's 0.6025 and no video
  moves more than 0.005 (21 0.6744, 22 0.6799, 36 0.5752, 47 0.5596, 82 0.5552).
- An fp16 placement can become a default only if HOTA ≥ 0.6005 and no video drops more than 0.01,
  and the 1,500-frame run (G3) passes too.

**G3, the README claim** (fp16 defaults only). The 1,500-frame held-out run must reach HOTA ≥ 0.567
(0.570 − 0.003), with no video more than 0.01 below 21 0.669, 22 0.669, 36 0.533, 47 0.484, 82 0.529.

**G4, speed** (E25, `summary.md`, interleaved, 3 rounds). The arm's whole-frame time must be below
0.95x the default arm's in all 3 rounds.

**Outcomes:**
- A placement that passes G1, G2, G4 (and G3 for fp16) becomes the Metal default.
- One that is faster (G4) but fails G2 or G3 ships only as an opt-in flag, with its measured HOTA
  stated next to it.
- One that fails G4 is recorded and not offered.
