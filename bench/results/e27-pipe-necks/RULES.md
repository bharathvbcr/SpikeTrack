# E27 rules (pre-stated 2026-11-04, before any run of necks on the pipeline worker)

## Candidate

With a pipelined CoreML trunk, `--necks coreml-*` now runs both neck programs on the trunk's worker
thread, right after that frame's trunk. Frame t + 1's necks then run while frame t is tracked. The
main thread only uploads the levels.

The neck programs are the ones E25 gated: G1 fp32 rel L2 1.3e-6 / 9.0e-7, and 32/32 id sets at
IoU 1.0000 against Ojas's necks. E25's necks32 failed G4 at 1.07x wall when the necks ran inline on
the main thread.

## Gates

- **N1, exactness of the move.** On every round, m16-n32 must reach 32/32 equal id sets against m16
  with min box IoU >= 0.999. That is E25's necks32 result, so the move cannot have changed the
  numbers.
- **N2, speed.** In all 3 rounds, wall per frame must be below 0.95x the same precision's arm
  without CoreML necks: m16-n32 against m16, and fast16-n16 against fast16.

## Outcomes

- If N1 and N2 pass, `--necks coreml-fp32` is offered with the pipeline. It becomes the Metal
  default only with G2 (`../track/held32-coreml/`, its own config), per
  `../e25-coreml-frame/RULES.md`.
- fast16-n16 is folded into the fast16 accuracy gates (G2/G3) if N2 passes.
- If N2 fails, the necks stay on Ojas.
