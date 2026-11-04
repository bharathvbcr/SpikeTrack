# E29 rules, written before any run (2026-11-04)

## Why

E28 (`bench/results/profile-fp16/`) profiled the shipped default on held-out VISEM 36 at 1.31 s/frame,
on a quiet machine. The trunk is fully hidden (3 ms of wait). Ojas's Metal device thread records
every op and commits to the GPU only at a read, an upload over 64 KB or a sync (each one waited), or
without a wait after every 256 dispatches (`ojas-metal/src/device.rs`, `OVERLAP_DISPATCHES`). A neck
is about 20 dispatches, so the GPU mostly starts only when the host blocks on a result, and the host
stages E28 shows (~260 ms/frame) do not overlap GPU work. The hypothesis is that committing sooner
lets the GPU run what was recorded while the host records the next ops or does host work.

`MetalBackend::set_overlap_dispatches` (new) and `track --overlap-dispatches N` (new) move only when
recorded work reaches the GPU. The ojas test `set_overlap_dispatches_refuses_zero_and_keeps_the_bits`
and the existing `overlap_commits_from_many_threads_match_a_serial_run_bit_for_bit` check that the
bits do not change.

## Arms

One binary. The shipped configuration as in E27/E28: LoRA v3 + `lorav3_exact_it_3p00.json`, prompt
sperm, `--boxes detector`, `--max-objects 64`, the default fp16 trunk (pipelined) and fp16 memory
attention. Video 36, first 32 frames, 3 rounds, order rotated per round.

- `o256`: no flag (Ojas's default, 256)
- `o32`: `--overlap-dispatches 32`
- `o8`: `--overlap-dispatches 8`
- `o1`: `--overlap-dispatches 1`

Wall is `(total - load) / frames` from run.json, steady frames >= 8 (`tools/analysis/e25_frame.py`).

## Gates

- **O1 (correctness):** every arm's `36.json` is byte-identical to the same round's `o256`, in all
  3 rounds. Any difference fails that arm outright, whatever its speed.
- **O2 (speed):** an arm's wall is below 0.95x the same round's `o256` in all 3 rounds.
- **Adoption:** an arm that passes O1 and O2 can become the default, through `open_metal` in
  `track_cmd.rs`, with the same comment style as `METAL_MEM_CAP`. If more than one passes, the one
  with the lowest median ratio is taken. If none passes, nothing changes, and the result is recorded
  as a negative.
- **Noise floor:** the spread of `o256` across the 3 rounds is reported beside the ratios.
