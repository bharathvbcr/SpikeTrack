# Memory attention: does the upstream parity test discriminate the head count? (2026-11-01)

Question (from `tools/reference/README.md`, "Memory attention head count"): `memory_attention_matches_upstream`
passes at `HEADS = 8`, but a comparison is evidence only if it could have come out the other way. The control runs the
same test with `HEADS = 1`.

Result: **it discriminates.** Same CPU test, same checkpoint, same fixtures, same bound (`> 0.999`, not adjusted):

| run | worst cosine | max abs / max ref | outcome | log |
|---|---|---|---|---|
| `HEADS = 8` (the code as committed) | 1.0000000 | 7.9e-6 to 6.6e-5 | ok | `baseline-heads8.log` |
| `HEADS = 1` (`HEAD_DIM = 256`) | 0.6070227 | 0.91 to 1.0 | FAILED at `tracker_parity.rs:180` (the cosine assert) | `control-heads1.log` |

At `HEADS = 1` all 7 frames fall between cosine 0.607 and 0.652, so it is a different function and not a near miss.
The failure is the test's own numeric bound, not a panic or a shape error.

## How it was run
`HEADS` is a `const` in `crates/st-model/src/tracker/memory_attention.rs:21`, and that file was not to be edited, so the
control ran on an untracked copy outside the repository, with one changed line (`pub const HEADS: usize = 1;`):

    cp Cargo.toml Cargo.lock ../SpikeTrack-headctl/ && rsync -a --exclude target crates ../SpikeTrack-headctl/
    sed -i '' 's/^pub const HEADS: usize = 8;/pub const HEADS: usize = 1;/' ../SpikeTrack-headctl/crates/st-model/src/tracker/memory_attention.rs
    cp -Rc target ../SpikeTrack-headctl/target && touch ../SpikeTrack-headctl/crates/st-model/src/tracker/memory_attention.rs
    SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors \
    SPIKETRACK_TRACKER_FIXTURES=fixtures/tracker/judo.safetensors \
    cargo test --release -p st-model --test tracker_parity -- --ignored --nocapture --test-threads=1 --exact memory_attention_matches_upstream

The same command, run in this repository, is the `HEADS = 8` baseline. No source file in the repository was changed.

## Caveats
- **A first control attempt was invalid and is not recorded as a result.** It shared this repository's `target/`, whose
  build was newer than the patched copy's file, so cargo reused the `HEADS = 8` binary and printed the baseline's
  numbers to the digit. The valid run uses its own cloned target dir and the log shows `Compiling st-model` from the
  copy. Anyone repeating this must give the copy its own target dir and touch the patched file.
- **Different ojas commits.** The shared ojas checkout was being edited by other sessions while this ran: the baseline
  built against ojas `46fb0ec` and the control against `be9e31a` (checkout HEAD `6baaf85` afterwards; the working tree
  was dirty). A gap of 0.61 against 1.0 is far larger than any change expected from that, but the two runs are not
  pinned to one commit.
- **CPU only, one clip.** CPU backend; the DAVIS `judo` clip, 8 frames (7 memory-attention calls), not the VISEM data.
  The Metal twin was not run.
- **Not a full-coverage result.** This shows the test is sensitive to the head count on this clip. It does not
  show the model is right on other inputs.
- Not committed as a Rust test: parameterising `HEADS` needs an edit to `memory_attention.rs`, which was out of scope.
