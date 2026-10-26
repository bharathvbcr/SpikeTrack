# SpikeTrack

SAM 3.1, Meta's Segment Anything with its Object Multiplex video tracker, built as one Rust binary
with no Python at runtime. It runs on Apple Silicon on tessl/Ojas. The evaluation is many-sperm
tracking in VISEM-Tracking. Built for the Meta hackathon, Oct 26 to Nov 4, 2026.

Status and plan: [HANDOFF.md](HANDOFF.md) (schedule, gates, task index), [tasks/](tasks/), open
questions in [gaps.jsonl](gaps.jsonl). What existed before the hackathon and what was built during
it: [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## Layout

| Path | What |
|---|---|
| `crates/st-weights` | The SAM 3.1 checkpoint (safetensors) mapped to typed structs. Every tensor is claimed exactly once. |
| `crates/st-model` | PE ViT trunk and the three FPN necks on the Ojas `Backend` |
| `crates/st-eval` | HOTA (TrackEval definition), DAVIS J&F, and the VISEM-Tracking loader |
| `tools/reference` | Offline Python only: the official SAM 3 code on CPU, fixtures, oracles, the zero-shot and exemplar checks |
| `bench/results` | Every reported number, with the command that produced it |

## Running

Weights and data are not in the repository. Get `facebook/sam3.1` from Hugging Face (gated, SAM
License) and VISEM-Tracking from Zenodo 7293726. Convert the checkpoint once, offline:
`tools/reference/.venv/bin/python -s tools/reference/convert_checkpoint.py`
(the venv setup is in `tools/reference/README.md`).

Heavy jobs run under the machine's lock: `tools/locked.sh <label> <command>`.

```sh
tools/locked.sh test cargo test --release -p st-weights -p st-eval -- --test-threads=2
# with the gated checkpoint and the fixtures (tools/reference/dump_fixtures.py):
SPIKETRACK_WEIGHTS=~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors \
SPIKETRACK_FIXTURES=fixtures/backbone \
tools/locked.sh parity cargo test --release -p st-model --test backbone_parity -- --ignored --nocapture
```

## Rules

Meta's Official Rules have not been read yet. They are on the hackathon portal, and only Bharath
can reach them (gaps.jsonl `GAP-OFFICIAL-RULES-UNREAD`). Their wording on prior work, required
Meta models, repository visibility and the demo video goes here once read.

## Licences

Our code is offered under MIT or Apache-2.0, pending GAP-SAM-LICENSE-PORT-TERMS. SAM 3.1 is under
the SAM License; see NOTICE. VISEM-Tracking is CC BY 4.0.
