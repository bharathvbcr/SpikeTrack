# Contributions: what existed before the hackathon, and what was built during it

Hackathon window: Mon 2026-10-26 to Wed 2026-11-04, 8:00 PM CT.
How to read the Official Rules on prior work is still open (gaps.jsonl `GAP-OFFICIAL-RULES-UNREAD`).
This file records dates and commits so they can be classified once the rules are read.

## Pre-existing libraries (used as dependencies, not hackathon work)

| Library | What SpikeTrack uses | Pinned commit |
|---|---|---|
| [Ojas](https://github.com/bharathvbcr/ojas) | `ojas-io` safetensors reader; Backend trait and CPU ops | base `83d7a29` (ojas main on the morning of 2026-10-26, before any SpikeTrack op work). **The workspace builds against `d2922fb`**, which is that base plus the Oct 26 SAM ops (see below). |
| tessl (Metal 4 runtime, part of the Ojas stack) | GEMM, attention, LayerNorm/GELU kernels (planned for st-model) | `789a132` |
| [sparsl](https://github.com/bharathvbcr/sparsl) | bitpacked spikes, gather/scatter (planned for st-gate) | `44edd92` |

## Third-party reference code and assets (not ours)

| Item | Use | Version |
|---|---|---|
| facebookresearch/sam3 (SAM License) | Offline reference only (`tools/reference`), never linked or run by the binary | `f1647e8` plus local CPU patch `tools/reference/sam3-cpu.patch` |
| SAM 3.1 weights `facebook/sam3.1` (SAM License, gated) | Model weights; never committed | `sam3.1_multiplex.pt`, SHA-256 `0567debe…41cb6` |
| VISEM-Tracking (CC BY 4.0, Zenodo 7293726) | Evaluation data; never committed | v4 train split, 20 videos |

## Built during the hackathon (this repository)

Everything in this repository other than the planning files is hackathon work. The planning files
(HANDOFF.md, COMPONENTS.md, IMPROVEMENTS.md, tasks/, an empty gaps.jsonl) were written on
2026-10-25, before the start, and contain no code. They were committed unchanged as the first
commit on 2026-10-26.

| Date | Work | Where |
|---|---|---|
| 2026-10-26 | Repository, workspace, licences, this file | root |
| 2026-10-26 | CPU-runnable reference: shims, device patch, model builder, checkpoint→safetensors converter, fp32 backbone fixtures, independent fixture checks | `tools/reference/` |
| 2026-10-26 | Typed checkpoint loader with claim-every-tensor checks | `crates/st-weights/` |

## Work in the pre-existing libraries during the window

Session ojas-78 is building SAM-oriented ops in Ojas from 2026-10-26: bidirectional and cross
attention, LayerNorm/GELU/bias, and conv/RoPE-2D/resize. That session logs the commit hashes and
timestamps. Whether that work counts as hackathon work or library work depends on the Official
Rules. It is listed here, with commits, before submission either way.

| Ojas range | Content | Dates | Classification |
|---|---|---|---|
| `83d7a29..d2922fb` | `attention_forward` (bidirectional/cross, bias, key_lens), `layer_norm`, `channel_layer_norm`, `gelu_erf`, `linear_bias`, `broadcast_*`, `conv2d`, `conv_transpose2d`, `max_pool2d`, bilinear interpolation, `rope_interleaved`, axial RoPE tables, window partition/merge, pixel shuffle (CPU; some Metal) | 2026-10-26 (commit times in ojas-78's report) | pending the Official Rules |

Reproducibility: `d2922fb` exists only in the local ojas repository (unpushed), so the `file://`
git dependencies in Cargo.toml build on this machine only. Before submission, the Ojas commits
SpikeTrack uses must be pushed, and Cargo.toml must point at the public repository.
