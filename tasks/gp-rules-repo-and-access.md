---
id: "gp-rules-repo-and-access"
title: "Rules, repository, licenses and gated access"
status: "ready"
priority: 0
severity: "critical"
type: "chore"
owner: "unassigned"
due: "2026-10-26"
labels:
  - "hackathon"
  - "compliance"
  - "repo"
repositories:
  - "spiketrack"
planned_files:
  - "README.md"
  - "CONTRIBUTIONS.md"
  - "LICENSE-MIT"
  - "LICENSE-APACHE"
  - "NOTICE"
  - "Cargo.toml"
  - ".gitignore"
  - "gaps.jsonl"
acceptance_criteria:
  - "Official Rules read; prior-work, Meta-model, repo and video requirements quoted with links in README 'Rules' section"
  - "HF access requested for facebook/sam3.1 and facebook/sam3; grant status recorded"
  - "SAM License redistribution/attribution terms summarized in NOTICE; weights and datasets ignored by git"
  - "VISEM-Tracking and SA-V license and access recorded; whether VISEM has per-object track IDs recorded [V]"
  - "Workspace builds empty crates with `cargo check` under mac_heavy.sh"
  - "CONTRIBUTIONS.md lists pre-existing libraries with pinned commits and states that all code in this repo is hackathon work"
---
# Task brief v1

## Title
Rules, repository, licenses and gated access

Task: gp-rules-repo-and-access
Type: chore
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-10-26
Labels: hackathon, compliance, repo

## Repositories
- spiketrack

## Description
Set up the hackathon repository so every later task lands in a compliant place.

- Read Meta's Official Rules (published before Oct 26) and the hackathon page. Record, with quotes under 15 words and links, what they say about: prior work and existing open-source code, required use of Meta models, repository visibility, demo video length, submission fields, judging criteria for the Indie track. [U] until read.
- Request access to `facebook/sam3.1` and `facebook/sam3` on Hugging Face (gated, SAM License, "other") [V]. Read the SAM License for redistribution and attribution terms; record them in NOTICE. Never commit weights.
- Check dataset licenses and access: VISEM-Tracking (Simula; 20 videos, ~30 s each, 29,196 frames, bounding-box annotations [V]; license and track IDs [U]); SA-V (Meta; access form [U]).
- Create the Cargo workspace (crates listed in gp-weights-loader and later briefs), dual MIT/Apache license for our code, `.gitignore` for weights/data/fixtures.
- `CONTRIBUTIONS.md` separates **pre-existing** (Ojas, tessl, sparsl, at named commits) from **built during the hackathon** (this repo). Pin the Ojas/tessl/sparsl commits used as git dependencies.
- Ask Bharath before creating the public GitHub repo or pushing.

### Second audit (2026-10-25)
- Official Rules: not found by search on Oct 25 [U]; read them from the hackathon portal after the kickoff email.
- No code before the hackathon starts on Oct 26. This repo holds only planning files (HANDOFF.md, tasks/, gaps.jsonl) made on Oct 25; say so in CONTRIBUTIONS.md.
- Do not use the unofficial weight mirror the CoreML port mentions; wait for gated access. If access is not granted by D2 evening, raise it in the hackathon Discord/office hours.

- Update [V, arXiv 2212.02842]: VISEM-Tracking has per-sperm track IDs, YOLO boxes, 3 classes, 45–50 fps, CC BY 4.0, Zenodo 7293726.

See HANDOFF.md for rules, schedule and dependencies.

## Acceptance criteria
- [ ] Official Rules read; prior-work, Meta-model, repo and video requirements quoted with links in README 'Rules' section
- [ ] HF access requested for facebook/sam3.1 and facebook/sam3; grant status recorded
- [ ] SAM License redistribution/attribution terms summarized in NOTICE; weights and datasets ignored by git
- [ ] VISEM-Tracking and SA-V license and access recorded; whether VISEM has per-object track IDs recorded [V]
- [ ] Workspace builds empty crates with `cargo check` under mac_heavy.sh
- [ ] CONTRIBUTIONS.md lists pre-existing libraries with pinned commits and states that all code in this repo is hackathon work

## Planned files
- README.md
- CONTRIBUTIONS.md
- LICENSE-MIT
- LICENSE-APACHE
- NOTICE
- Cargo.toml
- .gitignore
- gaps.jsonl
