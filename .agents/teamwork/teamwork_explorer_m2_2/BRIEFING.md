# BRIEFING — 2026-10-31T18:30:00Z

## Mission
Investigate and design authoritative budget management, cache trim ladder, and fail-closed placement enforcement for Milestone 2.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, synthesizer
- Working directory: /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_2
- Original parent: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Milestone: Milestone 2: Unified Architecture & Shared Memory Pools

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Feature 8: ojas_core::Budget as sole authoritative root ledger
- tessl::BufferPool under child budget derived from ojas_core::Budget
- Cache trim ladder trims pool.cache before OjasError::CapacityExceeded is returned
- Feature 10: Fail-closed placement enforcement (no silent fallback, return OjasError::Placement on un-transferred CPU access to Metal tensor)
- Compatibility with Tier 4 Scenario D (tier4_scenario_d_no_silent_fallback_placement_invariance)

## Current Parent
- Conversation ID: 7f1f02db-7cbf-40d1-8919-b2741643d465
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `ojas-core/src/budget.rs`: Budget hierarchy, atomic reservations, `charge_chain`, child budgets, scratch vectors
  - `ojas-core/src/tensor.rs`: Storage, payload (`Payload::Host`, `Payload::Device`), `f32_slice`, `host_window`, `to_host`, `from_device_reserved`
  - `ojas-core/src/backend.rs`: `BackendId`, `Backend::upload`, `Backend::download`, placement contracts
  - `ojas-core/src/error.rs`: `OjasError::Placement`, `OjasError::CapacityExceeded`, typed errors
  - `ojas-cpu/src/backend.rs` & `ojas-cpu/src/validate.rs`: CPU operand validation, `f32_slice()` placement refusal
  - `ojas-metal/src/backend.rs`, `src/device.rs`, `src/memory.rs`: `MetalBackend::arg`, `MetalBackend::outputs`, `device.rs::retrying`, string-scraping allocation classification
  - `tessl/src/runtime.rs`: `BufferPool`, `freelist`, `cached_bytes`, `max_cache_bytes`, `alloc`, `recycle`, `set_max_cache_bytes`
  - `tessl/Cargo.toml` & `ojas-metal/Cargo.toml`: dependency flow `ojas-metal -> tessl -> ojas-core`
  - `ojas-oracle/tests/e2e_tier4_production.rs`: Scenario C (pipeline budget isolation) and Scenario D (fail-closed placement invariance)
- **Key findings**:
  1. Feature 8: `tessl::BufferPool` currently has no connection to `ojas_core::Budget`, hoarding up to 2 GiB in `freelist` outside the budget ledger.
  2. Solution designed: `BufferPool` allocates under `child_budget = root_budget.child(max_cache_bytes)`. Each `CachedBuffer` in `freelist` holds a `Reservation`.
  3. Trim Ladder: `Budget` gains `BudgetTrimHandler`. When `try_reserve` would fail with `CapacityExceeded`, `Budget` queries `trim_handler` to release idle buffers before failing.
  4. String Scraping Elimination: `classify_alloc` and `is_new_buffer_failure` in `ojas-metal/src/device.rs` replaced with typed `AllocError` from `ojas-core`/`tessl`.
  5. Feature 10: Fail-closed placement enforcement verified. `host_window()` and `f32_slice()` refuse any tensor where `device() != None` with `OjasError::Placement`. Zero-copy UMA uses `to_host()` timeline fence without memcpy, preserving placement semantics.
  6. E2E Tier 4 test Scenario D passes and verified compatible with cross-backend placement assertions.
- **Unexplored areas**: None within scope. Ready for handoff report synthesis.

## Key Decisions Made
- Architecture verified: `ojas-metal -> tessl -> ojas-core` is clean and acyclic.
- `BudgetTrimHandler` in `ojas-core` allows lock-free `Budget` to request cache trims from `BufferPool`.
- `CachedBuffer` holding `Reservation` guarantees zero accounting drift when buffers move between active tensors and the freelist.
- Placement remains strictly explicit; zero-copy UMA is unlocked only through `to_host()` / `upload()` timeline fences.

## Artifact Index
- /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_2/DISPATCH.md — Dispatch log
- /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_2/BRIEFING.md — Situational awareness
- /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_2/progress.md — Liveness heartbeat
- /Users/bharath/Code/research/SpikeTrack/.agents/teamwork/teamwork_explorer_m2_2/handoff.md — Final investigation report
