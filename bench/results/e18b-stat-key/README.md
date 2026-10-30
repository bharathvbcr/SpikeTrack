# E18b: the CoreML package hash recorded under a stat key (2026-10-30)

Pre-stated in docs/SPEED_PLAN.md (E18b, 1175c14) before the run. The code is a35e80a. One mac_heavy
hold, 13:06-13:09Z, load 10.9-20.8. Binary shas are in `versions.txt`; the script is `job.sh`.

## Gate [V]

- **st-coreml unit tests: 7 of 7 pass.** They cover the stat key (including its documented blind
  spot), when a record is used or refused, and that a package changed mid-hash is not recorded.
- **`load_cache` (ignored, on a `cp -c` clone of the real fp32 package) passes.**
  - The first load compiles and hashes in full: 11.13 s total (hash 3.26 s, compile 0.17 s, model
    7.70 s).
  - The second load does neither, and reports the same SHA-256.
  - Their trunk predictions are bit-identical.
  - The second load timed 0.00 s in every phase. Core ML appears to hand back the model it already
    holds in the process [I].
  - A 0.17 s compile of a 1.7 GB package is far under the ~15 s seen on 2026-10-29
    [unexplained; `compiled true` is read from the missing directory, not from the timing].
- **`track` on VISEM 22, 2 frames:**
  - The first new run hashed in full (4.71 s), found the existing compiled model, and wrote
    `sam3.1_trunk_fp32.mlpackage.stat-fb8823d528c89ec8.sha256` (65 bytes) beside the package
    (`coreml-dir-before.txt`, `coreml-dir-after.txt`).
  - Every later new run read the recorded hash, with a 0.00 s hash phase. The limit was 0.2 s.
- **16 frames of 82 and 22: predictions are byte-identical** to `closing-e15/speed/v*-new-1`.

## Load time, interleaved, VISEM 22

| round | E18 trunk load (hash) | E18b trunk load (hash) | E18 `model loaded` | E18b `model loaded` |
|---|---|---|---|---|
| 1 | 7.5 s (3.73) | 3.1 s (0.00) | 8.0 s | 3.5 s |
| 2 | 5.8 s (3.61) | 0.7 s (0.00) | 6.4 s | 1.5 s |
| 3 | 6.1 s (5.23) | 2.1 s (0.00) | 6.6 s | 2.5 s |
| min of 3 | 5.8 s | **0.7 s** | 6.4 s | **1.5 s** |

- The 1-minute load was 17.5-20.8 for all six runs.
- What is left is Core ML's own model load: 0.38-3.07 s here, and up to 15 s earlier under load. The
  key cannot reach it.
- On a warm machine the CoreML default now starts within ~0.5 s of the Ojas trunk's ~1 s load
  [I: the Ojas start was not re-timed in this hold].
- So the CoreML default no longer loses on short clips the way GAP-COREML-TRUNK-COLD-START
  described (below ~10-17 frames). That break-even was computed from the old 3-12 s load.

Files: the run directories and logs (masks removed), the test and build logs, the coreml directory
listings, `job.sh` and `versions.txt`.
