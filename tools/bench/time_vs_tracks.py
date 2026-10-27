"""Seconds per frame against the number of active tracks, from `spiketrack track` logs.

Reads the per-frame progress lines `spiketrack track` prints
(`frame N (i/T): S s, K active track(s), R frame(s) released`) from one or more logs/<video>.log
files (tools/jobs/track_visem.sh writes them) and prints, per K, the frame count and the min,
median and max seconds per frame, plus the `/usr/bin/time -l` peak memory footprint of each log
when present. The first frame of each log is reported separately: it carries the prompt encode and
first-touch costs. Offline reporting only; nothing here runs at inference time.

Wall time on a shared machine: read every number against the load averages the job log records.

    python3 tools/bench/time_vs_tracks.py bench/results/track/<run>/logs/*.log
"""

import re
import statistics
import sys
from collections import defaultdict

FRAME = re.compile(r"^frame (\d+) \((\d+)/(\d+)\): ([0-9.]+) s, (\d+) active track\(s\)")
FOOTPRINT = re.compile(r"^\s*(\d+)\s+peak memory footprint")


def main(paths):
    if not paths:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    by_k = defaultdict(list)
    first = []
    for path in paths:
        footprint = None
        rows = 0
        with open(path) as f:
            for line in f:
                m = FRAME.match(line)
                if m:
                    secs, k = float(m.group(4)), int(m.group(5))
                    if m.group(2) == "1":
                        first.append((path, secs, k))
                    else:
                        by_k[k].append(secs)
                    rows += 1
                    continue
                m = FOOTPRINT.match(line)
                if m:
                    footprint = int(m.group(1))
        mem = f"{footprint / 2**30:.2f} GiB peak footprint" if footprint is not None else "no /usr/bin/time summary"
        print(f"{path}: {rows} frame lines, {mem}")
    if not by_k:
        print("no frame lines after the first frame of each log", file=sys.stderr)
        return 1
    print()
    print("| active tracks | frames | min s | median s | max s |")
    print("|---|---|---|---|---|")
    for k in sorted(by_k):
        v = by_k[k]
        print(f"| {k} | {len(v)} | {min(v):.2f} | {statistics.median(v):.2f} | {max(v):.2f} |")
    print()
    for path, secs, k in first:
        print(f"first frame of {path}: {secs:.2f} s with {k} active track(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
