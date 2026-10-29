"""Sum a `track --profile ops` report by (stage, op) and per steady frame (throwaway analysis;
nothing ships). Frame 0 rows ("frame 0: ...") are left out.

    python3 tools/analysis/profile_ops_by_kind.py OUT/82.profile-ops.md STEADY_FRAMES [STAGE...]
"""
import sys
from collections import defaultdict

path, steady = sys.argv[1], int(sys.argv[2])
want = sys.argv[3:]
by = defaultdict(float)
calls = defaultdict(int)
for line in open(path):
    cells = [c.strip() for c in line.split("|")[1:-1]]
    if len(cells) != 9 or cells[0] in ("stage", "---") or cells[0].startswith("frame 0:"):
        continue
    stage, op, _shapes, n, total = cells[0], cells[1], cells[2], int(cells[3]), float(cells[4])
    if want and stage not in want:
        continue
    by[(stage, op)] += total
    calls[(stage, op)] += n
stages = sorted({s for s, _ in by}, key=lambda s: -sum(v for (t, _), v in by.items() if t == s))
for s in stages:
    rows = sorted(((v, op) for (t, op), v in by.items() if t == s), reverse=True)
    tot = sum(v for v, _ in rows)
    print(f"{s}: {tot / steady:.1f} ms / steady frame (op-synced)")
    for v, op in rows[:8]:
        print(f"    {op:28s} {v / steady:7.1f} ms/frame  {calls[(s, op)] / steady:6.1f} calls/frame")
