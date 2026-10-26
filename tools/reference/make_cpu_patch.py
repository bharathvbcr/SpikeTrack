"""Rewrite the official SAM 3 tree's hardcoded CUDA placements to a device
chosen by SAM3_DEVICE (default "cuda", so CUDA runs are unchanged).

Run once on a branch of external/sam3; the resulting diff is committed here as
sam3-cpu.patch. Autocast contexts (device_type="cuda") are left alone: on a
CPU-only torch they disable themselves, which is what fp32 fixtures want.
"""

import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]) / "sam3"
helper = root / "_device.py"
helper.write_text(
    '"""Device for placements the upstream code hardcodes to CUDA (SpikeTrack CPU patch)."""\n\n'
    "import os\n\nimport torch\n\n\n"
    "def _sam3_device():\n"
    '    return torch.device(os.environ.get("SAM3_DEVICE", "cuda"))\n'
)

rules = [
    (re.compile(r"\.cuda\(\)"), ".to(_sam3_device())"),
    # .cuda(non_blocking=True) and friends (the memory path in video_tracking_multiplex.py).
    (re.compile(r"\.cuda\(([^)]+)\)"), r".to(_sam3_device(), \1)"),
    # io_utils' video loader asks for the current CUDA device unconditionally.
    (
        re.compile(r"else torch\.cuda\.current_device\(\)$"),
        "else (torch.cuda.current_device() if torch.cuda.is_available() else None)",
    ),
    (re.compile(r'torch\.device\("cuda"\)'), "_sam3_device()"),
    (re.compile(r'device="cuda"(?!\s+if)(?=\s*(?:[,)]|$))'), "device=_sam3_device()"),
]
multiline_cuda = re.compile(r"\.cuda\((\s*\n\s*)([^)]+?)(\s*)\)")
skip = {"_device.py"}
changed = []
for path in sorted(root.rglob("*.py")):
    if path.name in skip or "/train/" in str(path) or "/eval/" in str(path):
        continue
    src = path.read_text()
    out_lines = []
    n = 0
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        # Leave the builder's explicit `if device == "cuda": model.cuda()` branch
        # and any comment alone.
        if stripped.startswith("#") or 'if device == "cuda"' in line:
            out_lines.append(line)
            continue
        new = line
        for pat, rep in rules:
            new = pat.sub(rep, new)
        n += new != line
        out_lines.append(new)
    text = "".join(out_lines)
    # A call split over lines: `.cuda(\n    non_blocking=True\n)`.
    text, k = multiline_cuda.subn(r".to(\1_sam3_device(), \2\3)", text)
    n += k
    if n:
        # Insert the import after the module docstring / first import block.
        m = re.search(r"^(import |from )", text, re.M)
        assert m, path
        text = text[: m.start()] + "from sam3._device import _sam3_device\n" + text[m.start():]
        path.write_text(text)
        changed.append((str(path.relative_to(root.parent)), n))
for p, n in changed:
    print(f"{n:3d} {p}")
