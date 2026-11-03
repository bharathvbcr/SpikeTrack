"""Interpolate two spiketrack-lora-v1 adapters in weight space (a "soup"), exactly.

Per module the merged update is delta = c * s_a * B_a A_a + (1 - c) * s_b * B_b A_b (s = alpha / r),
formed in f64. Averaging the factors would be wrong (avg(B) avg(A) != avg(B A)), so the delta is
re-factored by thin SVD at rank min(r_a + r_b, rows, in), which holds it exactly, and written with
alpha = rank (scale 1). Both adapters must name the same modules with the same param, rows and
base_sha256. Refuses (exit 1) if any module's re-factored delta differs from the f64 target by more
than 1e-5 relative (f32 storage of the factors is the only loss).

With B given as the word `zero` the second update is zero, so the result is C * delta_a (a scaled
task vector; C in (0, 2]). That keeps A's factors and only multiplies each module's alpha by C, which
the loader turns into the scale alpha / r exactly.

usage: lora_soup.py A.safetensors B.safetensors|zero C OUT.safetensors
"""

import json
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

KEY = "spiketrack_lora"


def load(path):
    with safe_open(path, "np") as f:
        meta = json.loads(f.metadata()[KEY])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get("format") != "spiketrack-lora-v1":
        sys.exit(f"{path}: format {meta.get('format')!r}")
    return meta, {m["module"]: m for m in meta["modules"]}, tensors


def delta(mod, tensors):
    a = tensors[f"{mod['module']}.lora_A.weight"].astype(np.float64)
    b = tensors[f"{mod['module']}.lora_B.weight"].astype(np.float64)
    return (float(mod["alpha"]) / mod["rank"]) * (b @ a)


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    pa, pb, c, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    meta_a, mods_a, ta = load(pa)
    if pb == "zero":
        if not 0.0 < c <= 2.0:
            sys.exit(f"a scaled task vector needs c in (0, 2], got {c}")
        modules = [{**m, "alpha": float(m["alpha"]) * c} for m in mods_a.values()]
        meta = {**meta_a, "trainer": "tools/reference/lora_soup.py",
                "run": f"scale c={c:g} * {meta_a['run']}@{meta_a['step']}", "modules": modules}
        save_file(ta, out, metadata={KEY: json.dumps(meta)})
        print(f"wrote {out}: {len(modules)} modules, {meta['run']} (alpha scaled, factors unchanged)")
        return
    if not 0.0 <= c <= 1.0:
        sys.exit(f"c must be in [0, 1], got {c}")
    meta_b, mods_b, tb = load(pb)
    if meta_a["base_sha256"] != meta_b["base_sha256"]:
        sys.exit("adapters were trained on different base checkpoints")
    if list(mods_a) != list(mods_b):
        sys.exit("adapters name different modules")
    tensors, modules, worst = {}, [], 0.0
    for name, ma in mods_a.items():
        mb = mods_b[name]
        if (ma["param"], ma["rows"]) != (mb["param"], mb["rows"]):
            sys.exit(f"{name}: param/rows differ")
        target = c * delta(ma, ta) + (1.0 - c) * delta(mb, tb)
        rows, cols = target.shape
        rank = min(ma["rank"] + mb["rank"], rows, cols)
        u, s, vt = np.linalg.svd(target, full_matrices=False)
        root = np.sqrt(s[:rank])
        b_new = (u[:, :rank] * root).astype(np.float32)
        a_new = (root[:, None] * vt[:rank]).astype(np.float32)
        got = b_new.astype(np.float64) @ a_new.astype(np.float64)
        rel = np.linalg.norm(got - target) / max(np.linalg.norm(target), 1e-30)
        worst = max(worst, rel)
        if rel > 1e-5:
            sys.exit(f"{name}: re-factored delta rel error {rel:.3e} > 1e-5")
        tensors[f"{name}.lora_A.weight"] = np.ascontiguousarray(a_new)
        tensors[f"{name}.lora_B.weight"] = np.ascontiguousarray(b_new)
        modules.append({**ma, "rank": rank, "alpha": float(rank)})
    meta = {
        "format": "spiketrack-lora-v1",
        "base_sha256": meta_a["base_sha256"],
        "dropout": meta_a["dropout"],
        "trainer": "tools/reference/lora_soup.py",
        "step": 0,
        "seed": 0,
        "run": f"soup c={c:g} * {meta_a['run']}@{meta_a['step']} + {1 - c:g} * {meta_b['run']}@{meta_b['step']}",
        "modules": modules,
    }
    save_file(tensors, out, metadata={KEY: json.dumps(meta)})
    print(f"wrote {out}: {len(modules)} modules, c={c:g}, worst rel error {worst:.2e}")


if __name__ == "__main__":
    main()
