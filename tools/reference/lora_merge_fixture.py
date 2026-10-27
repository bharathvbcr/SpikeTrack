"""Fixture for the Rust merge (crates/st-weights/src/lora.rs): a small random adapter merged in PyTorch.

    tools/reference/.venv/bin/python -s tools/reference/lora_merge_fixture.py

Reads the real checkpoint tensors the adapter targets (selective `safe_open`, no 3.5 GB load), builds a
seeded random adapter over them (mixed ranks and alphas, so the per-module scale alpha/rank is exercised,
and row slices of the packed q/k/v `in_proj_weight`), writes it with the SAME writer the trainer uses
(`lora_common.save_adapter`), merges it with `lora_common.merge_into_state` (torch f32, W + (alpha/r) B A),
and writes the merged tensors. Output, git-ignored, in fixtures/lora/:
  adapter.safetensors   the adapter, spiketrack-lora-v1
  expected.safetensors  the merged f32 tensors by checkpoint name, plus `untouched.<name>` copies of two
                        tensors the adapter does not target
  MANIFEST.json         targets, checkpoint sha256, and how far torch's f32 result is from an f64 merge
                        (the rounding the Rust test's tolerance has to cover)
crates/st-weights/tests/lora_merge.rs checks the Rust merge against it (#[ignore], needs SPIKETRACK_WEIGHTS).
"""

import json
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lora_common as lc  # noqa: E402

OUT = lc.REPO / "fixtures/lora"
E = lc.DIM
ENC = "detector.transformer.encoder.layers.0.self_attn"
DEC_TEXT = "detector.transformer.decoder.layers.5.ca_text"
DEC_IMG = "detector.transformer.decoder.layers.2.cross_attn"
# (module, param, rows, rank, alpha)
SPEC = [
    (f"{ENC}.in_proj_q", f"{ENC}.in_proj_weight", (0, E), 16, 32.0),
    (f"{ENC}.in_proj_k", f"{ENC}.in_proj_weight", (E, 2 * E), 16, 32.0),
    (f"{ENC}.in_proj_v", f"{ENC}.in_proj_weight", (2 * E, 3 * E), 16, 32.0),
    (f"{ENC}.out_proj", f"{ENC}.out_proj.weight", (0, E), 16, 32.0),
    (f"{DEC_TEXT}.in_proj_q", f"{DEC_TEXT}.in_proj_weight", (0, E), 8, 12.0),
    (f"{DEC_TEXT}.in_proj_v", f"{DEC_TEXT}.in_proj_weight", (2 * E, 3 * E), 8, 12.0),
    (f"{DEC_TEXT}.out_proj", f"{DEC_TEXT}.out_proj.weight", (0, E), 4, 4.0),
    (f"{DEC_IMG}.in_proj_q", f"{DEC_IMG}.in_proj_weight", (0, E), 16, 32.0),
]
UNTOUCHED = [f"{ENC}.in_proj_bias", "detector.transformer.encoder.layers.1.self_attn.in_proj_weight"]


def main():
    g = torch.Generator().manual_seed(20261027)
    params = sorted({p for _, p, *_ in SPEC} | set(UNTOUCHED))
    with safe_open(str(lc.CKPT), "pt") as f:
        base = {p: f.get_tensor(p).clone() for p in params}
    entries = []
    for module, param, rows, rank, alpha in SPEC:
        a = torch.randn(rank, base[param].shape[1], generator=g) * 0.05
        b = torch.randn(rows[1] - rows[0], rank, generator=g) * 0.05
        entries.append((module, param, rows, rank, alpha, a, b))

    merged = {p: base[p].clone() for p in {p for _, p, *_ in SPEC}}
    lc.merge_into_state(merged, entries)

    # An f64 merge of the same adapter: how much of the torch-vs-Rust difference is just f32 rounding.
    f64 = {p: base[p].double() for p in merged}
    for module, param, rows, rank, alpha, a, b in entries:
        f64[param][rows[0]:rows[1]] += (alpha / rank) * (b.double() @ a.double())
    err = {p: (merged[p].double() - f64[p]).abs().max().item() for p in merged}
    changed = {p: bool((merged[p] != base[p]).any()) for p in merged}
    assert all(changed.values()), changed

    OUT.mkdir(parents=True, exist_ok=True)
    lc.save_adapter(OUT / "adapter.safetensors", entries, {"fixture": True, "seed": 20261027})
    expected = {p: t.contiguous() for p, t in merged.items()}
    expected.update({f"untouched.{p}": base[p].contiguous() for p in UNTOUCHED})
    save_file(expected, str(OUT / "expected.safetensors"))
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "command": "tools/reference/.venv/bin/python -s tools/reference/lora_merge_fixture.py",
        "checkpoint_sha256": lc.base_sha256(),
        "targets": [{"module": m, "param": p, "rows": list(r), "rank": k, "alpha": a} for m, p, r, k, a in SPEC],
        "merged_tensors": sorted(merged), "untouched": UNTOUCHED,
        "torch_f32_vs_f64_merge_max_abs_error": err,
        "rule": "W' = W + (alpha/rank) * (B @ A) on the row slice, f32; torch (BLAS) and Rust (sequential k) sum in a different order, "
                "so the Rust test allows |d| <= 1e-6 + 1e-6 * |expected| per element",
    }, indent=1))
    print("merged tensors:", {p.split("transformer.")[1]: f"{e:.1e}" for p, e in err.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
