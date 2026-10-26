"""Convert sam3.1_multiplex.pt to safetensors, offline (Python never runs at runtime).

    tools/reference/.venv/bin/python -s tools/reference/convert_checkpoint.py

Writes ~/Code/research/models/sam3.1/sam3.1_multiplex.safetensors (never
committed) and tools/reference/weights_manifest.json (names, shapes, dtypes;
no weight values), which crates/st-weights is generated against.

safetensors has no complex dtype: the 32 complex64 RoPE tables
(`...attn.freqs_cis`, [T, 32]) are stored as f32 [T, 32, 2] (re, im) under the
same name, and listed in the file metadata under "complex64_as_f32_pairs".
"""

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

MODELS = pathlib.Path("~/Code/research/models/sam3.1").expanduser()
SRC = MODELS / "sam3.1_multiplex.pt"
DST = MODELS / "sam3.1_multiplex.safetensors"
MANIFEST = pathlib.Path(__file__).parent / "weights_manifest.json"
SRC_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    got = sha256(SRC)
    assert got == SRC_SHA256, f"source checkpoint sha256 {got} != {SRC_SHA256}"
    ck = torch.load(SRC, map_location="cpu", weights_only=True)
    if "model" in ck and isinstance(ck["model"], dict):
        ck = ck["model"]
    out, complex_keys, entries = {}, [], []
    for k in sorted(ck):
        v = ck[k]
        src_dtype = str(v.dtype).removeprefix("torch.")
        if v.is_complex():
            assert v.dtype == torch.complex64, (k, v.dtype)
            v = torch.view_as_real(v)
            complex_keys.append(k)
        assert v.dtype == torch.float32, (k, v.dtype)
        out[k] = v.contiguous()
        entries.append({"name": k, "shape": list(v.shape), "dtype": "f32", "source_dtype": src_dtype})
    meta = {
        "source": SRC.name,
        "source_sha256": SRC_SHA256,
        "complex64_as_f32_pairs": json.dumps(complex_keys),
    }
    save_file(out, str(DST), metadata=meta)
    dst_sha = sha256(DST)
    MANIFEST.write_text(json.dumps({
        "source": SRC.name,
        "source_sha256": SRC_SHA256,
        "safetensors": DST.name,
        "safetensors_sha256": dst_sha,
        "tensors": len(entries),
        "params": int(sum(torch.tensor(e["shape"]).prod().item() for e in entries)),
        "complex64_as_f32_pairs": complex_keys,
        "entries": entries,
    }, indent=1) + "\n")
    print(f"{len(entries)} tensors -> {DST} ({DST.stat().st_size} bytes, sha256 {dst_sha})")


if __name__ == "__main__":
    main()
