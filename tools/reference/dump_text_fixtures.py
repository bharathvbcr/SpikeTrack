"""Text-path fixtures for gp-prompt-encoders: tokenizer ids and encoder outputs.

    SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_text_fixtures.py

Writes fixtures/text/prompts.safetensors (git-ignored) and records it in
fixtures/TEXT_MANIFEST.json. The ids are also written to
tools/reference/tokenizer_cases.json (committed: token ids, no weights), the
oracle for a Rust CLIP BPE.
"""

import hashlib
import json
import pathlib
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sam3_ref  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "fixtures/text"
PROMPTS = [
    "sperm", "sperm cell", "spermatozoon", "cell", "car", "cell nucleus",
    "a sperm cell swimming", "Sperm-Cell, (pinhead)!", "  white   space  ", "café naïve",
    "truck", "shoe", "x" * 60,
]


def main():
    sam3_ref.use_fp32_mlp()
    model, _, _ = sam3_ref.build_reference()
    lang = model.detector.backbone.language_backbone
    tok = lang.tokenizer(PROMPTS, context_length=lang.context_length)
    with torch.inference_mode():
        pooled, memory = lang.encoder(tok)  # memory [n, 32, 1024]
        mask, resized, embeds = lang(PROMPTS, device=torch.device("cpu"))
    t = {
        "token_ids": tok.to(torch.int64),
        "encoder.tokens": memory.float().contiguous(),
        "encoder.pooled": pooled.float().contiguous(),
        "resizer.out": resized.transpose(0, 1).float().contiguous(),  # [n, 32, 256]
        "inputs_embeds": embeds.transpose(0, 1).float().contiguous(),  # [n, 32, 1024]
        "pad_mask": mask.to(torch.uint8),  # 1 = padding (torch's key_padding convention)
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "prompts.safetensors"
    save_file(t, str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (REPO / "fixtures/TEXT_MANIFEST.json").write_text(json.dumps({
        "file": str(path.relative_to(REPO)), "sha256": sha, "prompts": PROMPTS,
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in t.items()},
        "command": "SAM3_DEVICE=cpu tools/reference/.venv/bin/python -s tools/reference/dump_text_fixtures.py",
    }, indent=1))
    cases = [{"text": p, "ids": [int(i) for i in row]} for p, row in zip(PROMPTS, tok.tolist())]
    (pathlib.Path(__file__).parent / "tokenizer_cases.json").write_text(json.dumps(
        {"context_length": lang.context_length, "source": "sam3 SimpleTokenizer, bpe_simple_vocab_16e6.txt.gz", "cases": cases},
        indent=1, ensure_ascii=False) + "\n")
    for c in cases[:4]:
        print(c["text"], [i for i in c["ids"] if i])


if __name__ == "__main__":
    main()
