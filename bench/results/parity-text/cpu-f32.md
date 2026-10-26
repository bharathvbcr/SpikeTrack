# Tokenizer and text encoder parity vs upstream (gp-prompt-encoders)

- 2026-10-26, M5 Pro, ojas d2922fb CpuBackend. Test: crates/st-model/tests/text_parity.rs; job tools/jobs/d2_text_detector.sh.
- Tokenizer: 13 prompts (tools/reference/tokenizer_cases.json: sperm, cell, mixed case and punctuation, extra whitespace, accents, a 60-char truncation) match upstream SimpleTokenizer's ids exactly.
- Text encoder, 13 prompts x 32 tokens, fp32:
  - encoder.tokens: cosine 1.000000000, max abs 1.049e-4, max abs / max ref 1.960e-6
  - resizer.out: cosine 1.000000000, max abs 4.864e-5, max abs / max ref 1.061e-5
