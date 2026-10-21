# Changelog

Tags are vX.Y.Z on the commit whose moon.mod carries that version.
Publish order: soehi/gguf -> soehi/moonlm -> soehi/edgeflow.

## 0.1.0 — 2026-10-21

- First release as its own module: read-only GGUF container parsing plus
  dequantization for seven storage types (f32, f16, q4_0, q5_0, q8_0, q4_K,
  q6_K).
- Extracted from the moonlm repository, which now consumes this module from
  the registry. The fixture generator and the fixture-staleness check moved
  here with the code.
- Seven storage types are decoded (five block-quantized plus f32/f16), not the
  six the README originally stated.
