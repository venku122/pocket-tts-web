# Model export (ONNX)

This directory contains tooling to export Pocket TTS to ONNX for browser use.

## Expected graph split

- `prompt_encoder.onnx` (optional for v1)
- `stream_step.onnx` (single-step inference with KV cache in/out)
- `decoder.onnx` (latent → PCM)

The web demo expects a `model_manifest.json` that maps input/output names and
includes the `voiceStateShape` produced by the prompt encoder.

## Usage

```bash
uv run python export_onnx.py \
  --output-dir /path/to/onnx \
  --manifest /path/to/onnx/model_manifest.json
```

Run parity checks:

```bash
uv run python parity_test.py \
  --onnx-dir /path/to/onnx \
  --manifest /path/to/onnx/model_manifest.json
```

## Notes

- Requires PyTorch 2.5+ (2.4.0 produces incorrect audio).
- This script logs boundary shapes for tokenizer, voice state, per-step latents,
  and decoder outputs.
