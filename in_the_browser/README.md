# Pocket TTS in the Browser (WASM)

This directory contains a **purely in-browser** Pocket TTS demo. The goal is to run
inference fully client-side with WebAssembly, using **onnxruntime-web** for model
execution and **SentencePiece WASM** for tokenization.

## What’s in here

```
in_the_browser/
  README.md
  web/              # Vite + React demo UI
  model_export/     # ONNX export + parity tools
  assets/           # Local-only model assets (ignored by git)
```

## Quick start (UI)

```bash
cd in_the_browser/web
npm install
npm run dev
```

Open the dev server URL in a modern browser (Chrome/Edge recommended).

## Model files (gated on Hugging Face)

The Pocket TTS weights are gated on HF, so **the demo does not fetch them**. Instead,
use the file picker to provide your local ONNX exports and metadata.

For local development you can also place files in:

```
in_the_browser/assets/models/
```

This directory is **gitignored** and should not contain weights in PRs.

## Cross‑origin isolation (threads)

If you enable WASM threads (recommended for performance), you **must** serve the app
with `COOP/COEP` headers so `SharedArrayBuffer` is available. The dev server sets
these automatically. For static hosting, configure:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Model export (ONNX)

See `model_export/README.md` for the export script, expected graph split, and parity
checks. A sample manifest lives at `web/public/model_manifest.example.json`.

## Notes

- Audio playback uses an `AudioWorklet` + ring buffer for smooth streaming.
- Tokenization uses a **SentencePiece WASM** wrapper to stay compatible with
  the original `tokenizer.model`.
- No model weights are committed in this directory.

## Manifest contract (model_manifest.json)

The demo relies on a manifest file to map ONNX graph IO names and shapes:

```json
{
  "sampleRate": 24000,
  "voiceStateShape": [1, 256, 80],
  "streamStep": {
    "inputs": {
      "tokens": "tokens",
      "voiceState": "voice_state",
      "kvCache": ["k_cache_0", "v_cache_0"]
    },
    "outputs": {
      "nextLatents": "next_latents",
      "kvCache": ["k_cache_0_out", "v_cache_0_out"]
    }
  },
  "decoder": {
    "inputs": { "latents": "latents" },
    "outputs": { "pcm": "pcm" }
  }
}
```

The `voiceStateShape` field is used to load the voice-state binary without
prompting for shape information in the UI.
