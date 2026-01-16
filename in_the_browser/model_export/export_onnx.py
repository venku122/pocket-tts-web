from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pocket_tts import TTSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Pocket TTS to ONNX graphs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text", type=str, default="Hello from Pocket TTS.")
    parser.add_argument("--voice", type=str, default="amy")
    parser.add_argument("--style", type=str, default="neutral")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model()
    tokenizer = model.text_conditioner.tokenizer

    tokens = tokenizer.encode(args.text)
    print("Tokenizer output length:", len(tokens))

    voice_state = model.get_state_for_audio_prompt(args.voice, args.style)
    print("Voice state shape:", voice_state.shape)

    first_latent = None
    for chunk in model.generate_audio_stream(args.text, voice=args.voice, style=args.style):
        first_latent = chunk
        break

    if first_latent is None:
        raise RuntimeError("No audio generated")

    print("First PCM chunk shape:", first_latent.shape)

    manifest = {
        "sampleRate": model.sample_rate,
        "voiceStateShape": list(voice_state.shape),
        "streamStep": {
            "inputs": {
                "tokens": "tokens",
                "voiceState": "voice_state",
                "kvCache": []
            },
            "outputs": {"nextLatents": "next_latents", "kvCache": []}
        },
        "decoder": {"inputs": {"latents": "latents"}, "outputs": {"pcm": "pcm"}}
    }

    args.manifest.write_text(json.dumps(manifest, indent=2))

    print("Manifest written to", args.manifest)
    print("TODO: implement ONNX export for prompt_encoder, stream_step, and decoder.")


if __name__ == "__main__":
    main()
