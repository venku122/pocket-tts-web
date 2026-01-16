from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parity checks for Pocket TTS ONNX")
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())

    stream_step = ort.InferenceSession(str(args.onnx_dir / "stream_step.onnx"))
    decoder = ort.InferenceSession(str(args.onnx_dir / "decoder.onnx"))

    print("Loaded stream_step inputs:", [i.name for i in stream_step.get_inputs()])
    print("Loaded decoder inputs:", [i.name for i in decoder.get_inputs()])

    print("TODO: implement parity comparison with torch outputs")


if __name__ == "__main__":
    main()
