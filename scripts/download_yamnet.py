#!/usr/bin/env python3
"""Download YAMNet TFLite model for baby cry detection."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/tfhub-lite-models/"
    "google/yamnet/classification/tflite/1.tflite"
)
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "yamnet.tflite"


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        print(f"Already present: {MODEL_PATH}")
        return 0

    print(f"Downloading YAMNet to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Saved {MODEL_PATH} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
