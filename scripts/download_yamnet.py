#!/usr/bin/env python3
"""Download YAMNet TFLite model for baby cry detection."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "audio_classifier/yamnet/float32/1/yamnet.tflite"
)
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "yamnet.tflite"
CLASS_MAP_PATH = MODEL_DIR / "yamnet_class_map.csv"


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    if not MODEL_PATH.is_file():
        try:
            _download(MODEL_URL, MODEL_PATH)
            size_kb = MODEL_PATH.stat().st_size / 1024
            print(f"Saved {MODEL_PATH} ({size_kb:.0f} KB)")
        except Exception as exc:
            print(f"Model download failed: {exc}", file=sys.stderr)
            ok = False
    else:
        print(f"Already present: {MODEL_PATH}")

    if not CLASS_MAP_PATH.is_file():
        try:
            _download(CLASS_MAP_URL, CLASS_MAP_PATH)
            print(f"Saved {CLASS_MAP_PATH}")
        except Exception as exc:
            print(f"Class map download failed: {exc}", file=sys.stderr)
            ok = False
    else:
        print(f"Already present: {CLASS_MAP_PATH}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
