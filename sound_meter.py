"""Track nursery microphone loudness for generic sound alerts."""

from __future__ import annotations

import threading
from typing import Dict

import numpy as np

_lock = threading.Lock()
_level = 0.0
_peak = 0.0


def feed_pcm(chunk: bytes) -> None:
    if not chunk:
        return
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return
    rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0)
    with _lock:
        global _level, _peak
        _level = _level * 0.85 + rms * 0.15
        if rms > _peak:
            _peak = rms


def get_status() -> Dict:
    with _lock:
        return {
            "available": True,
            "level": round(_level, 4),
            "peak": round(_peak, 4),
        }


def reset_peak() -> None:
    with _lock:
        global _peak
        _peak = 0.0
