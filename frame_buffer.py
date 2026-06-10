"""Thread-safe latest camera frame for local MJPEG and WebRTC."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

_lock = threading.Lock()
_latest_bgr: Optional[np.ndarray] = None


def update_frame(frame_bgr: np.ndarray) -> None:
    global _latest_bgr
    with _lock:
        _latest_bgr = frame_bgr.copy()


def get_frame_copy() -> Optional[np.ndarray]:
    with _lock:
        if _latest_bgr is None:
            return None
        return _latest_bgr.copy()
