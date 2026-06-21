"""Skip expensive pose inference when the camera scene has not changed."""

from __future__ import annotations

import time

import cv2
import numpy as np

# Downscaled grayscale diff — much cheaper than YOLO.
MOTION_SIZE = (160, 120)
MOTION_THRESHOLD = 4.0

# Re-run YOLO occasionally even when the scene looks static.
WATCHDOG_NO_PERSON_SEC = 4.0
WATCHDOG_PERSON_STATIC_SEC = 12.0


class InferenceGate:
    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._last_infer_time = 0.0
        self.last_motion_score = 0.0

    def motion_score(self, frame_bgr: np.ndarray) -> float:
        small = cv2.cvtColor(
            cv2.resize(frame_bgr, MOTION_SIZE, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if self._prev_gray is None:
            self._prev_gray = small
            self.last_motion_score = float("inf")
            return self.last_motion_score

        score = float(np.mean(cv2.absdiff(small, self._prev_gray)))
        self._prev_gray = small
        self.last_motion_score = score
        return score

    def should_run_inference(self, frame_bgr: np.ndarray, *, person_visible: bool) -> bool:
        now = time.time()
        score = self.motion_score(frame_bgr)

        if score >= MOTION_THRESHOLD:
            return True

        elapsed = now - self._last_infer_time
        watchdog = WATCHDOG_PERSON_STATIC_SEC if person_visible else WATCHDOG_NO_PERSON_SEC
        return elapsed >= watchdog

    def mark_inferred(self) -> None:
        self._last_infer_time = time.time()


_gate: InferenceGate | None = None


def get_inference_gate() -> InferenceGate:
    global _gate
    if _gate is None:
        _gate = InferenceGate()
    return _gate
