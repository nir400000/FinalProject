"""Center the camera on a detected person using pose keypoints."""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_KEYPOINTS = 10
KP_CONFIDENCE = 0.5
DEADZONE_PX = 28
MAX_STEP = 3.0
PAN_GAIN = 0.022
TILT_GAIN = 0.020

_lock = threading.Lock()
_enabled = False


def set_enabled(value: bool) -> None:
    global _enabled
    with _lock:
        _enabled = bool(value)
    logger.info("Auto track %s", "enabled" if value else "disabled")


def is_enabled() -> bool:
    with _lock:
        return _enabled


def get_status() -> Dict:
    return {"auto_track": is_enabled(), "min_keypoints": MIN_KEYPOINTS}


def _centroid(
    keypoints: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, int] | None:
    confident = [(x, y) for x, y, conf in keypoints if conf >= KP_CONFIDENCE]
    if len(confident) < MIN_KEYPOINTS:
        return None
    count = len(confident)
    cx = sum(p[0] for p in confident) / count
    cy = sum(p[1] for p in confident) / count
    return cx, cy, count


def update(
    keypoints: List[Tuple[float, float, float]],
    frame_w: int,
    frame_h: int,
) -> None:
    if not is_enabled():
        return
    if not keypoints or len(keypoints) < 17:
        return

    centroid = _centroid(keypoints)
    if centroid is None:
        return

    cx, cy, _ = centroid
    center_x = frame_w / 2.0
    center_y = frame_h / 2.0
    err_x = cx - center_x
    err_y = cy - center_y

    pan_delta = 0.0
    tilt_delta = 0.0
    if abs(err_x) > DEADZONE_PX:
        pan_delta = max(-MAX_STEP, min(MAX_STEP, -err_x * PAN_GAIN))
    if abs(err_y) > DEADZONE_PX:
        tilt_delta = max(-MAX_STEP, min(MAX_STEP, err_y * TILT_GAIN))

    if abs(pan_delta) < 1e-6 and abs(tilt_delta) < 1e-6:
        return

    from servo_controller import step_angles

    try:
        step_angles(pan_delta, tilt_delta)
    except RuntimeError as exc:
        logger.debug("Auto track servo step skipped: %s", exc)
