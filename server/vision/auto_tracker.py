"""Center the camera on a detected person using pose keypoints."""

from __future__ import annotations

import logging
import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_KEYPOINTS = 10
KP_CONFIDENCE = 0.5

# OV5647 / stream is 640x480; horizontal FOV ~75° (user measurement).
HFOV_DEG = 75.0

# Apply a fraction of the angular error each tick to avoid overshoot oscillation.
STEP_FRACTION = 0.11
DEADZONE_DEG = 2.5
MAX_STEP_DEG = 1.2

# Exponential smoothing on target position (reduces back-and-forth).
SMOOTH_ALPHA = 0.22

_lock = threading.Lock()
_enabled = False
_smooth_cx: Optional[float] = None
_smooth_cy: Optional[float] = None


def set_enabled(value: bool) -> None:
    global _enabled, _smooth_cx, _smooth_cy
    with _lock:
        _enabled = bool(value)
        if not value:
            _smooth_cx = None
            _smooth_cy = None
    logger.info("Auto track %s", "enabled" if value else "disabled")


def is_enabled() -> bool:
    with _lock:
        return _enabled


def get_status() -> Dict:
    return {
        "auto_track": is_enabled(),
        "min_keypoints": MIN_KEYPOINTS,
        "hfov_deg": HFOV_DEG,
    }


def _vertical_fov_deg(frame_w: int, frame_h: int) -> float:
    if frame_w <= 0:
        return HFOV_DEG
    return math.degrees(
        2.0 * math.atan(math.tan(math.radians(HFOV_DEG / 2.0)) * frame_h / frame_w)
    )


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


def _smooth_centroid(cx: float, cy: float) -> Tuple[float, float]:
    global _smooth_cx, _smooth_cy
    if _smooth_cx is None or _smooth_cy is None:
        _smooth_cx, _smooth_cy = cx, cy
    else:
        _smooth_cx = _smooth_cx + SMOOTH_ALPHA * (cx - _smooth_cx)
        _smooth_cy = _smooth_cy + SMOOTH_ALPHA * (cy - _smooth_cy)
    return _smooth_cx, _smooth_cy


def _pixel_error_to_step_deg(
    err_px: float,
    frame_span: int,
    fov_deg: float,
) -> float:
    if frame_span <= 0:
        return 0.0
    correction_deg = (err_px / frame_span) * fov_deg
    if abs(correction_deg) < DEADZONE_DEG:
        return 0.0
    step = correction_deg * STEP_FRACTION
    return max(-MAX_STEP_DEG, min(MAX_STEP_DEG, step))


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
    cx, cy = _smooth_centroid(cx, cy)

    center_x = frame_w / 2.0
    center_y = frame_h / 2.0
    err_x = cx - center_x
    err_y = cy - center_y

    vfov = _vertical_fov_deg(frame_w, frame_h)
    pan_delta = _pixel_error_to_step_deg(-err_x, frame_w, HFOV_DEG)
    tilt_delta = _pixel_error_to_step_deg(err_y, frame_h, vfov)

    if abs(pan_delta) < 1e-6 and abs(tilt_delta) < 1e-6:
        return

    from server.hardware.servo_controller import step_angles

    try:
        step_angles(pan_delta, tilt_delta)
    except RuntimeError as exc:
        logger.debug("Auto track servo step skipped: %s", exc)
