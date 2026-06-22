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
STEP_FRACTION = 0.14
DEADZONE_DEG = 2.2
MAX_STEP_DEG = 1.4

# Upper-body-only recovery: shoulders visible, hips/legs missing, upper body low in frame.
UPPER_KP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
LOWER_KP = (11, 12, 13, 14, 15, 16)
MIN_UPPER_KP = 6
MIN_LOWER_KP = 2
LOWER_FRAME_Y_RATIO = 0.52
TILT_DOWN_BOOST_DEG = 2.5
TILT_MARGIN_DEG = 2.0

# Exponential smoothing on target position (reduces back-and-forth).
SMOOTH_ALPHA = 0.28

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


def _confident_points(
    keypoints: Sequence[Tuple[float, float, float]],
    indices: Sequence[int],
) -> List[Tuple[float, float]]:
    return [
        (keypoints[i][0], keypoints[i][1])
        for i in indices
        if i < len(keypoints) and keypoints[i][2] >= KP_CONFIDENCE
    ]


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


def _is_upper_body_only(keypoints: Sequence[Tuple[float, float, float]]) -> bool:
    upper = _confident_points(keypoints, UPPER_KP)
    lower = _confident_points(keypoints, LOWER_KP)
    return len(upper) >= MIN_UPPER_KP and len(lower) < MIN_LOWER_KP


def _upper_body_mean_y(keypoints: Sequence[Tuple[float, float, float]]) -> Optional[float]:
    upper = _confident_points(keypoints, UPPER_KP)
    if not upper:
        return None
    return sum(p[1] for p in upper) / len(upper)


def _apply_upper_body_tilt_recovery(
    keypoints: Sequence[Tuple[float, float, float]],
    frame_h: int,
    err_y_px: float,
    tilt_angle: float,
) -> float:
    """Tilt down when only upper body is visible near the bottom of the frame."""
    if not _is_upper_body_only(keypoints):
        return err_y_px

    upper_y = _upper_body_mean_y(keypoints)
    if upper_y is None or upper_y < frame_h * LOWER_FRAME_Y_RATIO:
        return err_y_px

    can_tilt_down = tilt_angle < (90.0 - TILT_MARGIN_DEG)
    if not can_tilt_down:
        return err_y_px

    # Legs may be below the frame — prefer tilting down, not up.
    min_downward_px = frame_h * 0.10
    boosted_err_y = max(err_y_px, min_downward_px)

    vfov = _vertical_fov_deg(640, frame_h)
    boost_px = (TILT_DOWN_BOOST_DEG / vfov) * frame_h
    return boosted_err_y + boost_px


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

    from servo_controller import get_status, step_angles

    tilt_angle = float(get_status().get("tilt", 0.0))
    err_y = _apply_upper_body_tilt_recovery(keypoints, frame_h, err_y, tilt_angle)

    if _is_upper_body_only(keypoints) and err_y < 0:
        # Do not tilt up while lower body is missing — avoids the close-up loop.
        err_y = 0.0

    vfov = _vertical_fov_deg(frame_w, frame_h)
    pan_delta = _pixel_error_to_step_deg(-err_x, frame_w, HFOV_DEG)
    tilt_delta = _pixel_error_to_step_deg(err_y, frame_h, vfov)

    if abs(pan_delta) < 1e-6 and abs(tilt_delta) < 1e-6:
        return

    try:
        step_angles(pan_delta, tilt_delta)
    except RuntimeError as exc:
        logger.debug("Auto track servo step skipped: %s", exc)
