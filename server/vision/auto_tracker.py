"""Center the camera on a detected person using pose keypoints."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_KEYPOINTS = 3
KP_CONFIDENCE = 0.5

# Torso keypoints jitter less than limbs when a person is standing still.
TORSO_INDICES = (0, 5, 6, 11, 12)

# OV5647 / stream is 640x480; horizontal FOV ~75°.
HFOV_DEG = 75.0

# Spread each YOLO correction across the time until the next pose update.
# Keep in sync with inference_gate.AUTO_TRACK_PERSON_SEC (~0.28 s).
TRACK_CYCLE_SEC = 0.25
MICRO_STEP_INTERVAL_SEC = 0.05
MAX_MICRO_STEP_DEG = 1.0
MIN_MICRO_STEPS = 2
MAX_MICRO_STEPS = 8

# Fast when far off-center (acquisition); gentle near center to avoid oscillation.
ACQUIRE_ERROR_DEG = 12.0
STEP_FRACTION_ACQUIRE = 0.42
MAX_STEP_DEG_ACQUIRE = 3.2

STEP_FRACTION_FAR = 0.34
STEP_FRACTION_NEAR = 0.18
MAX_STEP_DEG_FAR = 2.6
MAX_STEP_DEG_NEAR = 0.85
FAR_ERROR_DEG = 6.0
NEAR_ERROR_DEG = 3.5

ENTER_DEADZONE_DEG = 3.0
EXIT_DEADZONE_DEG = 5.0
SETTLE_TICKS = 2

SMOOTH_ALPHA_FAR = 0.45
SMOOTH_ALPHA_NEAR = 0.18

_lock = threading.Lock()
_enabled = False
_smooth_cx: Optional[float] = None
_smooth_cy: Optional[float] = None
_settled = False
_settle_count = 0
_sweep_generation = 0


def set_enabled(value: bool) -> None:
    global _enabled, _smooth_cx, _smooth_cy, _settled, _settle_count, _sweep_generation
    with _lock:
        _enabled = bool(value)
        _smooth_cx = None
        _smooth_cy = None
        _settled = False
        _settle_count = 0
        _sweep_generation += 1
    from server.vision.inference_gate import get_inference_gate

    get_inference_gate().set_auto_track_active(bool(value))
    logger.info("Auto track %s", "enabled" if value else "disabled")


def is_enabled() -> bool:
    with _lock:
        return _enabled


def get_status() -> Dict:
    with _lock:
        settled = _settled
    return {
        "auto_track": is_enabled(),
        "min_keypoints": MIN_KEYPOINTS,
        "hfov_deg": HFOV_DEG,
        "settled": settled,
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
    torso = [
        (keypoints[i][0], keypoints[i][1])
        for i in TORSO_INDICES
        if i < len(keypoints) and keypoints[i][2] >= KP_CONFIDENCE
    ]
    if len(torso) >= MIN_KEYPOINTS:
        count = len(torso)
        cx = sum(p[0] for p in torso) / count
        cy = sum(p[1] for p in torso) / count
        return cx, cy, count

    confident = [(x, y) for x, y, conf in keypoints if conf >= KP_CONFIDENCE]
    if len(confident) < MIN_KEYPOINTS:
        return None
    count = len(confident)
    cx = sum(p[0] for p in confident) / count
    cy = sum(p[1] for p in confident) / count
    return cx, cy, count


def _smooth_centroid(cx: float, cy: float, error_mag_deg: float) -> Tuple[float, float]:
    global _smooth_cx, _smooth_cy

    if error_mag_deg >= ACQUIRE_ERROR_DEG:
        _smooth_cx, _smooth_cy = cx, cy
        return cx, cy

    if error_mag_deg >= FAR_ERROR_DEG:
        alpha = SMOOTH_ALPHA_FAR
    elif error_mag_deg <= NEAR_ERROR_DEG:
        alpha = SMOOTH_ALPHA_NEAR
    else:
        t = (error_mag_deg - NEAR_ERROR_DEG) / (FAR_ERROR_DEG - NEAR_ERROR_DEG)
        alpha = SMOOTH_ALPHA_NEAR + t * (SMOOTH_ALPHA_FAR - SMOOTH_ALPHA_NEAR)

    if _smooth_cx is None or _smooth_cy is None:
        _smooth_cx, _smooth_cy = cx, cy
    else:
        _smooth_cx = _smooth_cx + alpha * (cx - _smooth_cx)
        _smooth_cy = _smooth_cy + alpha * (cy - _smooth_cy)
    return _smooth_cx, _smooth_cy


def _correction_to_step(correction_deg: float) -> float:
    abs_err = abs(correction_deg)
    if abs_err < 1e-6:
        return 0.0

    if abs_err >= ACQUIRE_ERROR_DEG:
        fraction = STEP_FRACTION_ACQUIRE
        max_step = MAX_STEP_DEG_ACQUIRE
    elif abs_err >= FAR_ERROR_DEG:
        fraction = STEP_FRACTION_FAR
        max_step = MAX_STEP_DEG_FAR
    elif abs_err <= NEAR_ERROR_DEG:
        fraction = STEP_FRACTION_NEAR
        max_step = MAX_STEP_DEG_NEAR
    else:
        t = (abs_err - NEAR_ERROR_DEG) / (FAR_ERROR_DEG - NEAR_ERROR_DEG)
        fraction = STEP_FRACTION_NEAR + t * (STEP_FRACTION_FAR - STEP_FRACTION_NEAR)
        max_step = MAX_STEP_DEG_NEAR + t * (MAX_STEP_DEG_FAR - MAX_STEP_DEG_NEAR)

    step = correction_deg * fraction
    return max(-max_step, min(max_step, step))


def _plan_micro_steps(pan_total: float, tilt_total: float) -> Tuple[int, float, float]:
    """Split one correction into several timed micro-steps before the next YOLO frame."""
    steps_by_time = max(1, int(TRACK_CYCLE_SEC / MICRO_STEP_INTERVAL_SEC))
    max_axis = max(abs(pan_total), abs(tilt_total))
    if max_axis > 1e-6:
        steps_by_size = max(1, math.ceil(max_axis / MAX_MICRO_STEP_DEG))
    else:
        steps_by_size = 1

    count = max(steps_by_time, steps_by_size)
    count = max(MIN_MICRO_STEPS, min(MAX_MICRO_STEPS, count))
    return count, pan_total / count, tilt_total / count


def _micro_sweep_worker(
    pan_total: float,
    tilt_total: float,
    generation: int,
) -> None:
    from server.hardware.servo_controller import step_angles

    count, pan_step, tilt_step = _plan_micro_steps(pan_total, tilt_total)
    for index in range(count):
        if generation != _sweep_generation or not is_enabled():
            return
        try:
            step_angles(pan_step, tilt_step)
        except RuntimeError as exc:
            logger.debug("Auto track micro-step skipped: %s", exc)
            return
        if index < count - 1:
            time.sleep(MICRO_STEP_INTERVAL_SEC)


def _schedule_micro_sweep(pan_total: float, tilt_total: float) -> None:
    global _sweep_generation
    _sweep_generation += 1
    generation = _sweep_generation
    thread = threading.Thread(
        target=_micro_sweep_worker,
        args=(pan_total, tilt_total, generation),
        daemon=True,
        name="auto-track-sweep",
    )
    thread.start()


def update(
    keypoints: List[Tuple[float, float, float]],
    frame_w: int,
    frame_h: int,
) -> None:
    global _settled, _settle_count

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

    vfov = _vertical_fov_deg(frame_w, frame_h)
    pan_correction = (-err_x / frame_w) * HFOV_DEG if frame_w > 0 else 0.0
    tilt_correction = (err_y / frame_h) * vfov if frame_h > 0 else 0.0
    error_mag = math.hypot(pan_correction, tilt_correction)

    cx, cy = _smooth_centroid(cx, cy, error_mag)
    err_x = cx - center_x
    err_y = cy - center_y
    pan_correction = (-err_x / frame_w) * HFOV_DEG if frame_w > 0 else 0.0
    tilt_correction = (err_y / frame_h) * vfov if frame_h > 0 else 0.0

    with _lock:
        if _settled:
            if (
                abs(pan_correction) < EXIT_DEADZONE_DEG
                and abs(tilt_correction) < EXIT_DEADZONE_DEG
            ):
                return
            _settled = False
            _settle_count = 0

        if (
            abs(pan_correction) < ENTER_DEADZONE_DEG
            and abs(tilt_correction) < ENTER_DEADZONE_DEG
        ):
            _settle_count += 1
            if _settle_count >= SETTLE_TICKS:
                _settled = True
            return

        _settle_count = 0

    pan_delta = _correction_to_step(pan_correction)
    tilt_delta = _correction_to_step(tilt_correction)

    if abs(pan_delta) < 1e-6 and abs(tilt_delta) < 1e-6:
        return

    _schedule_micro_sweep(pan_delta, tilt_delta)
