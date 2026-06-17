"""Hardware PWM pan/tilt servo control for Raspberry Pi 5 (SG90)."""

from __future__ import annotations

import logging
import threading
from typing import Dict

logger = logging.getLogger(__name__)

MIN_ANGLE = -90
MAX_ANGLE = 90
STEP_DEG = 5
MIN_DUTY = 2.5
MAX_DUTY = 12.0

_lock = threading.Lock()
_pan_servo = None
_tilt_servo = None
_pan_angle = 0.0
_tilt_angle = 0.0
_initialized = False
_available = False


def angle_to_duty_cycle(angle: float) -> float:
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return ((angle - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)) * (MAX_DUTY - MIN_DUTY) + MIN_DUTY


def init_servos() -> bool:
    global _pan_servo, _tilt_servo, _initialized, _available

    if _initialized:
        return _available

    _initialized = True
    try:
        from rpi_hardware_pwm import HardwarePWM

        _pan_servo = HardwarePWM(pwm_channel=0, hz=50, chip=0)
        _tilt_servo = HardwarePWM(pwm_channel=1, hz=50, chip=0)
        _pan_servo.start(angle_to_duty_cycle(_pan_angle))
        _tilt_servo.start(angle_to_duty_cycle(_tilt_angle))
        _available = True
        logger.info("Pan/tilt servos initialized (GPIO 12 pan, GPIO 13 tilt)")
    except Exception as exc:
        logger.warning("Servo control unavailable: %s", exc)
        _available = False

    return _available


def set_angles(pan: float, tilt: float) -> Dict[str, float]:
    global _pan_angle, _tilt_angle

    pan = max(MIN_ANGLE, min(MAX_ANGLE, float(pan)))
    tilt = max(MIN_ANGLE, min(MAX_ANGLE, float(tilt)))

    with _lock:
        if not init_servos():
            raise RuntimeError("Servo hardware not available")
        _pan_angle = pan
        _tilt_angle = tilt
        _pan_servo.change_duty_cycle(angle_to_duty_cycle(pan))
        _tilt_servo.change_duty_cycle(angle_to_duty_cycle(tilt))

    return {"pan": pan, "tilt": tilt}


def step_angles(pan_delta: float = 0, tilt_delta: float = 0) -> Dict[str, float]:
    """Move servos relative to current position (same steps as servo_control_pi5.py keys)."""
    global _pan_angle, _tilt_angle

    with _lock:
        if not init_servos():
            raise RuntimeError("Servo hardware not available")

        if pan_delta:
            _pan_angle = max(MIN_ANGLE, min(MAX_ANGLE, _pan_angle + float(pan_delta)))
            _pan_servo.change_duty_cycle(angle_to_duty_cycle(_pan_angle))
        if tilt_delta:
            _tilt_angle = max(MIN_ANGLE, min(MAX_ANGLE, _tilt_angle + float(tilt_delta)))
            _tilt_servo.change_duty_cycle(angle_to_duty_cycle(_tilt_angle))

    return {"pan": _pan_angle, "tilt": _tilt_angle}


def get_status() -> Dict:
    with _lock:
        return {
            "available": _available,
            "pan": _pan_angle,
            "tilt": _tilt_angle,
            "min": MIN_ANGLE,
            "max": MAX_ANGLE,
            "step": STEP_DEG,
        }


def shutdown_servos() -> None:
    global _pan_servo, _tilt_servo, _available

    with _lock:
        for servo in (_pan_servo, _tilt_servo):
            if servo is not None:
                try:
                    servo.stop()
                except Exception:
                    pass
        _pan_servo = None
        _tilt_servo = None
        _available = False
