"""Combined nursery state for the Android alert service."""

from __future__ import annotations

import time
from typing import Dict

from inference_gate import get_inference_gate
from sleep_tracker import STATE_OUT, get_sleep_tracker


def get_alert_snapshot(inference_state: dict) -> Dict:
    from cry_detector import get_status as cry_status
    from sound_meter import get_status as sound_status

    cry = cry_status()
    sound = sound_status()
    gate = get_inference_gate()

    with inference_state["state_lock"]:
        pose = inference_state.get("last_label", "unknown")
        sleep_state = inference_state.get("sleep_state", STATE_OUT)

    tracker = get_sleep_tracker().get_status()
    motion = float(gate.last_motion_score)
    if motion == float("inf"):
        motion = 0.0

    return {
        "timestamp": time.time(),
        "audio": {
            "available": bool(cry.get("available")) and bool(sound.get("available")),
            "level": sound.get("level", 0.0),
            "peak": sound.get("peak", 0.0),
            "cry_score": cry.get("score", 0.0),
            "crying": cry.get("crying", False),
            "cry_label": cry.get("label", "idle"),
        },
        "camera": {
            "available": True,
            "motion_score": round(motion, 3),
            "pose": pose,
            "sleep_state": sleep_state,
            "activity_index": tracker.get("activity_index", 0.0),
            "person_visible": sleep_state != STATE_OUT,
        },
    }
