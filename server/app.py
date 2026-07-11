"""Flask HTTP server — video, audio, ML inference, and REST API."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2
from ultralytics import YOLO

from server.paths import YOLO_MODEL_PATH
from server.vision.auto_tracker import is_enabled as auto_track_enabled
from server.vision.auto_tracker import update as auto_track_update
from server.vision.frame_buffer import update_frame
from server.vision.inference_gate import get_inference_gate
from server.vision.pose import classify_pose, get_yolo_keypoints
from server.vision.sleep_tracker import (
    STATE_AWAKE,
    STATE_OUT,
    STATE_SLEEPING,
    get_sleep_tracker,
)

app = Flask(__name__)

camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
)
camera.start()

model = YOLO(str(YOLO_MODEL_PATH))

_inference_state = {
    "frame_for_inference": None,
    "last_kps": [],
    "last_kps_time": 0.0,
    "last_label": "unknown",
    "sleep_state": STATE_OUT,
    "stop_event": threading.Event(),
    "state_lock": threading.Lock(),
    "worker": None,
}


def _inference_worker() -> None:
    inference_size = (320, 240)
    last_infer = 0.0

    while not _inference_state["stop_event"].is_set():
        img = None
        with _inference_state["state_lock"]:
            if _inference_state["frame_for_inference"] is not None:
                img = _inference_state["frame_for_inference"]
                orig_w = _inference_state["frame_w"]
                orig_h = _inference_state["frame_h"]
                _inference_state["frame_for_inference"] = None

        if img is None:
            time.sleep(0.005)
            continue

        now = time.time()
        min_interval = 0.30 if auto_track_enabled() else 0.12
        if now - last_infer < min_interval:
            continue
        last_infer = now

        small = cv2.resize(img, inference_size)
        try:
            results = model(small, verbose=False)
        except Exception:
            continue

        kps_small = get_yolo_keypoints(results)
        scaled_kps = []
        if kps_small:
            sx = orig_w / inference_size[0]
            sy = orig_h / inference_size[1]
            for x, y, c in kps_small:
                scaled_kps.append((x * sx, y * sy, c))

        lbl = classify_pose(scaled_kps)
        sleep_state = get_sleep_tracker().update(scaled_kps, lbl, now)
        get_inference_gate().mark_inferred()
        with _inference_state["state_lock"]:
            _inference_state["last_kps"] = scaled_kps
            _inference_state["last_label"] = lbl
            _inference_state["sleep_state"] = sleep_state
            _inference_state["last_kps_time"] = now

        if auto_track_enabled():
            auto_track_update(scaled_kps, orig_w, orig_h)


def generate_frames():
    frame_count = 0
    last_display_time = time.time()
    last_tracker_tick = 0.0
    tracker_tick_interval = 0.5
    fps = 0.0
    gate = get_inference_gate()

    if _inference_state["worker"] is None or not _inference_state["worker"].is_alive():
        _inference_state["stop_event"].clear()
        _inference_state["worker"] = threading.Thread(
            target=_inference_worker, daemon=True
        )
        _inference_state["worker"].start()

    while True:
        frame_xrgb = camera.capture_array()
        frame_bgr = np.ascontiguousarray(frame_xrgb[:, :, :3])
        update_frame(frame_bgr)

        with _inference_state["state_lock"]:
            person_visible = _inference_state.get("sleep_state", STATE_OUT) != STATE_OUT

        motion_score = gate.motion_score(frame_bgr)
        now = time.time()
        if frame_count % 1 == 0:
            run_yolo = gate.should_run_inference_with_score(
                motion_score, person_visible=person_visible
            )
            if run_yolo:
                with _inference_state["state_lock"]:
                    _inference_state["frame_for_inference"] = frame_bgr.copy()
                    _inference_state["frame_w"] = frame_bgr.shape[1]
                    _inference_state["frame_h"] = frame_bgr.shape[0]
            elif now - last_tracker_tick >= tracker_tick_interval:
                with _inference_state["state_lock"]:
                    cached_kps = list(_inference_state.get("last_kps") or [])
                    cached_label = _inference_state.get("last_label", "unknown")
                sleep_state = get_sleep_tracker().update(cached_kps, cached_label, now)
                with _inference_state["state_lock"]:
                    _inference_state["sleep_state"] = sleep_state
                last_tracker_tick = now

        frame_count += 1

        now_display = time.time()
        dt = now_display - last_display_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = fps * 0.9 + inst_fps * 0.1 if fps else inst_fps
        last_display_time = now_display

        with _inference_state["state_lock"]:
            label = _inference_state["last_label"]
            sleep_state = _inference_state.get("sleep_state", STATE_OUT)

        cv2.putText(
            frame_bgr,
            f"Pose: {label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        sleep_color = (
            (0, 200, 255)
            if sleep_state == STATE_SLEEPING
            else (0, 165, 255) if sleep_state == STATE_AWAKE else (128, 128, 128)
        )
        cv2.putText(
            frame_bgr,
            f"Sleep: {sleep_state}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            sleep_color,
            2,
        )
        cv2.putText(
            frame_bgr,
            f"{fps:.1f}",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )

        _, buffer = cv2.imencode(".jpg", frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    from server.provisioning.wifi_manager import get_primary_ip

    return jsonify({"status": "ok", "ip": get_primary_ip(), "port": 5001})


@app.route("/pose")
def pose_json():
    with _inference_state["state_lock"]:
        label = _inference_state["last_label"]
        sleep_state = _inference_state.get("sleep_state", STATE_OUT)
    return jsonify({"pose": label, "sleep_state": sleep_state})


@app.route("/sleep/status")
def sleep_status():
    return jsonify(get_sleep_tracker().get_status())


@app.route("/sleep/analytics")
def sleep_analytics():
    return jsonify(get_sleep_tracker().get_full_report())


@app.route("/remote/info")
def remote_info():
    from server.provisioning.device_registry import get_remote_info_json

    return get_remote_info_json(), 200, {"Content-Type": "application/json"}


@app.route("/audio/info")
def audio_info():
    from server.audio.stream import get_audio_info

    return jsonify(get_audio_info())


@app.route("/audio_feed")
def audio_feed():
    from server.audio.stream import SAMPLE_RATE, generate_audio_stream

    try:
        return Response(
            generate_audio_stream(),
            mimetype=f"audio/L16; rate={SAMPLE_RATE}; channels=1",
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.route("/servo", methods=["GET", "POST"])
def servo_control():
    from server.hardware.servo_controller import (
        get_status,
        init_servos,
        set_angles,
        step_angles,
    )
    from server.vision.auto_tracker import get_status as auto_track_status
    from server.vision.auto_tracker import is_enabled, set_enabled

    if request.method == "GET":
        init_servos()
        return jsonify({**get_status(), **auto_track_status()})

    data = request.get_json(force=True, silent=True) or {}
    try:
        if "auto_track" in data:
            set_enabled(bool(data["auto_track"]))
            init_servos()
            return jsonify({"ok": True, **get_status(), **auto_track_status()})

        if is_enabled() and ("pan_delta" in data or "tilt_delta" in data):
            return jsonify({"ok": False, "error": "Auto track is enabled"}), 409

        if "pan_delta" in data or "tilt_delta" in data:
            result = step_angles(
                data.get("pan_delta", 0),
                data.get("tilt_delta", 0),
            )
        else:
            result = set_angles(data.get("pan", 0), data.get("tilt", 0))
        return jsonify({"ok": True, **result, **auto_track_status()})
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/alerts/snapshot")
def alerts_snapshot():
    from server.alerts.snapshot import get_alert_snapshot

    response = jsonify(get_alert_snapshot(_inference_state))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/cry/status")
def cry_status():
    from server.audio.cry_detector import get_status

    response = jsonify(get_status())
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def home():
    return (
        "<h1>Smart Baby Monitor</h1>"
        '<p>Live nursery feed:</p>'
        '<img src="/video_feed" style="width:640px; height:480px;" />'
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        try:
            from server.provisioning.ble_wifi_provision import start_ble_provisioning_thread

            start_ble_provisioning_thread()
            print("BLE Wi-Fi provisioning started (advertising as BabyMonitor)")
        except Exception as ble_exc:
            print(f"BLE Wi-Fi provisioning not available: {ble_exc}")

        try:
            from server.audio.cry_detector import init_cry_detector
            from server.audio.stream import start_capture_hub

            start_capture_hub()
            if init_cry_detector():
                print("Baby cry detection (YAMNet) ready")
            else:
                print("Baby cry detection unavailable — run: python scripts/download_yamnet.py")
            print("Nursery microphone capture started")
        except Exception as audio_exc:
            print(f"Nursery microphone not available: {audio_exc}")

        app.run(host="0.0.0.0", port=5001, threaded=True)
    finally:
        _inference_state["stop_event"].set()
        if _inference_state["worker"]:
            _inference_state["worker"].join(timeout=1.0)
        try:
            from server.hardware.servo_controller import shutdown_servos

            shutdown_servos()
        except Exception:
            pass
        try:
            from server.audio.stream import shutdown_capture_hub

            shutdown_capture_hub()
        except Exception:
            pass
