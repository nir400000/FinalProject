from flask import Flask, Response, jsonify, request
from frame_buffer import update_frame
from sleep_tracker import get_sleep_tracker, STATE_AWAKE, STATE_OUT, STATE_SLEEPING
from inference_gate import get_inference_gate
import cv2
from picamera2 import Picamera2
import numpy as np
import math
from ultralytics import YOLO
import threading
import time


app = Flask(__name__, static_folder='media', static_url_path='/media')

# --- HARDWARE SETUP ---
# Initialize the camera
camera = Picamera2()
# Configuring for XRGB8888 (4 channels), we will convert to BGR later
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()

# --- MODEL SETUP ---
# Load the Nano model (fastest for Pi). It will download automatically on first run.
model = YOLO('yolov8n-pose.pt')

# Global state for threaded inference
_inference_state = {
    'frame_for_inference': None,
    'last_kps': [],
    'last_label': 'unknown',
    'sleep_state': STATE_OUT,
    'stop_event': threading.Event(),
    'state_lock': threading.Lock(),
    'worker': None
}

def _inference_worker():
    """Background thread that runs YOLO inference on resized frames."""
    inference_size = (320, 240)
    min_inference_interval = 0.08
    last_infer = 0.0
    
    while not _inference_state['stop_event'].is_set():
        img = None
        with _inference_state['state_lock']:
            if _inference_state['frame_for_inference'] is not None:
                img = _inference_state['frame_for_inference']
                orig_w = _inference_state['frame_w']
                orig_h = _inference_state['frame_h']
                _inference_state['frame_for_inference'] = None
        
        if img is None:
            time.sleep(0.005)
            continue
        
        now = time.time()
        if now - last_infer < min_inference_interval:
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
        with _inference_state['state_lock']:
            _inference_state['last_kps'] = scaled_kps
            _inference_state['last_label'] = lbl
            _inference_state['sleep_state'] = sleep_state

        from auto_tracker import update as auto_track_update

        auto_track_update(scaled_kps, orig_w, orig_h)

def get_yolo_keypoints(results):
    """
    Extracts keypoints from YOLO results ensuring compatibility with your logic.
    YOLO keypoints shape: (1, 17, 3) -> [x, y, conf]
    """
    # Check if any person was detected
    if not results or len(results[0].boxes) == 0:
        return []

    # Get the keypoints for the first detected person
    # data[0] gives us the first person's keypoints
    kps = results[0].keypoints.data[0].cpu().numpy()
    
    # Convert to a list of tuples (x, y, conf) to match your previous structure
    formatted_kps = []
    for kp in kps:
        formatted_kps.append((float(kp[0]), float(kp[1]), float(kp[2])))
    
    return formatted_kps

def angle_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def joint_angle(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 180.0
    cosv = max(-1.0, min(1.0, dot/mag))
    return math.degrees(math.acos(cosv))


def classify_pose(kps):
    if not kps:
        return 'unknown'

    # COCO Keypoint Indices (YOLO uses the same as MoveNet/COCO):
    # 5: Left Shoulder, 6: Right Shoulder
    # 11: Left Hip, 12: Right Hip
    # 13: Left Knee, 14: Right Knee
    # 15: Left Ankle, 16: Right Ankle
    
    s_pts = [kps[5], kps[6]]
    h_pts = [kps[11], kps[12]]
    k_pts = [kps[13], kps[14]]
    a_pts = [kps[15], kps[16]]

    def valid_mean(pts):
        vals = [p for p in pts if p[2] > 0.5] # Increased threshold slightly for YOLO
        if not vals: return None
        x = sum(p[0] for p in vals)/len(vals)
        y = sum(p[1] for p in vals)/len(vals)
        return (x,y)

    shoulders = valid_mean(s_pts)
    hips = valid_mean(h_pts)
    
    # Torso Angle Calculation
    if shoulders and hips:
        torso_angle = abs(angle_between(shoulders, hips) - 90)
    else:
        torso_angle = 90

    # Knee Angle Calculation
    knee_angles = []
    for hip, knee, ankle in zip(h_pts, k_pts, a_pts):
        # Check confidence (YOLO confidence is index 2)19
        if hip[2] > 0.5 and knee[2] > 0.5 and ankle[2] > 0.5:
            knee_angles.append(joint_angle(hip, knee, ankle))
    
    knee_angle = sum(knee_angles)/len(knee_angles) if knee_angles else 180.0

    # Heuristics
    if torso_angle > 50:
        return 'lying'
    if knee_angle < 140:
        return 'sitting'
    return 'standing'

def generate_frames():
    one_in_Xframes = 1
    frame_count = 0
    last_display_time = time.time()
    last_tracker_tick = 0.0
    last_auto_track = 0.0
    tracker_tick_interval = 0.5
    auto_track_interval = 0.15
    fps = 0.0
    gate = get_inference_gate()
    
    # Start inference worker if not running
    if _inference_state['worker'] is None or not _inference_state['worker'].is_alive():
        _inference_state['stop_event'].clear()
        _inference_state['worker'] = threading.Thread(target=_inference_worker, daemon=True)
        _inference_state['worker'].start()
    
    while True:
        # Capture frame (Format is XRGB, 4 channels)
        frame_xrgb = camera.capture_array()
        frame_bgr = np.ascontiguousarray(frame_xrgb[:, :, :3])
        update_frame(frame_bgr)
        
        with _inference_state['state_lock']:
            person_visible = _inference_state.get('sleep_state', STATE_OUT) != STATE_OUT

        now = time.time()
        run_yolo = False
        if frame_count % one_in_Xframes == 0:
            run_yolo = gate.should_run_inference(frame_bgr, person_visible=person_visible)
            if run_yolo:
                with _inference_state['state_lock']:
                    _inference_state['frame_for_inference'] = frame_bgr.copy()
                    _inference_state['frame_w'] = frame_bgr.shape[1]
                    _inference_state['frame_h'] = frame_bgr.shape[0]
            elif now - last_tracker_tick >= tracker_tick_interval:
                with _inference_state['state_lock']:
                    cached_kps = list(_inference_state.get('last_kps') or [])
                    cached_label = _inference_state.get('last_label', 'unknown')
                sleep_state = get_sleep_tracker().update(cached_kps, cached_label, now)
                with _inference_state['state_lock']:
                    _inference_state['sleep_state'] = sleep_state
                last_tracker_tick = now

        from auto_tracker import is_enabled as auto_track_enabled, update as auto_track_update

        if auto_track_enabled() and now - last_auto_track >= auto_track_interval:
            with _inference_state['state_lock']:
                track_kps = list(_inference_state.get('last_kps') or [])
            auto_track_update(track_kps, frame_bgr.shape[1], frame_bgr.shape[0])
            last_auto_track = now
        
        frame_count += 1
        
        # Measure FPS
        now_display = time.time()
        dt = now_display - last_display_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = fps * 0.9 + inst_fps * 0.1 if fps else inst_fps
        last_display_time = now_display
        
        # Get latest inference results
        with _inference_state['state_lock']:
            label = _inference_state['last_label']
            sleep_state = _inference_state.get('sleep_state', STATE_OUT)
        
        # Draw text
        cv2.putText(frame_bgr, f"Pose: {label}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        sleep_color = (0, 200, 255) if sleep_state == STATE_SLEEPING else (
            (0, 165, 255) if sleep_state == STATE_AWAKE else (128, 128, 128)
        )
        cv2.putText(frame_bgr, f"Sleep: {sleep_state}", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sleep_color, 2)
        cv2.putText(frame_bgr, f"{fps:.1f}", (10, 470), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Encode for web streaming
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/health')
def health():
    """Lightweight check that the phone can reach the monitor over Wi-Fi."""
    from wifi_manager import get_primary_ip

    return jsonify({
        'status': 'ok',
        'ip': get_primary_ip(),
        'port': 5001,
    })


@app.route('/pose')
def pose_json():
    with _inference_state['state_lock']:
        label = _inference_state['last_label']
        sleep_state = _inference_state.get('sleep_state', STATE_OUT)
    return jsonify({'pose': label, 'sleep_state': sleep_state})


@app.route('/sleep/status')
def sleep_status():
    return jsonify(get_sleep_tracker().get_status())


@app.route('/sleep/analytics')
def sleep_analytics():
    return jsonify(get_sleep_tracker().get_full_report())

@app.route('/remote/info')
def remote_info():
    from device_registry import get_remote_info_json

    return get_remote_info_json(), 200, {"Content-Type": "application/json"}


@app.route('/audio/info')
def audio_info():
    from audio_stream import get_audio_info

    return jsonify(get_audio_info())


@app.route('/audio_feed')
def audio_feed():
    from audio_stream import SAMPLE_RATE, generate_audio_stream

    try:
        return Response(
            generate_audio_stream(),
            mimetype=f"audio/L16; rate={SAMPLE_RATE}; channels=1",
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.route('/servo', methods=['GET', 'POST'])
def servo_control():
    from auto_tracker import get_status as auto_track_status, is_enabled, set_enabled
    from servo_controller import get_status, init_servos, set_angles, step_angles

    if request.method == 'GET':
        init_servos()
        return jsonify({**get_status(), **auto_track_status()})

    data = request.get_json(force=True, silent=True) or {}
    try:
        if 'auto_track' in data:
            set_enabled(bool(data['auto_track']))
            init_servos()
            return jsonify({"ok": True, **get_status(), **auto_track_status()})

        if is_enabled() and ('pan_delta' in data or 'tilt_delta' in data):
            return jsonify({"ok": False, "error": "Auto track is enabled"}), 409

        if 'pan_delta' in data or 'tilt_delta' in data:
            result = step_angles(
                data.get('pan_delta', 0),
                data.get('tilt_delta', 0),
            )
        else:
            result = set_angles(data.get('pan', 0), data.get('tilt', 0))
        return jsonify({"ok": True, **result, **auto_track_status()})
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route('/cry/status')
def cry_status():
    from cry_detector import get_status

    return jsonify(get_status())


@app.route('/')
def home():
    return '<h1>Baby Monitor</h1><img src="/video_feed" style="width:640px; height:480px;" />'

if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        try:
            from ble_wifi_provision import start_ble_provisioning_thread
            start_ble_provisioning_thread()
            print('BLE Wi-Fi provisioning started (advertising as BabyMonitor)')
        except Exception as ble_exc:
            print(f'BLE Wi-Fi provisioning not available: {ble_exc}')

        try:
            from audio_stream import start_capture_hub, shutdown_capture_hub
            from cry_detector import init_cry_detector

            start_capture_hub()
            if init_cry_detector():
                print('Baby cry detection (YAMNet) ready')
            else:
                print('Baby cry detection unavailable — run scripts/download_yamnet.py')
            print('Nursery microphone capture started')
        except Exception as audio_exc:
            print(f'Nursery microphone not available: {audio_exc}')

        # Threaded=True is important for Flask with Video Streaming
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        _inference_state['stop_event'].set()
        if _inference_state['worker']:
            _inference_state['worker'].join(timeout=1.0)
        try:
            from servo_controller import shutdown_servos
            shutdown_servos()
        except Exception:
            pass
        try:
            from audio_stream import shutdown_capture_hub
            shutdown_capture_hub()
        except Exception:
            pass