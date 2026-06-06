import cv2
import numpy as np
import math
import threading
import time
from picamera2 import Picamera2
from ultralytics import YOLO
from sleep_tracker import get_sleep_tracker, STATE_AWAKE, STATE_OUT, STATE_SLEEPING

# --- HARDWARE SETUP ---
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()

# --- MODEL SETUP ---
model = YOLO('yolov8n-pose.pt')


def get_yolo_keypoints(results):
    """Extracts keypoints from YOLO results in (x, y, conf) format."""
    if not results or len(results[0].boxes) == 0:
        return []

    kps = results[0].keypoints.data[0].cpu().numpy()
    return [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in kps]


def angle_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def joint_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 180.0
    cosv = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cosv))


def classify_pose(kps):
    if not kps:
        return 'unknown'

    s_pts = [kps[5], kps[6]]
    h_pts = [kps[11], kps[12]]
    k_pts = [kps[13], kps[14]]
    a_pts = [kps[15], kps[16]]

    def valid_mean(pts):
        vals = [p for p in pts if p[2] > 0.5]
        if not vals:
            return None
        x = sum(p[0] for p in vals) / len(vals)
        y = sum(p[1] for p in vals) / len(vals)
        return (x, y)

    shoulders = valid_mean(s_pts)
    hips = valid_mean(h_pts)

    if shoulders and hips:
        torso_angle = abs(angle_between(shoulders, hips) - 90)
    else:
        torso_angle = 90

    knee_angles = []
    for hip, knee, ankle in zip(h_pts, k_pts, a_pts):
        if hip[2] > 0.5 and knee[2] > 0.5 and ankle[2] > 0.5:
            knee_angles.append(joint_angle(hip, knee, ankle))

    knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else 180.0

    if torso_angle > 50:
        return 'lying'
    if knee_angle < 140:
        return 'sitting'
    return 'standing'


def main():
    # Parameters to tune for performance
    one_in_Xframes = 1            # how often to post a frame for inference
    inference_size = (320, 240)   # run model on this smaller size for speed
    min_inference_interval = 0.08  # minimum seconds between inferences (~12.5 FPS)

    frame_count = 0
    label = 'unknown'

    # Shared state between capture loop and inference thread
    frame_for_inference = {'img': None, 'w': 640, 'h': 480}
    last_kps = []
    last_label = 'unknown'
    # holders so worker can update without rebinding outer vars
    last_label_holder = [last_label]
    last_sleep_state_holder = [STATE_OUT]
    last_activity_holder = [0.0]
    stop_event = threading.Event()
    state_lock = threading.Lock()

    def inference_worker():
        last_infer = 0.0
        while not stop_event.is_set():
            img = None
            with state_lock:
                if frame_for_inference['img'] is not None:
                    img = frame_for_inference['img']
                    orig_w = frame_for_inference['w']
                    orig_h = frame_for_inference['h']
                    frame_for_inference['img'] = None
            if img is None:
                time.sleep(0.005)
                continue

            now = time.time()
            if now - last_infer < min_inference_interval:
                # skip to limit inference rate
                continue
            last_infer = now

            # Resize for faster inference
            small = cv2.resize(img, inference_size)
            try:
                results = model(small, verbose=False)
            except Exception:
                continue

            kps_small = get_yolo_keypoints(results)
            # Scale keypoints back to original frame size
            scaled_kps = []
            if kps_small:
                sx = orig_w / inference_size[0]
                sy = orig_h / inference_size[1]
                for x, y, c in kps_small:
                    scaled_kps.append((x * sx, y * sy, c))

            lbl = classify_pose(scaled_kps)
            # Update sleep tracker
            now_ts = time.time()
            try:
                sleep_state = get_sleep_tracker().update(scaled_kps, lbl, now_ts)
                activity_index = get_sleep_tracker().get_status().get('activity_index', 0.0)
            except Exception:
                sleep_state = STATE_OUT
                activity_index = 0.0

            with state_lock:
                last_kps.clear()
                last_kps.extend(scaled_kps)
                last_label_holder[0] = lbl
                last_sleep_state_holder[0] = sleep_state
                last_activity_holder[0] = float(activity_index)

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    def draw_keypoints(frame, kps, conf_thresh=0.3):
        """Draw keypoints and a simple COCO skeleton on the frame."""
        if not kps:
            return
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        for a, b in skeleton:
            if kps[a][2] > conf_thresh and kps[b][2] > conf_thresh:
                pt1 = (int(kps[a][0]), int(kps[a][1]))
                pt2 = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        for i, (x, y, c) in enumerate(kps):
            if c > conf_thresh:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    last_kps = []

    # Display FPS measurement
    last_display_time = time.time()
    fps = 0.0

    while True:
        frame_xrgb = camera.capture_array()
        frame_bgr = np.ascontiguousarray(frame_xrgb[:, :, :3])

        if frame_count % one_in_Xframes == 0:
            # Post a copy for the worker to pick up
            with state_lock:
                frame_for_inference['img'] = frame_bgr.copy()
                frame_for_inference['w'] = frame_bgr.shape[1]
                frame_for_inference['h'] = frame_bgr.shape[0]

        frame_count += 1
        # Measure display FPS (smoothed)
        now_display = time.time()
        dt = now_display - last_display_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = fps * 0.9 + inst_fps * 0.1 if fps else inst_fps
        last_display_time = now_display

        cv2.putText(frame_bgr, f'Status: {label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame_bgr, f'FPS: {fps:.1f}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Draw last detected keypoints/skeleton and overlay tracker info
        with state_lock:
            draw_keypoints(frame_bgr, last_kps)
            label = last_label_holder[0]
            sleep_state = last_sleep_state_holder[0]
            activity_index = last_activity_holder[0]

        cv2.putText(frame_bgr, f'Sleep: {sleep_state} ({activity_index:.2f})', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow('Pi Camera Feed', frame_bgr)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # Shutdown worker
    stop_event.set()
    worker.join(timeout=1.0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
