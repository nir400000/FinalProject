from flask import Flask, Response, jsonify
import cv2
from picamera2 import Picamera2
import numpy as np
import math
from ultralytics import YOLO
import threading


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
        # Check confidence (YOLO confidence is index 2)
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
    while True:
        # Capture frame (Format is XRGB, 4 channels)
        frame_xrgb = camera.capture_array()
        
        # Drop the alpha/padding channel to get standard BGR for YOLO/OpenCV
        # Force memory to be contiguous so OpenCV can write on it
        frame_bgr = np.ascontiguousarray(frame_xrgb[:, :, :3])
        
        # Run YOLO inference
        # verbose=False keeps the terminal clean
        results = model(frame_bgr, verbose=False) 
        
        # Extract data
        kps = get_yolo_keypoints(results)
        label = classify_pose(kps)

        # Draw Visualization
        # YOLO has a built-in plotter if you want fancy skeletons:
        # frame_bgr = results[0].plot() 
        # OR stick to your simple text:
        cv2.putText(frame_bgr, f"Status: {label}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode for web streaming
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/pose')
def pose_json():
    frame_xrgb = camera.capture_array()
    frame_bgr = frame_xrgb[:, :, :3]
    
    results = model(frame_bgr, verbose=False)
    kps = get_yolo_keypoints(results)
    label = classify_pose(kps)
    return jsonify({'pose': label})

@app.route('/')
def home():
    return '<h1>Baby Monitor</h1><img src="/video_feed" style="width:640px; height:480px;" />'

if __name__ == '__main__':
    # Threaded=True is important for Flask with Video Streaming
    app.run(host='0.0.0.0', port=5001, threaded=True)