from flask import Flask, Response, jsonify
import cv2
from picamera2 import Picamera2
import numpy as np
import math

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite


app = Flask(__name__, static_folder='media', static_url_path='/media')


camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
camera.start()

# load TFLite MoveNet model (place model file in project)
MOVENET_MODEL = 'movenet_singlepose_lightning.tflite'
interpreter = tflite.Interpreter(model_path=MOVENET_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]

def movenet_keypoints(frame):
    img = cv2.resize(frame, (input_size, input_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    input_data = np.expand_dims(img_rgb / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    # MoveNet outputs shape (1,1,17,3) or (1,17,3)
    kps = out.reshape(-1, 3)[0:17] if out.ndim == 4 else out[0]
    # each keypoint: (y, x, score) -> convert to original frame coords
    h, w = frame.shape[:2]
    keypoints = []
    for kp in kps:
        y, x, s = kp
        keypoints.append((x * w, y * h, s))
    return keypoints

def angle_between(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def joint_angle(a, b, c):
    # angle at b formed by a-b-c in degrees
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 180.0
    cosv = max(-1.0, min(1.0, dot/mag))
    return math.degrees(math.acos(cosv))


def classify_pose(kps):
    # indices: 5 left shoulder,6 right shoulder,11 left hip,12 right hip,13 left knee,14 right knee,15 left ankle,16 right ankle
    s_pts = [kps[5], kps[6]]
    h_pts = [kps[11], kps[12]]
    k_pts = [kps[13], kps[14]]
    a_pts = [kps[15], kps[16]]
    # use average where available
    def valid_mean(pts):
        vals = [p for p in pts if p[2] > 0.2]
        if not vals: return None
        x = sum(p[0] for p in vals)/len(vals)
        y = sum(p[1] for p in vals)/len(vals)
        return (x,y)
    shoulders = valid_mean(s_pts)
    hips = valid_mean(h_pts)
    knees = valid_mean(k_pts)
    # torso angle
    if shoulders and hips:
        torso_angle = abs(angle_between(shoulders, hips) - 90)  # distance from vertical (deg)
    else:
        torso_angle = 90
    # knee angles
    knee_angles = []
    for hip, knee, ankle in zip(h_pts, k_pts, a_pts):
        if hip[2] > 0.2 and knee[2] > 0.2 and ankle[2] > 0.2:
            knee_angles.append(joint_angle(hip, knee, ankle))
    knee_angle = sum(knee_angles)/len(knee_angles) if knee_angles else 180.0
    # heuristics
    if torso_angle > 50:
        return 'lying'
    if knee_angle < 140:
        return 'sitting'
    return 'standing'

def generate_frames():
    while True:
        frame = camera.capture_array()
        kps = movenet_keypoints(frame)
        label = classify_pose(kps)
        cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/pose')
def pose_json():
    frame = camera.capture_array()
    kps = movenet_keypoints(frame)
    label = classify_pose(kps)
    return jsonify({'pose': label})
    
    
@app.route('/')
def home():
    return '<h1>Welcome to your RPi Web Server!</h1><video src="/media/video.mp4" alt="Baby" style="max-width: 500px;" controls autoplay>'


@app.route('/about')
def about():
    return '<h1>About This Server</h1><p>This server is running on a Raspberry Pi using Flask.</p>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)