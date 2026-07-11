"""
webcam_pose_tester.py
=====================
Opens the computer camera, detects a skeleton in real time with YOLOv8-Pose,
and classifies the pose (standing / sitting / lying) using the model trained by train_pose_classifier.py.

The detected pose is shown in a Tkinter GUI over the live video feed, including the
model confidence for the prediction.

Run:
    python webcam_pose_tester.py

Requirement: run pose_labeler.py first (to label data) and then train_pose_classifier.py (to train),
so that pose_classifier.joblib exists.
"""

import os
import sys
import time

import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

from pose_features import normalize_keypoints

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_classifier.joblib")

YOLO_WEIGHTS = "yolov8n-pose.pt"
PERSON_CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0
MAX_DISPLAY_SIZE = 780

DISPLAY_LABELS = {
    "standing": "Standing",
    "sitting": "Sitting",
    "lying": "Lying",
}

LABEL_COLORS = {
    "standing": "#2e7d32",
    "sitting": "#1565c0",
    "lying": "#b71c1c",
}


class WebcamPoseTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time pose classification - Camera")
        self.root.geometry("880x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Error",
                f"No trained model was found at:\n{MODEL_PATH}\n\n"
                "Please run: python train_pose_classifier.py",
            )
            root.destroy()
            sys.exit(1)

        print("Loading pose classification model...")
        bundle = joblib.load(MODEL_PATH)
        self.pipeline = bundle["pipeline"]
        self.label_encoder = bundle["label_encoder"]

        print("Loading YOLOv8-Pose model...")
        self.yolo_model = YOLO(YOLO_WEIGHTS)

        print("Opening camera...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the computer camera.")
            root.destroy()
            sys.exit(1)

        self.running = True
        self.current_photo = None
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self._build_ui()
        self._update_frame()

    # ------------------------------------------------------------------
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(pady=6)
        self.info_label = tk.Label(top, text="", font=("Arial", 11))
        self.info_label.pack()

        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=6)

        self.pose_label = tk.Label(
            self.root, text="Waiting for detection...", font=("Arial", 28, "bold"),
            fg="white", bg="#555555", width=20, height=2,
        )
        self.pose_label.pack(pady=12)

        self.warning_label = tk.Label(
            self.root, text="", font=("Arial", 12, "bold"), fg="red"
        )
        self.warning_label.pack()

        tk.Button(
            self.root, text="Close", font=("Arial", 12), width=12,
            command=self.on_close,
        ).pack(pady=10)

    # ------------------------------------------------------------------
    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.warning_label.configure(text="⚠ Could not read a frame from the camera")
            self.root.after(30, self._update_frame)
            return

        results = self.yolo_model(frame, verbose=False)
        r = results[0]

        num_persons = 0 if r.boxes is None else int(
            (r.boxes.conf >= PERSON_CONF_THRESHOLD).sum().item()
        )

        annotated_bgr = r.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        if num_persons == 1:
            person_idx = int(r.boxes.conf.argmax().item())
            kpts_xy = r.keypoints.xy[person_idx].cpu().numpy()
            kpts_conf = r.keypoints.conf[person_idx].cpu().numpy()

            features = normalize_keypoints(kpts_xy, kpts_conf).reshape(1, -1)
            pred_encoded = self.pipeline.predict(features)[0]
            pred_label_eng = self.label_encoder.inverse_transform([pred_encoded])[0]

            proba = self.pipeline.predict_proba(features)[0]
            confidence = float(np.max(proba))

            display_label = DISPLAY_LABELS.get(pred_label_eng, pred_label_eng)
            color = LABEL_COLORS.get(pred_label_eng, "#555555")
            self.pose_label.configure(
                text=f"{display_label}   ({confidence:.0%})", bg=color
            )
            self.warning_label.configure(text="")

        elif num_persons == 0:
            self.pose_label.configure(text="No person detected in frame", bg="#555555")
            self.warning_label.configure(text="")
        else:
            self.pose_label.configure(text="Waiting for detection...", bg="#555555")
            self.warning_label.configure(
                text=f"⚠ Detected {num_persons} people - exactly one is required for classification"
            )

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
        self.info_label.configure(
            text=f"People in frame: {num_persons}   |   FPS: {self.fps:.1f}"
        )

        pil_img = Image.fromarray(annotated_rgb)
        w, h = pil_img.size
        scale = MAX_DISPLAY_SIZE / max(w, h)
        if scale < 1:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        self.current_photo = ImageTk.PhotoImage(pil_img)
        self.video_label.configure(image=self.current_photo)

        self.root.after(15, self._update_frame)

    # ------------------------------------------------------------------
    def on_close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = WebcamPoseTester(root)
    root.mainloop()


if __name__ == "__main__":
    main()
