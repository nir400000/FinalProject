"""Normalize YOLO COCO keypoints into a fixed-length feature vector for pose classification."""

from __future__ import annotations

import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

NUM_KEYPOINTS = len(KEYPOINT_NAMES)
FEATURE_VECTOR_LENGTH = NUM_KEYPOINTS * 2 + NUM_KEYPOINTS

DEFAULT_CONF_THRESHOLD = 0.1


def normalize_keypoints(kpts_xy, kpts_conf, conf_threshold=DEFAULT_CONF_THRESHOLD):
    """Return a 51-D float32 vector: normalized (x, y) plus per-keypoint confidence."""
    kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)
    kpts_conf = np.asarray(kpts_conf, dtype=np.float32).reshape(NUM_KEYPOINTS)

    hip_center = (kpts_xy[LEFT_HIP] + kpts_xy[RIGHT_HIP]) / 2.0
    shoulder_center = (kpts_xy[LEFT_SHOULDER] + kpts_xy[RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(shoulder_center - hip_center))

    if scale < 1e-3:
        valid = kpts_conf >= conf_threshold
        if valid.sum() >= 2:
            pts = kpts_xy[valid]
            scale = float(
                max(
                    pts[:, 0].max() - pts[:, 0].min(),
                    pts[:, 1].max() - pts[:, 1].min(),
                )
            )
        if scale < 1e-3:
            scale = 1.0

    normalized = (kpts_xy - hip_center) / scale
    low_conf_mask = kpts_conf < conf_threshold
    normalized[low_conf_mask] = 0.0

    features = np.concatenate([normalized.flatten(), kpts_conf])
    return features.astype(np.float32)
