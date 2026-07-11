"""YOLO pose keypoint extraction and baby posture classification."""

from __future__ import annotations

import logging
import math
from typing import List, Sequence, Tuple

import joblib
import numpy as np

from server.paths import POSE_CLASSIFIER_FALLBACK_PATH, POSE_CLASSIFIER_PATH
from server.vision.pose_features import NUM_KEYPOINTS, normalize_keypoints

logger = logging.getLogger(__name__)

_classifier_bundle = None
_classifier_lookup_done = False
MIN_CLASSIFIER_CONFIDENCE = 0.4


def get_yolo_keypoints(results) -> List[Tuple[float, float, float]]:
    """Extract COCO keypoints from the highest-confidence YOLO pose detection."""
    if not results or len(results[0].boxes) == 0:
        return []

    result = results[0]
    person_idx = int(result.boxes.conf.argmax().item())
    kps = result.keypoints.data[person_idx].cpu().numpy()
    return [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in kps]


def classify_pose(kps: Sequence[Tuple[float, float, float]]) -> str:
    if not kps:
        return "unknown"

    label = _classify_with_model(kps)
    if label != "unknown":
        return label

    return _classify_pose_heuristic(kps)


def _load_classifier():
    global _classifier_bundle, _classifier_lookup_done
    if _classifier_lookup_done:
        return _classifier_bundle

    _classifier_lookup_done = True
    for path in (POSE_CLASSIFIER_PATH, POSE_CLASSIFIER_FALLBACK_PATH):
        if not path.exists():
            continue
        try:
            _classifier_bundle = joblib.load(path)
            logger.info("Loaded pose classifier from %s", path)
            return _classifier_bundle
        except Exception as exc:
            logger.warning("Failed to load pose classifier from %s: %s", path, exc)

    logger.warning(
        "Pose classifier not found; using heuristic fallback. "
        "Train with pose_dataset/train_pose_classifier.py"
    )
    return None


def _classify_with_model(kps: Sequence[Tuple[float, float, float]]) -> str:
    bundle = _load_classifier()
    if not bundle:
        return "unknown"

    if len(kps) != NUM_KEYPOINTS:
        return "unknown"

    kpts_xy = np.array([[x, y] for x, y, _ in kps], dtype=np.float32)
    kpts_conf = np.array([c for _, _, c in kps], dtype=np.float32)
    features = normalize_keypoints(kpts_xy, kpts_conf).reshape(1, -1)

    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    proba = pipeline.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    if float(proba[pred_idx]) < MIN_CLASSIFIER_CONFIDENCE:
        return "unknown"

    return str(label_encoder.inverse_transform([pred_idx])[0])


def _classify_pose_heuristic(kps: Sequence[Tuple[float, float, float]]) -> str:
    """Legacy angle-based fallback when the trained classifier is unavailable."""
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
        torso_angle = abs(_angle_between(shoulders, hips) - 90)
    else:
        torso_angle = 90

    knee_angles = []
    for hip, knee, ankle in zip(h_pts, k_pts, a_pts):
        if hip[2] > 0.5 and knee[2] > 0.5 and ankle[2] > 0.5:
            knee_angles.append(_joint_angle(hip, knee, ankle))

    knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else 180.0

    if torso_angle > 50:
        return "lying"
    if knee_angle < 140:
        return "sitting"
    return "standing"


def _angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def _joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 180.0
    cosv = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cosv))
