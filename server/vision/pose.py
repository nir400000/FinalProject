"""YOLO pose keypoint extraction and baby posture classification."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def get_yolo_keypoints(results) -> List[Tuple[float, float, float]]:
    """Extract COCO keypoints from YOLO pose results as (x, y, confidence)."""
    if not results or len(results[0].boxes) == 0:
        return []

    kps = results[0].keypoints.data[0].cpu().numpy()
    return [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in kps]


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 180.0
    cosv = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cosv))


def classify_pose(kps: Sequence[Tuple[float, float, float]]) -> str:
    if not kps:
        return "unknown"

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
        return "lying"
    if knee_angle < 140:
        return "sitting"
    return "standing"
