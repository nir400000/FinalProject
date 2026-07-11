"""
pose_features.py
=================
מודול משותף להמרת 17 נקודות המפתח (COCO keypoints) של YOLOv8-Pose
לוקטור פיצ'רים נורמלי, שמשמש גם לאימון המודל וגם לחיזוי בזמן אמת (וידאו).

חשוב: סדר נקודות המפתח כאן חייב להיות זהה לסדר בו נשמרו בקובץ labels.csv
(ראה KEYPOINT_NAMES ב-pose_labeler.py) - אחרת החיזוי יהיה שגוי.

איך הנרמול עובד (ולמה):
- נקודות המפתח הגולמיות (x, y) הן בפיקסלים, תלויות בגודל התמונה/מרחק המצלמה
  ובמיקום האדם בפריים. זה גרוע לפיצ'רים של מודל.
- לכן ממרכזים את כל הנקודות סביב מרכז האגן (hip center), ומנרמלים לפי אורך הפלג
  העליון (המרחק בין מרכז הכתפיים למרכז האגן). כך הפיצ'רים הופכים בלתי-תלויים
  בגודל התמונה, במרחק מהמצלמה ובמיקום האדם בפריים - ותלויים רק בתנוחת הגוף עצמה.
- נקודות עם ביטחון (confidence) נמוך מדי (למשל חלק גוף מוסתר) מאופסות ל-(0,0)
  כדי לא "להזיז" את הפיצ'רים בגלל זיהוי לא אמין, וערך ה-confidence עצמו נשמר
  כפיצ'ר נוסף כדי שהמודל ילמד להתחשב באמינות הנתון.
"""

import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

NUM_KEYPOINTS = len(KEYPOINT_NAMES)          # 17
FEATURE_VECTOR_LENGTH = NUM_KEYPOINTS * 2 + NUM_KEYPOINTS  # 34 קואורדינטות + 17 ביטחונות = 51

DEFAULT_CONF_THRESHOLD = 0.1


def normalize_keypoints(kpts_xy, kpts_conf, conf_threshold=DEFAULT_CONF_THRESHOLD):
    """
    kpts_xy:   מערך בגודל (17, 2) עם קואורדינטות x,y גולמיות (פיקסלים)
    kpts_conf: מערך בגודל (17,) עם ביטחון לכל נקודה

    מחזיר וקטור פיצ'רים באורך 51 (float32), מוכן להזנה למודל הסיווג.
    """
    kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)
    kpts_conf = np.asarray(kpts_conf, dtype=np.float32).reshape(NUM_KEYPOINTS)

    hip_center = (kpts_xy[LEFT_HIP] + kpts_xy[RIGHT_HIP]) / 2.0
    shoulder_center = (kpts_xy[LEFT_SHOULDER] + kpts_xy[RIGHT_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(shoulder_center - hip_center))

    if scale < 1e-3:
        # פאלבק: אם לא ניתן לחשב אורך פלג-עליון (למשל כתפיים/אגן לא זוהו טוב),
        # משתמשים בטווח הכללי של הנקודות התקינות כאומדן גודל.
        valid = kpts_conf >= conf_threshold
        if valid.sum() >= 2:
            pts = kpts_xy[valid]
            scale = float(max(pts[:, 0].max() - pts[:, 0].min(),
                               pts[:, 1].max() - pts[:, 1].min()))
        if scale < 1e-3:
            scale = 1.0

    normalized = (kpts_xy - hip_center) / scale

    low_conf_mask = kpts_conf < conf_threshold
    normalized[low_conf_mask] = 0.0

    features = np.concatenate([normalized.flatten(), kpts_conf])
    return features.astype(np.float32)


def feature_names():
    """שמות העמודות של וקטור הפיצ'רים - שימושי לדיווח feature importance."""
    names = []
    for kp in KEYPOINT_NAMES:
        names += [f"{kp}_x_norm", f"{kp}_y_norm"]
    for kp in KEYPOINT_NAMES:
        names.append(f"{kp}_conf")
    return names


def extract_from_yolo_result(result, person_index=0):
    """
    נוחות: מקבל אובייקט Results בודד מ-ultralytics YOLO-Pose ואינדקס של אדם
    (ברירת מחדל - האדם הראשון/היחיד), ומחזיר את וקטור הפיצ'רים המנורמל שלו.
    """
    kpts_xy = result.keypoints.xy[person_index].cpu().numpy()
    kpts_conf = result.keypoints.conf[person_index].cpu().numpy()
    return normalize_keypoints(kpts_xy, kpts_conf)
