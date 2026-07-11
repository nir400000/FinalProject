"""
pose_labeler.py
================
כלי גרפי (GUI) לתיוג תנוחות גוף על גבי תמונות, בעזרת מודל YOLOv8-Pose.

מה הכלי עושה:
- טוען את כל התמונות מתוך dataset/images/
- מריץ על כל תמונה את מודל היחס (Pose) של YOLOv8 כדי לזהות שלד (17 נקודות מפתח בפורמט COCO)
- מציג את התמונה עם השלד המצויר עליה
- מאפשר לך ללחוץ על אחד מכפתורי התנוחה: עומד / יושב / שוכב
- שומר שורה חדשה לקובץ dataset/labels.csv עם: שם הקובץ, התנוחה שנבחרה, וכל 17 נקודות המפתח (x, y, confidence)
- אם המודל זיהה יותר מבן אדם אחד (או אף אחד) בתמונה - התמונה מסומנת כ"בעייתית"
  וכפתורי התיוג ננעלים (כדי לא לשמור נתונים שגויים), ואפשר רק לדלג הלאה.
- דילוג על תמונה שלא תויגה (עם אישור בפופ-אפ) מוחק את קובץ התמונה לצמיתות
  מתיקיית images/, כך שרק תמונות שתויגו בפועל נשארות בסט הנתונים.

הרצה:
    pip install -r requirements.txt
    python pose_labeler.py

מבנה תיקיות מצופה:
    dataset/
    ├── images/
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   └── ...
    ├── labels.csv      <- נוצר אוטומטית תוך כדי העבודה
    └── pose_labeler.py
"""

import os
import csv
import sys
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


# ----------------------------------------------------------------------
# הגדרות כלליות
# ----------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
CSV_PATH = os.path.join(SCRIPT_DIR, "labels.csv")

# משקלות המודל. yolov8n-pose.pt קטן ומהיר, ניתן להחליף ל-yolov8s/m/l/x-pose.pt לדיוק גבוה יותר.
MODEL_WEIGHTS = "yolov8n-pose.pt"

# רק תיבות עם ביטחון (confidence) מעל הסף הזה ייחשבו כזיהוי "אדם" תקין.
PERSON_CONF_THRESHOLD = 0.5

MAX_DISPLAY_SIZE = 720  # גודל תצוגה מקסימלי (פיקסלים) בציר הארוך

# שמות 17 נקודות המפתח בפורמט COCO (הסדר קבוע ע"י YOLOv8-Pose)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

POSE_LABELS = [
    ("עומד", "standing"),
    ("יושב", "sitting"),
    ("שוכב", "lying"),
]

CSV_HEADER = ["image_name", "label", "num_persons_detected"]
for name in KEYPOINT_NAMES:
    CSV_HEADER += [f"{name}_x", f"{name}_y", f"{name}_conf"]


def build_image_list():
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.isdir(IMAGES_DIR):
        return []
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(valid_ext)]
    files.sort()
    return files


def load_existing_labels():
    """קורא labels.csv קיים (אם יש) ומחזיר set של תמונות שכבר תויגו."""
    labeled = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["image_name"])
    return labeled


class PoseLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("תיוג תנוחות גוף - YOLOv8 Pose")
        self.root.geometry("900x820")

        self.images = build_image_list()
        if not self.images:
            messagebox.showerror(
                "שגיאה",
                f"לא נמצאו תמונות בתיקייה:\n{IMAGES_DIR}\n\n"
                "ודא שהתמונות נמצאות בתיקיית dataset/images/",
            )
            root.destroy()
            sys.exit(1)

        self.labeled_set = load_existing_labels()
        self._ensure_csv_exists()

        print("טוען מודל YOLOv8-Pose... (בפעם הראשונה ירד אוטומטית)")
        self.model = YOLO(MODEL_WEIGHTS)

        self.index = 0
        # מתחילים מהתמונה הראשונה שעדיין לא תויגה, אם יש כזו
        for i, name in enumerate(self.images):
            if name not in self.labeled_set:
                self.index = i
                break

        self.current_result = None  # תוצאת המודל לתמונה הנוכחית
        self.current_num_persons = 0
        self.current_display_img = None  # שומר רפרנס ל-ImageTk כדי שלא יימחק ע"י ה-GC

        self._build_ui()
        self._load_current_image()

    # ------------------------------------------------------------------
    # בניית ממשק המשתמש
    # ------------------------------------------------------------------
    def _build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=8)

        self.status_label = tk.Label(top_frame, text="", font=("Arial", 12))
        self.status_label.pack()

        self.canvas = tk.Label(self.root)
        self.canvas.pack(pady=8)

        self.warning_label = tk.Label(
            self.root, text="", font=("Arial", 12, "bold"), fg="red"
        )
        self.warning_label.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        self.pose_buttons = []
        for i, (heb_label, eng_label) in enumerate(POSE_LABELS):
            b = tk.Button(
                btn_frame,
                text=heb_label,
                font=("Arial", 16),
                width=10,
                height=2,
                command=lambda e=eng_label: self.on_label_selected(e),
            )
            b.grid(row=0, column=i, padx=10)
            self.pose_buttons.append(b)

        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)

        tk.Button(
            nav_frame, text="⬅ הקודם", font=("Arial", 12), width=12,
            command=self.on_previous,
        ).grid(row=0, column=0, padx=8)

        tk.Button(
            nav_frame, text="דלג ומחק ⏭", font=("Arial", 12), width=14,
            command=self.on_skip,
        ).grid(row=0, column=1, padx=8)

        # קיצורי מקלדת: 1=עומד, 2=יושב, 3=שוכב, מקש שמאלה/ימינה לניווט
        self.root.bind("1", lambda e: self.on_label_selected("standing"))
        self.root.bind("2", lambda e: self.on_label_selected("sitting"))
        self.root.bind("3", lambda e: self.on_label_selected("lying"))
        self.root.bind("<Left>", lambda e: self.on_previous())
        self.root.bind("<Right>", lambda e: self.on_skip())

    # ------------------------------------------------------------------
    # csv
    # ------------------------------------------------------------------
    def _ensure_csv_exists(self):
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

    def _append_row(self, row):
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ------------------------------------------------------------------
    # טעינה והצגה של תמונה נוכחית
    # ------------------------------------------------------------------
    def _load_current_image(self):
        if self.index >= len(self.images):
            messagebox.showinfo("סיום", "כל התמונות תויגו/נסקרו! הקובץ labels.csv מוכן.")
            self.root.quit()
            return

        image_name = self.images[self.index]
        image_path = os.path.join(IMAGES_DIR, image_name)

        results = self.model(image_path, verbose=False)
        r = results[0]
        self.current_result = r

        num_persons = 0 if r.boxes is None else int(
            (r.boxes.conf >= PERSON_CONF_THRESHOLD).sum().item()
        )
        self.current_num_persons = num_persons

        # r.plot() מצייר עבורנו את השלד וה-bounding box, מחזיר תמונה בפורמט BGR
        annotated_bgr = r.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb)

        # שינוי גודל לתצוגה נוחה (בלי לשנות את קובץ המקור)
        w, h = pil_img.size
        scale = MAX_DISPLAY_SIZE / max(w, h)
        if scale < 1:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        self.current_display_img = ImageTk.PhotoImage(pil_img)
        self.canvas.configure(image=self.current_display_img)

        already = " (כבר תויגה בעבר - תישמר תגית חדשה) " if image_name in self.labeled_set else ""
        self.status_label.configure(
            text=f"תמונה {self.index + 1} / {len(self.images)}   |   {image_name}{already}"
        )

        if num_persons == 1:
            self.warning_label.configure(text="")
            for b in self.pose_buttons:
                b.configure(state=tk.NORMAL)
        else:
            self.warning_label.configure(
                text=f"⚠ זוהו {num_persons} אנשים בתמונה (נדרש בדיוק 1) - לא ניתן לתייג, יש לדלג"
            )
            for b in self.pose_buttons:
                b.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # שמירת תיוג
    # ------------------------------------------------------------------
    def on_label_selected(self, label_eng):
        if self.current_num_persons != 1:
            messagebox.showwarning(
                "לא ניתן לתייג",
                "יש לוודא שבתמונה מזוהה בדיוק אדם אחד לפני התיוג.",
            )
            return

        image_name = self.images[self.index]
        r = self.current_result

        # נקודות המפתח של האדם היחיד שזוהה (בהתאם ל-index עם הביטחון הגבוה ביותר)
        person_idx = int(r.boxes.conf.argmax().item())
        kpts_xy = r.keypoints.xy[person_idx].cpu().numpy()      # (17, 2)
        kpts_conf = r.keypoints.conf[person_idx].cpu().numpy()  # (17,)

        row = [image_name, label_eng, self.current_num_persons]
        for i in range(len(KEYPOINT_NAMES)):
            x, y = kpts_xy[i]
            conf = kpts_conf[i]
            row += [round(float(x), 2), round(float(y), 2), round(float(conf), 4)]

        self._append_row(row)
        self.labeled_set.add(image_name)

        self.index += 1
        self._load_current_image()

    def on_skip(self):
        image_name = self.images[self.index]
        image_path = os.path.join(IMAGES_DIR, image_name)

        confirm = messagebox.askyesno(
            "מחיקת תמונה",
            f"התמונה '{image_name}' לא תויגה.\n"
            "דילוג עליה ימחק אותה לצמיתות מתיקיית images.\n\n"
            "להמשיך?",
        )
        if not confirm:
            return

        try:
            os.remove(image_path)
        except OSError as e:
            messagebox.showerror("שגיאה במחיקה", f"לא הצלחתי למחוק את הקובץ:\n{e}")
            return

        # מסירים מהרשימה בזיכרון (לא מזיזים את האינדקס - התמונה הבאה "נכנסת" למקום שהתפנה)
        del self.images[self.index]
        self.labeled_set.discard(image_name)

        self._load_current_image()

    def on_previous(self):
        if self.index > 0:
            self.index -= 1
            self._load_current_image()


def main():
    root = tk.Tk()
    app = PoseLabelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
