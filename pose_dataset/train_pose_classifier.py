"""
train_pose_classifier.py
=========================
Trains a pose classification model (standing / sitting / lying) from COCO keypoints
saved in labels.csv by pose_labeler.py.

The model is a StandardScaler + RandomForestClassifier pipeline from scikit-learn.
It is not a neural network; it is small, fast to train (seconds), and strong enough
for this problem (classifying pose from 51 geometric features). It also works well with
small datasets (tens to hundreds of samples).

Run:
    python train_pose_classifier.py

Output:
    pose_classifier.joblib  <- trained model (loaded by webcam_pose_tester.py)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pose_features import KEYPOINT_NAMES, normalize_keypoints, feature_names

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "labels.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_classifier.joblib")

MIN_SAMPLES_RECOMMENDED = 60  # roughly 20 samples per pose, as a guideline


def load_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} was not found. Run pose_labeler.py first to label images.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if len(df) == 0:
        print("Error: labels.csv is empty - no images have been labeled yet.")
        sys.exit(1)

    X = []
    y = []
    for _, row in df.iterrows():
        kpts_xy = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
        kpts_conf = np.zeros((len(KEYPOINT_NAMES),), dtype=np.float32)
        for i, name in enumerate(KEYPOINT_NAMES):
            kpts_xy[i, 0] = row[f"{name}_x"]
            kpts_xy[i, 1] = row[f"{name}_y"]
            kpts_conf[i] = row[f"{name}_conf"]

        features = normalize_keypoints(kpts_xy, kpts_conf)
        X.append(features)
        y.append(row["label"])

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    print("Loading dataset from labels.csv ...")
    X, y = load_dataset()
    n_samples = len(X)
    print(f"Loaded {n_samples} labeled samples.")

    unique, counts = np.unique(y, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

    if n_samples < MIN_SAMPLES_RECOMMENDED:
        print(
            f"\n⚠ Warning: only {n_samples} samples are available (recommended at least {MIN_SAMPLES_RECOMMENDED}, "
            "about 20 per pose). The model may be less accurate. You can continue labeling more images "
            "and run this script again."
        )
    if len(unique) < 3:
        print(
            f"\n⚠ Warning: only {len(unique)} unique labels are available out of 3 possible labels "
            "(standing/sitting/lying). Make sure you label examples for all classes."
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # If there are very few samples, a split that is too small may fail; guard against that.
    test_size = 0.2 if n_samples >= 20 else max(1 / n_samples, 0.1)

    stratify = y_encoded if min(counts) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=stratify
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on the test set: {acc:.2%}")
    print("\nDetailed classification report:")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(f"Label order: {list(label_encoder.classes_)}")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation on the full dataset for a more stable estimate if enough samples exist.
    if n_samples >= 30 and min(counts) >= 5:
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=5)
        print(f"\n5-fold cross validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    # Save the full model bundle (including the scaler and label encoder) in one file.
    bundle = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "keypoint_names": KEYPOINT_NAMES,
        "feature_names": feature_names(),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print("You can now run: python webcam_pose_tester.py")


if __name__ == "__main__":
    main()
