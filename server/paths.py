"""Central path constants for the Smart Baby Monitor server."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = CONFIG_DIR / "device_config.json"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-pose.pt"
POSE_CLASSIFIER_PATH = MODELS_DIR / "pose_classifier.joblib"
POSE_CLASSIFIER_FALLBACK_PATH = PROJECT_ROOT / "pose_dataset" / "pose_classifier.joblib"
SLEEP_DB_PATH = DATA_DIR / "sleep_data.db"
YAMNET_MODEL_PATH = MODELS_DIR / "yamnet.tflite"
YAMNET_CLASS_MAP_PATH = MODELS_DIR / "yamnet_class_map.csv"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
