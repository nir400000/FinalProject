"""Baby cry detection using YAMNet TFLite on nursery PCM audio."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
PATCH_SAMPLES = 15600  # ~0.975 s window required by YAMNet
PATCH_BYTES = PATCH_SAMPLES * 2

# AudioSet / YAMNet class indices
CLASS_BABY_CRY = 20
CLASS_CRYING = 21
CRY_CLASSES = (CLASS_BABY_CRY, CLASS_CRYING)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yamnet.tflite"
SCORE_THRESHOLD = 0.45
CLEAR_THRESHOLD = 0.30
CONFIRM_HITS = 2
CLEAR_SEC = 5.0

_lock = threading.Lock()
_buffer = bytearray()
_interpreter = None
_input_index = 0
_output_index = 0
_input_2d = False
_available = False

_crying = False
_score = 0.0
_label = "idle"
_since: Optional[float] = None
_last_detect_ts = 0.0
_high_hits = 0


def _load_interpreter() -> bool:
    global _interpreter, _input_index, _output_index, _input_2d, _available

    if _interpreter is not None:
        return _available
    if not MODEL_PATH.is_file():
        logger.warning(
            "YAMNet model not found at %s — run scripts/download_yamnet.py",
            MODEL_PATH,
        )
        _available = False
        return False

    try:
        Interpreter = None
        backend = "unknown"
        try:
            from ai_edge_litert.interpreter import Interpreter

            backend = "ai-edge-litert"
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter

                backend = "tflite-runtime"
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

                backend = "tensorflow"

        interpreter = Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        _interpreter = interpreter
        _input_index = int(input_details[0]["index"])
        _output_index = int(output_details[0]["index"])
        shape = input_details[0].get("shape")
        _input_2d = bool(shape is not None and len(shape) == 2)
        _available = True
        logger.info("YAMNet cry detector loaded (%s via %s)", MODEL_PATH.name, backend)
        return True
    except Exception as exc:
        logger.warning("YAMNet cry detector unavailable: %s", exc)
        _available = False
        return False


def _pcm_to_waveform(pcm_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def _predict_cry_score(waveform: np.ndarray) -> float:
    if _interpreter is None or len(waveform) < PATCH_SAMPLES:
        return 0.0

    patch = waveform[:PATCH_SAMPLES]
    tensor = patch.reshape(1, -1) if _input_2d else patch
    _interpreter.set_tensor(_input_index, tensor)
    _interpreter.invoke()
    scores = _interpreter.get_tensor(_output_index)

    if scores.ndim == 2:
        frame = scores[0]
    else:
        frame = scores

    return float(max(frame[i] for i in CRY_CLASSES))


def _set_state(crying: bool, score: float, label: str, now: float) -> None:
    global _crying, _score, _label, _since, _last_detect_ts, _high_hits

    _score = score
    _label = label
    _last_detect_ts = now

    if crying and not _crying:
        _crying = True
        _since = now
        logger.info("Baby cry detected (score=%.2f)", score)
    elif not crying and _crying:
        _crying = False
        _since = None
        _high_hits = 0
        logger.info("Baby cry cleared (score=%.2f)", score)


def _update_state(score: float, now: float) -> None:
    global _high_hits

    if score >= SCORE_THRESHOLD:
        _high_hits += 1
        if _high_hits >= CONFIRM_HITS:
            _set_state(True, score, "baby_cry", now)
        return

    _high_hits = 0
    if _crying:
        if score <= CLEAR_THRESHOLD and (now - (_since or now)) >= CLEAR_SEC:
            _set_state(False, score, "quiet", now)
        else:
            _score = score
            _label = "baby_cry"
    else:
        _score = score
        _label = "quiet"


def feed_pcm(chunk: bytes) -> None:
    if not chunk:
        return
    if not _load_interpreter():
        return

    with _lock:
        _buffer.extend(chunk)
        while len(_buffer) >= PATCH_BYTES:
            window = bytes(_buffer[:PATCH_BYTES])
            del _buffer[:PATCH_BYTES]
            waveform = _pcm_to_waveform(window)
            score = _predict_cry_score(waveform)
            _update_state(score, time.time())


def get_status() -> Dict:
    with _lock:
        return {
            "available": _available,
            "crying": _crying,
            "score": round(_score, 3),
            "label": _label,
            "since": _since,
            "threshold": SCORE_THRESHOLD,
            "model": str(MODEL_PATH.name) if MODEL_PATH.is_file() else None,
        }


def init_cry_detector() -> bool:
    """Load YAMNet if the model file is present."""
    return _load_interpreter()
