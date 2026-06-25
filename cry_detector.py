"""Baby cry detection using YAMNet TFLite on nursery PCM audio."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
PATCH_SAMPLES = 15600  # ~0.975 s window required by YAMNet
PATCH_BYTES = PATCH_SAMPLES * 2

# AudioSet / YAMNet class indices
CLASS_BABY_CRY = 20
CLASS_CRYING = 19
CRY_CLASSES = (CLASS_BABY_CRY, CLASS_CRYING)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "yamnet.tflite"
CLASS_MAP_PATH = Path(__file__).resolve().parent / "models" / "yamnet_class_map.csv"
SCORE_THRESHOLD = 0.3
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


def _predict_scores(waveform: np.ndarray) -> np.ndarray:
    if _interpreter is None or len(waveform) < PATCH_SAMPLES:
        return np.zeros(521, dtype=np.float32)

    patch = waveform[:PATCH_SAMPLES]
    tensor = patch.reshape(1, -1) if _input_2d else patch
    _interpreter.set_tensor(_input_index, tensor)
    _interpreter.invoke()
    scores = _interpreter.get_tensor(_output_index)

    if scores.ndim == 2:
        frame = scores[0]
    else:
        frame = scores
    return np.asarray(frame, dtype=np.float32)


def _predict_cry_score(waveform: np.ndarray) -> float:
    frame = _predict_scores(waveform)
    if frame.size == 0:
        return 0.0
    return float(max(frame[i] for i in CRY_CLASSES if i < len(frame)))


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
    global _high_hits, _score, _label

    _score = score

    if score >= SCORE_THRESHOLD:
        _high_hits += 1
        if _high_hits >= CONFIRM_HITS:
            _set_state(True, score, "baby_cry", now)
        else:
            _label = "possible_cry"
        return

    _high_hits = 0
    if _crying:
        if score <= CLEAR_THRESHOLD and (now - (_since or now)) >= CLEAR_SEC:
            _set_state(False, score, "quiet", now)
        else:
            _label = "baby_cry"
    else:
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
            "updated_at": _last_detect_ts,
            "threshold": SCORE_THRESHOLD,
            "model": str(MODEL_PATH.name) if MODEL_PATH.is_file() else None,
        }


def init_cry_detector() -> bool:
    """Load YAMNet if the model file is present."""
    return _load_interpreter()


_class_names: Optional[List[str]] = None


def load_class_names() -> List[str]:
    global _class_names
    if _class_names is not None:
        return _class_names

    names: List[str] = []
    if CLASS_MAP_PATH.is_file():
        for line in CLASS_MAP_PATH.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.strip().split(",", 2)
            if len(parts) == 3:
                names.append(parts[2].strip('"'))
    if not names:
        names = [f"class_{i}" for i in range(521)]
    _class_names = names
    return names


def analyze_waveform(waveform: np.ndarray, top_n: int = 8) -> Dict:
    """Return top YAMNet classes and cry scores for one audio window."""
    if not _load_interpreter():
        raise RuntimeError(
            f"YAMNet not available — install ai-edge-litert and run scripts/download_yamnet.py"
        )

    scores = _predict_scores(waveform)
    names = load_class_names()
    limit = min(len(scores), len(names))
    ranked = sorted(
        ((int(i), float(scores[i]), names[i]) for i in range(limit)),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    baby_cry = float(scores[CLASS_BABY_CRY]) if CLASS_BABY_CRY < len(scores) else 0.0
    crying = float(scores[CLASS_CRYING]) if CLASS_CRYING < len(scores) else 0.0
    cry_score = max(baby_cry, crying)

    return {
        "top": [{"index": i, "name": name, "score": round(score, 4)} for i, score, name in ranked],
        "baby_cry_score": round(baby_cry, 4),
        "crying_score": round(crying, 4),
        "cry_score": round(cry_score, 4),
        "would_alert": cry_score >= SCORE_THRESHOLD,
        "threshold": SCORE_THRESHOLD,
    }


def analyze_pcm_window(pcm_bytes: bytes, top_n: int = 8) -> Dict:
    waveform = _pcm_to_waveform(pcm_bytes)
    if len(waveform) < PATCH_SAMPLES:
        raise ValueError(f"Need at least {PATCH_SAMPLES} samples, got {len(waveform)}")
    return analyze_waveform(waveform[:PATCH_SAMPLES], top_n=top_n)


def print_analysis(result: Dict) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"\n[{stamp}] sound analysis")
    for item in result["top"]:
        marker = "  *" if item["index"] in CRY_CLASSES else ""
        print(f"  {item['score']:.3f}  [{item['index']:>3}] {item['name']}{marker}")
    alert = "YES" if result["would_alert"] else "no"
    print(
        f"  cry score={result['cry_score']:.3f} "
        f"(baby={result['baby_cry_score']:.3f}, sob={result['crying_score']:.3f}) "
        f"alert={alert} threshold={result['threshold']}"
    )


def _run_live_demo(device: str, top_n: int, seconds: float) -> int:
    import subprocess

    from audio_stream import detect_capture_device

    if not init_cry_detector():
        print("YAMNet failed to load.", flush=True)
        return 1

    dev = device or detect_capture_device()
    cmd = [
        "arecord",
        "-D",
        dev,
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        "1",
        "-t",
        "raw",
        "-q",
        "-",
    ]
    print(f"Listening on {dev} — Ctrl+C to stop", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = bytearray()
    started = time.time()
    try:
        assert proc.stdout is not None
        while proc.stdout.readable():
            if seconds > 0 and (time.time() - started) >= seconds:
                break
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            buffer.extend(chunk)
            while len(buffer) >= PATCH_BYTES:
                window = bytes(buffer[:PATCH_BYTES])
                del buffer[:PATCH_BYTES]
                result = analyze_pcm_window(window, top_n=top_n)
                print_analysis(result)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=2)
    return 0


def _run_file_demo(path: Path, top_n: int) -> int:
    import wave

    if not init_cry_detector():
        print("YAMNet failed to load.", flush=True)
        return 1

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        width = wf.getsampwidth()
        if channels != 1 or rate != SAMPLE_RATE or width != 2:
            print(
                f"Expected mono 16 kHz 16-bit WAV, got "
                f"{channels}ch {rate}Hz {width * 8}-bit",
                flush=True,
            )
            return 1
        pcm = wf.readframes(wf.getnframes())

    offset = 0
    window_idx = 0
    while offset + PATCH_BYTES <= len(pcm):
        window = pcm[offset : offset + PATCH_BYTES]
        offset += PATCH_BYTES
        window_idx += 1
        result = analyze_pcm_window(window, top_n=top_n)
        print(f"--- window {window_idx} ---")
        print_analysis(result)
    return 0


def demo_main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug YAMNet baby cry detection on live mic or WAV file.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Analyze live microphone (default if no --file)",
    )
    parser.add_argument("--file", type=Path, help="Analyze a mono 16 kHz 16-bit WAV")
    parser.add_argument(
        "--device",
        default="",
        help="ALSA device for --live (default: from device_config / auto-detect)",
    )
    parser.add_argument("--top", type=int, default=8, help="How many classes to print")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Stop after N seconds in live mode (0 = until Ctrl+C)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.file:
        return _run_file_demo(args.file, top_n=args.top)
    return _run_live_demo(device=args.device, top_n=args.top, seconds=args.seconds)


if __name__ == "__main__":
    raise SystemExit(demo_main())
