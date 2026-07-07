"""Stream nursery audio from the Raspberry Pi USB microphone over HTTP."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from typing import Iterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
# 4096 frames ≈ 256 ms at 16 kHz; larger buffers reduce crackling on Pi (see PortAudio/PyAudio notes).
CHUNK_SIZE = 4096
CAPTURE_QUEUE_SIZE = 32
SUBSCRIBER_QUEUE_SIZE = 32
from server.paths import CONFIG_PATH

_resolved_device: str | None = None
_resolved_device_index: int | None = None

try:
    import pyaudio

    _PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None  # type: ignore[assignment]
    _PYAUDIO_AVAILABLE = False


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def detect_capture_device() -> str:
    env = os.environ.get("BABYMONITOR_AUDIO_DEVICE", "").strip()
    if env:
        return env

    configured = str(_load_config().get("audio_device", "") or "").strip()
    if configured:
        return configured

    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "card" not in line:
                    continue
                match = re.search(r"card (\d+):", line)
                if match and "usb" in line.lower():
                    return f"plughw:{match.group(1)},0"

            cards = re.findall(r"card (\d+):", result.stdout)
            if cards:
                return f"plughw:{cards[-1]},0"
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not auto-detect microphone: %s", exc)

    return "plughw:2,0"


def get_capture_device() -> str:
    global _resolved_device
    if _resolved_device is None:
        _resolved_device = detect_capture_device()
        logger.info("Using microphone device: %s", _resolved_device)
    return _resolved_device


def _card_number_from_device(device: str) -> int | None:
    match = re.search(r":(\d+)", device)
    return int(match.group(1)) if match else None


def _resolve_pyaudio_device_index() -> int | None:
    global _resolved_device_index
    if _resolved_device_index is not None:
        return _resolved_device_index

    config = _load_config()
    configured_index = config.get("audio_device_index")
    if configured_index is not None:
        _resolved_device_index = int(configured_index)
        logger.info("Using configured PyAudio device index: %s", _resolved_device_index)
        return _resolved_device_index

    if not _PYAUDIO_AVAILABLE:
        return None

    card_num = _card_number_from_device(get_capture_device())
    pa = pyaudio.PyAudio()
    try:
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) < 1:
                continue
            name = str(info.get("name", "")).lower()
            if card_num is not None and (
                f"hw:{card_num}," in name
                or f"card {card_num}" in name
                or f":{card_num}," in name
            ):
                _resolved_device_index = index
                logger.info(
                    "PyAudio input device %s: %s",
                    index,
                    info.get("name"),
                )
                return _resolved_device_index
            if card_num is None and "usb" in name:
                _resolved_device_index = index
                logger.info(
                    "PyAudio USB input device %s: %s",
                    index,
                    info.get("name"),
                )
                return _resolved_device_index
    finally:
        pa.terminate()

    logger.warning("Could not map %s to a PyAudio device index", get_capture_device())
    return None


def get_audio_info() -> dict:
    backend = "pyaudio" if _PYAUDIO_AVAILABLE and _resolve_pyaudio_device_index() is not None else "arecord"
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "format": "S16_LE",
        "device": get_capture_device(),
        "backend": backend,
        "url_path": "/audio_feed",
    }


def _arecord_command(device: str) -> list[str]:
    return [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        str(CHANNELS),
        "-t",
        "raw",
        "--period-size=4096",
        "--buffer-size=16384",
        "-q",
        "-",
    ]


def _enqueue_chunk(capture_queue: queue.Queue[bytes], chunk: bytes) -> None:
    try:
        capture_queue.put_nowait(chunk)
    except queue.Full:
        try:
            capture_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            capture_queue.put_nowait(chunk)
        except queue.Full:
            pass


def _fanout_chunk(subscribers: list[queue.Queue[bytes]], chunk: bytes) -> None:
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(chunk)
        except queue.Full:
            try:
                subscriber.get_nowait()
            except queue.Empty:
                pass
            try:
                subscriber.put_nowait(chunk)
            except queue.Full:
                pass

    try:
        from server.audio.cry_detector import feed_pcm

        feed_pcm(chunk)
    except Exception as exc:
        logger.debug("Cry detector feed failed: %s", exc)

    try:
        from server.audio.sound_meter import feed_pcm as feed_sound

        feed_sound(chunk)
    except Exception as exc:
        logger.debug("Sound meter feed failed: %s", exc)


class AudioCaptureHub:
    """Single mic capture shared by all listeners; capture never waits on network."""

    _instance: "AudioCaptureHub | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._hub_lock = threading.Lock()
        self._subscribers: list[queue.Queue[bytes]] = []
        self._capture_queue: queue.Queue[bytes] = queue.Queue(maxsize=CAPTURE_QUEUE_SIZE)
        self._proc: subprocess.Popen | None = None
        self._pyaudio: object | None = None
        self._pyaudio_stream: object | None = None
        self._capture_thread: threading.Thread | None = None
        self._distributor_thread: threading.Thread | None = None
        self._active = False
        self._backend = "none"

    @classmethod
    def get(cls) -> "AudioCaptureHub":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def ensure_running(self) -> None:
        with self._hub_lock:
            if self._active:
                return
            self._active = True
            self._distributor_thread = threading.Thread(
                target=self._distributor_loop,
                name="audio-distributor",
                daemon=True,
            )
            self._distributor_thread.start()

            if self._start_pyaudio_capture():
                self._backend = "pyaudio"
            else:
                self._backend = "arecord"
                self._capture_thread = threading.Thread(
                    target=self._arecord_loop,
                    name="audio-arecord",
                    daemon=True,
                )
                self._capture_thread.start()

            logger.info("Audio capture hub started (backend=%s)", self._backend)

    def subscribe(self) -> queue.Queue[bytes]:
        self.ensure_running()
        subscriber: queue.Queue[bytes] = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        with self._hub_lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[bytes]) -> None:
        with self._hub_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def shutdown(self) -> None:
        with self._hub_lock:
            self._active = False
            self._stop_pyaudio_capture()
            self._kill_proc()
            logger.info("Audio capture hub stopped")

    def _start_pyaudio_capture(self) -> bool:
        if not _PYAUDIO_AVAILABLE:
            return False

        device_index = _resolve_pyaudio_device_index()
        if device_index is None:
            return False

        hub = self

        def callback(in_data, frame_count, time_info, status):  # noqa: ARG001
            if hub._active and in_data:
                _enqueue_chunk(hub._capture_queue, bytes(in_data))
            return (None, pyaudio.paContinue)

        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=callback,
            )
            stream.start_stream()
            self._pyaudio = pa
            self._pyaudio_stream = stream
            logger.info(
                "PyAudio capture started (device_index=%s, rate=%s, chunk=%s)",
                device_index,
                SAMPLE_RATE,
                CHUNK_SIZE,
            )
            return True
        except Exception as exc:
            logger.warning("PyAudio capture unavailable, falling back to arecord: %s", exc)
            self._stop_pyaudio_capture()
            return False

    def _stop_pyaudio_capture(self) -> None:
        stream = self._pyaudio_stream
        pa = self._pyaudio
        self._pyaudio_stream = None
        self._pyaudio = None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass

    def _distributor_loop(self) -> None:
        while True:
            with self._hub_lock:
                if not self._active:
                    break
            try:
                chunk = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._hub_lock:
                subscribers = list(self._subscribers)
            _fanout_chunk(subscribers, chunk)

    def _arecord_loop(self) -> None:
        while True:
            with self._hub_lock:
                if not self._active:
                    break
                proc = self._open_proc()
                if proc is None:
                    time.sleep(1.0)
                    continue

            assert proc.stdout is not None
            try:
                while self._active:
                    chunk = proc.stdout.read(CHUNK_SIZE * 2)
                    if not chunk:
                        stderr = ""
                        if proc.stderr is not None:
                            stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                        logger.warning("Microphone capture ended: %s", stderr or "no data")
                        break
                    _enqueue_chunk(self._capture_queue, chunk)
            finally:
                with self._hub_lock:
                    self._kill_proc()

            if not self._active:
                break
            time.sleep(0.5)

    def _open_proc(self) -> subprocess.Popen | None:
        device = get_capture_device()
        try:
            self._proc = subprocess.Popen(
                _arecord_command(device),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            logger.info(
                "arecord capture started (device=%s, rate=%s, channels=%s)",
                device,
                SAMPLE_RATE,
                CHANNELS,
            )
            return self._proc
        except OSError as exc:
            logger.error("Failed to start arecord: %s", exc)
            self._proc = None
            return None

    def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)


def generate_audio_stream() -> Iterator[bytes]:
    hub = AudioCaptureHub.get()
    subscriber = hub.subscribe()
    try:
        while True:
            chunk = subscriber.get()
            yield chunk
    finally:
        hub.unsubscribe(subscriber)


def start_capture_hub() -> None:
    AudioCaptureHub.get().ensure_running()


def shutdown_capture_hub() -> None:
    AudioCaptureHub.get().shutdown()
