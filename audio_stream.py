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
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096
SUBSCRIBER_QUEUE_SIZE = 160
CONFIG_PATH = Path(__file__).resolve().parent / "device_config.json"

_resolved_device: str | None = None


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


def get_audio_info() -> dict:
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "format": "S16_LE",
        "device": get_capture_device(),
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
        "--period-size=1024",
        "--buffer-size=4096",
        "-q",
        "-",
    ]


class AudioCaptureHub:
    """One arecord process shared by all listeners; keeps running while server is up."""

    _instance: "AudioCaptureHub | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._hub_lock = threading.Lock()
        self._subscribers: list[queue.Queue[bytes]] = []
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._active = False

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
            self._reader_thread = threading.Thread(
                target=self._capture_loop,
                name="audio-capture",
                daemon=True,
            )
            self._reader_thread.start()
            logger.info("Audio capture hub started")

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
            self._kill_proc()
            logger.info("Audio capture hub stopped")

    def _capture_loop(self) -> None:
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
                    chunk = proc.stdout.read(CHUNK_SIZE)
                    if not chunk:
                        stderr = ""
                        if proc.stderr is not None:
                            stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                        logger.warning("Microphone capture ended: %s", stderr or "no data")
                        break
                    with self._hub_lock:
                        subscribers = list(self._subscribers)
                    for subscriber in subscribers:
                        subscriber.put(chunk, block=True)
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
                "Audio capture started (device=%s, rate=%s, channels=%s)",
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
