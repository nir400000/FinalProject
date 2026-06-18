"""Stream nursery audio from the Raspberry Pi USB microphone over HTTP."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 2048
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
    """Resolve ALSA capture device for the USB microphone."""
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
        "-q",
        "-",
    ]


class AudioCaptureHub:
    """Single arecord process shared by all /audio_feed clients."""

    _instance: "AudioCaptureHub | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._hub_lock = threading.Lock()
        self._subscribers: list[queue.Queue[bytes | None]] = []
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._running = False

    @classmethod
    def get(cls) -> "AudioCaptureHub":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def subscribe(self) -> queue.Queue[bytes | None]:
        subscriber: queue.Queue[bytes | None] = queue.Queue(maxsize=48)
        with self._hub_lock:
            self._subscribers.append(subscriber)
            if not self._running:
                self._start_capture()
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[bytes | None]) -> None:
        with self._hub_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
            if not self._subscribers:
                self._stop_capture()

    def _start_capture(self) -> None:
        if self._running:
            return

        device = get_capture_device()
        cmd = _arecord_command(device)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            logger.error("Failed to start arecord: %s", exc)
            raise RuntimeError(str(exc)) from exc

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="audio-capture",
            daemon=True,
        )
        self._reader_thread.start()
        logger.info(
            "Audio capture started (device=%s, rate=%s, channels=%s)",
            device,
            SAMPLE_RATE,
            CHANNELS,
        )

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None

        try:
            while self._running:
                chunk = self._proc.stdout.read(CHUNK_SIZE)
                if not chunk:
                    stderr = ""
                    if self._proc.stderr is not None:
                        stderr = self._proc.stderr.read().decode("utf-8", errors="ignore").strip()
                    logger.warning("Microphone capture ended: %s", stderr or "no data")
                    break

                with self._hub_lock:
                    subscribers = list(self._subscribers)
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
        finally:
            with self._hub_lock:
                self._notify_end()
                self._cleanup_process()

    def _notify_end(self) -> None:
        with self._hub_lock:
            for subscriber in self._subscribers:
                try:
                    subscriber.put_nowait(None)
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                        subscriber.put_nowait(None)
                    except queue.Empty:
                        pass

    def _stop_capture(self) -> None:
        self._running = False
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
        self._cleanup_process()
        logger.info("Audio capture stopped")

    def _cleanup_process(self) -> None:
        self._proc = None
        self._reader_thread = None
        self._running = False


def generate_audio_stream() -> Iterator[bytes]:
    """Yield PCM chunks from the shared microphone capture."""
    hub = AudioCaptureHub.get()
    subscriber = hub.subscribe()
    try:
        while True:
            chunk = subscriber.get()
            if chunk is None:
                break
            yield chunk
    finally:
        hub.unsubscribe(subscriber)
