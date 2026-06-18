"""Stream nursery audio from the Raspberry Pi USB microphone over HTTP."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
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


def generate_audio_stream() -> Iterator[bytes]:
    """Yield PCM chunks from the USB microphone."""
    device = get_capture_device()
    cmd = _arecord_command(device)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdout is not None

    logger.info(
        "Audio stream started (device=%s, rate=%s, channels=%s)",
        device,
        SAMPLE_RATE,
        CHANNELS,
    )

    try:
        while True:
            chunk = proc.stdout.read(CHUNK_SIZE)
            if not chunk:
                stderr = ""
                if proc.stderr is not None:
                    stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if proc.poll() is not None:
                    raise RuntimeError(stderr or "Microphone stream ended unexpectedly")
                continue
            yield chunk
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
        logger.info("Audio stream stopped")
