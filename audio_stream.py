"""Stream nursery audio from the Raspberry Pi USB microphone over HTTP."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Generator, Iterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # S16_LE
CHUNK_SIZE = 2048
AUDIO_DEVICE = os.environ.get("BABYMONITOR_AUDIO_DEVICE", "default").strip() or "default"


def get_audio_info() -> dict:
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "format": "S16_LE",
        "device": AUDIO_DEVICE,
    }


def _arecord_command() -> list[str]:
    return [
        "arecord",
        "-D",
        AUDIO_DEVICE,
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
    cmd = _arecord_command()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdout is not None

    logger.info(
        "Audio stream started (device=%s, rate=%s, channels=%s)",
        AUDIO_DEVICE,
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
